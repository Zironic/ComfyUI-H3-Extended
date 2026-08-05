"""Self-tests for the two-stage SM89 efficient Sage backend.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_sage_mem_eff.py

CPU tests use injected quantization and kernel stand-ins. A CUDA section checks
that deleting the source views makes the fused allocation disappear before
``execute`` starts; it skips cleanly without a GPU.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention import stats  # noqa: E402
from h3_attention.sage_mem_eff import (  # noqa: E402
    EfficientSageError,
    SageSM89API,
    SM89SageMemoryEfficientBackend,
    first_unsafe_v_length,
    guard_v_stride,
)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakeKernel:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def __call__(self, q8, k8, v8, out, q_scale, k_scale, v_scale,
                 layout, causal, granularity, scale, return_lse):
        self.calls += 1
        if self.fail:
            raise RuntimeError("synthetic kernel failure")
        check(layout == 1, "kernel receives HND layout flag")
        check(causal == 0 and granularity == 3 and return_lse == 0,
              "kernel receives non-causal per-thread flags")
        out.zero_()
        return None


def fake_quantizer(q, k, km=None, **kwargs):
    assert km is None
    q8 = torch.zeros(q.shape, dtype=torch.int8, device=q.device)
    k8 = torch.zeros(k.shape, dtype=torch.int8, device=k.device)
    b, h, s, _ = q.shape
    q_scale = torch.ones((b, h, ((s + 127) // 128) * 32), dtype=torch.float32, device=q.device)
    k_scale = torch.ones((b, h, ((s + 63) // 64) * 4), dtype=torch.float32, device=q.device)
    return q8, q_scale, k8, k_scale


def fake_v_prepare(v, tensor_layout, scale_max, smooth_v):
    assert tensor_layout == "HND"
    assert scale_max == 2.25 and smooth_v is False
    # Independent storage is the requirement; exact format belongs to Sage.
    v8 = torch.zeros(v.shape, dtype=torch.int8, device=v.device)
    scale = torch.ones((v.shape[0], v.shape[1], v.shape[3]), dtype=torch.float32, device=v.device)
    return v8, scale, None


def fused_hnd(sequence=16, heads=2, device="cpu"):
    inner = heads * 128
    fused = torch.randn(sequence, inner * 3, dtype=torch.bfloat16, device=device)
    q, k, v = fused.split(inner, dim=-1)
    return (
        q.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
        k.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
        v.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
    )


def build_backend(kernel=None, allow_cpu=True):
    kernel = kernel or FakeKernel()
    api = SageSM89API(
        version="2.2.test",
        per_channel_fp8=fake_v_prepare,
        kernel=kernel,
        kernel_name="fake_sm89",
    )
    return SM89SageMemoryEfficientBackend(
        api=api,
        quantizer=fake_quantizer,
        allow_cpu_for_tests=allow_cpu,
    ), kernel


def test_prepare_execute():
    print("prepare / execute")
    stats.reset_stats()
    backend, kernel = build_backend()
    q, k, v = fused_hnd()
    source_storage = q.untyped_storage().data_ptr()

    prepared = backend.prepare(q, k, v, layer_index=7, transformer_options={})
    check(prepared.q_int8.untyped_storage().data_ptr() != source_storage,
          "prepared Q does not retain fused storage")
    check(prepared.k_int8.untyped_storage().data_ptr() != source_storage,
          "prepared K does not retain fused storage")
    check(prepared.v_fp8.untyped_storage().data_ptr() != source_storage,
          "prepared V does not retain fused storage")
    check(prepared.layer_index == 7 and prepared.sequence == 16,
          "prepared metadata carries layer and sequence")

    del q, k, v
    out = backend.execute(prepared)
    check(out.shape == (1, 2, 16, 128), "execute returns HND output")
    check(out.dtype == torch.bfloat16, "output dtype matches source")
    check(kernel.calls == 1, "low-level kernel called exactly once")

    got = stats.get_stats()
    check(got["configured"] == 1 and got["prepared"] == 1 and got["executed"] == 1,
          "dispatch counters record the path")


def test_validation():
    print("validation")
    backend, _ = build_backend()
    q, k, v = fused_hnd()
    try:
        backend.prepare(q[..., :64], k[..., :64], v[..., :64],
                        layer_index=0, transformer_options={})
    except EfficientSageError as exc:
        check("head_dim 128" in str(exc), "wrong head dimension raises clearly")
    else:
        raise AssertionError("wrong head dimension must fail")

    q2 = q.expand(2, -1, -1, -1)
    try:
        backend.prepare(q2, k.expand_as(q2), v.expand_as(q2),
                        layer_index=0, transformer_options={})
    except EfficientSageError as exc:
        check("batch 1" in str(exc), "wrong attention batch raises clearly")
    else:
        raise AssertionError("wrong attention batch must fail")


def test_error_context():
    print("kernel error context")
    backend, _ = build_backend(kernel=FakeKernel(fail=True))
    q, k, v = fused_hnd(sequence=8)
    prepared = backend.prepare(q, k, v, layer_index=41, transformer_options={})
    del q, k, v
    try:
        backend.execute(prepared)
    except EfficientSageError as exc:
        text = str(exc)
        check("layer=41" in text and "sequence=8" in text and "fake_sm89" in text,
              "kernel failure reports layer, sequence, and symbol")
    else:
        raise AssertionError("kernel failure must not silently fall back")


def test_v_boundary():
    print("V offset boundary")
    check(first_unsafe_v_length() == 199730,
          "first unsafe H3 V sequence length is 199,730")
    q, _, v = fused_hnd(sequence=16)
    check(guard_v_stride(v) is v, "normal H3 V passes without a copy")
    del q, v


def test_cuda_lifetime():
    print("CUDA lifetime")
    if not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
        return
    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    q, k, v = fused_hnd(sequence=4096, heads=56, device=device)
    after_fused = torch.cuda.memory_allocated(device)

    backend, _ = build_backend(allow_cpu=True)
    prepared = backend.prepare(q, k, v, layer_index=0, transformer_options={})
    del q, k, v
    torch.cuda.synchronize(device)
    after_delete = torch.cuda.memory_allocated(device)
    check(after_delete < after_fused,
          "deleting source views releases the fused QKV allocation before execute")
    backend.execute(prepared)
    del prepared
    torch.cuda.synchronize(device)
    check(torch.cuda.memory_allocated(device) >= base,
          "CUDA lifetime test completed without allocator corruption")


def main():
    with torch.no_grad():
        test_prepare_execute()
        test_validation()
        test_error_context()
        test_v_boundary()
        test_cuda_lifetime()
    print("\nall efficient Sage self-tests passed")


if __name__ == "__main__":
    main()
