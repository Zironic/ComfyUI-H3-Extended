"""CPU tests for prepared Sage architecture backends."""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(
    0,
    os.path.abspath(os.path.join(_HERE, "..", "..", "..")),
)

from h3_attention.sage_arch import (  # noqa: E402
    KernelBinding,
    SM80API,
    SM86API,
    SM90API,
    SM12xAPI,
    SageSM80MemoryEfficientBackend,
    SageSM86MemoryEfficientBackend,
    SageSM90MemoryEfficientBackend,
    SageSM12xMemoryEfficientBackend,
)


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def fused_hnd(sequence=65, heads=2, dtype=torch.bfloat16):
    inner = heads * 128
    fused = torch.randn(sequence, inner * 3, dtype=dtype)
    q, k, v = fused.split(inner, dim=-1)
    return (
        q.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
        k.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
        v.view(sequence, heads, 128).transpose(0, 1).unsqueeze(0),
    )


def fake_thread_quantizer(q, k, km=None, **kwargs):
    assert km is None
    q8 = torch.zeros(q.shape, dtype=torch.int8)
    k8 = torch.zeros(k.shape, dtype=torch.int8)
    b, h, s, _ = q.shape
    q_scale = torch.ones((b, h, max(1, (s + 63) // 64) * 32))
    k_scale = torch.ones((b, h, max(1, (s + 63) // 64) * 4))
    return q8, q_scale, k8, k_scale


class FakeKernel:
    def __init__(self, expected_granularity):
        self.expected_granularity = expected_granularity
        self.calls = 0
        self.v_dtype = None

    def __call__(
        self,
        q8,
        k8,
        v,
        out,
        q_scale,
        k_scale,
        *args,
    ):
        self.calls += 1
        self.v_dtype = v.dtype
        layout, causal, granularity, _scale, return_lse = args[-5:]
        assert layout == 1
        assert causal == 0
        assert granularity == self.expected_granularity
        assert return_lse == 0
        out.zero_()


class FakeFP8:
    def __init__(self):
        self.scale_max = None
        self.sequence = None

    def __call__(self, v, tensor_layout, scale_max, smooth_v):
        assert tensor_layout == "HND"
        assert smooth_v is False
        self.scale_max = scale_max
        self.sequence = int(v.shape[2])
        fake = torch.zeros(v.shape, dtype=torch.int8)
        scale = torch.ones(
            (v.shape[0], v.shape[1], v.shape[3]),
            dtype=torch.float32,
        )
        return fake, scale, None


def assert_independent(prepared, source_ptr):
    for tensor, name in (
        (prepared.q_int8, "Q"),
        (prepared.k_int8, "K"),
        (prepared.v_source, "V"),
    ):
        check(
            tensor.untyped_storage().data_ptr() != source_ptr,
            "prepared %s does not retain fused QKV storage" % name,
        )


def test_sm80():
    print("SM80")
    kernel = FakeKernel(expected_granularity=3)
    api = SM80API(
        version="2.2.test",
        kernel=KernelBinding(kernel, "fake_sm80", "test"),
    )
    backend = SageSM80MemoryEfficientBackend(
        api=api,
        quantizer=fake_thread_quantizer,
        allow_cpu_for_tests=True,
    )
    q, k, v = fused_hnd()
    source_ptr = q.untyped_storage().data_ptr()
    prepared = backend.prepare(
        q,
        k,
        v,
        layer_index=0,
        transformer_options={},
    )
    assert_independent(prepared, source_ptr)
    del q, k, v
    output = backend.execute(prepared)
    check(output.dtype == torch.bfloat16, "SM80 output preserves BF16")
    check(kernel.v_dtype == torch.float16, "SM80 kernel receives FP16 V")


def test_sm86():
    print("SM86")
    seen = {}

    def quantizer(q, k, **kwargs):
        seen["sm_scale"] = kwargs["sm_scale"]
        return fake_thread_quantizer(q, k)

    def attention(
        q8,
        k8,
        v,
        q_scale,
        k_scale,
        **kwargs,
    ):
        seen["v_dtype"] = v.dtype
        seen["layout"] = kwargs["tensor_layout"]
        return (
            torch.zeros(q8.shape, dtype=kwargs["output_dtype"]),
            torch.empty(0),
        )

    api = SM86API("2.2.test", quantizer, attention)
    backend = SageSM86MemoryEfficientBackend(
        api=api,
        allow_cpu_for_tests=True,
    )
    q, k, v = fused_hnd()
    prepared = backend.prepare(
        q,
        k,
        v,
        layer_index=1,
        transformer_options={},
    )
    del q, k, v
    output = backend.execute(prepared)
    check(output.dtype == torch.bfloat16, "SM86 output preserves BF16")
    check(seen["v_dtype"] == torch.float16, "SM86 Triton path receives FP16 V")
    check(seen["layout"] == "HND", "SM86 Triton path receives HND")


def test_sm90():
    print("SM90")
    kernel = FakeKernel(expected_granularity=3)
    fp8 = FakeFP8()
    api = SM90API(
        "2.2.test",
        fp8,
        KernelBinding(kernel, "fake_sm90", "test"),
    )
    backend = SageSM90MemoryEfficientBackend(
        api=api,
        quantizer=fake_thread_quantizer,
        allow_cpu_for_tests=True,
    )
    q, k, v = fused_hnd(sequence=65)
    prepared = backend.prepare(
        q,
        k,
        v,
        layer_index=2,
        transformer_options={},
    )
    del q, k, v
    output = backend.execute(prepared)
    check(output.shape == (1, 2, 65, 128), "SM90 output keeps original length")
    check(fp8.sequence == 128, "SM90 pads V to a multiple of 128")
    check(fp8.scale_max == 448.0, "SM90 uses E4M3 scale range")


def test_sm12x():
    print("SM120/121")
    kernel = FakeKernel(expected_granularity=2)
    fp8 = FakeFP8()
    seen = {}

    def per_warp(q, k, km=None, **kwargs):
        seen.update(kwargs)
        return fake_thread_quantizer(q, k)

    api = SM12xAPI(
        "2.2.test",
        per_warp,
        fp8,
        KernelBinding(kernel, "fake_sm12x", "test"),
    )
    backend = SageSM12xMemoryEfficientBackend(
        api=api,
        allow_cpu_for_tests=True,
    )
    q, k, v = fused_hnd()
    prepared = backend.prepare(
        q,
        k,
        v,
        layer_index=3,
        transformer_options={},
    )
    del q, k, v
    output = backend.execute(prepared)
    check(output.dtype == torch.bfloat16, "SM12x output preserves BF16")
    check(seen["WARPQ"] == 32, "SM12x uses upstream per-warp Q")
    check(fp8.scale_max == 2.25, "SM12x uses FP32+FP16 V scale range")


def main():
    with torch.no_grad():
        test_sm80()
        test_sm86()
        test_sm90()
        test_sm12x()
    print("\nall Sage architecture backend tests passed")


if __name__ == "__main__":
    main()
