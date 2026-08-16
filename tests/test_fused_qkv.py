"""CPU contracts for fused H3 QKV carriers and projected Sparse Sage."""

import inspect
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import h3_attention.hybrid.fused_qkv as fused_qkv  # noqa: E402
from h3_attention.hybrid.fused_qkv import (  # noqa: E402
    FusedQKVError,
    FusedQKVProjector,
    PreparedFusedQKV,
    validate_prepared_fused_qkv,
)
from h3_attention.hybrid.backend import HybridSparseBackend  # noqa: E402
from h3_attention.hybrid.config import (  # noqa: E402
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
)
from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    SparseSageKernelSpec,
    SparseSageExecutor,
)
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def sparse_spec(kernel=None):
    return SparseSageKernelSpec(
        version="fake-sparge",
        architecture="sm89",
        capability=(8, 9),
        q_tile=128,
        kv_tile=64,
        v_format="fp8",
        kernel=kernel or (lambda *args: None),
        accumulator="f32",
        fused_v_ops="fake-v-fused",
        kernel_name="fake_sparse_kernel",
    )


def expect_error(fn, text):
    try:
        fn()
    except FusedQKVError as exc:
        check(text in str(exc), "invalid carrier is rejected with a clear error")
    else:
        raise AssertionError("expected FusedQKVError containing %r" % text)


def layout(sequence=350, video_start=96):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=[
            (0, video_start - 32, "text"),
            (video_start - 32, video_start, "audio"),
            (video_start, sequence, "video"),
        ],
        video_shape=(1, 1, sequence - video_start),
        audio_t=16,
    )


def options(sequence=350, video_start=96):
    return {
        RUNTIME_KEY: RuntimeSnapshot(
            request_id=11,
            step_index=2,
            total_steps=20,
            sigma=0.5,
            branch=(0,),
            layout=layout(sequence, video_start),
            layout_signature=(sequence, video_start),
            compute_dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
    }


def projected(sequence=350, heads=2):
    q_blocks = (sequence + 127) // 128
    k_blocks = (sequence + 63) // 64
    shape = (1, heads, sequence, 128)
    return PreparedFusedQKV(
        q_int8=torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int8).reshape(shape),
        q_scale=torch.arange(heads * q_blocks, dtype=torch.float32).reshape(1, heads, q_blocks),
        k_int8=torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int8).reshape(shape),
        k_scale=torch.arange(heads * k_blocks, dtype=torch.float32).reshape(1, heads, k_blocks),
        v=torch.randn(shape, dtype=torch.bfloat16),
        q_summary=torch.randn((1, heads, q_blocks, 128), dtype=torch.bfloat16),
        k_summary=torch.randn((1, heads, k_blocks, 128), dtype=torch.bfloat16),
        output_dtype=torch.bfloat16,
        sequence=sequence,
        heads=heads,
        head_dim=128,
        layer_index=7,
        smooth_k=True,
    )


def test_prepared_validation():
    print("prepared fused QKV validation")
    prepared = projected()
    check(validate_prepared_fused_qkv(prepared) is prepared,
          "valid shapes, dtypes, devices, and contiguity are accepted")

    wrong_shape = replace(prepared, q_int8=prepared.q_int8[..., :127])
    expect_error(lambda: validate_prepared_fused_qkv(wrong_shape), "carrier shape is invalid")

    wrong_dtype = replace(prepared, q_int8=prepared.q_int8.to(torch.uint8))
    expect_error(lambda: validate_prepared_fused_qkv(wrong_dtype), "carriers must be INT8")

    noncontiguous = replace(
        prepared,
        q_int8=torch.empty((1, prepared.heads, prepared.sequence, 1), dtype=torch.int8)
        .expand(1, prepared.heads, prepared.sequence, prepared.head_dim),
    )
    expect_error(lambda: validate_prepared_fused_qkv(noncontiguous), "carriers must be contiguous")


def test_mode_selection():
    print("fused QKV mode selection")
    api = sparse_spec()
    established = HybridSparseBackend(
        HybridSparseConfig(mode=MODE_SAGE128),
        kernel_spec=api,
        allow_cpu_for_tests=True,
    )
    fused = HybridSparseBackend(
        HybridSparseConfig(mode=MODE_SAGE128_FUSED_QKV),
        kernel_spec=api,
        allow_cpu_for_tests=True,
    )
    check(established.projector is None, "established sage128 mode retains BF16 projection")
    check(fused.projector is not None and fused.projector.name == "h3_fused_qkv",
          "fused mode owns the H3 QKV projector")


def test_projector_tensor_core_injection():
    print("fused QKV tensor-core injection")
    calls = []
    sentinel = object()

    def fake_run(module, x, rope, *, layer_index, tensor_core=None):
        calls.append((module, x, rope, layer_index, tensor_core))
        return sentinel

    import h3_attention.hybrid.fused_qkv as fused_qkv
    original = fused_qkv.run_fused_qkv
    try:
        fused_qkv.run_fused_qkv = fake_run
        projector = FusedQKVProjector(sentinel)
        result = projector.project(
            "module", "x", "rope", layer_index=9, transformer_options={},
        )
    finally:
        fused_qkv.run_fused_qkv = original
    check(result is sentinel, "projector preserves the standard project return")
    check(calls == [("module", "x", "rope", 9, sentinel)],
          "injected tensor core reaches the fused projection boundary")


def test_summary_router_matches_direct():
    print("summary-based fused QKV routing")
    torch.manual_seed(19)
    q = torch.randn((1, 2, 350, 128), dtype=torch.bfloat16)
    k = torch.randn_like(q)
    router = SparseTileRouter()
    geometry = layout()
    direct = router.build_lut(q, k, geometry, 0.5)
    summaries = router.build_lut_from_summaries(
        router._mean_pool(q, 128),
        router._mean_pool(k, 64),
        geometry,
        0.5,
    )
    check(torch.equal(direct[0], summaries[0]), "summary route LUT exactly matches direct routing")
    check(torch.equal(direct[1], summaries[1]), "summary route valid counts exactly match direct routing")
    check(direct[2] == summaries[2], "summary route metadata exactly matches direct routing")
    check(direct[0].shape[-1] == 6 and direct[0].shape[-2] == 3,
          "partial final Q and KV tiles are represented")


def test_projected_sparse_sage():
    print("projected Sparse Sage preparation and execution")
    projected_qkv = projected()
    router = SparseTileRouter()
    lut, valid, _ = router.build_lut_from_summaries(
        projected_qkv.q_summary,
        projected_qkv.k_summary,
        layout(),
        0.5,
    )
    v_calls = []

    def fake_v_preparer(v, fused):
        v_calls.append((v, fused))
        padded = (v.shape[-2] + 127) // 128 * 128
        carrier = torch.zeros((1, v.shape[1], v.shape[3], padded), dtype=torch.float8_e4m3fn)
        scale = torch.ones((1, v.shape[1], v.shape[3]), dtype=torch.float32)
        return carrier, scale

    kernel_calls = []

    def fake_kernel(*args):
        kernel_calls.append(args)
        args[3].fill_(3)

    api = sparse_spec(fake_kernel)
    executor = SparseSageExecutor(
        api,
        allow_cpu_for_tests=True,
        v_preparer=fake_v_preparer,
        low_level_selector=lambda _q: fake_kernel,
    )
    prepared = executor.prepare_projected(
        projected_qkv,
        lut,
        valid,
        metadata={"route": "test"},
    )
    check(prepared.q_int8 is projected_qkv.q_int8 and prepared.k_int8 is projected_qkv.k_int8,
          "projected preparation preserves exact Q/K INT8 carrier objects")
    check(prepared.q_scale is projected_qkv.q_scale and prepared.k_scale is projected_qkv.k_scale,
          "projected preparation preserves exact Q/K scales")
    check(len(v_calls) == 1 and v_calls[0][0] is projected_qkv.v,
          "projected preparation injects the V preparer")
    check(prepared.metadata["route"] == "test"
          and prepared.metadata["qkv_projection"] == "fused_int8"
          and prepared.metadata["smooth_k"] is True,
          "projected preparation carries fused projection metadata")
    output = executor.execute(prepared)
    check(len(kernel_calls) == 1, "projected execution invokes the injected low-level kernel")
    check(torch.all(output == 3), "injected low-level kernel controls the CPU output")


def test_projected_backend_integration():
    print("projected hybrid backend integration")
    projected_qkv = projected()

    def fake_v_preparer(v, fused):
        padded = (v.shape[-2] + 127) // 128 * 128
        return (
            torch.zeros((1, v.shape[1], v.shape[3], padded), dtype=torch.float8_e4m3fn),
            torch.ones((1, v.shape[1], v.shape[3]), dtype=torch.float32),
        )

    def fake_kernel(*args):
        args[3].fill_(4)

    backend = HybridSparseBackend(
        HybridSparseConfig(mode=MODE_SAGE128_FUSED_QKV, video_budget=0.5),
        kernel_spec=sparse_spec(fake_kernel),
        allow_cpu_for_tests=True,
        v_preparer=fake_v_preparer,
        low_level_selector=lambda _q: fake_kernel,
    )
    prepared = backend.prepare_projected(
        projected_qkv,
        layer_index=7,
        transformer_options=options(),
    )
    output = backend.execute(prepared)
    check(prepared.sparse.metadata["request_id"] == 11
          and prepared.sparse.metadata["layer"] == 7,
          "runtime and layer metadata reach the projected Sparse Sage carrier")
    check(torch.all(output == 4), "hybrid backend executes the projected carrier")


def test_v_projection_is_a_separate_kernel():
    print("dedicated V projection")
    core_source = inspect.getsource(fused_qkv._fused_qkv_tensor_core)
    check("for kind, block_k, num_warps, num_stages" in core_source,
          "Q/K retain separate constexpr-specialized launches")
    check("_fused_v_kernel[v_grid]" in core_source,
          "V uses its own launch configuration")
    v_source = inspect.getsource(fused_qkv._fused_v_kernel.fn)
    check("q_norm_ptr" not in v_source
          and "k_norm_ptr" not in v_source
          and "rope_ptr" not in v_source,
          "V carries no Q/K-only norm or RoPE inputs")
    defaults = inspect.signature(fused_qkv._fused_qkv_tensor_core).parameters
    check(defaults["q_block_k"].default == 128
          and defaults["k_block_k"].default == 128,
          "Q and K independently use the exact-gated K128 specialization")
    check((defaults["v_block_m"].default,
           defaults["v_block_n"].default,
           defaults["v_block_k"].default) == (128, 256, 128),
          "V uses the measured M128/N256/K128 geometry")
    check(defaults["v_num_warps"].default == 8
          and defaults["v_num_stages"].default == 3,
          "V uses the lower-register eight-warp three-stage launch")


def test_qdata_stride_is_preserved_without_a_hot_path_copy():
    print("QData stride")
    source = inspect.getsource(fused_qkv)
    check("qdata = qdata.contiguous()" not in source,
          "fused projection never copies full QKV QData")
    for kernel in (fused_qkv._fused_qk_kernel, fused_qkv._fused_v_kernel):
        kernel_source = inspect.getsource(kernel.fn)
        check("weight_stride_output" in kernel_source
              and "weight_stride_inner" in kernel_source,
              "%s indexes the checkpoint QData with its actual strides" % kernel.fn.__name__)


def main():
    test_prepared_validation()
    test_mode_selection()
    test_projector_tensor_core_injection()
    test_summary_router_matches_direct()
    test_projected_sparse_sage()
    test_projected_backend_integration()
    test_v_projection_is_a_separate_kernel()
    test_qdata_stride_is_preserved_without_a_hot_path_copy()
    print("\nall fused QKV tests passed")


if __name__ == "__main__":
    main()
