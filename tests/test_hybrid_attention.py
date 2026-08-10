"""CPU contracts plus explicitly opt-in CUDA checks for Hybrid Sparse."""

import os
import sys
import inspect
from types import SimpleNamespace
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_attention.hybrid import (  # noqa: E402
    DeferredCudaTiming,
    HybridSparseBackend,
    HybridSparseConfig,
    HybridStatsCollector,
    TIMING_STAGES,
    SparseSageKernelSpec,
    SparseSageError,
    SparseSageExecutor,
    SparseTileRouter,
    load_sparse_sage_spec,
    resolve_sparse_sage_spec,
)
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttention  # noqa: E402
from h3_attention.hybrid.report import render  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    _load_qattn_surface,
    quantize_qk,
)
from h3_attention.hybrid import sparse_quant  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def layout(sequence=384, video_start=128):
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


def options(sequence=384, video_start=128, device=None):
    token_layout = layout(sequence, video_start)
    snapshot = RuntimeSnapshot(
        request_id=2,
        step_index=3,
        total_steps=20,
        sigma=0.5,
        branch=(0,),
        layout=token_layout,
        layout_signature=(sequence, video_start),
        compute_dtype=torch.bfloat16,
        device=device or torch.device("cpu"),
    )
    return {RUNTIME_KEY: snapshot}


def fused_hnd(sequence=384, heads=2, head_dim=128, device="cpu", dtype=torch.bfloat16):
    inner = heads * head_dim
    fused = torch.randn(sequence, inner * 3, dtype=dtype, device=device)
    q, k, v = fused.split(inner, dim=-1)
    return (
        q.view(sequence, heads, head_dim).transpose(0, 1).unsqueeze(0),
        k.view(sequence, heads, head_dim).transpose(0, 1).unsqueeze(0),
        v.view(sequence, heads, head_dim).transpose(0, 1).unsqueeze(0),
    )


class FakeSparseKernel:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def __call__(self, q, k, v, output, *args):
        if self.fail:
            raise RuntimeError("synthetic sparse failure")
        self.calls.append((q, k, v, output, args))
        output.zero_()


def kernel_spec(kernel=None, *, capability=(8, 9), q_tile=128, kv_tile=64,
                v_format="fp8", accumulator="f32", fused=object()):
    kernel = kernel or FakeSparseKernel()
    return SparseSageKernelSpec(
        version="0.1.test",
        architecture="sm%d%d" % capability,
        capability=capability,
        q_tile=q_tile,
        kv_tile=kv_tile,
        v_format=v_format,
        kernel=kernel,
        accumulator=accumulator,
        fused_v_ops=fused if v_format == "fp8" else None,
        kernel_name="fake_sparse_kernel",
    )


def fake_v_preparer(v, *_):
    b, h, sequence, dim = v.shape
    padded = (sequence + 127) // 128 * 128
    return (torch.zeros((b, h, dim, padded), dtype=torch.float8_e4m3fn),
            torch.ones((b, h, dim), dtype=torch.float32))


class Collector:
    def __init__(self):
        self.records = []

    def record(self, metadata):
        self.records.append(dict(metadata))


class FakeTimingEvent:
    def __init__(self, clock, events):
        self.clock = clock
        self.events = events
        self.value = None

    def record(self):
        self.clock[0] += 1.0
        self.value = self.clock[0]
        self.events.append(("record", self.value))

    def synchronize(self):
        self.events.append(("synchronize", self.value))

    def elapsed_time(self, other):
        return other.value - self.value


def test_deferred_timing():
    print("deferred CUDA timing seam")
    clock = [0.0]
    events = []
    factory = lambda **kwargs: FakeTimingEvent(clock, events)
    collector = HybridStatsCollector("output", "timing")
    config = HybridSparseConfig(timing=True)
    api = kernel_spec()
    hybrid = HybridSparseBackend(
        config, kernel_spec=api, collector=collector, event_factory=factory,
        allow_cpu_for_tests=True,
        v_preparer=fake_v_preparer,
        qk_quantizer=quantize_qk,
    )
    collector.on_request_reset(2)
    q, k, v = fused_hnd()
    prepared = hybrid.prepare(q, k, v, layer_index=0, transformer_options=options())
    hybrid.execute(prepared)
    with mock.patch("h3_attention.hybrid.report.os.makedirs"), \
            mock.patch("h3_attention.hybrid.report.open", mock.mock_open()):
        collector.on_request_end(2, 2.0)
    check(sum(item[0] == "synchronize" for item in events) == 1,
          "request end synchronizes only the last completed event")
    summary = collector._timing._resolved
    check(summary["call_count"] == 1 and all(
        summary["stages"][stage]["count"] == 1
        for stage in ("direct_lut_construction", "v_preparation",
                      "q_k_int8_quantization", "sparse_sage_low_level_kernel",
                      "total_hybrid_attention")
    ), "all deferred timing stages resolve once")
    check(summary["attention_cuda_to_request_wall_ratio"] is not None,
          "timing summary includes the caveated CUDA/wall ratio")
    check(tuple(summary["stages"]) == TIMING_STAGES,
          "timing summary resolves every declared stage key")
    report_text = render({
        "mode": "sage128",
        "summary": {
            "requested_video_budget": 0.5,
            "layer_count": 1,
            "expected_layer_count": 50,
            "step_count": 1,
            "mean_video_tile_density": 0.5,
            "mean_full_mask_density": 0.6,
            "min_full_mask_density": 0.6,
            "max_full_mask_density": 0.6,
            "timing": summary,
        },
    })
    check(report_text.count("q_k_int8_quantization:") == 1
          and "Stage times are nested" in report_text,
          "human report lists each timing stage once with nesting caveat")

    disabled_factory_calls = []
    disabled = DeferredCudaTiming(
        False,
        event_factory=lambda **kwargs: disabled_factory_calls.append(kwargs),
    )
    disabled.begin_request(3, cuda=True)
    disabled.end(disabled.begin("total_hybrid_attention"))
    disabled_summary = disabled.resolve(1.0)
    check(not disabled_factory_calls and disabled_summary["call_count"] == 0,
          "disabled timing allocates no CUDA events")


def test_per_step_timing():
    print("per-step CUDA timing")
    clock = [0.0]
    events = []
    timer = DeferredCudaTiming(
        True,
        event_factory=lambda **kwargs: FakeTimingEvent(clock, events),
    )
    timer.begin_request(4, cuda=False)

    def record(stage, step_index, branch):
        snapshot = options()[RUNTIME_KEY]
        snapshot = RuntimeSnapshot(
            request_id=4,
            step_index=step_index,
            total_steps=snapshot.total_steps,
            sigma=snapshot.sigma,
            branch=branch,
            layout=snapshot.layout,
            layout_signature=snapshot.layout_signature,
            compute_dtype=snapshot.compute_dtype,
            device=snapshot.device,
        )
        timer.set_context(snapshot)
        token = timer.begin(stage)
        timer.end(token)

    record("total_dit_block", 0, (0,))
    record("total_dit_block", 0, (1,))
    record("model_forward", 0, (0,))
    record("total_hybrid_attention", 1, (0,))
    record("total_dit_block", -1, (0,))
    summary = timer.resolve(5.0)

    check(sum(item[0] == "synchronize" for item in events) == 1,
          "per-step resolution synchronizes once")
    check(summary["stages"]["total_dit_block"]["count"] == 3,
          "request aggregate retains events without a known step")
    check([step["step_index"] for step in summary["per_step"]] == [0, 1],
          "per-step summary excludes unknown steps and keeps sampler order")
    first = summary["per_step"][0]
    check(first["stages"]["total_dit_block"]["count"] == 2,
          "step summary rolls CFG branches together")
    check([branch["branch"] for branch in first["branches"]] == [[0], [1]],
          "step summary preserves exact CFG branch buckets")
    check(first["stages"]["model_forward"]["count"] == 1,
          "step summary includes model-forward timing")


def backend(kernel=None, collector=None, budget=0.5):
    kernel = kernel or FakeSparseKernel()
    api = kernel_spec(kernel)
    config = HybridSparseConfig(video_budget=budget)
    return HybridSparseBackend(
        config,
        kernel_spec=api,
        collector=collector,
        allow_cpu_for_tests=True,
        v_preparer=fake_v_preparer,
        qk_quantizer=quantize_qk,
    ), kernel


def test_prepare_execute_lifetime():
    print("prepare / execute ownership")
    collector = Collector()
    hybrid, kernel = backend(collector=collector)
    q, k, v = fused_hnd()
    source = q.untyped_storage().data_ptr()
    prepared = hybrid.prepare(q, k, v, layer_index=7, transformer_options=options())
    sparse = prepared.sparse
    check(all(
        not torch.is_tensor(value) or value.untyped_storage().data_ptr() != source
        for value in vars(sparse).values()
    ), "prepared carrier retains no view into fused QKV storage")
    check(sparse.q_int8.dtype == torch.int8 and sparse.k_int8.dtype == torch.int8,
          "prepared Q/K are quantized int8 buffers")
    check(sparse.v_carrier.dtype == torch.float8_e4m3fn,
          "prepared V is FP8")
    check(sparse.lut.shape == (1, 2, 3, 6) and sparse.lut.is_contiguous()
          and sparse.valid_block_num.dtype == torch.int32,
          "prepared LUT and valid counts are contiguous int32 geometry")
    del q, k, v
    output = hybrid.execute(prepared)
    check(output.shape == (1, 2, 384, 128), "hybrid backend returns HND output")
    check(output.dtype == torch.bfloat16, "output dtype matches H3 input")
    check(len(kernel.calls) == 1 and kernel.calls[0][4][6:9] == (1, 0, 1),
          "low-level Sparse Sage ABI receives HND non-causal mode")
    check(len(collector.records) == 1 and collector.records[0]["layer"] == 7,
          "successful execution records layer structural statistics")


def test_strict_errors():
    print("strict failures")
    hybrid, _ = backend()
    q, k, v = fused_hnd()
    try:
        hybrid.prepare(q, k, v, layer_index=0, transformer_options={})
    except SparseSageError as exc:
        check("runtime snapshot" in str(exc), "missing runtime layout raises explicitly")
    else:
        raise AssertionError("missing runtime snapshot must fail")

    bad = options()
    bad[RUNTIME_KEY] = RuntimeSnapshot(
        request_id=0, step_index=0, total_steps=20, sigma=1.0, branch=(0,),
        layout=None, layout_signature=None, compute_dtype=None,
        device=torch.device("cpu"), error="synthetic layout failure",
    )
    try:
        hybrid.prepare(q, k, v, layer_index=0, transformer_options=bad)
    except SparseSageError as exc:
        check("synthetic layout failure" in str(exc), "invalid layout error is preserved")
    else:
        raise AssertionError("invalid runtime layout must fail")

    failing, _ = backend(kernel=FakeSparseKernel(fail=True))
    prepared = failing.prepare(q, k, v, layer_index=41, transformer_options=options())
    try:
        failing.execute(prepared)
    except SparseSageError as exc:
        check("layer=41" in str(exc) and "0.1.test" in str(exc),
              "kernel failure names layer and dependency version")
    else:
        raise AssertionError("kernel failure must never fall back")


def test_dependency_and_disabled_node():
    print("dependency and disabled node")
    with mock.patch("importlib.import_module", side_effect=ModuleNotFoundError("missing")):
        try:
            load_sparse_sage_spec()
        except SparseSageError as exc:
            check("compiled" in str(exc), "missing SpargeAttention raises explicit dependency error")
        else:
            raise AssertionError("missing SpargeAttention must fail")

    marker = object()
    result = MiniMaxH3HybridSparseAttention.execute(marker, enabled=False)
    check(result.args[0] is marker, "disabled node is an exact model pass-through")


def test_kernel_spec_resolution():
    print("portable kernel spec resolution")
    ampere = FakeSparseKernel()
    ada_f16 = FakeSparseKernel()
    ada_f32 = FakeSparseKernel()
    hopper = FakeSparseKernel()
    qattn = SimpleNamespace()
    setattr(
        qattn,
        "qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold",
        ampere,
    )
    setattr(
        qattn,
        "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        ada_f16,
    )
    setattr(
        qattn,
        "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        ada_f32,
    )
    setattr(
        qattn,
        "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90",
        hopper,
    )
    fused = object()

    for capability in ((8, 0), (8, 6), (8, 7)):
        spec = resolve_sparse_sage_spec(
            qattn, fused, capability=capability, version="test",
            cuda_version=(12, 6),
        )
        check(
            spec.capability == capability and spec.q_tile == 128
            and spec.kv_tile == 64 and spec.v_format == "fp16"
            and spec.kernel is ampere,
            "%s resolves Ampere FP16-V geometry" % spec.architecture,
        )

    ada = resolve_sparse_sage_spec(
        qattn, fused, capability=(8, 9), version="test",
        cuda_version=(12, 6),
    )
    check(
        ada.kernel is ada_f32 and ada.accumulator == "f32"
        and ada.v_format == "fp8" and ada.v_quant_bound == 2.25
        and (ada.q_tile, ada.kv_tile) == (128, 64),
        "SM89 CUDA 12.6 resolves the FP8-V f32 accumulator",
    )
    ada_128 = resolve_sparse_sage_spec(
        qattn, fused, capability=(8, 9), version="test",
        cuda_version=(12, 8),
    )
    check(
        ada_128.kernel is ada_f16 and ada_128.accumulator == "f16",
        "SM89 CUDA 12.8 prefers the Sage2++ f16 accumulator",
    )
    sm90 = resolve_sparse_sage_spec(
        qattn, fused, capability=(9, 0), version="test",
        cuda_version=(12, 6),
    )
    check(
        sm90.kernel is hopper and sm90.v_format == "fp8"
        and sm90.v_quant_bound == 2.25
        and (sm90.q_tile, sm90.kv_tile) == (64, 128),
        "SM90 resolves Hopper FP8-V 64Q x 128KV geometry",
    )

    sm120 = resolve_sparse_sage_spec(
        qattn, fused, capability=(12, 0), version="test",
        cuda_version=(12, 8),
    )
    check(
        sm120.kernel is ada_f16 and sm120.architecture == "sm120"
        and sm120.accumulator == "f16" and sm120.v_format == "fp8"
        and sm120.v_quant_bound == 2.25
        and (sm120.q_tile, sm120.kv_tile) == (128, 64),
        "SM120 resolves the SM89-family FP8-V 128Q x 64KV f16 ABI",
    )
    try:
        resolve_sparse_sage_spec(
            qattn, fused, capability=(12, 0), version="test",
            cuda_version=(12, 7),
        )
    except SparseSageError as exc:
        check("CUDA 12.8" in str(exc), "SM120 rejects CUDA versions below 12.8")
    else:
        raise AssertionError("SM120 must reject CUDA versions below 12.8")
    try:
        resolve_sparse_sage_spec(
            qattn, None, capability=(12, 0), version="test",
            cuda_version=(12, 8),
        )
    except SparseSageError as exc:
        check("_fused" in str(exc), "SM120 reports a missing compiled _fused extension")
    else:
        raise AssertionError("SM120 must require the compiled _fused extension")
    try:
        resolve_sparse_sage_spec(
            SimpleNamespace(), fused, capability=(12, 0), version="test",
            cuda_version=(12, 8),
        )
    except SparseSageError as exc:
        check("SM120" in str(exc) and "required kernel" in str(exc),
              "SM120 reports a missing block-sparse symbol")
    else:
        raise AssertionError("SM120 must require its block-sparse symbol")

    for capability, expected in (((9, 0), "_sm90"),):
        surface = SimpleNamespace() if capability == (9, 0) else qattn
        try:
            resolve_sparse_sage_spec(
                surface, fused, capability=capability, version="test",
                cuda_version=(12, 8),
            )
        except SparseSageError as exc:
            check(expected in str(exc), "unsupported or missing architecture fails explicitly")
        else:
            raise AssertionError("invalid Sparse Sage architecture was accepted")

    split = SimpleNamespace(split=True)

    def split_import(name):
        if name == "spas_sage_attn._qattn":
            raise ModuleNotFoundError(name=name)
        if name == "spas_sage_attn._qattn_sm89":
            return SimpleNamespace()
        raise AssertionError("unexpected split extension import %s" % name)

    with mock.patch(
            "h3_attention.hybrid.sparse_sage.importlib.import_module",
            side_effect=split_import,
    ), mock.patch.object(
            torch.ops, "spas_sage_attn_qattn_sm89", split, create=True,
    ):
        check(_load_qattn_surface((8, 9)) == (split, "split"),
              "architecture registry normalizes split SM89 extension packages")
        check(_load_qattn_surface((12, 0)) == (split, "split"),
              "SM120 reuses the split SM89 extension namespace")

    direct_split = SimpleNamespace(**{
        "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold":
            lambda *args: None,
    })
    with mock.patch(
            "h3_attention.hybrid.sparse_sage.importlib.import_module",
            return_value=direct_split,
    ):
        check(
            _load_qattn_surface((12, 0)) == (direct_split, "split"),
            "split extension direct Python kernel exports are preserved",
        )

    monolithic = SimpleNamespace(monolithic=True)
    with mock.patch(
            "h3_attention.hybrid.sparse_sage.importlib.import_module",
            return_value=monolithic,
    ):
        check(
            _load_qattn_surface((9, 0)) == (monolithic, "monolithic"),
            "architecture registry prefers current upstream's monolithic extension",
        )

    fused_surface = SimpleNamespace(
        transpose_pad_permute_cuda=lambda *args: None,
        scale_fuse_quant_cuda=lambda *args: None,
    )

    def sm120_import(name):
        if name in ("spas_sage_attn", "spas_sage_attn._qattn_sm89"):
            return SimpleNamespace()
        if name == "spas_sage_attn._fused":
            return fused_surface
        raise AssertionError("unexpected SM120 extension import %s" % name)

    with mock.patch(
            "h3_attention.hybrid.sparse_sage.importlib.import_module",
            side_effect=sm120_import,
    ), mock.patch.object(
            torch.ops, "spas_sage_attn_qattn_sm89", qattn, create=True,
    ):
        loaded = load_sparse_sage_spec(
            capability=(12, 0), cuda_version=(12, 8),
        )
    check(
        loaded.capability == (12, 0) and loaded.extension_layout == "split"
        and loaded.kernel is ada_f16,
        "load accepts SM120 and resolves the split SM89-family ABI",
    )


def test_architecture_specific_carriers_and_abis():
    print("architecture-specific carriers and ABIs")
    ampere_kernel = FakeSparseKernel()
    ampere = kernel_spec(
        ampere_kernel, capability=(8, 0), v_format="fp16", accumulator="f16",
    )
    q, k, v = fused_hnd()
    ampere_lut = torch.zeros((1, 2, 3, 6), dtype=torch.int32)
    ampere_valid = torch.zeros((1, 2, 3), dtype=torch.int32)
    ampere_executor = SparseSageExecutor(ampere, allow_cpu_for_tests=True)
    prepared = ampere_executor.prepare(
        q, k, v, ampere_lut, ampere_valid, layer_index=0, metadata={},
    )
    check(
        prepared.v_carrier.dtype == torch.float16
        and prepared.v_carrier.is_contiguous()
        and prepared.v_scale.numel() == 0,
        "Ampere prepares an independent contiguous FP16 HND V carrier",
    )
    ampere_executor.execute(prepared)
    check(
        len(ampere_kernel.calls) == 1
        and ampere_kernel.calls[0][4][5:8] == (1, 0, 1),
        "Ampere dispatch omits the FP8 V scale argument",
    )

    hopper_kernel = FakeSparseKernel()
    hopper = kernel_spec(
        hopper_kernel, capability=(9, 0), q_tile=64, kv_tile=128,
    )
    router = SparseTileRouter(HybridSparseConfig(), spec=hopper)
    lut, valid, metadata = router.build_lut(q, k, layout(), 0.5)
    check(
        lut.shape == (1, 2, 6, 3)
        and metadata.q_tiles == 6 and metadata.kv_tiles == 3,
        "Hopper router uses 64Q x 128KV geometry",
    )
    hopper_executor = SparseSageExecutor(
        hopper, allow_cpu_for_tests=True, v_preparer=fake_v_preparer,
    )
    prepared = hopper_executor.prepare(
        q, k, v, lut, valid, layer_index=0, metadata={},
    )
    check(
        prepared.q_scale.shape == (1, 2, 6)
        and prepared.k_scale.shape == (1, 2, 3),
        "Hopper Q/K quantization uses the same resolved geometry",
    )
    hopper_executor.execute(prepared)
    check(
        len(hopper_kernel.calls) == 1
        and hopper_kernel.calls[0][4][6:9] == (1, 0, 1),
        "Hopper dispatch includes the FP8 V scale argument",
    )

    try:
        HybridSparseBackend(
            HybridSparseConfig(mode="sage128_fused_qkv"), kernel_spec=hopper,
            allow_cpu_for_tests=True, v_preparer=fake_v_preparer,
        )
    except SparseSageError as exc:
        check("SM89" in str(exc), "fused QKV rejects non-Ada geometry before patching")
    else:
        raise AssertionError("fused QKV accepted Hopper geometry")

    try:
        HybridSparseBackend(
            HybridSparseConfig(), kernel_spec=hopper, router=SparseTileRouter(),
            allow_cpu_for_tests=True, v_preparer=fake_v_preparer,
        )
    except SparseSageError as exc:
        check("router geometry" in str(exc), "injected router must match the resolved ABI")
    else:
        raise AssertionError("mismatched injected router geometry was accepted")


def test_node_mode_schema():
    print("node mode schema")
    schema = MiniMaxH3HybridSparseAttention.define_schema()
    check(
        "SM89" not in schema.description and "128Q x 64KV" not in schema.description,
        "node describes architecture-native routing rather than one GPU geometry",
    )
    mode = next(item for item in schema.inputs if item.id == "mode")
    check(mode.options == ["sage128", "sage128_fused_qkv"],
          "node exposes established and fused-QKV modes")
    check(mode.default == "sage128",
          "established mode remains the backward-compatible default")

    from h3_memory_optimizer.config import ACTIVATION_MODES
    activation = next(item for item in schema.inputs if item.id == "activation")
    check(
        "mlp_chunked_convrot_2slice" in activation.options
        and "mlp_chunked_convrot_2slice" in ACTIVATION_MODES,
        "Hybrid Sparse activation exposes the two-slice ConvRot mode",
    )
    compile_backend = next(
        item for item in schema.inputs if item.id == "compile_backend"
    )
    check(
        compile_backend.options == ["off", "inductor"]
        and compile_backend.default == "off",
        "shared block compilation is explicit and backward compatible",
    )
    chunk_rows = next(item for item in schema.inputs if item.id == "chunk_rows")
    check(chunk_rows.min == 256, "node chunk minimum matches activation validation")


def test_node_rejects_invalid_compile_configuration_before_preflight():
    print("node compile validation")
    marker = object()
    with mock.patch(
        "h3_sparse_attention.nodes.RuntimeEnvironment.detect",
        side_effect=AssertionError("preflight must not run"),
    ):
        cases = (
            ({"compile_backend": "bogus"}, "compile backend"),
            ({"compile_backend": "inductor"}, "sage128_fused_qkv"),
            ({
                "compile_backend": "inductor",
                "mode": "sage128_fused_qkv",
            }, "convrot_2slice"),
            ({"chunk_rows": 128}, "chunk_rows"),
        )
        for kwargs, expected in cases:
            try:
                MiniMaxH3HybridSparseAttention.execute(marker, **kwargs)
            except ValueError as exc:
                check(expected in str(exc), "%s fails before preflight" % expected)
            else:
                raise AssertionError("invalid node configuration was accepted")


def test_sparse_quant_uses_i64_pointer_arithmetic():
    print("sparse quant pointer width")
    source = inspect.getsource(sparse_quant._quantize_blocks.fn)
    check(source.count(".to(tl.int64)") >= 15,
          "Q/K source, mean, output, and scale offsets use int64")


def test_report_files():
    print("request report")
    output_root = os.path.join("output", "h3_hybrid_sparse")
    with mock.patch("h3_attention.hybrid.report.os.makedirs") as makedirs, \
            mock.patch("h3_attention.hybrid.report.open", mock.mock_open()) as opened:
        collector = HybridStatsCollector(output_root, "hybrid50")
        collector.on_request_reset(0)
        collector.record({
            "layer": 0,
            "step": 0,
            "requested_video_budget": 0.5,
            "actual_video_tile_density": 0.5,
            "full_mask_density": 0.6,
        })
        report_dir = collector.on_request_end(0, 1.25)
        names = [call.args[0] for call in opened.call_args_list]
        check(makedirs.call_args.args[0] == report_dir,
              "request end creates one tagged report directory")
        check(os.path.join(report_dir, "report.json") in names,
              "request end writes report.json")
        check(os.path.join(report_dir, "report.txt") in names,
              "request end writes report.txt")


def _dense_reference(q, k, v, lut, valid):
    sequence = q.shape[-2]
    block_mask = torch.zeros(lut.shape, dtype=torch.bool, device=lut.device)
    for q_index in range(lut.shape[-2]):
        count = valid[..., q_index].max().item()
        if count:
            indices = torch.cumsum(lut[..., q_index, :int(count)], dim=-1)
            block_mask[..., q_index, :] = torch.nn.functional.one_hot(
                indices.long(), num_classes=lut.shape[-1]).any(dim=-2)
    token_mask = block_mask.repeat_interleave(128, dim=-2).repeat_interleave(64, dim=-1)
    token_mask = token_mask[..., :sequence, :sequence]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (128 ** -0.5)
    scores.masked_fill_(~token_mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def optional_cuda_numerical():
    if os.environ.get("H3_RUN_SPARSE_SAGE_CUDA_TESTS") != "1":
        print("CUDA numerical parity: SKIP (set H3_RUN_SPARSE_SAGE_CUDA_TESTS=1 after authorization)")
        return
    api = load_sparse_sage_spec()
    for budget in (1.0, 0.5):
        q, k, v = fused_hnd(sequence=256, heads=2, device="cuda", dtype=torch.float16)
        config = HybridSparseConfig(video_budget=budget)
        hybrid = HybridSparseBackend(config, kernel_spec=api)
        prepared = hybrid.prepare(
            q, k, v, layer_index=0,
            transformer_options=options(256, 128, torch.device("cuda")),
        )
        reference = _dense_reference(q, k, v, prepared.sparse.lut,
                                     prepared.sparse.valid_block_num)
        output = hybrid.execute(prepared)
        error = ((output.float() - reference.float()).square().mean().sqrt()
                 / reference.float().square().mean().sqrt().clamp_min(1e-8)).item()
        check(error < 0.08, "Sparse Sage %.0f%% matches explicit masked attention" % (100 * budget))


def main():
    test_kernel_spec_resolution()
    test_architecture_specific_carriers_and_abis()
    test_prepare_execute_lifetime()
    test_strict_errors()
    test_dependency_and_disabled_node()
    test_node_mode_schema()
    test_node_rejects_invalid_compile_configuration_before_preflight()
    test_sparse_quant_uses_i64_pointer_arithmetic()
    test_report_files()
    test_deferred_timing()
    test_per_step_timing()
    optional_cuda_numerical()
    print("\nall hybrid attention tests passed")


if __name__ == "__main__":
    main()
