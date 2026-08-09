"""CPU contract tests plus explicitly opt-in CUDA checks for Phase A."""

import os
import sys
from types import SimpleNamespace
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

from h3_attention.hybrid import (  # noqa: E402
    DeferredCudaTiming,
    HybridSparseBackend,
    HybridSparseConfig,
    HybridStatsCollector,
    TIMING_STAGES,
    SparseSageAPI,
    SparseSageError,
    load_sparse_sage_api,
)
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttention  # noqa: E402
from h3_attention.hybrid.report import render  # noqa: E402
from h3_attention.hybrid.sparse_sage import quantize_qk  # noqa: E402


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
    api = SparseSageAPI(version="0.1.test", low_level_f16=FakeSparseKernel(),
                        low_level_f32=FakeSparseKernel())
    hybrid = HybridSparseBackend(
        config, api=api, collector=collector, event_factory=factory,
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
        for stage in ("direct_lut_construction", "v_fp8_preparation",
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


def backend(kernel=None, collector=None, budget=0.5):
    kernel = kernel or FakeSparseKernel()
    api = SparseSageAPI(version="0.1.test", low_level_f16=kernel,
                        low_level_f32=kernel)
    config = HybridSparseConfig(video_budget=budget)
    return HybridSparseBackend(
        config,
        api=api,
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
    check(sparse.v_fp8.dtype == torch.float8_e4m3fn,
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
            load_sparse_sage_api()
        except SparseSageError as exc:
            check("compiled" in str(exc), "missing SpargeAttention raises explicit dependency error")
        else:
            raise AssertionError("missing SpargeAttention must fail")

    marker = object()
    result = MiniMaxH3HybridSparseAttention.execute(marker, enabled=False)
    check(result.args[0] is marker, "disabled node is an exact model pass-through")


def test_node_mode_schema():
    print("node mode schema")
    schema = MiniMaxH3HybridSparseAttention.define_schema()
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
    api = load_sparse_sage_api()
    for budget in (1.0, 0.5):
        q, k, v = fused_hnd(sequence=256, heads=2, device="cuda", dtype=torch.float16)
        config = HybridSparseConfig(video_budget=budget)
        hybrid = HybridSparseBackend(config, api=api)
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
    test_prepare_execute_lifetime()
    test_strict_errors()
    test_dependency_and_disabled_node()
    test_node_mode_schema()
    test_report_files()
    test_deferred_timing()
    optional_cuda_numerical()
    print("\nall hybrid attention tests passed")


if __name__ == "__main__":
    main()
