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
    HybridSparseBackend,
    HybridSparseConfig,
    HybridStatsCollector,
    SparseSageAPI,
    SparseSageError,
    load_sparse_sage_api,
)
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttention  # noqa: E402


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

    def __call__(self, q, k, v, **kwargs):
        if self.fail:
            raise RuntimeError("synthetic sparse failure")
        self.calls.append((q, k, v, kwargs))
        return q.clone()


class Collector:
    def __init__(self):
        self.records = []

    def record(self, metadata):
        self.records.append(dict(metadata))


def backend(kernel=None, collector=None, budget=0.5):
    kernel = kernel or FakeSparseKernel()
    api = SparseSageAPI(version="0.1.test", block_sparse=kernel)
    config = HybridSparseConfig(video_budget=budget)
    return HybridSparseBackend(
        config,
        api=api,
        collector=collector,
        allow_cpu_for_tests=True,
    ), kernel


def test_prepare_execute_lifetime():
    print("prepare / execute ownership")
    collector = Collector()
    hybrid, kernel = backend(collector=collector)
    q, k, v = fused_hnd()
    source = q.untyped_storage().data_ptr()
    prepared = hybrid.prepare(q, k, v, layer_index=7, transformer_options=options())
    sparse = prepared.sparse
    check(sparse.q.untyped_storage().data_ptr() != source,
          "prepared Q does not retain fused QKV storage")
    check(sparse.k.untyped_storage().data_ptr() != source,
          "prepared K does not retain fused QKV storage")
    check(sparse.v.untyped_storage().data_ptr() != source,
          "prepared V does not retain fused QKV storage")
    check(sparse.mask_id.shape == (1, 2, 3, 6) and sparse.mask_id.is_contiguous(),
          "prepared mask is contiguous per-head 128Q x 64KV geometry")
    del q, k, v
    output = hybrid.execute(prepared)
    check(output.shape == (1, 2, 384, 128), "hybrid backend returns HND output")
    check(output.dtype == torch.bfloat16, "output dtype matches H3 input")
    check(len(kernel.calls) == 1 and kernel.calls[0][3]["tensor_layout"] == "HND",
          "public Sparse Sage API is called once in HND layout")
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


def _dense_reference(q, k, v, block_mask):
    sequence = q.shape[-2]
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
        reference = _dense_reference(q, k, v, prepared.sparse.mask_id)
        output = hybrid.execute(prepared)
        error = ((output.float() - reference.float()).square().mean().sqrt()
                 / reference.float().square().mean().sqrt().clamp_min(1e-8)).item()
        check(error < 0.08, "Sparse Sage %.0f%% matches explicit masked attention" % (100 * budget))


def main():
    test_prepare_execute_lifetime()
    test_strict_errors()
    test_dependency_and_disabled_node()
    test_report_files()
    optional_cuda_numerical()
    print("\nall hybrid attention tests passed")


if __name__ == "__main__":
    main()
