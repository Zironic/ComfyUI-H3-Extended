"""CPU contracts for adaptive-budget 128Q x 64KV Sparse-Sage routing."""

import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_attention.hybrid.config import (  # noqa: E402
    DENSITY_ADAPTIVE_BUDGET,
    DENSITY_FIXED,
    HybridSparseConfig,
)
from h3_attention.hybrid.report import render, summarize  # noqa: E402
from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402
from h3_attention.hybrid.stats import (  # noqa: E402
    ROUTE_HISTOGRAM_KEY,
    build_route_histogram,
    resolve_route_telemetry,
)


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


def decode(lut, valid):
    """Decode every head/query row with its own valid count."""
    indices = torch.cumsum(lut, dim=-1).long()
    rank = torch.arange(lut.shape[-1], device=lut.device)
    active = rank.view(1, 1, 1, -1) < valid[..., None]
    encoded = F.one_hot(indices.clamp(0, lut.shape[-1] - 1), lut.shape[-1]).bool()
    return (encoded & active[..., None]).any(dim=-2)


def summaries():
    q = torch.zeros((1, 2, 3, 4), dtype=torch.float32)
    k = torch.zeros((1, 2, 6, 4), dtype=torch.float32)

    # Each pure-video Q tile reads a different score component from its head's
    # K summaries, giving deliberately different concentration profiles.
    q[0, 0, 1] = torch.tensor((1.0, 0.0, 0.0, 0.0))
    q[0, 0, 2] = torch.tensor((0.0, 1.0, 0.0, 0.0))
    q[0, 1, 1] = torch.tensor((1.0, 0.0, 0.0, 0.0))
    q[0, 1, 2] = torch.tensor((0.0, 1.0, 0.0, 0.0))

    # head 0: concentrated first row, diffuse second row
    k[0, 0, 2:] = torch.tensor((
        (10.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
    ))
    # head 1: two graded rows
    k[0, 1, 2:] = torch.tensor((
        (4.0, 4.0, 0.0, 0.0),
        (3.0, 3.0, 0.0, 0.0),
        (2.0, 2.0, 0.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
    ))
    return q, k


def adaptive_config(**overrides):
    values = {
        "video_budget": 0.5,
        "density_mode": DENSITY_ADAPTIVE_BUDGET,
        "min_video_density": 0.25,
        "max_video_density": 1.0,
        "adaptive_temperature": 1.0,
    }
    values.update(overrides)
    return HybridSparseConfig(**values)


def test_fixed_parity():
    print("fixed mode parity")
    torch.manual_seed(31)
    q = torch.randn((1, 2, 3, 8))
    k = torch.randn((1, 2, 6, 8))
    implicit = SparseTileRouter().build_lut_from_summaries(q, k, layout(), 0.5)
    explicit = SparseTileRouter(HybridSparseConfig(
        video_budget=0.5,
        density_mode=DENSITY_FIXED,
    )).build_lut_from_summaries(q, k, layout(), 0.5)
    check(torch.equal(implicit[0], explicit[0]), "explicit fixed LUT is byte-identical")
    check(torch.equal(implicit[1], explicit[1]), "explicit fixed valid counts are identical")
    check(implicit[2].actual_video_tile_density == explicit[2].actual_video_tile_density,
          "fixed metadata retains the established density")


def test_exact_budget_and_variable_counts():
    print("adaptive exact aggregate budget")
    q, k = summaries()
    router = SparseTileRouter(adaptive_config())
    lut, valid, metadata = router.build_lut_from_summaries(q, k, layout(), 0.5)
    video_counts = valid[..., 1:] - 2
    target_k = math.ceil(0.5 * 4)
    rows = video_counts.numel()
    check(int(video_counts.sum()) == target_k * rows,
          "adaptive row counts preserve the fixed route's exact total block budget")
    check(int(video_counts.min()) >= 1 and int(video_counts.max()) <= 4,
          "adaptive row counts respect quantized minimum and maximum rails")
    check(video_counts.unique().numel() > 1,
          "different score concentration produces different row densities")
    check(metadata.density_mode == DENSITY_ADAPTIVE_BUDGET,
          "metadata identifies adaptive-budget routing")
    check(metadata.actual_video_tile_density == 0.5,
          "aggregate metadata reports the exact executed mean density")
    check(metadata.allocation == "mass_bisection_exact_budget",
          "metadata names the allocation rule")

    mask = decode(lut, valid)
    check(mask[..., :2].all(), "all context KV tiles remain dense")
    check(mask[:, :, 0].all(), "all non-video Q tiles remain dense")
    for head in range(2):
        for row in range(2):
            count = int(video_counts[0, head, row])
            actual = torch.where(mask[0, head, row + 1, 2:])[0]
            check(actual.numel() == count,
                  "decoded adaptive row contains exactly its valid video block count")


def test_rows_retain_score_prefix():
    print("adaptive rows retain their own top-K prefix")
    q, k = summaries()
    router = SparseTileRouter(adaptive_config())
    lut, valid, _metadata = router.build_lut_from_summaries(q, k, layout(), 0.5)
    mask = decode(lut, valid)
    scores = torch.matmul(q[..., 1:, :], k[..., 2:, :].transpose(-1, -2))
    counts = valid[..., 1:] - 2
    for head in range(scores.shape[1]):
        for row in range(scores.shape[2]):
            count = int(counts[0, head, row])
            selected = set(torch.where(mask[0, head, row + 1, 2:])[0].tolist())
            expected = set(torch.topk(scores[0, head, row], count).indices.tolist())
            check(selected == expected, "variable row keeps its score-ranked top-K blocks")


def test_determinism():
    print("adaptive route determinism")
    q, k = summaries()
    router = SparseTileRouter(adaptive_config())
    first = router.build_lut_from_summaries(q, k, layout(), 0.5)
    second = router.build_lut_from_summaries(q, k, layout(), 0.5)
    check(torch.equal(first[0], second[0]), "adaptive LUT is deterministic")
    check(torch.equal(first[1], second[1]), "adaptive row counts are deterministic")


def test_direct_and_summary_routes_match():
    print("adaptive direct / fused-summary equivalence")
    q_summary, k_summary = summaries()
    q = q_summary.repeat_interleave(128, dim=-2)[..., :384, :].clone()
    k = k_summary.repeat_interleave(64, dim=-2)[..., :384, :].clone()
    router = SparseTileRouter(adaptive_config())
    direct = router.build_lut(q, k, layout(), 0.5)
    summary = router.build_lut_from_summaries(q_summary, k_summary, layout(), 0.5)
    check(torch.equal(direct[0], summary[0]), "direct and fused-summary adaptive LUTs match")
    check(torch.equal(direct[1], summary[1]), "direct and fused-summary valid counts match")
    check(direct[2] == summary[2], "direct and fused-summary metadata matches")


def test_full_budget_fast_path():
    print("adaptive 100% fast path")

    class NoPoolingRouter(SparseTileRouter):
        @staticmethod
        def _mean_pool(x, block):
            raise AssertionError("100% adaptive target must not pool Q/K")

    config = adaptive_config(
        video_budget=1.0,
        min_video_density=0.25,
        max_video_density=1.0,
    )
    q = torch.randn((1, 2, 384, 8))
    lut, valid, metadata = NoPoolingRouter(config).build_lut(q, q, layout(), 1.0)
    check(decode(lut, valid).all(), "100% adaptive target produces a dense block mask")
    check(metadata.full_mask_density == 1.0,
          "100% adaptive metadata reports a fully dense executable mask")


def test_reporting_telemetry():
    print("adaptive row-density reporting")
    q, k = summaries()
    router = SparseTileRouter(adaptive_config())
    _lut, valid, metadata = router.build_lut_from_summaries(q, k, layout(), 0.5)
    histogram = build_route_histogram(valid, metadata)
    check(histogram is not None and int(histogram.sum()) == 4,
          "deferred route histogram contains every head/query row")

    record = metadata.as_dict()
    record.update({
        "step": 3,
        "layer": 7,
        "request_id": 1,
        ROUTE_HISTOGRAM_KEY: histogram,
    })
    records, routing = resolve_route_telemetry([record])
    check(ROUTE_HISTOGRAM_KEY not in records[0],
          "private tensor telemetry is removed before JSON serialization")
    check(routing["adaptive_reallocation_observed"],
          "request summary explicitly says adaptive reallocation occurred")
    check(routing["unique_video_kv_tile_counts"] > 1
          and routing["min_video_kv_tiles"] < routing["max_video_kv_tiles"],
          "request summary exposes actual variable row K values")
    check(abs(routing["mean_video_kv_tiles"] - 2.0) < 1e-9,
          "reported row-K mean preserves the exact 50% budget")
    check(routing["rows_below_target"] > 0 and routing["rows_above_target"] > 0,
          "report counts rows that donated and received attention blocks")
    check(routing["per_step"][0]["step_index"] == 3
          and routing["per_layer"][0]["layer"] == 7,
          "report carries the row-K distribution into step and layer summaries")
    check("actual_row_min_video_kv_tiles" in records[0]
          and "actual_row_p50_video_kv_tiles" in records[0],
          "per-call records contain resolved row-K statistics")

    payload = {
        "mode": "sage128",
        "summary": summarize(records, route_summary=routing),
    }
    text = render(payload)
    check("adaptive reallocation observed: yes" in text
          and "actual row K:" in text
          and "adaptive step 3:" in text,
          "human report makes adaptive behavior directly visible")


def test_configuration_guards():
    print("adaptive configuration guards")
    cases = (
        ({"min_video_density": 0.6, "max_video_density": 0.5}, "min"),
        ({"video_budget": 0.2, "min_video_density": 0.3}, "between"),
        ({"adaptive_temperature": 0.0}, "temperature"),
        ({"adaptive_target_mass": 0.0}, "target_mass"),
        ({"density_mode": "unknown"}, "density_mode"),
    )
    for values, text in cases:
        try:
            adaptive_config(**values)
        except ValueError as exc:
            check(text in str(exc), "invalid adaptive configuration fails clearly")
        else:
            raise AssertionError("invalid adaptive configuration was accepted")


def main():
    test_fixed_parity()
    test_exact_budget_and_variable_counts()
    test_rows_retain_score_prefix()
    test_determinism()
    test_direct_and_summary_routes_match()
    test_full_budget_fast_path()
    test_reporting_telemetry()
    test_configuration_guards()
    print("\nall adaptive hybrid router tests passed")


if __name__ == "__main__":
    main()
