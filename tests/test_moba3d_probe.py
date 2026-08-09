"""CPU self-test for the H3 3D MoBA-style probe. No checkpoint required.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_moba3d_probe.py
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention.observer import notify_attention, observing  # noqa: E402
from h3_mask import predictor  # noqa: E402
from h3_probe import layout as h3_layout  # noqa: E402
from h3_probe import latent_dynamics, moba3d, moba_capture, moba_report  # noqa: E402
from tools import analyze_h3_active_masks  # noqa: E402

TEXT_LEN = 24
LATENT_T, LATENT_H, LATENT_W = 4, 8, 12
AUDIO_T = 8
HEADS, DIM = 2, 8


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def build_layout():
    # Keep this probe CPU-isolated: importing core PackedLayout pulls optional
    # model dependencies (including comfy_aimdo) that are not needed here.
    video_rows = LATENT_T * (LATENT_H // 2) * (LATENT_W // 2)
    audio_start = TEXT_LEN
    video_start = audio_start + 2 * AUDIO_T
    packed = SimpleNamespace(
        signature=(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T),
        seq_len=video_start + video_rows,
        segments=[
            (0, TEXT_LEN, "text"),
            (audio_start, video_start, "audio"),
            (video_start, video_start + video_rows, "video"),
        ],
    )
    return h3_layout.from_packed_layout(packed)


def test_budgets():
    check(moba3d.parse_budgets("10,25,50") == (0.1, 0.25, 0.5), "percent budgets parse")
    check(moba3d.parse_budgets("0.1,0.5") == (0.1, 0.5), "fraction budgets parse")
    check(
        moba3d.parse_budgets((0.1, 0.25, 0.5)) == (0.1, 0.25, 0.5),
        "already-normalized tuple budgets are idempotent",
    )
    check(
        moba3d.parse_budgets([10, 25, 50]) == (0.1, 0.25, 0.5),
        "numeric iterable percent budgets parse",
    )
    check(moba3d.parse_budgets("auto") == moba3d.DEFAULT_BUDGETS, "auto budgets use defaults")


def test_block_map(lay):
    ids, counts, grid, n = moba3d._video_block_map(lay, 2, 2, 3, torch.device("cpu"))
    check(ids.numel() == lay.video_range[1] - lay.video_range[0], "every video token gets a 3D block")
    check(int(counts.sum()) == ids.numel(), "3D block counts conserve video tokens")
    check(n == grid[0] * grid[1] * grid[2], "block-grid dimensions multiply to block count")
    check(int(ids.max()) < n, "block ids stay in range")


def test_observer_carries_v():
    seen = []
    q = torch.zeros(1, 2, 4, 8)
    k = torch.ones(1, 2, 4, 8)
    v = torch.full((1, 2, 4, 8), 2.0)

    def observer(q_, k_, v_, layer_index):
        seen.append((float(q_.sum()), float(k_.sum()), float(v_.sum()), layer_index))

    options = {}
    with observing(options, observer):
        notify_attention(q, k, v, layer_index=7, transformer_options=options)
    check(
        seen == [(0.0, float(k.sum()), float(v.sum()), 7)],
        "attention observer receives Q/K/V and layer index",
    )


def test_per_query_routing_and_output(lay):
    """Different query tokens must be allowed to choose different video blocks."""
    seq = lay.seq_len
    q = torch.zeros(1, HEADS, seq, DIM)
    k = torch.zeros(1, HEADS, seq, DIM)
    v = torch.zeros(1, HEADS, seq, DIM)

    prepared = moba3d.prepare_video_router(
        k,
        lay,
        block_t=lay.video_shape[0],
        block_h=lay.video_shape[1],
        block_w=lay.video_shape[2] // 2,
    )
    ids = prepared["block_ids"]
    check(prepared["n_blocks"] == 2, "test geometry has two video blocks")

    v0, _v1 = lay.video_range
    left = torch.nonzero(ids == 0, as_tuple=False).flatten() + v0
    right = torch.nonzero(ids == 1, as_tuple=False).flatten() + v0

    # Query token 0 wants the left block; query token 1 wants the right block.
    q[:, :, 0, 0] = 8.0
    q[:, :, 1, 1] = 8.0
    k[:, :, left, 0] = 10.0
    k[:, :, right, 1] = 10.0

    # Distinct values make a wrong route visible as output error.
    v[:, :, left, 0] = 2.0
    v[:, :, right, 1] = -3.0

    prepared = moba3d.prepare_video_router(
        k,
        lay,
        block_t=lay.video_shape[0],
        block_h=lay.video_shape[1],
        block_w=lay.video_shape[2] // 2,
    )
    result = moba3d.analyze_routing(
        q,
        k,
        v,
        lay,
        0,
        2,
        block_t=lay.video_shape[0],
        block_h=lay.video_shape[1],
        block_w=lay.video_shape[2] // 2,
        budgets="50,100",
        head_chunk=1,
        prepared=prepared,
    )

    half = result["budgets"][0]
    full = result["budgets"][1]
    check(
        half["video_blocks"] == 2 and half["keep_blocks"] == 1,
        "50% budget keeps one of two video blocks per query token",
    )
    check(half["routed_mass_mean"] > 0.999, "per-token router retains the planted dense mass")
    check(
        half["sparse_output_rel_l2_max_head"] < 1e-4,
        "per-token routed sparse output matches dense output",
    )
    check(half["routing_regret_mean"] < 1e-5, "per-token planted case is effectively oracle")
    check(full["routed_mass_min"] > 0.999999, "100% video budget retains all dense probability mass")
    check(
        full["sparse_output_rel_l2_max_head"] < 1e-6,
        "100% video budget reproduces dense attention output",
    )


def test_random_output_metrics(lay):
    torch.manual_seed(17)
    seq = lay.seq_len
    q = torch.randn(1, HEADS, seq, DIM)
    k = torch.randn(1, HEADS, seq, DIM)
    v = torch.randn(1, HEADS, seq, DIM)
    q0, _ = lay.video_frame_range(0)

    result = moba3d.analyze_routing(
        q,
        k,
        v,
        lay,
        q0,
        q0 + min(8, lay.frame_rows),
        block_t=1,
        block_h=2,
        block_w=2,
        budgets=(0.25, 0.5),
        head_chunk=1,
    )

    for row in result["budgets"]:
        check(0.0 <= row["routed_mass_mean"] <= 1.0001, "routed mass is a probability")
        check(
            row["oracle_mass_mean"] + 1e-6 >= row["routed_mass_mean"],
            "oracle never underperforms router on retained dense mass",
        )
        check(row["routing_regret_mean"] >= -1e-6, "routing regret is non-negative")
        check(0.0 <= row["oracle_block_overlap_mean"] <= 1.0, "oracle overlap is bounded")
        check(
            0.0 < row["effective_token_density_mean"] <= 1.0,
            "effective token density is bounded",
        )
        check(row["sparse_output_rel_l2_max_head"] >= 0.0, "sparse output relative-L2 is non-negative")
        check(len(row["head_rel_l2"]) == HEADS, "per-head sparse output errors are retained")
        check(len(row["oracle_head_rel_l2"]) == HEADS, "per-head oracle output errors are retained")


def test_sage_sparse_execution_geometry(lay):
    """Sparse-Sage coarsening uses global packed Q/KV tiles, not local regions."""
    v0, v1 = lay.video_range
    video_tokens = v1 - v0
    logical = torch.zeros(1, 8, video_tokens, dtype=torch.bool)
    # Two queries in one global Q tile choose different video KV tiles.
    logical[0, 0, 0] = True
    logical[0, 1, 5] = True
    logical[0, 2, 0] = True
    logical[0, 3, 5] = True
    keep, meta = moba3d._execution_mask(
        logical,
        2,
        3,
        lay.seq_len,
        lay.video_range,
        q_tile=8,
        kv_tile=4,
        aligned_start=0,
        aligned_end=8,
    )
    check(meta["q_range"] == [0, 8], "sage Q range is globally aligned, not sampled-local")
    check(bool(keep[0, 0, v0 + 0]) and bool(keep[0, 0, v0 + 5]),
          "queries sharing one Q tile union per-head KV selections")

    # With 64-token KV tiles, the first tile contains context and video rows;
    # context density forces the complete tile on.
    mixed_logical = torch.zeros(1, 8, video_tokens, dtype=torch.bool)
    mixed_logical[0, 0, 0] = True
    mixed, _ = moba3d._execution_mask(
        mixed_logical,
        1,
        2,
        lay.seq_len,
        lay.video_range,
        q_tile=8,
        kv_tile=64,
        aligned_start=0,
        aligned_end=8,
    )
    check(bool(mixed[0, 0, :64].all()),
          "KV tile mixing non-video and video is fully enabled")

    torch.manual_seed(23)
    q = torch.randn(1, HEADS, lay.seq_len, DIM)
    k = torch.randn_like(q)
    val = torch.randn_like(q)
    result = moba3d.analyze_routing(
        q, k, val, lay, 27, 35,
        block_t=1, block_h=2, block_w=2,
        budgets=(0.25,), execution_geometry="sage_sparse",
        sage_q_tile=8, sage_kv_tile=4, head_chunk=1,
    )
    row = result["budgets"][0]
    check(result["requested_q_range"] == [27, 35],
          "sage execution preserves the caller's requested Q range")
    check(result["evaluated_q_range"] == [24, 32]
          and result["execution_q_range"] == [24, 32]
          and result["execution_q_tiles"] == 1,
          "one-tile misaligned request evaluates exactly one aligned Q tile")
    check(row["execution_geometry"] == "sage_sparse",
          "sage execution geometry is retained in each budget row")
    check(row["executable_effective_token_density_mean"] + 1e-6 >= row["effective_token_density_mean"],
          "executable density is at least logical density")
    check(0.0 <= row["executable_sparse_output_rel_l2_max_head"],
          "executable sparse output metric is present and bounded")
    check(len(row["executable_head_rel_l2"]) == HEADS,
          "executable per-head errors are retained")

    larger = moba3d.analyze_routing(
        q, k, val, lay, 27, 43,
        block_t=1, block_h=2, block_w=2,
        budgets=(0.25,), execution_geometry="sage_sparse",
        sage_q_tile=8, sage_kv_tile=4, head_chunk=1,
    )
    check(larger["requested_q_range"] == [27, 43]
          and larger["evaluated_q_range"] == [24, 40]
          and larger["execution_q_range"] == [24, 40]
          and larger["execution_q_tiles"] == 2,
          "larger request evaluates the expected whole number of aligned Q tiles")


def _prediction_row(result, threshold, profile):
    return next(
        row for row in result["rows"]
        if row["threshold"] == threshold and row["profile"] == profile
    )


def test_active_mask_predictor():
    print("causal active-mask predictor")
    previous = torch.zeros(3, 5, 5)
    previous[1, 2, 2] = 0.10
    current = torch.zeros_like(previous)
    energy = torch.zeros_like(previous)
    for position, value in (
        ((1, 2, 2), 1.0),
        ((1, 2, 3), 4.0),
        ((2, 2, 2), 9.0),
        ((0, 0, 0), 16.0),
    ):
        current[position] = 0.10
        energy[position] = value

    result = predictor.evaluate_predictability(
        previous, current, energy, torch.ones_like(energy), total_values=300
    )
    exact = _prediction_row(result, 0.05, "exact")
    spatial = _prediction_row(result, 0.05, "spatial_1")
    temporal = _prediction_row(result, 0.05, "temporal_1")
    spatiotemporal = _prediction_row(result, 0.05, "spatiotemporal_1")

    check(abs(exact["predicted_active_fraction"] - 1 / 75) < 1e-9,
          "exact mask retains only the previously active token")
    check(abs(exact["current_active_fraction"] - 4 / 75) < 1e-9
          and abs(exact["next_active_recall"] - 0.25) < 1e-9,
          "same-threshold next-active recall uses only prior information")
    check(abs(exact["captured_energy_fraction"] - 1 / 30) < 1e-9,
          "exact mask captures only the planted stationary update energy")
    check(abs(spatial["captured_energy_fraction"] - 5 / 30) < 1e-9,
          "spatial_1 captures the planted same-frame neighbour")
    check(abs(temporal["captured_energy_fraction"] - 10 / 30) < 1e-9,
          "temporal_1 captures the planted next-frame neighbour")
    check(abs(spatiotemporal["captured_energy_fraction"] - 14 / 30) < 1e-9,
          "spatiotemporal_1 captures both planted halo neighbours")
    check(abs(exact["false_freeze_rate"] - 3 / 75) < 1e-9,
          "false-freeze fraction counts missed active tokens over all tokens")
    transition = result["transition_probabilities"]["2%"]["5%"]
    check(transition["denominator"] == 74 and transition["count"] == 3,
          "transition matrix excludes the previously active token causally")
    zero = predictor.evaluate_predictability(
        torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)
    )
    zero_exact = _prediction_row(zero, 0.05, "exact")
    check(zero_exact["captured_energy_fraction"] == 1.0
          and zero_exact["missed_energy_fraction"] == 0.0,
          "a zero-energy update is fully preserved by convention")


def test_tracker_causal_staging():
    print("tracker causal staging")
    run = SimpleNamespace(
        anchor_frames=[],
        layout=SimpleNamespace(video_range=(0, 27)),
        latent_activity_maps=[],
        latent_energy_maps=[],
    )
    tracker = latent_dynamics.LatentDynamicsTracker()
    baseline = torch.ones(1, 1, 3, 6, 6)
    check(tracker.capture(run, 0, baseline, baseline, 3, None, ()) is None,
          "first callback only establishes the x0 baseline")

    first = baseline.clone()
    first[:, :, 1, 2:4, 2:4] += 0.10
    staged = tracker.capture(run, 1, first, first, 3, None, ())
    check(staged["predictability"] is None and len(run.latent_activity_maps) == 1,
          "second callback stages activity without scoring it against itself")

    second = first.clone()
    second[:, :, 1, 2:4, 4:6] += 0.20
    evaluated = tracker.capture(run, 2, second, second, 3, None, ())
    check(evaluated["from_step"] == 1 and evaluated["to_step"] == 2,
          "third callback labels the prior-to-current causal transition")
    exact = _prediction_row(evaluated["predictability"], 0.05, "exact")
    spatial = _prediction_row(evaluated["predictability"], 0.05, "spatial_1")
    check(exact["captured_energy_fraction"] < 1e-4,
          "exact prior mask misses essentially all newly moving-patch energy")
    check(spatial["captured_energy_fraction"] > 0.9999,
          "spatial_1 causally captures the planted moving patch")
    check(len(run.latent_activity_maps) == 2 and len(run.latent_energy_maps) == 2,
          "every x0 delta is retained once for offline calibration")
    tracker.close()
    check(tracker.previous_prediction_activity is None,
          "closing a run clears predictor state")


def test_tracker_excludes_forecast_predictions():
    print("tracker forecast exclusion")
    run = SimpleNamespace(
        anchor_frames=[],
        layout=SimpleNamespace(video_range=(0, 27)),
        latent_activity_maps=[],
        latent_energy_maps=[],
    )
    tracker = latent_dynamics.LatentDynamicsTracker()
    baseline = torch.ones(1, 1, 3, 6, 6)
    tracker.capture(run, 0, baseline, baseline, 3, None, ())

    synthetic = baseline + 10.0
    forecast = tracker.capture(
        run, 1, synthetic, synthetic, 3, None, (),
        callback_metadata={
            "h3_vector_forecast": True,
            "h3_vector_true_nfe": 1,
        },
    )
    check(forecast["h3_vector_forecast"] is True
          and forecast["global"]["prediction"] is None,
          "forecast callback is labelled and excluded from prediction deltas")
    check(tracker.previous_prediction_step == 0
          and len(run.latent_activity_maps) == 0,
          "synthetic x0 does not become an actual prediction anchor")

    actual = baseline + 0.25
    resumed = tracker.capture(
        run, 2, actual, actual, 3, None, (),
        callback_metadata={"h3_vector_forecast": False},
    )
    check(resumed["global"]["prediction"]["update_rms"] < 1.0,
          "next genuine prediction compares with the prior genuine anchor")
    check(tracker.previous_prediction_step == 2
          and len(run.latent_activity_maps) == 1,
          "genuine callback resumes the actual-only prediction stream")


def test_attention_bypass_keeps_dynamics_layout(lay):
    print("attention bypass")
    session = moba_capture.MobaProbeSession(
        tag="bypass", layers_spec="auto", steps_spec="auto", n_time=2,
        n_spatial=1, query_block=16, include_audio=False, include_text=False,
        capture_uncond=False, capture_latent_dynamics=True, block_t=1,
        block_h=2, block_w=2, budgets="25", base_dir=".",
        capture_attention=False,
    )
    run = session.begin()
    original = moba_capture.h3_layout.resolve_layout
    moba_capture.h3_layout.resolve_layout = lambda *_args, **_kwargs: lay
    marker = object()

    def executor(*_args, **_kwargs):
        return marker

    try:
        wrapped = moba_capture.make_wrapper(session)
        options = {"sample_sigmas": torch.tensor([1.0, 0.5, 0.0])}
        got = wrapped(executor, torch.zeros(1), None, torch.zeros(1), options)
    finally:
        moba_capture.h3_layout.resolve_layout = original
        session.run = None

    check(got is marker and run.layout is lay,
          "attention-off path delegates densely after resolving the shared layout")
    check(run.anchor_frames == [] and bool(run.dynamics_queries),
          "attention-off path still initializes anchors and dynamics queries")
    check(run.records == [], "attention-off path creates no Q/K/V routing records")


def test_latent_dynamics_math():
    previous = torch.ones(1, 2, 3, 4, 4)
    current = previous.clone()
    current[:, :, 1] += 0.01
    current[:, :, 2] += 0.20

    stream = latent_dynamics._stream_update(previous, current)
    updates = [row["update_rel_l2"] for row in stream["frames"]]
    check(abs(updates[0]) < 1e-8, "unchanged latent frame has zero update")
    check(updates[0] < updates[1] < updates[2], "frame update metric tracks planted change magnitude")
    check(stream["patch_shape"] == (3, 2, 2), "latent dynamics reduce to H3 1x2x2 patches")
    check(
        abs(stream["global"]["stable_patch_fraction"]["1%"] - (2.0 / 3.0)) < 1e-6,
        "stable-patch fraction counts unchanged and 1% frames",
    )

    tiny_layout = SimpleNamespace(video_range=(100, 112))
    region = latent_dynamics._region_metrics(stream, tiny_layout, 104, 108)
    check(abs(region["update_rel_l2"] - 0.01) < 1e-5, "query-region update matches planted middle frame")

    anchors = latent_dynamics.resolve_anchor_frames(
        {
            "frame_count": 125,
            "keyframes": [
                {"resolved_frame_index": 0},
                {"resolved_frame_index": 124},
            ],
        },
        62,
    )
    check(anchors == [0, 61], "pixel first/last keyframes map to latent endpoints")
    check(latent_dynamics.anchor_distance(30, anchors) == 30, "anchor distance uses nearest explicit keyframe")


def test_latent_dynamics_summary():
    frames = [
        {
            "frame": i,
            "anchor_distance": i,
            "sample": {"update_rel_l2": 0.01 * (i + 1)},
            "prediction": {"update_rel_l2": 0.02 * (i + 1)},
        }
        for i in range(4)
    ]
    dynamics = [
        {
            "step": 3,
            "anchor_frames": [0],
            "frames": frames,
            "query_regions": [],
            "global": {
                "sample": {"update_rel_l2": 0.02},
                "prediction": {"update_rel_l2": 0.04},
            },
        }
    ]
    summary = latent_dynamics.summarize_dynamics(dynamics)
    corr = summary["by_step"]["3"]["anchor_distance_vs_prediction_update_pearson"]
    check(corr is not None and corr > 0.999, "anchor-distance/update correlation is reported")


def test_report(lay):
    class Run:
        tag = "unit"
        layout = lay
        layers = {0}
        steps = {0}
        notes = {"num_layers": 1, "total_steps": 1}
        block_t, block_h, block_w = 1, 2, 2
        budgets = (0.25,)
        execution_geometry = "sage_sparse"
        sage_q_tile, sage_kv_tile = 128, 64
        capture_latent_dynamics = True
        anchor_frames = [0]

    row = {
        "budget": 0.25,
        "keep_blocks": 4,
        "video_blocks": 16,
        "video_block_density": 0.25,
        "routed_mass_mean": 0.9,
        "routed_mass_min": 0.82,
        "oracle_mass_mean": 0.94,
        "oracle_mass_min": 0.88,
        "routing_regret_mean": 0.04,
        "routing_regret_max": 0.08,
        "oracle_block_overlap_mean": 0.75,
        "oracle_block_overlap_min": 0.6,
        "effective_token_density_mean": 0.5,
        "effective_token_density_max": 0.52,
        "sparse_output_rel_l2_mean_head": 0.012,
        "sparse_output_rel_l2_median_head": 0.01,
        "sparse_output_rel_l2_max_head": 0.03,
        "sparse_output_mean_abs_mean_head": 0.001,
        "sparse_output_max_abs": 0.02,
        "oracle_output_rel_l2_mean_head": 0.005,
        "oracle_output_rel_l2_max_head": 0.01,
        "oracle_output_mean_abs_mean_head": 0.0005,
        "oracle_output_max_abs": 0.01,
        "head_rel_l2": [0.03, 0.005],
        "oracle_head_rel_l2": [0.01, 0.002],
        "heads_rel_l2_gt_1pct": [0],
        "heads_rel_l2_gt_2pct": [0],
        "heads_rel_l2_gt_5pct": [],
        "worst_heads": [
            {"head": 0, "rel_l2": 0.03},
            {"head": 1, "rel_l2": 0.005},
        ],
        "execution_geometry": "sage_sparse",
        "executable_effective_token_density_mean": 0.75,
        "executable_effective_token_density_max": 0.78,
        "executable_sparse_output_rel_l2_mean_head": 0.02,
        "executable_sparse_output_rel_l2_max_head": 0.04,
        "executable_head_rel_l2": [0.04, 0.01],
        "executable_heads_rel_l2_gt_1pct": [0],
        "executable_heads_rel_l2_gt_2pct": [0],
        "executable_heads_rel_l2_gt_5pct": [],
        "executable_worst_heads": [
            {"head": 0, "rel_l2": 0.04},
            {"head": 1, "rel_l2": 0.01},
        ],
    }
    rec = {
        "kind": "video",
        "frame": 0,
        "start": lay.video_range[0],
        "stop": lay.video_range[0] + 4,
        "layer": 0,
        "step": 0,
        "sigma": 1.0,
        "cond_or_uncond": 0,
        "moba3d": {
            "routing_granularity": "per-query-token",
            "execution_geometry": "sage_sparse",
            "execution_q_tile": 128,
            "execution_kv_tile": 64,
            "requested_q_range": [27, 155],
            "evaluated_q_range": [0, 128],
            "execution_q_range": [0, 128],
            "execution_q_tiles": 1,
            "block_shape": [1, 2, 2],
            "block_grid": [4, 2, 2],
            "video_blocks": 16,
            "video_tokens": lay.video_range[1] - lay.video_range[0],
            "nonvideo_tokens": lay.seq_len - (lay.video_range[1] - lay.video_range[0]),
            "dense_nonvideo_mass_mean": 0.2,
            "dense_video_mass_mean": 0.8,
            "budgets": [row],
        },
    }
    Run.records = [rec]
    Run.latent_dynamics = [
        {
            "step": 0,
            "total_steps": 2,
            "anchor_frames": [0],
            "frames": [
                {
                    "frame": 0,
                    "anchor_distance": 0,
                    "sample": {"update_rel_l2": 0.01},
                    "prediction": {"update_rel_l2": 0.02},
                }
            ],
            "query_regions": [
                {
                    "kind": "video",
                    "frame": 0,
                    "spatial_offset": 0,
                    "start": rec["start"],
                    "stop": rec["stop"],
                    "anchor_distance": 0,
                    "sample": {"update_rel_l2": 0.01},
                    "prediction": {"update_rel_l2": 0.02},
                }
            ],
            "global": {
                "sample": {
                    "update_rel_l2": 0.01,
                    "stable_patch_fraction": {"1%": 0.5, "2%": 0.75, "5%": 1.0},
                },
                "prediction": {
                    "update_rel_l2": 0.02,
                    "stable_patch_fraction": {"1%": 0.25, "2%": 0.5, "5%": 0.9},
                },
            },
            "patch_shape": [4, 4, 6],
        }
    ]
    summary = moba_report.summarize(Run.records)
    dyn_summary = latent_dynamics.summarize_dynamics(Run.latent_dynamics, Run.records)
    text = moba_report.render(Run, summary, dyn_summary)
    check(
        "sparse output rel-L2" in text
        and "LAYER / HEAD DIAGNOSTICS" in text
        and "oracle" in text,
        "report exposes output error, layer/head diagnostics and oracle",
    )
    check(
        "execution:    sage_sparse" in text
        and "logical effective KV" in text
        and "executable effective KV" in text,
        "report labels logical and executable geometry separately",
    )
    check(
        "sage Q range: requested 27-155 | evaluated 0-128 | Q tiles 1" in text,
        "report exposes requested versus evaluated sage Q ranges",
    )
    check(
        "LATENT DYNAMICS" in text and "latent dynamics: sample" in text,
        "report exposes sampler convergence beside matching attention queries",
    )


def test_dynamics_archive_and_offline_analyzer(lay, tmp):
    print("raw dynamics archive and offline analyzer")
    activity0 = torch.zeros(2, 5, 5)
    activity0[0, 0, 0] = 0.10
    activity1 = torch.zeros_like(activity0)
    activity1[0, 0, 1] = 0.10
    energy0 = torch.zeros_like(activity0)
    energy0[0, 0, 0] = 1.0
    energy1 = torch.zeros_like(activity0)
    energy1[0, 0, 1] = 4.0
    predictability = predictor.evaluate_predictability(
        activity0, activity1, energy1, torch.ones_like(energy1), total_values=200
    )
    predictability.update({"from_step": 1, "to_step": 2})

    run = SimpleNamespace(
        tag="offline", out_dir=tmp, layout=lay,
        layers=set(), steps=set(), notes={"total_steps": 3, "num_layers": 50},
        block_t=1, block_h=2, block_w=2, budgets=(0.25,), records=[],
        capture_attention=False, capture_latent_dynamics=True, anchor_frames=[],
        latent_dynamics=[{
            "step": 2, "total_steps": 3, "anchor_frames": [], "frames": [],
            "query_regions": [], "global": {"sample": None, "prediction": None},
            "patch_shape": [2, 5, 5], "from_step": 1, "to_step": 2,
            "predictability": predictability,
        }],
        latent_activity_maps=[
            {"index": 0, "step": 1, "activity": activity0},
            {"index": 1, "step": 2, "activity": activity1},
        ],
        latent_energy_maps=[
            {"index": 0, "step": 1, "energy": energy0},
            {"index": 1, "step": 2, "energy": energy1},
        ],
    )
    report_path = moba_report.write_run(run)
    archive = os.path.join(run.out_dir, "latent_dynamics.npz")
    check(os.path.exists(report_path) and os.path.exists(archive),
          "dynamics-only report writes text and compressed raw maps")
    with np.load(archive) as saved:
        check(saved["activity_0000_step1"].dtype == np.float32
              and saved["energy_0001_step2"].dtype == np.float32,
              "raw activity and update energy stay float32 in the archive")

    result = analyze_h3_active_masks.analyze(archive)
    check(len(result["configurations"]) == 7 * 3 * 3 * 3
          and bool(result["pareto_frontier"]),
          "offline analyzer sweeps all 189 policies and produces a Pareto frontier")
    tiled = analyze_h3_active_masks._mask(activity0.numpy(), 0.05, 2, 1, 0)
    check(int(tiled.sum()) == 16,
          "offline spatial halo expands on the tile grid before returning to tokens")
    json_path, text_path = analyze_h3_active_masks.write_analysis(archive, result)
    check(os.path.basename(json_path) == "active_mask_analysis.json"
          and os.path.exists(json_path) and os.path.exists(text_path),
          "offline Pareto artifacts use stable adjacent filenames")
    text = open(report_path, encoding="utf-8").read()
    check("attention:    OFF" in text and "ACTIVE-SET PREDICTABILITY (x0)" in text
          and "PER QUERY BLOCK" not in text,
          "dynamics-only text is compact and emphasizes causal active-set results")


def main():
    temp_root = os.path.abspath(os.path.join(_HERE, "..", ".agent", "tmp"))
    os.makedirs(temp_root, exist_ok=True)
    try:
        lay = build_layout()
        print("layout: %s" % lay.describe())
        test_budgets()
        test_block_map(lay)
        test_observer_carries_v()
        test_per_query_routing_and_output(lay)
        test_random_output_metrics(lay)
        test_sage_sparse_execution_geometry(lay)
        test_active_mask_predictor()
        test_tracker_causal_staging()
        test_tracker_excludes_forecast_predictions()
        test_attention_bypass_keeps_dynamics_layout(lay)
        test_latent_dynamics_math()
        test_latent_dynamics_summary()
        test_report(lay)
        test_dynamics_archive_and_offline_analyzer(lay, temp_root)
        print("\nall 3D MoBA probe self-tests passed")
    finally:
        for name in (
            "moba3d_report.txt", "moba3d_summary.json", "latent_dynamics.npz",
            "active_mask_analysis.json", "active_mask_analysis.txt",
        ):
            path = os.path.join(temp_root, name)
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    main()
