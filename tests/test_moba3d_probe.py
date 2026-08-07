"""CPU self-test for the H3 3D MoBA-style probe. No checkpoint required.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_moba3d_probe.py
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention.observer import notify_attention, observing  # noqa: E402
from h3_probe import layout as h3_layout  # noqa: E402
from h3_probe import moba3d, moba_report  # noqa: E402

TEXT_LEN = 24
LATENT_T, LATENT_H, LATENT_W = 4, 8, 12
AUDIO_T = 8
HEADS, DIM = 2, 8


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def build_layout():
    from comfy.ldm.minimax.model import PackedLayout

    packed = PackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T)
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


def test_report(lay):
    class Run:
        tag = "unit"
        layout = lay
        layers = {0}
        steps = {0}
        notes = {"num_layers": 1, "total_steps": 1}
        block_t, block_h, block_w = 1, 2, 2
        budgets = (0.25,)

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
    summary = moba_report.summarize(Run.records)
    text = moba_report.render(Run, summary)
    check(
        "sparse output rel-L2" in text
        and "LAYER / HEAD DIAGNOSTICS" in text
        and "oracle" in text,
        "report exposes output error, layer/head diagnostics and oracle",
    )


def main():
    lay = build_layout()
    print("layout: %s" % lay.describe())
    test_budgets()
    test_block_map(lay)
    test_observer_carries_v()
    test_per_query_routing_and_output(lay)
    test_random_output_metrics(lay)
    test_report(lay)
    print("\nall 3D MoBA probe self-tests passed")


if __name__ == "__main__":
    main()
