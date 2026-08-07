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
    check(moba3d.parse_budgets("auto") == moba3d.DEFAULT_BUDGETS, "auto budgets use defaults")


def test_block_map(lay):
    ids, counts, grid, n = moba3d._video_block_map(lay, 2, 2, 3, torch.device("cpu"))
    check(ids.numel() == lay.video_range[1] - lay.video_range[0], "every video token gets a 3D block")
    check(int(counts.sum()) == ids.numel(), "3D block counts conserve video tokens")
    check(n == grid[0] * grid[1] * grid[2], "block-grid dimensions multiply to block count")
    check(int(ids.max()) < n, "block ids stay in range")


def test_router_prefers_planted_block(lay):
    seq = lay.seq_len
    q = torch.zeros(1, HEADS, seq, DIM)
    k = torch.zeros(1, HEADS, seq, DIM)

    # Every selected query points along dimension zero. Plant one strong key
    # region in the final latent frame. Mean-pooled routing and exact dense mass
    # should agree that the corresponding 3D block matters.
    q[..., 0] = 6.0
    t0, _ = lay.video_frame_range(lay.latent_t - 1)
    _, ph, pw = lay.video_shape
    planted = []
    for y in range(min(2, ph)):
        for x in range(min(2, pw)):
            planted.append(t0 + y * pw + x)
    k[:, :, planted, 0] = 10.0

    q0, _ = lay.video_frame_range(0)
    result = moba3d.analyze_routing(
        q, k, lay, q0, q0 + min(8, lay.frame_rows),
        block_t=1, block_h=2, block_w=2, budgets="10,25,50", head_chunk=1,
    )

    check(result["video_blocks"] > 1, "test geometry has multiple video blocks")
    for row in result["budgets"]:
        check(0.0 <= row["routed_mass_mean"] <= 1.0001, "routed mass is a probability")
        check(row["oracle_mass_mean"] + 1e-6 >= row["routed_mass_mean"], "oracle never underperforms router")
        check(row["routing_regret_mean"] >= -1e-6, "routing regret is non-negative")
        check(0.0 <= row["oracle_block_overlap_mean"] <= 1.0, "oracle overlap is bounded")
        check(0.0 < row["effective_token_density_mean"] <= 1.0, "effective token density is bounded")

    first = result["budgets"][0]
    check(first["routed_mass_mean"] > 0.95, "mean-pooled router captures planted high-mass block at smallest budget")
    check(first["routing_regret_mean"] < 1e-3, "planted case is near oracle")


def test_report(lay):
    class Run:
        tag = "unit"
        layout = lay
        layers = {0}
        steps = {0}
        notes = {"num_layers": 1, "total_steps": 1}
        block_t, block_h, block_w = 1, 2, 2
        budgets = (0.25,)

    rec = {
        "kind": "video", "frame": 0, "start": lay.video_range[0], "stop": lay.video_range[0] + 4,
        "layer": 0, "step": 0, "sigma": 1.0, "cond_or_uncond": 0,
        "moba3d": {
            "block_shape": [1, 2, 2], "block_grid": [4, 2, 2], "video_blocks": 16,
            "video_tokens": lay.video_range[1] - lay.video_range[0],
            "nonvideo_tokens": lay.seq_len - (lay.video_range[1] - lay.video_range[0]),
            "dense_nonvideo_mass_mean": 0.2, "dense_video_mass_mean": 0.8,
            "budgets": [{
                "budget": 0.25, "keep_blocks": 4, "video_blocks": 16,
                "video_block_density": 0.25, "routed_mass_mean": 0.9,
                "routed_mass_min_head": 0.88, "oracle_mass_mean": 0.94,
                "oracle_mass_min_head": 0.92, "routing_regret_mean": 0.04,
                "routing_regret_max_head": 0.05, "oracle_block_overlap_mean": 0.75,
                "effective_token_density_mean": 0.5,
            }],
        },
    }
    Run.records = [rec]
    summary = moba_report.summarize(Run.records)
    text = moba_report.render(Run, summary)
    check("routed min" in text and "oracle min" in text and "max regret" in text,
          "report exposes routing quality and oracle ceiling")


def main():
    lay = build_layout()
    print("layout: %s" % lay.describe())
    test_budgets()
    test_block_map(lay)
    test_router_prefers_planted_block(lay)
    test_report(lay)
    print("\nall 3D MoBA probe self-tests passed")


if __name__ == "__main__":
    main()
