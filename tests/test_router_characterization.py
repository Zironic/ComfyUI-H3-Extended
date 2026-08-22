"""CPU self-test for HASTE/static-topology router characterization.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_router_characterization.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")

from h3_probe import moba3d, router_characterization as rc  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def layout():
    return SimpleNamespace(
        seq_len=8,
        video_range=(0, 8),
        video_shape=(1, 2, 4),
        segments=[(0, 8, "video")],
        audio_t=0,
    )


def test_direct_router_full_budget_is_dense():
    lay = layout()
    torch.manual_seed(73)
    q = torch.randn(1, 2, lay.seq_len, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    metrics = rc._direct_tile_calibration(
        q,
        k,
        v,
        lay,
        0,
        8,
        budgets=(0.5, 1.0),
        head_chunk=1,
        q_tile=4,
        kv_tile=2,
    )
    full = metrics[1.0]
    half = metrics[0.5]
    check(full["direct_tile_keep_video_kv_tiles"] == 4,
          "100% budget keeps every pure-video KV tile")
    check(full["direct_tile_effective_token_density_mean"] > 0.999999,
          "100% direct tile route is fully dense")
    check(full["direct_tile_sparse_output_rel_l2_max_head"] < 1e-6,
          "100% direct tile route reproduces dense attention")
    check(half["direct_tile_keep_video_kv_tiles"] == 2,
          "50% direct tile route keeps exactly half the KV tiles")
    check(len(half["direct_tile_head_rel_l2"]) == 2,
          "direct calibration retains per-head sensitivity")


def test_wrapped_probe_exposes_direct_calibration():
    lay = layout()
    torch.manual_seed(79)
    q = torch.randn(1, 2, lay.seq_len, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    result = moba3d.analyze_routing(
        q,
        k,
        v,
        lay,
        0,
        8,
        block_t=1,
        block_h=1,
        block_w=1,
        budgets=(0.5,),
        head_chunk=1,
        execution_geometry="sage_sparse",
        sage_q_tile=4,
        sage_kv_tile=2,
    )
    row = result["budgets"][0]
    check(result.get("direct_tile_calibration") is True,
          "sage_sparse snapshots automatically include direct-tile calibration")
    check("direct_tile_head_rel_l2" in row,
          "budget row contains direct production-router per-head error")
    check("executable_q_tile_density_sweep" in row,
          "existing Q-mask sharing sweep remains composed with calibration")


def test_temporal_reuse_metrics_and_topology_archive():
    lay = layout()
    torch.manual_seed(83)
    q = torch.randn(1, 2, lay.seq_len, 4)
    k = torch.randn_like(q)

    tracker = rc.RouterDynamicsTracker(
        q_tile=4,
        kv_tile=2,
        budget=0.5,
        topology_q_samples=2,
    )
    first = tracker.capture(
        q,
        k,
        lay,
        step=0,
        sigma=1.0,
        branch=0,
        layer=3,
    )
    second = tracker.capture(
        q,
        k,
        lay,
        step=1,
        sigma=0.5,
        branch=0,
        layer=3,
    )

    check(first["q_cosine_mean"] is None,
          "first observation has no fabricated temporal comparison")
    check(abs(second["q_cosine_mean"] - 1.0) < 1e-6,
          "identical Q summaries report unit temporal cosine")
    check(abs(second["k_cosine_mean"] - 1.0) < 1e-6,
          "identical K summaries report unit temporal cosine")
    check(abs(second["exact_route_reuse_fraction_mean"] - 1.0) < 1e-6,
          "identical routes are recognized as exactly reusable")
    check(abs(second["sampled_route_jaccard_mean"] - 1.0) < 1e-6,
          "sampled topology Jaccard is exact for an unchanged route")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "router_topology.npz")
        tracker.write_topology(path)
        check(os.path.exists(path), "sampled static-topology archive is written")
        with np.load(path) as saved:
            counts = saved["layer_03_counts"]
            observations = int(saved["layer_03_observations"][0])
            check(observations == 2,
                  "topology archive records the number of accumulated denoising observations")
            check(counts.shape[:2] == (2, 2),
                  "topology archive preserves head and sampled-Q dimensions")
            check(int(counts.max()) == 2,
                  "unchanged selected tiles accumulate on both observations")


def test_changed_route_is_visible():
    lay = layout()
    q0 = torch.zeros(1, 1, 8, 4)
    k = torch.zeros_like(q0)
    # Four two-token KV tiles get distinct basis directions.
    k[:, :, 0:2, 0] = 8.0
    k[:, :, 2:4, 1] = 8.0
    k[:, :, 4:6, 2] = 8.0
    k[:, :, 6:8, 3] = 8.0
    q0[:, :, :, 0] = 8.0
    q1 = q0.clone()
    q1.zero_()
    q1[:, :, :, 3] = 8.0

    tracker = rc.RouterDynamicsTracker(
        q_tile=4,
        kv_tile=2,
        budget=0.25,
        topology_q_samples=2,
    )
    tracker.capture(q0, k, lay, step=0, sigma=1.0, branch=0, layer=0)
    changed = tracker.capture(q1, k, lay, step=1, sigma=0.5, branch=0, layer=0)
    check(changed["exact_route_reuse_fraction_mean"] < 0.01,
          "a planted route change is not classified as reusable")
    check(changed["sampled_route_jaccard_mean"] < 0.01,
          "a planted disjoint route has near-zero sampled Jaccard")


def test_precision_teacher_reports_changed_rows():
    lay = SimpleNamespace(
        seq_len=128,
        video_range=(0, 128),
        video_shape=(1, 8, 16),
        segments=[(0, 128, "video")],
        audio_t=0,
    )
    metrics = None
    for seed in range(32):
        torch.manual_seed(seed)
        q = torch.randn(1, 2, lay.seq_len, 32, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        candidate = rc._direct_tile_calibration(
            q,
            k,
            v,
            lay,
            0,
            lay.seq_len,
            budgets=(0.5,),
            head_chunk=1,
            q_tile=8,
            kv_tile=4,
        )[0.5]["router_precision_teacher"]
        if candidate["changed_route_q_tile_rows"]:
            metrics = candidate
            break

    check(metrics is not None,
          "deterministic BF16 activations contain a precision-sensitive route")
    check(metrics["summary_arms"]["bf16"].startswith("BF16 pooling"),
          "BF16 arm pools and scores in BF16")
    check(metrics["summary_arms"]["fp32"].startswith("FP32 pooling"),
          "FP32 arm pools and scores in FP32")
    check(metrics["selected_slot_substitution_fraction"] > 0.0,
          "route report counts selected-slot substitutions")
    check(metrics["changed_rows"]["sampled_row_head_count"] > 0,
          "teacher report isolates sampled rows whose routes changed")
    check(metrics["arms"]["bf16"]["retained_dense_attention_mass"]["mean"] > 0.0,
          "BF16 route reports retained dense attention mass")
    check(metrics["arms"]["fp32"]["sparse_output_rel_l2_by_row"]["p95"] >= 0.0,
          "FP32 route reports row-level p95 output error")
    check(metrics["boundary_margin"]["bf16"]["changed"] is not None,
          "BF16 cutoff margins are split by changed routes")


def test_adaptive_teacher_uses_real_luts_and_exact_budget():
    lay = SimpleNamespace(
        seq_len=128,
        video_range=(0, 128),
        video_shape=(1, 8, 16),
        segments=[(0, 128, "video")],
        audio_t=0,
    )
    torch.manual_seed(907)
    q = torch.randn(1, 2, lay.seq_len, 32, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    metrics = rc._direct_tile_calibration(
        q,
        k,
        v,
        lay,
        0,
        lay.seq_len,
        budgets=(0.5,),
        head_chunk=1,
        q_tile=8,
        kv_tile=4,
    )[0.5]["adaptive_teacher"]
    route = metrics["full_router_call"]
    sampled = metrics["sampled_teacher_rows"]
    controls = metrics["sampled_allocation_controls"]
    check(route["exact_budget_match"],
          "fixed and adaptive production LUTs preserve the exact full-call budget")
    check(route["adaptive_k"]["min"] >= 1,
          "adaptive production LUT reports valid per-row K")
    check(sampled["fixed"]["micro_relative_l2"] >= 0.0
          and sampled["adaptive"]["micro_relative_l2"] >= 0.0,
          "paired dense-teacher arms report micro relative L2")
    check(sampled["fixed"]["retained_dense_attention_mass"]["p05"] >= 0.0,
          "paired dense-teacher arms report the retained-mass bottom tail")
    outcomes = sampled["adaptive_rel_l2_outcomes"]
    check(abs(outcomes["win_fraction"] + outcomes["tie_fraction"]
              + outcomes["loss_fraction"] - 1.0) < 1e-6,
          "paired adaptive outcomes partition wins, ties, and losses")
    check(metrics["demand_teacher"]["oracle_k95"]["min"] >= 1,
          "exact dense block mass produces per-row oracle K95")
    check("spearman_adaptive_k_vs_total_pure_video_attention_mass"
          in metrics["demand_teacher"],
          "adaptive K is correlated against absolute video-vs-context mass")
    check(controls["target_selected_video_tiles"]
          == controls["output_allocation_control"]["selected_video_tiles"],
          "sampled output-allocation control repairs to its exact local budget")
    check(controls["output_allocation_control"]["squared_error"]
          <= controls["output_allocation_control"]["uniform_fixed_squared_error"] + 1e-8,
          "sampled output-allocation control is no worse than uniform fixed K")


def main():
    print("router characterization")
    test_direct_router_full_budget_is_dense()
    test_wrapped_probe_exposes_direct_calibration()
    test_temporal_reuse_metrics_and_topology_archive()
    test_changed_route_is_visible()
    test_precision_teacher_reports_changed_rows()
    test_adaptive_teacher_uses_real_luts_and_exact_budget()
    print("\nall router characterization self-tests passed")


if __name__ == "__main__":
    main()
