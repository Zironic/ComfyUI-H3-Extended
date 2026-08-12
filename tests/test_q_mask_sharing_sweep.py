"""CPU self-test for the Q-mask sharing diagnostic.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_q_mask_sharing_sweep.py
"""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_probe import moba3d, q_mask_sharing_sweep  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def test_exact_union_curve():
    logical = torch.zeros(1, 8, 8, dtype=torch.bool)
    for query in range(8):
        logical[0, query, query] = True

    sweep = q_mask_sharing_sweep.execution_density_for_q_tiles(
        logical,
        seq_len=8,
        video_range=(0, 8),
        q_tiles=(8, 4, 2, 1),
        kv_tile=1,
        aligned_start=0,
        aligned_end=8,
    )
    means = {q_tile: float(density.mean()) for q_tile, density in sweep.items()}

    check(abs(means[8] - 1.0) < 1e-7,
          "8 queries sharing one route union all eight independent choices")
    check(abs(means[4] - 0.5) < 1e-7,
          "4-query sharing retains half the KV in the planted case")
    check(abs(means[2] - 0.25) < 1e-7,
          "2-query sharing retains one quarter of the KV in the planted case")
    check(abs(means[1] - 0.125) < 1e-7,
          "per-query routing retains only each query's own KV choice")


def test_sweep_sizes():
    check(
        q_mask_sharing_sweep.q_tile_sweep_sizes(128)
        == (128, 64, 32, 16, 8),
        "SM89 Q=128 produces the supported executable Q rows",
    )


def test_probe_wrapper_attaches_sweep():
    layout = SimpleNamespace(
        seq_len=32,
        video_range=(0, 32),
        video_shape=(1, 4, 8),
    )
    torch.manual_seed(41)
    q = torch.randn(1, 2, 32, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    result = moba3d.analyze_routing(
        q,
        k,
        v,
        layout,
        0,
        32,
        block_t=1,
        block_h=1,
        block_w=1,
        budgets=(0.5,),
        head_chunk=1,
        execution_geometry="sage_sparse",
        sage_q_tile=128,
        sage_kv_tile=64,
    )

    row = result["budgets"][0]
    matrix = row.get("executable_q_kv_matrix")
    check(matrix is not None, "sage_sparse probe records contain the Q x KV matrix")
    check(len(matrix) == 15, "matrix contains all 15 supported Q x KV geometries")
    check(not any(key.startswith(("1x", "2x", "4x")) for key in matrix),
          "matrix does not compute legacy Q=1/2/4 geometries")

    exact = {"128x64", "64x32", "32x32", "16x32", "64x16", "32x16", "16x16"}
    check(
        {key for key, value in matrix.items() if "executable_head_rel_l2" in value}
        == exact,
        "exact executable errors exist only for the seven requested geometries",
    )
    check(
        matrix["128x64"]["executable_rel_l2_mean_head"]
        == row["executable_sparse_output_rel_l2_mean_head"],
        "128x64 matrix control reuses the existing executable metrics",
    )
    check(
        len(matrix["128x64"]["executable_head_rel_l2"]) == 2,
        "head error arrays aggregate across head chunks",
    )
    check(
        matrix["128x64"]["executable_density_max"]
        >= matrix["128x64"]["executable_density_mean"],
        "density arrays aggregate across head chunks",
    )


def main():
    print("Q-mask sharing sweep")
    test_exact_union_curve()
    test_sweep_sizes()
    test_probe_wrapper_attaches_sweep()
    print("\nall Q-mask sharing sweep self-tests passed")


if __name__ == "__main__":
    main()
