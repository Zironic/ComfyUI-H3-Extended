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
        == (128, 64, 32, 16, 8, 4, 2, 1),
        "SM89 Q=128 produces the intended complete sharing ladder",
    )


def test_probe_wrapper_attaches_sweep():
    layout = SimpleNamespace(
        seq_len=8,
        video_range=(0, 8),
        video_shape=(1, 2, 4),
    )
    torch.manual_seed(41)
    q = torch.randn(1, 2, 8, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    result = moba3d.analyze_routing(
        q,
        k,
        v,
        layout,
        0,
        8,
        block_t=1,
        block_h=1,
        block_w=1,
        budgets=(0.5,),
        head_chunk=1,
        execution_geometry="sage_sparse",
        sage_q_tile=8,
        sage_kv_tile=1,
    )

    row = result["budgets"][0]
    sweep = row.get("executable_q_tile_density_sweep")
    check(sweep is not None,
          "sage_sparse probe records contain the Q-sharing density sweep")
    check(list(sweep) == ["8", "4", "2", "1"],
          "record preserves the Q-sharing ladder in descending execution size")
    check(result["execution_q_tile_density_sweep"] == [8, 4, 2, 1],
          "record metadata identifies all tested Q-sharing sizes")
    check(result["execution_q_tile_density_sweep_kv_tile"] == 1,
          "record metadata identifies the fixed KV granularity")

    means = [sweep[str(q_tile)]["mean"] for q_tile in (1, 2, 4, 8)]
    check(all(a <= b + 1e-7 for a, b in zip(means, means[1:])),
          "executable density cannot increase as Q-mask sharing gets finer")


def main():
    print("Q-mask sharing sweep")
    test_exact_union_curve()
    test_sweep_sizes()
    test_probe_wrapper_attaches_sweep()
    print("\nall Q-mask sharing sweep self-tests passed")


if __name__ == "__main__":
    main()
