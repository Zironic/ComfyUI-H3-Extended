"""Branch-local diagnostic for Sparse-Sage Q-mask sharing granularity.

The original MoBA3D probe routes every query token independently. Sparse Sage
cannot execute those independent masks: one execution Q tile shares one KV-tile
route, so the logical masks are unioned across the Q tile. On H3 that union was
observed to turn useful logical sparsity back into ~dense execution.

The executable Q x KV closure is computed by :func:`moba3d.analyze_routing`
while its dense tensors are live. This module retains only the small mask
helper used to test the coarsening behavior.
"""

from __future__ import annotations

import torch

from . import moba3d


def q_tile_sweep_sizes(q_tile):
    """Return the supported Q-sharing rows of the executable closure matrix."""
    q_tile = max(1, int(q_tile))
    return tuple(value for value in moba3d.EXECUTION_Q_TILES if value <= q_tile)


def execution_density_for_q_tiles(
    logical_video_keep,
    *,
    seq_len,
    video_range,
    q_tiles,
    kv_tile,
    aligned_start,
    aligned_end,
):
    """Coarsen one logical mask at several Q-sharing granularities.

    ``logical_video_keep`` is shaped ``[heads, queries, video_tokens]`` and is
    still the original per-query route. KV granularity stays fixed so the
    resulting density curve isolates information lost specifically by sharing a
    route across neighbouring Q tokens.
    """
    out = {}
    for q_tile in q_tiles:
        q_tile = max(1, int(q_tile))
        if aligned_start % q_tile:
            raise ValueError(
                "Q-mask sweep requires aligned_start divisible by every Q tile"
            )
        keep, _ = moba3d._execution_mask(
            logical_video_keep,
            aligned_start,
            aligned_end,
            int(seq_len),
            video_range,
            q_tile,
            int(kv_tile),
            aligned_start,
            aligned_end,
        )
        out[q_tile] = keep.float().mean(-1)
    return out
