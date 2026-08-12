"""Branch-local diagnostic for Sparse-Sage Q-mask sharing granularity.

The original MoBA3D probe routes every query token independently. Sparse Sage
cannot execute those independent masks: one execution Q tile shares one KV-tile
route, so the logical masks are unioned across the Q tile. On H3 that union was
observed to turn useful logical sparsity back into ~dense execution.

This module leaves the existing probe and its dense/error measurements intact.
It wraps :func:`moba3d.analyze_routing` and, only for ``sage_sparse`` records,
recomputes the inexpensive logical router once and asks how much executable KV
would remain if the backend could share a route across progressively smaller Q
groups while keeping the current Sparse-Sage KV granularity fixed.

The sweep is diagnostic only. It does not change inference or the production
Sparse-Sage router.
"""

from __future__ import annotations

import math

import torch

from . import moba3d


_INSTALLED = False
_ORIGINAL_ANALYZE_ROUTING = None


def q_tile_sweep_sizes(q_tile):
    """Return a power-of-two Q-sharing ladder down to per-query routing."""
    q_tile = max(1, int(q_tile))
    sizes = []
    value = q_tile
    while value > 1:
        sizes.append(value)
        value = max(1, value // 2)
    sizes.append(1)
    return tuple(dict.fromkeys(sizes))


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


def _new_accumulator(q_tiles):
    return {
        int(q_tile): {"sum": 0.0, "count": 0, "max": 0.0}
        for q_tile in q_tiles
    }


def _accumulate(accumulator, q_tile, density):
    values = density.detach().float()
    bucket = accumulator[int(q_tile)]
    bucket["sum"] += float(values.sum())
    bucket["count"] += int(values.numel())
    bucket["max"] = max(bucket["max"], float(values.max()))


def _finish(accumulator):
    return {
        str(q_tile): {
            "mean": (
                bucket["sum"] / bucket["count"]
                if bucket["count"]
                else 0.0
            ),
            "max": bucket["max"],
        }
        for q_tile, bucket in accumulator.items()
    }


def measure_q_mask_sharing(
    q,
    k,
    layout,
    evaluated_start,
    evaluated_end,
    *,
    block_t,
    block_h,
    block_w,
    budgets,
    prepared,
    head_chunk,
    sage_q_tile,
    sage_kv_tile,
):
    """Return executable-density curves for the existing logical router.

    This deliberately reproduces only the route-scoring portion of
    ``moba3d.analyze_routing``. It does *not* repeat dense attention, softmax, V
    application, or error calculation, so one normal probe run can characterize
    every candidate Q-sharing size.
    """
    budgets = moba3d.parse_budgets(budgets)
    q_tiles = q_tile_sweep_sizes(sage_q_tile)
    accumulators = {frac: _new_accumulator(q_tiles) for frac in budgets}

    if prepared is None:
        prepared = moba3d.prepare_video_router(
            k,
            layout,
            block_t=block_t,
            block_h=block_h,
            block_w=block_w,
        )

    requested_shape = (int(block_t), int(block_h), int(block_w))
    if tuple(prepared["block_shape"]) != requested_shape:
        raise ValueError(
            "prepared router block geometry does not match requested geometry"
        )

    ids = prepared["block_ids"]
    pooled = prepared["pooled_keys"]
    n_blocks = int(prepared["n_blocks"])
    v0, v1 = layout.video_range
    heads = int(q.shape[1])
    scale = q.shape[-1] ** -0.5
    head_chunk = max(1, int(head_chunk))

    for h0 in range(0, heads, head_chunk):
        h1 = min(heads, h0 + head_chunk)
        qh = q[0, h0:h1, evaluated_start:evaluated_end, :].float()
        route_scores = torch.einsum(
            "hqd,hbd->hqb", qh, pooled[h0:h1]
        ) * scale

        for frac in budgets:
            keep_blocks = max(
                1,
                min(n_blocks, int(math.ceil(float(frac) * n_blocks))),
            )
            routed_idx = torch.topk(
                route_scores,
                k=keep_blocks,
                dim=-1,
            ).indices
            routed_blocks = torch.zeros(
                route_scores.shape,
                dtype=torch.bool,
                device=route_scores.device,
            )
            routed_blocks.scatter_(2, routed_idx, True)
            logical_video_keep = routed_blocks.index_select(2, ids)

            sweep = execution_density_for_q_tiles(
                logical_video_keep,
                seq_len=layout.seq_len,
                video_range=(v0, v1),
                q_tiles=q_tiles,
                kv_tile=sage_kv_tile,
                aligned_start=evaluated_start,
                aligned_end=evaluated_end,
            )
            for q_tile, density in sweep.items():
                _accumulate(accumulators[frac], q_tile, density)

        del route_scores

    return {
        float(frac): _finish(accumulators[frac])
        for frac in budgets
    }, q_tiles


def _analyze_routing_with_q_mask_sweep(
    q,
    k,
    v,
    layout,
    qs,
    qe,
    *,
    block_t=1,
    block_h=4,
    block_w=4,
    budgets=moba3d.DEFAULT_BUDGETS,
    head_chunk=4,
    prepared=None,
    execution_geometry="logical",
    sage_q_tile=128,
    sage_kv_tile=64,
):
    result = _ORIGINAL_ANALYZE_ROUTING(
        q,
        k,
        v,
        layout,
        qs,
        qe,
        block_t=block_t,
        block_h=block_h,
        block_w=block_w,
        budgets=budgets,
        head_chunk=head_chunk,
        prepared=prepared,
        execution_geometry=execution_geometry,
        sage_q_tile=sage_q_tile,
        sage_kv_tile=sage_kv_tile,
    )

    if result.get("execution_geometry") != "sage_sparse":
        return result

    evaluated = result.get("evaluated_q_range") or result.get("execution_q_range")
    if not evaluated:
        return result
    evaluated_start, evaluated_end = (int(evaluated[0]), int(evaluated[1]))

    sweep_by_budget, q_tiles = measure_q_mask_sharing(
        q,
        k,
        layout,
        evaluated_start,
        evaluated_end,
        block_t=block_t,
        block_h=block_h,
        block_w=block_w,
        budgets=budgets,
        prepared=prepared,
        head_chunk=head_chunk,
        sage_q_tile=sage_q_tile,
        sage_kv_tile=sage_kv_tile,
    )

    for row in result.get("budgets", ()):
        frac = float(row["budget"])
        row["executable_q_tile_density_sweep"] = sweep_by_budget[frac]

    result["execution_q_tile_density_sweep"] = [int(x) for x in q_tiles]
    result["execution_q_tile_density_sweep_kv_tile"] = int(sage_kv_tile)
    return result


def install():
    """Install the diagnostic wrapper once for this experiment branch."""
    global _INSTALLED, _ORIGINAL_ANALYZE_ROUTING
    if _INSTALLED:
        return
    current = moba3d.analyze_routing
    if getattr(current, "_h3_q_mask_sharing_sweep", False):
        _INSTALLED = True
        _ORIGINAL_ANALYZE_ROUTING = getattr(current, "_h3_original", current)
        return

    _ORIGINAL_ANALYZE_ROUTING = current
    _analyze_routing_with_q_mask_sweep._h3_q_mask_sharing_sweep = True
    _analyze_routing_with_q_mask_sweep._h3_original = current
    moba3d.analyze_routing = _analyze_routing_with_q_mask_sweep
    _INSTALLED = True


install()
