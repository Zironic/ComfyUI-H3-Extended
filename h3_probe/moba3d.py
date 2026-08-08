"""Probe-only 3D MoBA-style routing simulation for MiniMax H3.

This module does not change inference. It evaluates a parameter-free content
router against exact dense attention on selected H3 query tokens. Only target-
video KV tokens are sparsification candidates; non-video context stays dense.

The public H3 description says block importance is probed with mean pooling and
3D sparsification is applied to video tokens. The optional ``sage_sparse``
execution geometry coarsens those logical masks to globally aligned packed
Q/KV tiles for measurement; it does not claim to reproduce MiniMax's
unreleased training-aware backend or kernel implementation.
"""

from __future__ import annotations

import math

import torch


DEFAULT_BUDGETS = (0.10, 0.20, 0.30, 0.40)


def parse_budgets(spec):
    """Normalize a budget string or numeric iterable to sorted fractions.

    The node passes a string (``"10,20,30"``), while the probe session stores
    the normalized tuple and may pass that tuple back through this function.
    Accepting both makes normalization idempotent.
    """
    if spec is None:
        return DEFAULT_BUDGETS

    if isinstance(spec, str):
        text = spec.strip()
        if not text or text.lower() == "auto":
            return DEFAULT_BUDGETS
        parts = text.split(",")
    else:
        try:
            parts = list(spec)
        except TypeError:
            parts = [spec]

    values = []
    for part in parts:
        if isinstance(part, str):
            part = part.strip()
            if not part:
                continue
        value = float(part)
        if value > 1.0:
            value /= 100.0
        if 0.0 < value <= 1.0:
            values.append(value)
    if not values:
        raise ValueError("moba3d budgets must contain at least one value in (0, 100]")
    return tuple(sorted(set(values)))


def _video_block_map(layout, bt, bh, bw, device):
    bt, bh, bw = int(bt), int(bh), int(bw)
    if min(bt, bh, bw) <= 0:
        raise ValueError("3D block dimensions must be positive")
    t, h, w = layout.video_shape
    nt, nh, nw = math.ceil(t / bt), math.ceil(h / bh), math.ceil(w / bw)
    tt = torch.arange(t, device=device)[:, None, None]
    yy = torch.arange(h, device=device)[None, :, None]
    xx = torch.arange(w, device=device)[None, None, :]
    ids = ((tt // bt) * nh * nw + (yy // bh) * nw + (xx // bw)).expand(t, h, w)
    flat = ids.reshape(-1).long()
    n_blocks = nt * nh * nw
    counts = torch.bincount(flat, minlength=n_blocks)
    return flat, counts, (nt, nh, nw), n_blocks


def prepare_video_router(k, layout, *, block_t=1, block_h=4, block_w=4):
    """Build the mean-pooled video-block key index once per attention call."""
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[2] != layout.seq_len:
        raise ValueError("expected k shaped [1, heads, seq, dim]")
    ids, counts, grid, n_blocks = _video_block_map(
        layout, block_t, block_h, block_w, k.device
    )
    v0, v1 = layout.video_range
    if v1 - v0 != ids.numel():
        raise ValueError("video token count does not match the resolved 3D lattice")

    kv = k[0, :, v0:v1, :].float()
    heads, _tokens, dim = kv.shape
    pooled = torch.zeros(
        heads, n_blocks, dim, device=k.device, dtype=torch.float32
    )
    index = ids.view(1, -1, 1).expand(heads, -1, dim)
    pooled.scatter_add_(1, index, kv)
    pooled /= counts.clamp_min(1).float().view(1, -1, 1)

    return {
        "block_ids": ids,
        "counts": counts,
        "grid": grid,
        "n_blocks": n_blocks,
        "pooled_keys": pooled,
        "block_shape": (int(block_t), int(block_h), int(block_w)),
    }


def _block_mass(video_probs, block_ids, n_blocks):
    """Reduce exact dense video probabilities to [head, query, 3D-block]."""
    heads, queries, _video_tokens = video_probs.shape
    out = torch.zeros(
        heads, queries, n_blocks,
        device=video_probs.device, dtype=torch.float32,
    )
    index = block_ids.view(1, 1, -1).expand(heads, queries, -1)
    out.scatter_add_(2, index, video_probs)
    return out


def _renormalized_sparse_output(probs, values, video_keep, video_range):
    """Exact masked-attention output from dense probabilities.

    If logits are unchanged, masking keys and applying softmax again is exactly
    equivalent to taking the original dense probabilities on retained keys and
    renormalizing them by their retained mass. Doing that here avoids allocating
    another full masked-logit tensor for every budget.
    """
    v0, v1 = video_range
    video_probs = probs[:, :, v0:v1]
    selected_video = video_probs * video_keep.to(video_probs.dtype)

    retained = selected_video.sum(-1)
    numerator = torch.matmul(selected_video, values[:, v0:v1, :])

    if v0:
        retained = retained + probs[:, :, :v0].sum(-1)
        numerator = numerator + torch.matmul(
            probs[:, :, :v0], values[:, :v0, :]
        )
    if v1 < probs.shape[-1]:
        retained = retained + probs[:, :, v1:].sum(-1)
        numerator = numerator + torch.matmul(
            probs[:, :, v1:], values[:, v1:, :]
        )

    retained = retained.clamp_min(1e-12)
    return retained, numerator / retained.unsqueeze(-1)


def _renormalized_masked_output(probs, values, keep):
    """Exact masked-and-renormalized output for a full packed KV mask."""
    masked = probs * keep.to(probs.dtype)
    retained = masked.sum(-1).clamp_min(1e-12)
    return retained, torch.matmul(masked, values) / retained.unsqueeze(-1)


def _execution_mask(
    logical_video_keep,
    qs,
    qe,
    seq_len,
    video_range,
    q_tile,
    kv_tile,
    aligned_start,
    aligned_end,
):
    """Coarsen logical per-query video masks to global Q/KV tile masks."""
    heads, aligned_queries, video_tokens = logical_video_keep.shape
    v0, v1 = video_range
    if aligned_queries != aligned_end - aligned_start:
        raise ValueError("aligned query mask does not match execution range")

    q_tile = max(1, int(q_tile))
    kv_tile = max(1, int(kv_tile))
    q_tile_count = (aligned_queries + q_tile - 1) // q_tile
    kv_tile_count = (seq_len + kv_tile - 1) // kv_tile
    video_global = torch.arange(v0, v1, device=logical_video_keep.device)
    kv_video_ids = torch.div(video_global, kv_tile, rounding_mode="floor")
    tile_enabled = torch.zeros(
        heads, q_tile_count, kv_tile_count,
        dtype=torch.bool,
        device=logical_video_keep.device,
    )
    for q_index in range(q_tile_count):
        a = q_index * q_tile
        b = min(aligned_queries, a + q_tile)
        selected = logical_video_keep[:, a:b].any(dim=1)
        for head in range(heads):
            tile_enabled[head, q_index, kv_video_ids[selected[head]]] = True

    # A KV tile containing any context token is fully dense, including video
    # rows that happen to share that global tile.
    global_k = torch.arange(seq_len, device=logical_video_keep.device)
    kv_ids = torch.div(global_k, kv_tile, rounding_mode="floor")
    nonvideo = (global_k < v0) | (global_k >= v1)
    tile_has_nonvideo = torch.zeros(
        kv_tile_count, dtype=torch.bool, device=logical_video_keep.device
    )
    tile_has_nonvideo[kv_ids[nonvideo]] = True
    tile_enabled |= tile_has_nonvideo.view(1, 1, -1)

    q_global = torch.arange(qs, qe, device=logical_video_keep.device)
    q_tile_ids = torch.div(q_global, q_tile, rounding_mode="floor") - (
        aligned_start // q_tile
    )
    per_query_tiles = tile_enabled.index_select(1, q_tile_ids)
    full_keep = per_query_tiles.gather(
        2, kv_ids.view(1, 1, -1).expand(heads, qe - qs, -1)
    )
    return full_keep, {
        "q_range": [int(aligned_start), int(aligned_end)],
        "q_tiles": int(q_tile_count),
        "q_tile": int(q_tile),
        "kv_tile": int(kv_tile),
        "kv_tiles": int(kv_tile_count),
    }


def _per_head_error(got, want):
    """Return per-head relative-L2, mean-absolute, and max-absolute errors."""
    diff = got - want
    heads = diff.shape[0]
    flat_diff = diff.reshape(heads, -1)
    flat_want = want.reshape(heads, -1)
    rel_l2 = torch.linalg.vector_norm(flat_diff, dim=-1) / torch.linalg.vector_norm(
        flat_want, dim=-1
    ).clamp_min(1e-12)
    mean_abs = flat_diff.abs().mean(-1)
    max_abs = flat_diff.abs().amax(-1)
    return rel_l2, mean_abs, max_abs


def _threshold_heads(rel_l2):
    values = rel_l2.tolist()
    return {
        "heads_rel_l2_gt_1pct": [i for i, x in enumerate(values) if x > 0.01],
        "heads_rel_l2_gt_2pct": [i for i, x in enumerate(values) if x > 0.02],
        "heads_rel_l2_gt_5pct": [i for i, x in enumerate(values) if x > 0.05],
    }


def _worst_heads(rel_l2, limit=5):
    order = torch.argsort(rel_l2, descending=True)[: min(limit, rel_l2.numel())]
    return [
        {"head": int(i), "rel_l2": float(rel_l2[i])}
        for i in order.tolist()
    ]


def analyze_routing(
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
    budgets=DEFAULT_BUDGETS,
    head_chunk=4,
    prepared=None,
    execution_geometry="logical",
    sage_q_tile=128,
    sage_kv_tile=64,
):
    """Evaluate per-query-token 3D routing and exact sparse-output error.

    q/k/v are H3 attention tensors in [1, heads, seq, dim]. Routing is performed
    independently for every query token and head, as in MoBA-style top-k block
    selection. For every budget we compare the routed block set with an oracle
    that selects blocks using exact dense attention mass, then compute both
    routed and oracle sparse outputs after proper softmax renormalization.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape != k.shape
        or q.shape != v.shape
        or q.shape[0] != 1
    ):
        raise ValueError(
            "expected matching q/k/v tensors shaped [1, heads, seq, dim]"
        )
    if q.shape[2] != layout.seq_len:
        raise ValueError("q/k/v sequence length does not match the packed layout")
    if not (0 <= qs < qe <= layout.seq_len):
        raise ValueError("query range is outside the packed sequence")

    execution_geometry = str(execution_geometry or "logical").strip().lower()
    if execution_geometry not in ("logical", "sage_sparse"):
        raise ValueError("execution_geometry must be logical or sage_sparse")
    sage_q_tile = max(1, int(sage_q_tile))
    sage_kv_tile = max(1, int(sage_kv_tile))
    budgets = parse_budgets(budgets)
    prepared = prepared or prepare_video_router(
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
    counts = prepared["counts"]
    grid = prepared["grid"]
    n_blocks = int(prepared["n_blocks"])
    pooled = prepared["pooled_keys"]
    v0, v1 = layout.video_range
    scale = q.shape[-1] ** -0.5
    heads = q.shape[1]
    head_chunk = max(1, int(head_chunk))
    if execution_geometry == "sage_sparse":
        requested_width = qe - qs
        execution_q_tiles = max(1, (requested_width + sage_q_tile - 1) // sage_q_tile)
        evaluated_start = (qs // sage_q_tile) * sage_q_tile
        evaluated_end = min(
            layout.seq_len,
            evaluated_start + execution_q_tiles * sage_q_tile,
        )
    else:
        evaluated_start, evaluated_end = qs, qe
        execution_q_tiles = None
    q_eval_all = q[0, :, evaluated_start:evaluated_end, :]

    accum = {
        frac: {
            "routed_mass": [],
            "oracle_mass": [],
            "overlap": [],
            "density": [],
            "rel_l2": [],
            "mean_abs": [],
            "max_abs": [],
            "oracle_rel_l2": [],
            "oracle_mean_abs": [],
            "oracle_max_abs": [],
            "executable_density": [],
            "executable_rel_l2": [],
            "executable_mean_abs": [],
            "executable_max_abs": [],
        }
        for frac in budgets
    }
    dense_nonvideo = []
    dense_video = []

    for h0 in range(0, heads, head_chunk):
        h1 = min(heads, h0 + head_chunk)
        qh = q_eval_all[h0:h1].float()
        kh = k[0, h0:h1].float()
        vh = v[0, h0:h1].float()

        scores = torch.matmul(qh, kh.transpose(-1, -2)) * scale
        probs = torch.softmax(scores, dim=-1)
        del scores

        dense_out = torch.matmul(probs, vh)
        video_probs = probs[:, :, v0:v1]
        exact_blocks = _block_mass(video_probs, ids, n_blocks)
        nonvideo_mass = probs[:, :, :v0].sum(-1)
        if v1 < probs.shape[-1]:
            nonvideo_mass = nonvideo_mass + probs[:, :, v1:].sum(-1)
        dense_nonvideo.append(nonvideo_mass.detach().cpu())
        dense_video.append(video_probs.sum(-1).detach().cpu())

        route_scores_all = torch.einsum(
            "hqd,hbd->hqb", qh, pooled[h0:h1]
        ) * scale

        for frac in budgets:
            keep = max(1, min(n_blocks, int(math.ceil(float(frac) * n_blocks))))
            routed_idx_all = torch.topk(
                route_scores_all, k=keep, dim=-1
            ).indices
            oracle_idx = torch.topk(exact_blocks, k=keep, dim=-1).indices

            routed_blocks_all = torch.zeros(
                route_scores_all.shape,
                dtype=torch.bool,
                device=route_scores_all.device,
            )
            routed_blocks_all.scatter_(2, routed_idx_all, True)
            routed_keep_all = routed_blocks_all.index_select(2, ids)
            routed_blocks = routed_blocks_all
            routed_keep = routed_keep_all
            oracle_blocks = torch.zeros_like(exact_blocks, dtype=torch.bool)
            oracle_blocks.scatter_(2, oracle_idx, True)

            oracle_keep = oracle_blocks.index_select(2, ids)

            routed_mass, routed_out = _renormalized_sparse_output(
                probs, vh, routed_keep, (v0, v1)
            )
            oracle_mass, oracle_out = _renormalized_sparse_output(
                probs, vh, oracle_keep, (v0, v1)
            )

            rel_l2, mean_abs, max_abs = _per_head_error(routed_out, dense_out)
            oracle_rel_l2, oracle_mean_abs, oracle_max_abs = _per_head_error(
                oracle_out, dense_out
            )

            overlap = (routed_blocks & oracle_blocks).sum(-1).float() / keep
            selected_tokens = counts[routed_idx_all].sum(-1).float()
            effective_density = (
                layout.seq_len - (v1 - v0) + selected_tokens
            ) / layout.seq_len

            bucket = accum[frac]
            bucket["routed_mass"].append(routed_mass.detach().cpu())
            bucket["oracle_mass"].append(oracle_mass.detach().cpu())
            bucket["overlap"].append(overlap.detach().cpu())
            bucket["density"].append(effective_density.detach().cpu())
            bucket["rel_l2"].append(rel_l2.detach().cpu())
            bucket["mean_abs"].append(mean_abs.detach().cpu())
            bucket["max_abs"].append(max_abs.detach().cpu())
            bucket["oracle_rel_l2"].append(oracle_rel_l2.detach().cpu())
            bucket["oracle_mean_abs"].append(oracle_mean_abs.detach().cpu())
            bucket["oracle_max_abs"].append(oracle_max_abs.detach().cpu())

            if execution_geometry == "sage_sparse":
                executable_keep, _execution_meta = _execution_mask(
                    routed_keep_all,
                    evaluated_start,
                    evaluated_end,
                    layout.seq_len,
                    (v0, v1),
                    sage_q_tile,
                    sage_kv_tile,
                    evaluated_start,
                    evaluated_end,
                )
                _exec_mass, executable_out = _renormalized_masked_output(
                    probs, vh, executable_keep
                )
                executable_rel_l2, executable_mean_abs, executable_max_abs = _per_head_error(
                    executable_out, dense_out
                )
                bucket["executable_density"].append(
                    executable_keep.float().mean(-1).detach().cpu()
                )
                bucket["executable_rel_l2"].append(
                    executable_rel_l2.detach().cpu()
                )
                bucket["executable_mean_abs"].append(
                    executable_mean_abs.detach().cpu()
                )
                bucket["executable_max_abs"].append(
                    executable_max_abs.detach().cpu()
                )

            del (
                routed_blocks,
                oracle_blocks,
                routed_keep,
                oracle_keep,
                routed_mass,
                routed_out,
                oracle_mass,
                oracle_out,
                routed_blocks_all,
                routed_keep_all,
            )

        del probs, dense_out, video_probs, exact_blocks, route_scores_all

    dense_nonvideo = torch.cat(dense_nonvideo, dim=0)
    dense_video = torch.cat(dense_video, dim=0)

    rows = []
    for frac in budgets:
        bucket = accum[frac]
        routed_mass = torch.cat(bucket["routed_mass"], dim=0)
        oracle_mass = torch.cat(bucket["oracle_mass"], dim=0)
        overlap = torch.cat(bucket["overlap"], dim=0)
        density = torch.cat(bucket["density"], dim=0)
        rel_l2 = torch.cat(bucket["rel_l2"], dim=0)
        mean_abs = torch.cat(bucket["mean_abs"], dim=0)
        max_abs = torch.cat(bucket["max_abs"], dim=0)
        oracle_rel_l2 = torch.cat(bucket["oracle_rel_l2"], dim=0)
        oracle_mean_abs = torch.cat(bucket["oracle_mean_abs"], dim=0)
        oracle_max_abs = torch.cat(bucket["oracle_max_abs"], dim=0)

        regret = oracle_mass - routed_mass
        keep = max(1, min(n_blocks, int(math.ceil(float(frac) * n_blocks))))

        row = {
            "budget": float(frac),
            "keep_blocks": int(keep),
            "video_blocks": int(n_blocks),
            "video_block_density": float(keep / n_blocks),
            "routed_mass_mean": float(routed_mass.mean()),
            "routed_mass_min": float(routed_mass.min()),
            "oracle_mass_mean": float(oracle_mass.mean()),
            "oracle_mass_min": float(oracle_mass.min()),
            "routing_regret_mean": float(regret.mean()),
            "routing_regret_max": float(regret.max()),
            "oracle_block_overlap_mean": float(overlap.mean()),
            "oracle_block_overlap_min": float(overlap.min()),
            "effective_token_density_mean": float(density.mean()),
            "effective_token_density_max": float(density.max()),
            "sparse_output_rel_l2_mean_head": float(rel_l2.mean()),
            "sparse_output_rel_l2_median_head": float(rel_l2.median()),
            "sparse_output_rel_l2_max_head": float(rel_l2.max()),
            "sparse_output_mean_abs_mean_head": float(mean_abs.mean()),
            "sparse_output_max_abs": float(max_abs.max()),
            "oracle_output_rel_l2_mean_head": float(oracle_rel_l2.mean()),
            "oracle_output_rel_l2_max_head": float(oracle_rel_l2.max()),
            "oracle_output_mean_abs_mean_head": float(oracle_mean_abs.mean()),
            "oracle_output_max_abs": float(oracle_max_abs.max()),
            "head_rel_l2": [float(x) for x in rel_l2.tolist()],
            "oracle_head_rel_l2": [float(x) for x in oracle_rel_l2.tolist()],
            "worst_heads": _worst_heads(rel_l2),
            "execution_geometry": execution_geometry,
            "execution_q_tile": int(sage_q_tile) if execution_geometry == "sage_sparse" else None,
            "execution_kv_tile": int(sage_kv_tile) if execution_geometry == "sage_sparse" else None,
        }
        if execution_geometry == "sage_sparse":
            executable_density = torch.cat(bucket["executable_density"], dim=0)
            executable_rel_l2 = torch.cat(bucket["executable_rel_l2"], dim=0)
            executable_mean_abs = torch.cat(bucket["executable_mean_abs"], dim=0)
            executable_max_abs = torch.cat(bucket["executable_max_abs"], dim=0)
            row.update(
                {
                    "executable_effective_token_density_mean": float(executable_density.mean()),
                    "executable_effective_token_density_max": float(executable_density.max()),
                    "executable_sparse_output_rel_l2_mean_head": float(executable_rel_l2.mean()),
                    "executable_sparse_output_rel_l2_median_head": float(executable_rel_l2.median()),
                    "executable_sparse_output_rel_l2_max_head": float(executable_rel_l2.max()),
                    "executable_sparse_output_mean_abs_mean_head": float(executable_mean_abs.mean()),
                    "executable_sparse_output_max_abs": float(executable_max_abs.max()),
                    "executable_head_rel_l2": [float(x) for x in executable_rel_l2.tolist()],
                    "executable_heads_rel_l2_gt_1pct": _threshold_heads(executable_rel_l2)["heads_rel_l2_gt_1pct"],
                    "executable_heads_rel_l2_gt_2pct": _threshold_heads(executable_rel_l2)["heads_rel_l2_gt_2pct"],
                    "executable_heads_rel_l2_gt_5pct": _threshold_heads(executable_rel_l2)["heads_rel_l2_gt_5pct"],
                    "executable_worst_heads": _worst_heads(executable_rel_l2),
                }
            )
        else:
            row["executable_metrics"] = None
        row.update(_threshold_heads(rel_l2))
        rows.append(row)

    result = {
        "routing_granularity": "per-query-token",
        "execution_geometry": execution_geometry,
        "execution_q_tile": int(sage_q_tile) if execution_geometry == "sage_sparse" else None,
        "execution_kv_tile": int(sage_kv_tile) if execution_geometry == "sage_sparse" else None,
        "sage_q_tile": int(sage_q_tile) if execution_geometry == "sage_sparse" else None,
        "sage_kv_tile": int(sage_kv_tile) if execution_geometry == "sage_sparse" else None,
        "execution_q_range": [int(evaluated_start), int(evaluated_end)] if execution_geometry == "sage_sparse" else None,
        "execution_q_tiles": int(execution_q_tiles) if execution_geometry == "sage_sparse" else None,
        "execution_kv_tiles": int((layout.seq_len + sage_kv_tile - 1) // sage_kv_tile) if execution_geometry == "sage_sparse" else None,
        "block_shape": list(prepared["block_shape"]),
        "block_grid": [int(x) for x in grid],
        "video_blocks": int(n_blocks),
        "video_tokens": int(v1 - v0),
        "nonvideo_tokens": int(layout.seq_len - (v1 - v0)),
        "dense_nonvideo_mass_mean": float(dense_nonvideo.mean()),
        "dense_video_mass_mean": float(dense_video.mean()),
        "budgets": rows,
    }
    if execution_geometry == "sage_sparse":
        result.update(
            {
                "requested_q_range": [int(qs), int(qe)],
                "evaluated_q_range": [int(evaluated_start), int(evaluated_end)],
            }
        )
    return result
