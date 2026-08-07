"""Probe-only 3D MoBA-style routing simulation for MiniMax H3.

This module does not change inference. It evaluates a parameter-free content
router against exact dense attention on selected H3 query blocks. Only target-
video KV tokens are sparsification candidates; non-video context stays dense.
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
    """Build the mean-pooled video-block index once for one attention call."""
    if k.ndim != 4 or k.shape[0] != 1 or k.shape[2] != layout.seq_len:
        raise ValueError("expected k shaped [1, heads, seq, dim]")
    ids, counts, grid, n_blocks = _video_block_map(
        layout, block_t, block_h, block_w, k.device)
    v0, v1 = layout.video_range
    if v1 - v0 != ids.numel():
        raise ValueError("video token count does not match the resolved 3D lattice")
    kv = k[0, :, v0:v1, :].float()
    heads, _tokens, dim = kv.shape
    pooled = torch.zeros(heads, n_blocks, dim, device=k.device, dtype=torch.float32)
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


def _mean_dense_mass(q_sel, k_all, layout, block_ids, n_blocks, head_chunk):
    heads = q_sel.shape[0]
    scale = q_sel.shape[-1] ** -0.5
    v0, v1 = layout.video_range
    nonvideo, video_blocks = [], []
    for h0 in range(0, heads, head_chunk):
        h1 = min(heads, h0 + head_chunk)
        scores = torch.matmul(
            q_sel[h0:h1].float(), k_all[h0:h1].float().transpose(-1, -2)
        ) * scale
        probs = torch.softmax(scores, dim=-1).mean(dim=1)
        nv = probs[:, :v0].sum(-1) + probs[:, v1:].sum(-1)
        vb = torch.zeros(h1 - h0, n_blocks, device=probs.device, dtype=torch.float32)
        vb.scatter_add_(1, block_ids.view(1, -1).expand(h1 - h0, -1), probs[:, v0:v1])
        nonvideo.append(nv)
        video_blocks.append(vb)
    return torch.cat(nonvideo), torch.cat(video_blocks)


def analyze_routing(q, k, layout, qs, qe, *, block_t=1, block_h=4, block_w=4,
                    budgets=DEFAULT_BUDGETS, head_chunk=4, prepared=None):
    """Compare mean-pooled 3D routing with exact dense attention."""
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape or q.shape[0] != 1:
        raise ValueError("expected matching q/k tensors shaped [1, heads, seq, dim]")
    if not (0 <= qs < qe <= layout.seq_len):
        raise ValueError("query range is outside the packed sequence")
    budgets = parse_budgets(budgets)
    prepared = prepared or prepare_video_router(
        k, layout, block_t=block_t, block_h=block_h, block_w=block_w)
    if tuple(prepared["block_shape"]) != (int(block_t), int(block_h), int(block_w)):
        raise ValueError("prepared router block geometry does not match requested geometry")

    ids = prepared["block_ids"]
    counts = prepared["counts"]
    grid = prepared["grid"]
    n_blocks = prepared["n_blocks"]
    pooled = prepared["pooled_keys"]
    v0, v1 = layout.video_range

    q_sel = q[0, :, qs:qe, :]
    q_probe = q_sel.float().mean(dim=1)
    route_scores = torch.einsum("hd,hbd->hb", q_probe, pooled) * (q.shape[-1] ** -0.5)
    dense_nonvideo, dense_video_blocks = _mean_dense_mass(
        q_sel, k[0], layout, ids, n_blocks, max(1, int(head_chunk)))
    total = dense_nonvideo + dense_video_blocks.sum(-1)
    if not torch.allclose(total, torch.ones_like(total), atol=2e-4, rtol=2e-4):
        raise RuntimeError("dense attention mass did not sum to one")

    rows = []
    for frac in budgets:
        keep = max(1, min(n_blocks, int(math.ceil(float(frac) * n_blocks))))
        routed_idx = torch.topk(route_scores, k=keep, dim=-1).indices
        oracle_idx = torch.topk(dense_video_blocks, k=keep, dim=-1).indices
        routed_mask = torch.zeros_like(dense_video_blocks, dtype=torch.bool)
        oracle_mask = torch.zeros_like(dense_video_blocks, dtype=torch.bool)
        routed_mask.scatter_(1, routed_idx, True)
        oracle_mask.scatter_(1, oracle_idx, True)
        routed_mass = dense_nonvideo + (dense_video_blocks * routed_mask).sum(-1)
        oracle_mass = dense_nonvideo + (dense_video_blocks * oracle_mask).sum(-1)
        overlap = (routed_mask & oracle_mask).sum(-1).float() / keep
        selected_tokens = counts[routed_idx].sum(-1).float()
        effective_density = (layout.seq_len - (v1 - v0) + selected_tokens) / layout.seq_len
        rows.append({
            "budget": float(frac), "keep_blocks": int(keep), "video_blocks": int(n_blocks),
            "video_block_density": float(keep / n_blocks),
            "routed_mass_mean": float(routed_mass.mean().item()),
            "routed_mass_min_head": float(routed_mass.min().item()),
            "oracle_mass_mean": float(oracle_mass.mean().item()),
            "oracle_mass_min_head": float(oracle_mass.min().item()),
            "routing_regret_mean": float((oracle_mass - routed_mass).mean().item()),
            "routing_regret_max_head": float((oracle_mass - routed_mass).max().item()),
            "oracle_block_overlap_mean": float(overlap.mean().item()),
            "effective_token_density_mean": float(effective_density.mean().item()),
        })

    return {
        "block_shape": list(prepared["block_shape"]),
        "block_grid": [int(x) for x in grid],
        "video_blocks": int(n_blocks), "video_tokens": int(v1 - v0),
        "nonvideo_tokens": int(layout.seq_len - (v1 - v0)),
        "dense_nonvideo_mass_mean": float(dense_nonvideo.mean().item()),
        "dense_video_mass_mean": float(dense_video_blocks.sum(-1).mean().item()),
        "budgets": rows,
    }
