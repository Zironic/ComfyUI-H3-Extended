"""Probe-only 3D MoBA-style routing simulation for MiniMax H3.

This module does not change inference.  It evaluates a parameter-free content
router against exact dense attention on selected H3 query blocks:

* only target-video KV tokens are sparsification candidates;
* target video is partitioned in its native (T, H, W) patch lattice;
* each candidate block is represented by the mean of its post-RoPE keys;
* a query-block mean scores those pooled block representatives;
* text/reference/audio tokens remain dense;
* routed top-k coverage is compared with an oracle that selects blocks by the
  exact dense attention mass they actually receive.

The design intentionally mirrors only what the H3 team publicly described.  It
is a measurement tool, not a claim to reproduce their unreleased backend.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


DEFAULT_BUDGETS = (0.10, 0.20, 0.30, 0.40)


def parse_budgets(spec):
    """Parse comma-separated percentages/fractions into sorted fractions.

    ``"10,20,30"`` and ``"0.1,0.2,0.3"`` are equivalent. Values are clamped to
    (0, 1].
    """
    if spec is None:
        return DEFAULT_BUDGETS
    text = str(spec).strip()
    if not text or text.lower() == "auto":
        return DEFAULT_BUDGETS
    values = []
    for part in text.split(","):
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


def _validate_geometry(layout, bt, bh, bw):
    bt, bh, bw = int(bt), int(bh), int(bw)
    if min(bt, bh, bw) <= 0:
        raise ValueError("3D block dimensions must be positive")
    t, h, w = layout.video_shape
    return bt, bh, bw, t, h, w


def _video_block_map(layout, bt, bh, bw, device):
    """Return token->3D-block ids and block metadata for target video tokens."""
    bt, bh, bw, t, h, w = _validate_geometry(layout, bt, bh, bw)
    nt = math.ceil(t / bt)
    nh = math.ceil(h / bh)
    nw = math.ceil(w / bw)

    tt = torch.arange(t, device=device)[:, None, None]
    yy = torch.arange(h, device=device)[None, :, None]
    xx = torch.arange(w, device=device)[None, None, :]
    block_ids = ((tt // bt) * nh * nw + (yy // bh) * nw + (xx // bw)).expand(t, h, w)
    flat = block_ids.reshape(-1).to(torch.long)
    n_blocks = nt * nh * nw
    counts = torch.bincount(flat, minlength=n_blocks)
    return flat, counts, (nt, nh, nw), n_blocks


def _pooled_video_keys(k_video, block_ids, counts, n_blocks):
    """Mean-pool video keys into 3D blocks.

    ``k_video`` is [heads, video_tokens, dim].
    """
    heads, _tokens, dim = k_video.shape
    out = torch.zeros(heads, n_blocks, dim, device=k_video.device, dtype=torch.float32)
    idx = block_ids.view(1, -1, 1).expand(heads, -1, dim)
    out.scatter_add_(1, idx, k_video.to(torch.float32))
    out /= counts.clamp_min(1).to(out.dtype).view(1, -1, 1)
    return out


def _mean_dense_mass(q_sel, k_all, layout, block_ids, n_blocks, head_chunk):
    """Exact dense softmax mass reduced to non-video + 3D video blocks."""
    heads = q_sel.shape[0]
    scale = q_sel.shape[-1] ** -0.5
    v0, v1 = layout.video_range
    nonvideo = []
    video_blocks = []

    for h0 in range(0, heads, head_chunk):
        h1 = min(heads, h0 + head_chunk)
        scores = torch.matmul(
            q_sel[h0:h1].to(torch.float32),
            k_all[h0:h1].to(torch.float32).transpose(-1, -2),
        ) * scale
        probs = torch.softmax(scores, dim=-1).mean(dim=1)  # [hc, seq]
        del scores

        nv = probs[:, :v0].sum(-1) + probs[:, v1:].sum(-1)
        vb = torch.zeros(h1 - h0, n_blocks, device=probs.device, dtype=torch.float32)
        vb.scatter_add_(1, block_ids.view(1, -1).expand(h1 - h0, -1), probs[:, v0:v1])
        nonvideo.append(nv)
        video_blocks.append(vb)
        del probs

    return torch.cat(nonvideo), torch.cat(video_blocks)


def analyze_routing(
    q,
    k,
    layout,
    qs,
    qe,
    *,
    block_t=1,
    block_h=4,
    block_w=4,
    budgets=DEFAULT_BUDGETS,
    head_chunk=4,
):
    """Compare mean-pooled 3D routing with exact dense attention.

    q/k are H3 post-RoPE tensors in [1, heads, seq, dim].  Results are ordinary
    Python values so they can go directly into JSON reports.
    """
    if q.ndim != 4 or k.ndim != 4 or q.shape != k.shape:
        raise ValueError("expected matching q/k tensors shaped [1, heads, seq, dim]")
    if q.shape[0] != 1:
        raise ValueError("moba3d probe currently expects batch size 1")
    if not (0 <= qs < qe <= layout.seq_len):
        raise ValueError("query range is outside the packed sequence")

    budgets = parse_budgets(budgets)
    block_ids, counts, grid, n_blocks = _video_block_map(
        layout, block_t, block_h, block_w, q.device)
    v0, v1 = layout.video_range
    if v1 - v0 != block_ids.numel():
        raise ValueError("video token count does not match the resolved 3D lattice")

    q_sel = q[0, :, qs:qe, :]
    q_probe = q_sel.to(torch.float32).mean(dim=1)  # [heads, dim]
    k_all = k[0]
    k_video = k_all[:, v0:v1, :]
    pooled = _pooled_video_keys(k_video, block_ids, counts, n_blocks)
    route_scores = torch.einsum("hd,hbd->hb", q_probe, pooled) * (q.shape[-1] ** -0.5)

    dense_nonvideo, dense_video_blocks = _mean_dense_mass(
        q_sel, k_all, layout, block_ids, n_blocks, max(1, int(head_chunk)))

    dense_total = dense_nonvideo + dense_video_blocks.sum(-1)
    if not torch.allclose(dense_total, torch.ones_like(dense_total), atol=2e-4, rtol=2e-4):
        raise RuntimeError("dense attention mass did not sum to one")

    budget_rows = []
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
        overlap = (routed_mask & oracle_mask).sum(-1).to(torch.float32) / keep

        selected_tokens = counts[routed_idx].sum(-1).to(torch.float32)
        nonvideo_tokens = layout.seq_len - (v1 - v0)
        effective_density = (nonvideo_tokens + selected_tokens) / layout.seq_len

        budget_rows.append({
            "budget": float(frac),
            "keep_blocks": int(keep),
            "video_blocks": int(n_blocks),
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
        "block_shape": [int(block_t), int(block_h), int(block_w)],
        "block_grid": [int(x) for x in grid],
        "video_blocks": int(n_blocks),
        "video_tokens": int(v1 - v0),
        "nonvideo_tokens": int(layout.seq_len - (v1 - v0)),
        "dense_nonvideo_mass_mean": float(dense_nonvideo.mean().item()),
        "dense_video_mass_mean": float(dense_video_blocks.sum(-1).mean().item()),
        "budgets": budget_rows,
    }
