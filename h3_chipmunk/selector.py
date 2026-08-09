from __future__ import annotations

import math
import torch

MEASURE_FRACTIONS = (0.10, 0.20, 0.25, 0.30, 0.40, 0.50)


def logical_swiglu(fc1_out: torch.Tensor) -> torch.Tensor:
    gate, up = fc1_out.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul(up)


def group_scores(delta: torch.Tensor, feature_group: int) -> torch.Tensor:
    """RMS delta score per [token-group, feature-group]."""
    if delta.ndim != 2:
        raise ValueError("delta must be [rows, features]")
    rows, features = delta.shape
    if features % feature_group:
        raise ValueError("feature width must be divisible by feature_group")
    return (
        delta.float()
        .reshape(rows, features // feature_group, feature_group)
        .square()
        .mean(-1)
        .sqrt()
    )


def token_group_scores(scores: torch.Tensor, token_group_rows: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must be [rows, feature_groups]")
    rows = scores.shape[0]
    groups = math.ceil(rows / token_group_rows)
    out = []
    for gi in range(groups):
        a = gi * token_group_rows
        b = min(rows, a + token_group_rows)
        out.append(scores[a:b].mean(dim=0))
    return torch.stack(out, dim=0)


def select_top_groups(
    scores: torch.Tensor,
    top_fraction: float,
    random_fraction: float = 0.0,
):
    """Return rectangular int32 group indices and per-row counts on-device."""
    if scores.ndim != 2:
        raise ValueError("scores must be [token_groups, feature_groups]")
    n_groups = scores.shape[-1]
    keep = max(1, min(n_groups, int(math.ceil(n_groups * float(top_fraction)))))
    values, indices = torch.topk(scores, keep, dim=-1, sorted=True)
    del values
    if random_fraction > 0.0 and keep < n_groups:
        extra = max(1, int(math.ceil(n_groups * float(random_fraction))))
        rnd = torch.rand_like(scores)
        rnd.scatter_(1, indices, -1.0)
        extra_idx = torch.topk(
            rnd,
            min(extra, n_groups - keep),
            dim=-1,
        ).indices
        indices = torch.cat((indices, extra_idx), dim=-1)
    indices = torch.sort(indices, dim=-1).values.to(torch.int32).contiguous()
    counts = torch.full(
        (indices.shape[0],),
        indices.shape[1],
        dtype=torch.int32,
        device=indices.device,
    )
    return indices, counts


def energy_capture_by_fraction(scores: torch.Tensor, fractions=MEASURE_FRACTIONS):
    """Return on-device scalar tensors describing group-delta concentration."""
    if scores.ndim != 2:
        raise ValueError("scores must be [token_groups, feature_groups]")
    energy = scores.float().square()
    total = energy.sum()
    groups = int(scores.shape[-1])
    captures = {}
    for fraction in fractions:
        keep = max(1, min(groups, int(math.ceil(groups * float(fraction)))))
        kept = torch.topk(energy, keep, dim=-1, sorted=False).values.sum()
        captures[f"{float(fraction):.2f}"] = torch.where(
            total > 0,
            kept / total,
            torch.ones_like(total),
        )
    return captures


def selected_mask(
    indices: torch.Tensor,
    counts: torch.Tensor,
    feature_groups: int,
) -> torch.Tensor:
    """Build a selection mask without reading device counts on the host."""
    if indices.ndim != 2 or counts.ndim != 1:
        raise ValueError("indices/counts must be [rows,width] and [rows]")
    valid = (
        torch.arange(indices.shape[1], device=indices.device)[None, :]
        < counts.to(device=indices.device, dtype=torch.long)[:, None]
    )
    mask = torch.zeros(
        (indices.shape[0], feature_groups),
        dtype=torch.bool,
        device=indices.device,
    )
    mask.scatter_(1, indices.long(), valid)
    return mask


def expand_selection(
    mask: torch.Tensor,
    rows: int,
    token_group_rows: int,
    feature_group: int,
) -> torch.Tensor:
    """Expand token-group/feature-group mask to [rows, logical_features]."""
    feature_mask = mask.repeat_interleave(feature_group, dim=-1)
    row_mask = feature_mask.repeat_interleave(token_group_rows, dim=0)
    return row_mask[:rows]
