"""Torch mask and energy operations shared by H3 probes."""

import torch
import torch.nn.functional as F

PATCH_H = 2
PATCH_W = 2


def token_score(score, patch_h=PATCH_H, patch_w=PATCH_W):
    if score.ndim != 3:
        raise ValueError("score must be [T,H,W], got %s" % list(score.shape))
    t, h, w = score.shape
    pad_h = (-h) % patch_h
    pad_w = (-w) % patch_w
    x = score.reshape(t, 1, h, w)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=float(score.min()))
    pooled = F.max_pool2d(x, kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w))
    return pooled.reshape(t, (h + pad_h) // patch_h, (w + pad_w) // patch_w)


def tile_grid(patch_h, patch_w, tile_h, tile_w):
    return (patch_h + tile_h - 1) // tile_h, (patch_w + tile_w - 1) // tile_w


def to_tiles(token_mask, tile_h, tile_w):
    if tile_h == 1 and tile_w == 1:
        return token_mask.clone()
    t, ph, pw = token_mask.shape
    pad_h = (-ph) % tile_h
    pad_w = (-pw) % tile_w
    x = token_mask.float().reshape(t, 1, ph, pw)
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), value=0.0)
    pooled = F.max_pool2d(x, kernel_size=(tile_h, tile_w), stride=(tile_h, tile_w))
    nth, ntw = tile_grid(ph, pw, tile_h, tile_w)
    return pooled.reshape(t, nth, ntw) > 0.5


def from_tiles(tile_mask, tile_h, tile_w, patch_h, patch_w):
    if tile_h == 1 and tile_w == 1:
        return tile_mask[:, :patch_h, :patch_w].clone()
    expanded = tile_mask.repeat_interleave(tile_h, dim=1).repeat_interleave(tile_w, dim=2)
    return expanded[:, :patch_h, :patch_w].contiguous()


def dilate_spatial(mask, halo):
    if halo <= 0:
        return mask
    t = mask.shape[0]
    x = mask.float().reshape(t, 1, mask.shape[1], mask.shape[2])
    x = F.max_pool2d(x, kernel_size=2 * halo + 1, stride=1, padding=halo)
    return x.reshape(mask.shape) > 0.5


def dilate_temporal(mask, halo):
    if halo <= 0:
        return mask
    t, a, b = mask.shape
    x = mask.float().permute(1, 2, 0).reshape(1, a * b, t)
    x = F.max_pool1d(x, kernel_size=2 * halo + 1, stride=1, padding=halo)
    return x.reshape(a, b, t).permute(2, 0, 1).contiguous() > 0.5


def build_mask(token_scores, threshold, tile_h, tile_w, spatial_halo, temporal_halo):
    core = token_scores >= float(threshold)
    _, ph, pw = token_scores.shape
    tiles = to_tiles(core, tile_h, tile_w)
    tiles = dilate_spatial(tiles, spatial_halo)
    tiles = dilate_temporal(tiles, temporal_halo)
    return core, from_tiles(tiles, tile_h, tile_w, ph, pw), tiles


def active_fraction(mask):
    n = mask.numel()
    return float(mask.sum().item()) / n if n else 0.0


def jaccard(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    union = float((a | b).sum().item())
    return 1.0 if union == 0.0 else float((a & b).sum().item()) / union


def escaped_fraction(later, earlier):
    if later is None or earlier is None or later.shape != earlier.shape:
        return None
    n = float(later.sum().item())
    return 0.0 if n == 0.0 else float((later & ~earlier).sum().item()) / n


def coverage_fraction(reference, candidate):
    escaped = escaped_fraction(reference, candidate)
    return None if escaped is None else 1.0 - escaped


def missed_score_mass(scores, candidate, threshold=0.0):
    if scores is None or candidate is None or scores.shape != candidate.shape:
        return None
    excess = (scores.float() - float(threshold)).clamp_min(0)
    total = float(excess.sum().item())
    return 0.0 if total == 0.0 else float(excess[~candidate].sum().item()) / total


def quantiles(t, qs):
    flat = t.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {str(q): None for q in qs}
    vals = torch.quantile(flat, torch.tensor(list(qs), dtype=torch.float32))
    return {"%g" % q: float(v) for q, v in zip(qs, vals)}


def threshold_sweep(token_scores, thresholds, tile_h, tile_w, spatial_halo, temporal_halo):
    return [
        {"threshold": float(thr), "active_core": active_fraction(core),
         "active_expanded": active_fraction(expanded)}
        for thr in thresholds
        for core, expanded, _ in [build_mask(token_scores, thr, tile_h, tile_w, spatial_halo, temporal_halo)]
    ]


def captured_energy_fraction(energy, mask):
    if energy is None or mask is None or energy.shape != mask.shape:
        return None
    total = float(energy.float().sum().item())
    return 1.0 if total == 0.0 else float(energy.float()[mask].sum().item()) / total
