"""Build and analyse conservative H3 Ref2V edit masks.

The measurement stage stores the raw per-cell error and source magnitude in
float32.  Relative scores, floors, token pooling, thresholds, tiles and halos can
therefore be recalibrated offline without rerunning an expensive generation.
"""

import torch
import torch.nn.functional as F

PATCH_H = 2
PATCH_W = 2


def rms_channels(t):
    """Per-cell RMS over channels: ``[B,C,T,H,W] -> [T,H,W]``."""
    return t.float().pow(2).mean(dim=1).sqrt().mean(dim=0)


def score_components(x0, source):
    """Return float32 ``(error_rms, source_rms)`` maps."""
    if x0.shape != source.shape:
        raise ValueError("x0 %s and source %s must have the same shape"
                         % (list(x0.shape), list(source.shape)))
    return rms_channels(x0 - source), rms_channels(source)


def relative_score(error_rms, source_rms, absolute_floor):
    if error_rms.shape != source_rms.shape:
        raise ValueError("error/source score maps must have the same shape")
    return error_rms.float() / (source_rms.float() + float(absolute_floor))


def latent_score(x0, source, absolute_floor):
    error, scale = score_components(x0, source)
    return relative_score(error, scale, absolute_floor)


def spatial_saliency(token_scores, eps=1e-6):
    """Robust per-frame excess over ordinary full-frame reconstruction drift.

    Returns ``(score - median) / (MAD + eps)`` at token resolution.  This is a
    diagnostic only; Stage 0 does not use it to alter inference.
    """
    if token_scores.ndim != 3:
        raise ValueError("token_scores must be [T,H,W]")
    flat = token_scores.float().reshape(token_scores.shape[0], -1)
    med = flat.median(dim=1).values[:, None]
    mad = (flat - med).abs().median(dim=1).values[:, None]
    return ((flat - med) / (mad + float(eps))).reshape_as(token_scores)


def token_score(score, patch_h=PATCH_H, patch_w=PATCH_W):
    """Latent-cell scores -> one score per H3 target-video row."""
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
    if union == 0.0:
        return 1.0
    return float((a & b).sum().item()) / union


def escaped_fraction(later, earlier):
    if later is None or earlier is None or later.shape != earlier.shape:
        return None
    n = float(later.sum().item())
    if n == 0.0:
        return 0.0
    return float((later & ~earlier).sum().item()) / n


def coverage_fraction(reference, candidate):
    """Fraction of ``reference`` active cells covered by ``candidate``."""
    escaped = escaped_fraction(reference, candidate)
    return None if escaped is None else 1.0 - escaped


def missed_score_mass(scores, candidate):
    """Share of non-negative score mass outside ``candidate``."""
    if scores is None or candidate is None or scores.shape != candidate.shape:
        return None
    scores = scores.float().clamp_min(0)
    total = float(scores.sum().item())
    if total == 0.0:
        return 0.0
    return float(scores[~candidate].sum().item()) / total


def quantiles(t, qs):
    flat = t.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {str(q): None for q in qs}
    qt = torch.tensor(list(qs), dtype=torch.float32)
    vals = torch.quantile(flat, qt)
    return {"%g" % q: float(v) for q, v in zip(qs, vals)}


def threshold_sweep(token_scores, thresholds, tile_h, tile_w, spatial_halo, temporal_halo):
    out = []
    for thr in thresholds:
        core, expanded, _ = build_mask(token_scores, thr, tile_h, tile_w,
                                       spatial_halo, temporal_halo)
        out.append({
            "threshold": float(thr),
            "active_core": active_fraction(core),
            "active_expanded": active_fraction(expanded),
        })
    return out
