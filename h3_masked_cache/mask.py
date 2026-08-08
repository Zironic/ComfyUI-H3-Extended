"""Build and analyse conservative H3 Ref2V edit masks.

Source-relative scoring remains owned by the masked-cache package; generic
mask operations live in :mod:`h3_mask.ops` and are re-exported here for the
existing public imports.
"""

import torch

try:
    from h3_mask.ops import (active_fraction, build_mask, captured_energy_fraction,
                             coverage_fraction, dilate_spatial, dilate_temporal,
                             escaped_fraction, from_tiles, jaccard, missed_score_mass,
                             quantiles, threshold_sweep, tile_grid, to_tiles,
                             token_score)
except ImportError:
    from ..h3_mask.ops import (active_fraction, build_mask, captured_energy_fraction,
                               coverage_fraction, dilate_spatial, dilate_temporal,
                               escaped_fraction, from_tiles, jaccard, missed_score_mass,
                               quantiles, threshold_sweep, tile_grid, to_tiles,
                               token_score)

PATCH_H = 2
PATCH_W = 2


def rms_channels(t):
    """Per-cell RMS over channels: ``[B,C,T,H,W] -> [T,H,W]``."""
    return t.float().pow(2).mean(dim=1).sqrt().mean(dim=0)


def score_components(x0, source):
    """Return float32 ``(error_rms, source_rms)`` maps."""
    if x0.shape != source.shape:
        raise ValueError("x0 %s and source %s must have the same shape" %
                         (list(x0.shape), list(source.shape)))
    return rms_channels(x0 - source), rms_channels(source)


def relative_score(error_rms, source_rms, absolute_floor):
    if error_rms.shape != source_rms.shape:
        raise ValueError("error/source score maps must have the same shape")
    return error_rms.float() / (source_rms.float() + float(absolute_floor))


def latent_score(x0, source, absolute_floor):
    error, scale = score_components(x0, source)
    return relative_score(error, scale, absolute_floor)


def spatial_saliency(token_scores, eps=1e-6):
    """Robust per-frame excess over ordinary full-frame reconstruction drift."""
    if token_scores.ndim != 3:
        raise ValueError("token_scores must be [T,H,W]")
    flat = token_scores.float().reshape(token_scores.shape[0], -1)
    med = flat.median(dim=1).values[:, None]
    mad = (flat - med).abs().median(dim=1).values[:, None]
    return ((flat - med) / (mad + float(eps))).reshape_as(token_scores)


__all__ = [
    "PATCH_H", "PATCH_W", "rms_channels", "score_components", "relative_score",
    "latent_score", "spatial_saliency", "token_score", "tile_grid", "to_tiles",
    "from_tiles", "dilate_spatial", "dilate_temporal", "build_mask",
    "active_fraction", "jaccard", "escaped_fraction", "coverage_fraction",
    "missed_score_mass", "quantiles", "threshold_sweep", "captured_energy_fraction",
]
