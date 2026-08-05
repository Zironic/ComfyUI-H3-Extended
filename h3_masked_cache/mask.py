"""Turn a predicted-clean latent and its source into an active-edit mask.

The chain is one direction only, and every stage is coarser than the last:

    x0, source  [B,24,T,H,W]
      -> latent score          [T, H, W]        relative per-cell difference
      -> token score           [T, H/2, W/2]    max over each 1x2x2 DiT patch
      -> core mask             [T, H/2, W/2]    threshold
      -> tile mask             [T, nth, ntw]    any active token activates its tile
      -> spatial dilation      [T, nth, ntw]    halo in tiles
      -> temporal dilation     [T, nth, ntw]    halo in latent frames
      -> active token mask     [T, H/2, W/2]    one row per target-video token

Every reduction is a `max`, never a mean: the mask decides what may be *dropped*
from the computation, so a single changed cell inside a patch has to keep the
whole patch. Averaging would let a small bright edit vanish into a large
unchanged patch, which is precisely the failure this ordering exists to prevent.

Nothing here touches model state or reads anything but the two tensors it is
given, which is what makes `measure` mode provably output-neutral.
"""

import torch
import torch.nn.functional as F

PATCH_H = 2
PATCH_W = 2


# --------------------------------------------------------------------------
# scores
# --------------------------------------------------------------------------

def rms_channels(t):
    """Per-cell RMS over the channel dim: `[B,C,T,H,W] -> [T,H,W]`.

    Batch is reduced by mean; H3 is batch-1 in practice, and a batched call
    still has to produce one mask because one mask is what the sequence layout
    can express.
    """
    return t.float().pow(2).mean(dim=1).sqrt().mean(dim=0)


def latent_score(x0, source, absolute_floor):
    """Relative difference between the predicted clean latent and the source.

    Normalizing by the source's own magnitude keeps a dark, low-variance region
    from reading as unchanged merely because its absolute error is small. The
    floor bounds the ratio where the source is genuinely flat - without it, an
    empty background divides a small error by a smaller scale and scores higher
    than the edit.
    """
    if x0.shape != source.shape:
        raise ValueError("x0 %s and source %s must have the same shape"
                         % (list(x0.shape), list(source.shape)))
    error = rms_channels(x0 - source)
    scale = rms_channels(source)
    return error / (scale + float(absolute_floor))


def token_score(score, patch_h=PATCH_H, patch_w=PATCH_W):
    """Latent-cell scores -> one score per target-video sequence row.

    The DiT patch is 1x2x2, so each token owns a 2x2 latent area of one frame.
    Odd latent sizes are padded the way core pads the video input before
    patching, and the pad is filled with the score's own minimum so it can never
    invent an active token.
    """
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


# --------------------------------------------------------------------------
# tiles and halos
# --------------------------------------------------------------------------

def tile_grid(patch_h, patch_w, tile_h, tile_w):
    """Tile-grid size for a token grid, rounding up (edge tiles are kept whole)."""
    return (patch_h + tile_h - 1) // tile_h, (patch_w + tile_w - 1) // tile_w


def to_tiles(token_mask, tile_h, tile_w):
    """`[T,ph,pw]` token mask -> `[T,nth,ntw]` tile mask; any active token wins.

    Edge tiles are padded with inactive tokens rather than dropped, so a token
    in a partial tile still activates its tile and no token is ever unrepresented.
    """
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
    """`[T,nth,ntw]` tile mask -> `[T,ph,pw]` token mask; the pad is cropped off."""
    if tile_h == 1 and tile_w == 1:
        return tile_mask[:, :patch_h, :patch_w].clone()
    expanded = tile_mask.repeat_interleave(tile_h, dim=1).repeat_interleave(tile_w, dim=2)
    return expanded[:, :patch_h, :patch_w].contiguous()


def dilate_spatial(mask, halo):
    """Grow every active cell by `halo` cells in each spatial direction."""
    if halo <= 0:
        return mask
    t = mask.shape[0]
    x = mask.float().reshape(t, 1, mask.shape[1], mask.shape[2])
    x = F.max_pool2d(x, kernel_size=2 * halo + 1, stride=1, padding=halo)
    return x.reshape(mask.shape) > 0.5


def dilate_temporal(mask, halo):
    """Grow every active cell by `halo` latent frames in each direction.

    Latent frames are not uniform in pixel-frame terms - H3's temporal spans
    widen after the first frame - so one unit of temporal halo covers a
    different amount of real time early in the clip than late in it. The halo is
    still expressed in latent indices because that is the only axis the token
    layout has; the report records the resulting span so the asymmetry stays
    visible rather than implied.
    """
    if halo <= 0:
        return mask
    t, a, b = mask.shape
    x = mask.float().permute(1, 2, 0).reshape(1, a * b, t)
    x = F.max_pool1d(x, kernel_size=2 * halo + 1, stride=1, padding=halo)
    return x.reshape(a, b, t).permute(2, 0, 1).contiguous() > 0.5


def build_mask(token_scores, threshold, tile_h, tile_w, spatial_halo, temporal_halo):
    """Full threshold -> tile -> halo chain.

    Returns `(core, expanded, tiles)`: the raw thresholded token mask, the final
    token mask after tiling and both halos, and the tile-resolution mask the
    halos were applied at.
    """
    core = token_scores >= float(threshold)
    _, ph, pw = token_scores.shape
    tiles = to_tiles(core, tile_h, tile_w)
    tiles = dilate_spatial(tiles, spatial_halo)
    tiles = dilate_temporal(tiles, temporal_halo)
    return core, from_tiles(tiles, tile_h, tile_w, ph, pw), tiles


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def active_fraction(mask):
    n = mask.numel()
    return float(mask.sum().item()) / n if n else 0.0


def jaccard(a, b):
    """Intersection over union of two boolean masks; 1.0 for two empty masks."""
    if a is None or b is None or a.shape != b.shape:
        return None
    union = float((a | b).sum().item())
    if union == 0.0:
        return 1.0
    return float((a & b).sum().item()) / union


def escaped_fraction(later, earlier):
    """Share of `later`'s active cells that `earlier` did not already cover.

    This is the number Stage 0 exists to produce: if an early mask is to be
    frozen and used for the rest of the run, what matters is not how similar
    consecutive masks look but how much of the late edit appears outside the
    early one.
    """
    if later is None or earlier is None or later.shape != earlier.shape:
        return None
    n = float(later.sum().item())
    if n == 0.0:
        return 0.0
    return float((later & ~earlier).sum().item()) / n


def quantiles(t, qs):
    """Score quantiles, computed on CPU float32 to keep the result comparable."""
    flat = t.detach().float().reshape(-1).cpu()
    if flat.numel() == 0:
        return {str(q): None for q in qs}
    qt = torch.tensor(list(qs), dtype=torch.float32)
    vals = torch.quantile(flat, qt)
    return {"%g" % q: float(v) for q, v in zip(qs, vals)}


def threshold_sweep(token_scores, thresholds, tile_h, tile_w, spatial_halo, temporal_halo):
    """Active fraction before and after expansion, for each candidate threshold.

    The sweep is what a default threshold gets chosen from, so it reports both
    ends of the chain: a threshold that looks selective at token resolution can
    be worthless once a 4x4 tile and a halo have been applied to it.
    """
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
