#!/usr/bin/env python3
"""Re-score an H3 masked-cache ``mask.npz`` without rerunning generation.

Example:

    python tools/analyze_h3_edit_masks.py output/h3_masked_cache/run/mask.npz \
        --floor 0.01 --threshold 2.0 --burn-in 2 --warmup 2 \
        --tile-size 2 --spatial-halo 1 --temporal-halo 1

The archive contains float32 ``error_rms_*`` and ``source_rms_*`` maps, so the
relative score and floor can be recalibrated offline. This tool intentionally
uses NumPy only and does not import ComfyUI.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def pool_tokens(x):
    t, h, w = x.shape
    ph, pw = (-h) % 2, (-w) % 2
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), constant_values=float(x.min()))
    return x.reshape(t, x.shape[1] // 2, 2, x.shape[2] // 2, 2).max(axis=(2, 4))


def to_tiles(mask, tile):
    if tile == 1:
        return mask.copy()
    t, h, w = mask.shape
    ph, pw = (-h) % tile, (-w) % tile
    if ph or pw:
        mask = np.pad(mask, ((0, 0), (0, ph), (0, pw)), constant_values=False)
    return mask.reshape(t, mask.shape[1] // tile, tile,
                        mask.shape[2] // tile, tile).max(axis=(2, 4))


def from_tiles(tiles, tile, h, w):
    if tile == 1:
        return tiles[:, :h, :w].copy()
    return np.repeat(np.repeat(tiles, tile, axis=1), tile, axis=2)[:, :h, :w]


def dilate_spatial(mask, halo):
    out = mask.copy()
    for _ in range(halo):
        p = np.pad(out, ((0, 0), (1, 1), (1, 1)), constant_values=False)
        out = np.logical_or.reduce([
            p[:, i:i + out.shape[1], j:j + out.shape[2]]
            for i in range(3) for j in range(3)
        ])
    return out


def dilate_temporal(mask, halo):
    out = mask.copy()
    for _ in range(halo):
        p = np.pad(out, ((1, 1), (0, 0), (0, 0)), constant_values=False)
        out = p[:-2] | p[1:-1] | p[2:]
    return out


def build_mask(token_score, threshold, tile, spatial, temporal):
    _, h, w = token_score.shape
    tiles = to_tiles(token_score >= threshold, tile)
    tiles = dilate_spatial(tiles, spatial)
    tiles = dilate_temporal(tiles, temporal)
    return from_tiles(tiles, tile, h, w)


def escaped(reference, candidate):
    n = reference.sum()
    return 0.0 if n == 0 else float((reference & ~candidate).sum() / n)


def missed_excess_mass(token_score, candidate, threshold):
    excess = np.maximum(token_score - threshold, 0.0)
    total = excess.sum()
    return 0.0 if total == 0 else float(excess[~candidate].sum() / total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--floor", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--burn-in", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--tile-size", type=int, choices=(1, 2, 4), default=2)
    p.add_argument("--spatial-halo", type=int, default=1)
    p.add_argument("--temporal-halo", type=int, default=1)
    args = p.parse_args()

    if args.burn_in < 0:
        raise SystemExit("--burn-in must be >= 0")
    if args.warmup < 1:
        raise SystemExit("--warmup must be >= 1")

    with np.load(args.archive, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        labels = [k[len("error_rms_"):] for k in z.files
                  if k.startswith("error_rms_s")]
        labels.sort()
        token_scores = []
        masks = []
        for label in labels:
            relative = z["error_rms_" + label].astype(np.float32) / (
                z["source_rms_" + label].astype(np.float32) + args.floor)
            token = pool_tokens(relative)
            token_scores.append(token)
            masks.append(build_mask(token, args.threshold, args.tile_size,
                                    args.spatial_halo, args.temporal_halo))
        if not masks:
            raise SystemExit("archive contains no guided/error maps")

        stop = args.burn_in + args.warmup
        if len(masks) < stop:
            raise SystemExit(
                "need at least %d observations for burn-in %d + warmup %d; archive has %d"
                % (stop, args.burn_in, args.warmup, len(masks)))
        frozen = np.logical_or.reduce(masks[args.burn_in:stop])

        print("tag:", index.get("tag"))
        print("guided observations:", len(masks))
        print("policy: discard %d, freeze union of observations %d..%d" % (
            args.burn_in, args.burn_in, stop - 1))
        print("tile/halo: %dx%d, spatial %d tiles, temporal %d frames" % (
            args.tile_size, args.tile_size, args.spatial_halo, args.temporal_halo))
        print("frozen active: %.2f%%" % (100 * frozen.mean()))
        for i, (token, mask) in enumerate(zip(token_scores, masks)):
            print("step %02d active %6.2f%% escaped frozen %6.2f%% excess missed %6.2f%%" % (
                i, 100 * mask.mean(), 100 * escaped(mask, frozen),
                100 * missed_excess_mass(token, frozen, args.threshold)))

        if "error_rms_final" in z and "source_rms_final" in z:
            relative = z["error_rms_final"].astype(np.float32) / (
                z["source_rms_final"].astype(np.float32) + args.floor)
            final_score = pool_tokens(relative)
            final = build_mask(final_score, args.threshold, args.tile_size,
                               args.spatial_halo, args.temporal_halo)
            print("final active: %.2f%%" % (100 * final.mean()))
            print("final escaped frozen: %.2f%%" % (100 * escaped(final, frozen)))
            print("final excess missed: %.2f%%" % (
                100 * missed_excess_mass(final_score, frozen, args.threshold)))
        else:
            print("final sample: not present")


if __name__ == "__main__":
    main()
