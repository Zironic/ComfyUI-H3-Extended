#!/usr/bin/env python3
"""Re-score an H3 masked-cache ``mask.npz`` without rerunning generation.

Example:

    python tools/analyze_h3_edit_masks.py output/h3_masked_cache/run/mask.npz \
        --floor 0.01 --threshold 2.0 --warmup 2

The archive contains float32 ``error_rms_*`` and ``source_rms_*`` maps, so the
relative score and floor can be recalibrated offline.  This tool intentionally
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


def dilate(mask, spatial, temporal):
    out = mask.copy()
    for _ in range(spatial):
        p = np.pad(out, ((0, 0), (1, 1), (1, 1)))
        out = np.logical_or.reduce([
            p[:, i:i + out.shape[1], j:j + out.shape[2]]
            for i in range(3) for j in range(3)
        ])
    for _ in range(temporal):
        p = np.pad(out, ((1, 1), (0, 0), (0, 0)))
        out = p[:-2] | p[1:-1] | p[2:]
    return out


def build_mask(score, threshold, spatial, temporal):
    return dilate(pool_tokens(score) >= threshold, spatial, temporal)


def escaped(reference, candidate):
    n = reference.sum()
    return 0.0 if n == 0 else float((reference & ~candidate).sum() / n)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("archive", type=Path)
    p.add_argument("--floor", type=float, default=1e-3)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--spatial-halo", type=int, default=1)
    p.add_argument("--temporal-halo", type=int, default=1)
    args = p.parse_args()

    with np.load(args.archive, allow_pickle=False) as z:
        index = json.loads(str(z["index"]))
        labels = [k[len("error_rms_"):] for k in z.files
                  if k.startswith("error_rms_") and k != "error_rms_final"]
        labels.sort()
        masks = []
        for label in labels:
            score = z["error_rms_" + label].astype(np.float32) / (
                z["source_rms_" + label].astype(np.float32) + args.floor)
            masks.append(build_mask(score, args.threshold,
                                    args.spatial_halo, args.temporal_halo))
        if not masks:
            raise SystemExit("archive contains no guided/error maps")

        warm = masks[:max(1, args.warmup)]
        frozen = np.logical_or.reduce(warm)
        print("tag:", index.get("tag"))
        print("guided observations:", len(masks))
        print("frozen active: %.2f%%" % (100 * frozen.mean()))
        for i, mask in enumerate(masks):
            print("step %02d active %6.2f%% escaped frozen %6.2f%%" % (
                i, 100 * mask.mean(), 100 * escaped(mask, frozen)))

        if "error_rms_final" in z and "source_rms_final" in z:
            score = z["error_rms_final"].astype(np.float32) / (
                z["source_rms_final"].astype(np.float32) + args.floor)
            final = build_mask(score, args.threshold,
                               args.spatial_halo, args.temporal_halo)
            print("final active: %.2f%%" % (100 * final.mean()))
            print("final escaped frozen: %.2f%%" % (100 * escaped(final, frozen)))
        else:
            print("final sample: not present")


if __name__ == "__main__":
    main()
