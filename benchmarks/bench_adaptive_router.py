"""Benchmark fixed and adaptive-budget H3 Sparse-Sage routing.

This benchmark exercises only the production 128Q x 64KV summary router.  It
reports route-construction latency and the actual per-row video-block counts; it
does not invoke the Sparse-Sage attention kernel.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_attention.hybrid.config import (  # noqa: E402
    DENSITY_ADAPTIVE_BUDGET,
    DENSITY_FIXED,
    HybridSparseConfig,
)
from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402


def make_layout(video_q_tiles, context_kv_tiles=4):
    context_tokens = int(context_kv_tiles) * 64
    video_tokens = int(video_q_tiles) * 128
    sequence = context_tokens + video_tokens
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(context_tokens, sequence),
        segments=[
            (0, max(0, context_tokens - 64), "text"),
            (max(0, context_tokens - 64), context_tokens, "audio"),
            (context_tokens, sequence, "video"),
        ],
        video_shape=(1, 1, video_tokens),
        audio_t=32,
    )


def make_summaries(layout, heads, dim, seed, device):
    generator = torch.Generator(device=device).manual_seed(int(seed))
    q_tiles = (layout.seq_len + 127) // 128
    kv_tiles = (layout.seq_len + 63) // 64
    q = torch.randn(
        (1, int(heads), q_tiles, int(dim)),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    k = torch.randn(
        (1, int(heads), kv_tiles, int(dim)),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )

    # Give alternating rows different score concentration without changing
    # tensor geometry.  This makes variable row allocation observable even in a
    # synthetic benchmark.
    q_start = (layout.video_range[0] + 127) // 128
    k_start = (layout.video_range[0] + 63) // 64
    q_video = q[..., q_start:, :]
    k_video = k[..., k_start:, :]
    if q_video.numel() and k_video.numel():
        q_video[..., 0::2, :].mul_(2.0)
        k_video[..., 0::4, :].mul_(3.0)
    return q, k


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(router, q, k, layout, budget, warmups, repeats):
    times = []
    result = None
    for index in range(int(warmups) + int(repeats)):
        synchronize(q.device)
        started = time.perf_counter()
        result = router.build_lut_from_summaries(q, k, layout, budget)
        synchronize(q.device)
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= warmups:
            times.append(elapsed)
    lut, valid, metadata = result
    context = metadata.q_tiles - metadata.pure_video_q_tiles
    context_kv = metadata.kv_tiles - metadata.pure_video_kv_tiles
    counts = valid[..., context:] - context_kv
    return {
        "latency_ms_median": statistics.median(times),
        "latency_ms_min": min(times),
        "density_mode": metadata.density_mode,
        "requested_budget": float(budget),
        "mean_video_blocks": float(counts.float().mean()),
        "min_video_blocks": int(counts.min()),
        "max_video_blocks": int(counts.max()),
        "std_video_blocks": float(counts.float().std(unbiased=False)),
        "mean_video_density": float(counts.float().mean() / metadata.pure_video_kv_tiles),
        "pure_video_q_tiles": int(metadata.pure_video_q_tiles),
        "pure_video_kv_tiles": int(metadata.pure_video_kv_tiles),
        "rows": int(counts.numel()),
        "lut_shape": list(lut.shape),
    }


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--video-q-tiles", type=int, default=416)
    value.add_argument("--context-kv-tiles", type=int, default=4)
    value.add_argument("--heads", type=int, default=56)
    value.add_argument("--dim", type=int, default=128)
    value.add_argument("--budget", type=float, default=0.20)
    value.add_argument("--min-density", type=float, default=0.05)
    value.add_argument("--max-density", type=float, default=0.50)
    value.add_argument("--temperature", type=float, default=1.0)
    value.add_argument("--target-mass", type=float, default=0.80)
    value.add_argument("--warmups", type=int, default=2)
    value.add_argument("--repeats", type=int, default=10)
    value.add_argument("--seed", type=int, default=6841)
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    value.add_argument("--json", dest="json_path")
    return value


def main():
    args = parser().parse_args()
    if min(args.video_q_tiles, args.context_kv_tiles, args.heads, args.dim) <= 0:
        raise ValueError("tile, head, and dimension arguments must be positive")
    if args.warmups < 0 or args.repeats <= 0:
        raise ValueError("warmups must be non-negative and repeats positive")
    device = torch.device(args.device)
    layout = make_layout(args.video_q_tiles, args.context_kv_tiles)
    q, k = make_summaries(layout, args.heads, args.dim, args.seed, device)

    fixed = SparseTileRouter(HybridSparseConfig(
        video_budget=args.budget,
        density_mode=DENSITY_FIXED,
    ))
    adaptive = SparseTileRouter(HybridSparseConfig(
        video_budget=args.budget,
        density_mode=DENSITY_ADAPTIVE_BUDGET,
        min_video_density=args.min_density,
        max_video_density=args.max_density,
        adaptive_temperature=args.temperature,
        adaptive_target_mass=args.target_mass,
    ))
    payload = {
        "device": str(device),
        "geometry": {
            "sequence": int(layout.seq_len),
            "video_q_tiles": int(args.video_q_tiles),
            "heads": int(args.heads),
            "head_dim": int(args.dim),
        },
        "fixed": measure(
            fixed, q, k, layout, args.budget, args.warmups, args.repeats
        ),
        "adaptive_budget": measure(
            adaptive, q, k, layout, args.budget, args.warmups, args.repeats
        ),
    }
    print(json.dumps(payload, indent=2))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
