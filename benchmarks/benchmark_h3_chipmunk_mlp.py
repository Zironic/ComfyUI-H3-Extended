"""Real-checkpoint H3 ConvRot group-delta MLP microbenchmark.

Run from the ComfyUI root with the environment used for H3 inference.
"""

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from benchmarks.benchmark_h3_activation_memory import (
    resolve_checkpoint,
    load_block_mlp_tensors,
    load_convrot_mlp,
)
from h3_chipmunk.selector import logical_swiglu


def parse_fractions(value):
    values = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not values or any(not (0.0 < x <= 1.0) for x in values):
        raise ValueError("fractions must be comma-separated values in (0, 1]")
    return values


def sync():
    torch.cuda.synchronize()


def timed(fn, warmup=3, repeats=8):
    for _ in range(warmup):
        fn()
    sync()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        sync()
        samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def ck_linear(x, q, scale, groupsize):
    import comfy.quant_ops
    return comfy.quant_ops.ck.int8_linear(
        x, q, scale, None, x.dtype,
        convrot=True, convrot_groupsize=groupsize,
    )


def selected_rows(ffn, group_size, fraction, device):
    groups = ffn // group_size
    keep = max(1, min(groups, int(round(groups * fraction))))
    gids = torch.arange(keep, device=device, dtype=torch.long)
    off = torch.arange(group_size, device=device, dtype=torch.long)
    return (gids[:, None] * group_size + off[None]).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--fractions", default="0.10,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    raw = load_block_mlp_tensors(checkpoint, args.block_index)
    mlp = load_convrot_mlp(raw)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(args.seed)

    fc1q = mlp["fc1"]["weight"].to(device)
    fc1s = mlp["fc1"]["weight_scale"].to(device)
    fc2q = mlp["fc2"]["weight"].to(device)
    fc2s = mlp["fc2"]["weight_scale"].to(device)
    gs = int(mlp["fc2"]["group_size"])
    hidden = int(mlp["hidden_width"])
    ffn = int(mlp["ffn_width"])
    if gs != 256:
        raise ValueError(f"expected H3 ConvRot-256, got {gs}")

    x0 = torch.randn((args.rows, hidden), device=device, dtype=dtype)
    x1 = x0 + 0.01 * torch.randn_like(x0)

    def dense(x):
        mid = ck_linear(x, fc1q, fc1s, gs)
        act = logical_swiglu(mid)
        return ck_linear(act, fc2q, fc2s, gs)

    dense0 = dense(x0)
    dense1 = dense(x1)
    dense_ms = timed(lambda: dense(x1))
    results = [{"path": "dense", "ms": dense_ms, "speedup": 1.0}]

    for fraction in parse_fractions(args.fractions):
        idx = selected_rows(ffn, gs, fraction, device)
        fc1_rows = torch.cat((idx, idx + ffn), dim=0)
        q1 = fc1q.index_select(0, fc1_rows).contiguous()
        s1 = fc1s if fc1s.numel() == 1 else fc1s.index_select(0, fc1_rows).contiguous()
        q2 = fc2q.index_select(1, idx).contiguous()

        def selected_act(x):
            return logical_swiglu(ck_linear(x, q1, s1, gs))

        old_act = selected_act(x0)

        def delta_once():
            new_act = selected_act(x1)
            old_part = ck_linear(old_act, q2, fc2s, gs)
            new_part = ck_linear(new_act, q2, fc2s, gs)
            return dense0 - old_part + new_part

        approx = delta_once()
        sparse_ms = timed(delta_once)
        rel = float((approx.float() - dense1.float()).norm() / dense1.float().norm())
        results.append({
            "path": "group_delta",
            "fraction_requested": fraction,
            "fraction_actual": float(idx.numel() / ffn),
            "selected_features": int(idx.numel()),
            "ms": sparse_ms,
            "speedup": dense_ms / sparse_ms,
            "relative_l2_vs_dense_next": rel,
        })

    payload = {
        "checkpoint": checkpoint,
        "block_index": args.block_index,
        "rows": args.rows,
        "hidden": hidden,
        "ffn": ffn,
        "group_size": gs,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
