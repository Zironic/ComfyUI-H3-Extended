"""Synthetic MLP memory/runtime sweep for H3 activation chunking.

Run from the ComfyUI root. Defaults are deliberately small; pass H3 dimensions
and a CUDA device for the real allocation geometry:

    python custom_nodes/ComfyUI-H3-Extended/benchmarks/benchmark_h3_activation_memory.py \
        --device cuda --dtype bf16 --seq 45990 --hidden 5376 --ffn 14336 \
        --chunks 1024,2048,4096,8192
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

from h3_activation_memory.linear import swiglu_eager  # noqa: E402


def dtype_from_name(name):
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, warmup, iterations):
    for _ in range(warmup):
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    started = time.perf_counter()
    checksum = None
    for _ in range(iterations):
        checksum = float(fn().float().mean())
    synchronize(device)
    elapsed = (time.perf_counter() - started) / iterations
    peak = (
        torch.cuda.max_memory_allocated(device) - baseline
        if device.type == "cuda"
        else 0
    )
    return {"seconds": elapsed, "peak_bytes": peak, "checksum": checksum}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype", choices=("fp32", "fp16", "bf16"), default="fp32"
    )
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--ffn", type=int, default=1024)
    parser.add_argument("--chunks", default="1024,2048,4096,8192")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(0)
    x = torch.randn(args.seq, args.hidden, device=device, dtype=dtype)
    fc1 = torch.nn.Linear(
        args.hidden,
        args.ffn * 2,
        bias=False,
        device=device,
        dtype=dtype,
    )
    fc2 = torch.nn.Linear(
        args.ffn,
        args.hidden,
        bias=False,
        device=device,
        dtype=dtype,
    )

    def stock():
        return fc2(swiglu_eager(fc1(x)))

    baseline = measure(stock, device, args.warmup, args.iterations)
    rows = [{"mode": "full", "chunk_rows": args.seq, **baseline}]

    for chunk_rows in (int(v) for v in args.chunks.split(",") if v.strip()):
        def chunked(chunk_rows=chunk_rows):
            output = torch.empty_like(x)
            for start in range(0, args.seq, chunk_rows):
                stop = min(args.seq, start + chunk_rows)
                output[start:stop] = fc2(
                    swiglu_eager(fc1(x[start:stop]))
                )
            return output

        result = measure(chunked, device, args.warmup, args.iterations)
        result["max_abs_vs_full"] = float(
            (chunked().float() - stock().float()).abs().max()
        )
        rows.append(
            {"mode": "chunked", "chunk_rows": chunk_rows, **result}
        )

    print(
        json.dumps(
            {
                "shape": {
                    "seq": args.seq,
                    "hidden": args.hidden,
                    "ffn": args.ffn,
                    "dtype": str(dtype),
                    "device": str(device),
                },
                "results": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with torch.no_grad():
        main()
