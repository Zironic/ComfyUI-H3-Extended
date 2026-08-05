"""Compare stock SM89 SageAttention with the H3 two-stage backend.

Run from the ComfyUI root. The benchmark uses H3's real fused-QKV stride and
reports steady-state latency plus complete-call peak allocated memory.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention.sage_mem_eff import SM89SageMemoryEfficientBackend

GIB = 1024 ** 3
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FUSED_STRIDE = INNER * 3
SEQUENCES = {22: 13617, 73: 37898, 90: 45990}


def fused_qkv(sequence, device):
    q, k, v = torch.randn(
        sequence, FUSED_STRIDE,
        dtype=torch.bfloat16,
        device=device,
    ).split(INNER, dim=-1)
    return (
        q.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        k.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        v.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
    )


def run_stock(q, k, v):
    import sageattention.core as core
    return core.sageattn_qk_int8_pv_fp8_cuda(
        q, k, v,
        tensor_layout="HND",
        is_causal=False,
        qk_quant_gran="per_thread",
        pv_accum_dtype="fp32+fp16",
        smooth_k=False,
        smooth_v=False,
        return_lse=False,
    )


def measure_stock(sequence, repeats, device):
    times = []
    peaks = []
    outputs = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        q, k, v = fused_qkv(sequence, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        out = run_stock(q, k, v)
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000)
        peaks.append((torch.cuda.max_memory_allocated(device) - base) / GIB)
        outputs.append(out)
        del q, k, v, out
    return statistics.median(times), statistics.median(peaks), outputs[-1]


def measure_custom(sequence, repeats, device):
    backend = SM89SageMemoryEfficientBackend()
    times = []
    peaks = []
    outputs = []
    for _ in range(repeats):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        q, k, v = fused_qkv(sequence, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        prepared = backend.prepare(q, k, v, layer_index=0, transformer_options={})
        del q, k, v
        out = backend.execute(prepared)
        del prepared
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000)
        peaks.append((torch.cuda.max_memory_allocated(device) - base) / GIB)
        outputs.append(out)
        del out
    return statistics.median(times), statistics.median(peaks), outputs[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, choices=sorted(SEQUENCES), default=90)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (8, 9):
        raise SystemExit("this benchmark is SM89-only")

    sequence = SEQUENCES[args.frames]
    # Warm both compiled paths outside measurements.
    q, k, v = fused_qkv(1024, device)
    run_stock(q, k, v)
    backend = SM89SageMemoryEfficientBackend()
    prepared = backend.prepare(q, k, v, layer_index=0, transformer_options={})
    del q, k, v
    backend.execute(prepared)
    del prepared
    torch.cuda.synchronize(device)

    stock_ms, stock_peak, stock_out = measure_stock(sequence, args.repeats, device)
    custom_ms, custom_peak, custom_out = measure_custom(sequence, args.repeats, device)
    delta = (custom_out.float() - stock_out.float()).abs()
    rel = float(delta.mean() / stock_out.float().pow(2).mean().sqrt())
    result = {
        "frames": args.frames,
        "sequence": sequence,
        "stock_ms": stock_ms,
        "custom_ms": custom_ms,
        "stock_peak_gib": stock_peak,
        "custom_peak_gib": custom_peak,
        "peak_saved_gib": stock_peak - custom_peak,
        "relative_error": rel,
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print("%-22s %s" % (key, value))
        verdict = "PASS" if result["peak_saved_gib"] >= 0.90 else "FAIL"
        print("C=90 target >= 0.90 GiB: %s" % verdict)


if __name__ == "__main__":
    main()
