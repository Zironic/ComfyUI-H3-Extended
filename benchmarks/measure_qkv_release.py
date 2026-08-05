"""PLAN.md §2.1: does dropping the fused-QKV views actually free the memory?

This is the gate for the whole kernel effort. The fused bf16 QKV projection is
36% of the measured sampling transient (1.518 GB at C=73), and q/k/v are views
into it, so core's forward cannot release it. An H3-owned forward can - but only
if the allocator gives the memory back, and ComfyUI runs on `cudaMallocAsync`,
where a freed block need not shrink the reserved pool.

No model and no DiT are involved: the question is purely about the allocation
pattern, so it is reproduced directly at H3's real shapes and answered in
seconds instead of instrumenting a multi-hour sampling run.

    python custom_nodes/ComfyUI-H3-Extended/benchmarks/measure_qkv_release.py --dry-run
    python custom_nodes/ComfyUI-H3-Extended/benchmarks/measure_qkv_release.py --frames 73

Decision threshold (PLAN.md §2.1): a realized saving of >= 0.90 GB at C=73 beats
the headroom that chunked_ref2v §4.6 currently buys with +2.7% compute, and
Commits 3-5 are worth writing. Below that, they are not.
"""

import argparse
import json
import sys

import torch

GB = 1024 ** 3
HEADS, HEAD_DIM = 56, 128
INNER = HEADS * HEAD_DIM                 # 7168
QKV_STRIDE = INNER * 3                   # 21504 - the stride the int32 overflow turns on

# Measured packed sequence lengths at 1216x672, from chunked_ref2v/PLAN.md §4.3.
# The per-rung step is not perfectly uniform (8092/8096/8092/8092), so the
# measured values are used where they exist rather than re-derived from a step.
MEASURED_SEQ = {39: 21710, 56: 29802, 73: 37898, 90: 45990, 107: 54082}
RUNG_STEP = 8093                          # mean step, for rungs outside the table


def seq_len_for(frames):
    if (frames - 5) % 17:
        raise SystemExit("C=%d is not on the H3 grid (needs C %% 17 == 5)" % frames)
    if frames in MEASURED_SEQ:
        return MEASURED_SEQ[frames]
    anchor = min(MEASURED_SEQ, key=lambda c: abs(c - frames))
    return MEASURED_SEQ[anchor] + ((frames - anchor) // 17) * RUNG_STEP


def predict(seq):
    """Byte sizes of each tensor in the attention peak, in GB."""
    return {
        "fused_qkv_bf16": seq * QKV_STRIDE * 2 / GB,
        "q_int8": seq * INNER / GB,
        "k_int8": seq * INNER / GB,
        "v_fp8": seq * INNER / GB,
        "out_bf16": seq * INNER * 2 / GB,
    }


def _quantize_int8(x):
    """Per-token amax scaling. Stands in for the real kernel: same output size,
    same read of the strided source, no dependency on sageattention."""
    scale = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-6) / 127.0
    return (x.float() / scale).round().clamp(-127, 127).to(torch.int8), scale


def run_once(seq, release, device):
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.memory_allocated(device)
    base_reserved = torch.cuda.memory_reserved(device)

    # exactly core's shape: the fused projection is never bound to a local, so
    # q/k/v views are the only thing keeping the whole buffer alive
    q, k, v = torch.empty(seq, QKV_STRIDE, dtype=torch.bfloat16,
                          device=device).normal_().split(INNER, dim=-1)
    q = q.view(seq, HEADS, HEAD_DIM)
    k = k.view(seq, HEADS, HEAD_DIM)
    v = v.view(seq, HEADS, HEAD_DIM)
    after_proj = torch.cuda.memory_allocated(device) - base_alloc

    q8, q_scale = _quantize_int8(q)
    k8, k_scale = _quantize_int8(k)
    v8 = v.to(torch.float8_e4m3fn)

    if release:
        del q, k, v                      # drops the last refs to the fused buffer
    torch.cuda.synchronize(device)
    after_quant = torch.cuda.memory_allocated(device) - base_alloc

    # the attention kernel's own output, allocated while the above is resident
    out = torch.empty(seq, INNER, dtype=torch.bfloat16, device=device)
    torch.cuda.synchronize(device)

    peak_alloc = torch.cuda.max_memory_allocated(device) - base_alloc
    peak_reserved = torch.cuda.max_memory_reserved(device) - base_reserved
    del q8, k8, v8, q_scale, k_scale, out
    if not release:
        del q, k, v
    return {
        "after_projection_gb": after_proj / GB,
        "after_quantize_gb": after_quant / GB,
        "peak_allocated_gb": peak_alloc / GB,
        "peak_reserved_gb": peak_reserved / GB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=73, help="chunk frames C (grid: C %% 17 == 5)")
    ap.add_argument("--repeats", type=int, default=3, help="runs per mode; the median is reported")
    ap.add_argument("--dry-run", action="store_true", help="print predicted sizes, allocate nothing")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    seq = seq_len_for(args.frames)
    sizes = predict(seq)
    print("C=%d  S=%d  seq stride=%d" % (args.frames, seq, QKV_STRIDE))
    for name, gb in sizes.items():
        print("   %-16s %7.3f GB" % (name, gb))
    print("   %-16s %7.3f GB   <- the release target" % ("releasable", sizes["fused_qkv_bf16"]))

    if args.dry_run:
        print("\ndry run: nothing allocated")
        return 0
    if not torch.cuda.is_available():
        print("\nno CUDA device", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    free, total = torch.cuda.mem_get_info(device)
    need = sum(sizes.values())
    print("\nGPU free %.2f / %.2f GB, this run needs ~%.2f GB"
          % (free / GB, total / GB, need))
    if free / GB < need * 1.3:
        print("not enough free VRAM; refusing to run", file=sys.stderr)
        return 1

    results = {}
    for mode, release in (("hold", False), ("release", True)):
        runs = [run_once(seq, release, device) for _ in range(args.repeats)]
        runs.sort(key=lambda r: r["peak_allocated_gb"])
        results[mode] = runs[len(runs) // 2]
        torch.cuda.empty_cache()

    print("\n%-10s %14s %14s %14s %14s"
          % ("mode", "after proj", "after quant", "peak alloc", "peak reserved"))
    for mode, r in results.items():
        print("%-10s %11.3f GB %11.3f GB %11.3f GB %11.3f GB"
              % (mode, r["after_projection_gb"], r["after_quantize_gb"],
                 r["peak_allocated_gb"], r["peak_reserved_gb"]))

    saved_alloc = results["hold"]["peak_allocated_gb"] - results["release"]["peak_allocated_gb"]
    saved_reserved = results["hold"]["peak_reserved_gb"] - results["release"]["peak_reserved_gb"]
    print("\nrealized saving:  allocated %.3f GB   reserved %.3f GB"
          % (saved_alloc, saved_reserved))
    print("predicted ceiling: %.3f GB" % sizes["fused_qkv_bf16"])
    verdict = "PROCEED to Commits 3-5" if saved_reserved >= 0.90 else "STOP - below the §2.1 threshold"
    print("threshold 0.90 GB reserved  ->  %s" % verdict)

    if args.json:
        print(json.dumps({"frames": args.frames, "seq_len": seq, "sizes_gb": sizes,
                          "results": results, "saved_reserved_gb": saved_reserved}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
