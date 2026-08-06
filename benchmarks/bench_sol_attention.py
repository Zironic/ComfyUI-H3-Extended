"""Compare prepared dense Sage with explicit Sol-Attn on H3 packed shapes.

Run from the ComfyUI root, for example:

    python custom_nodes/ComfyUI-H3-Extended/benchmarks/bench_sol_attention.py --frames 90

Sol-Attn is optional.  This benchmark forces its sparse path (no dense warmup or
dense layers), uses the released H3 prefix-sink policy, and feeds identical
post-RoPE-shaped BF16 Q/K/V to both backends.
"""

import argparse
import json
import os
import statistics
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.abspath(os.path.join(_ROOT, "..", "..")))

import _minimax_vram_probe_base as probe  # noqa: E402
from h3_memory_optimizer.attention import (  # noqa: E402
    ATTENTION_AUTO,
    ATTENTION_SOL,
    FALLBACK_ERROR,
    RuntimeEnvironment,
    resolve_attention,
)
from h3_probe.layout import TokenLayout  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_runtime.metrics import tensor_error_metrics  # noqa: E402

GIB = 1024 ** 3
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FUSED_STRIDE = INNER * 3
BASE_SEED = 6841


def token_layout(frames, width, height, text_len):
    raw = probe.build_layout(frames, width, height, text_len)
    text = audio = video = None
    refs = []
    for start, stop, kind in raw["segments"]:
        if kind == "text":
            text = (start, stop)
        elif kind == "audio":
            audio = (start, stop)
        elif kind == "video":
            video = (start, stop)
        else:
            refs.append((kind, start, stop))
    return TokenLayout(
        seq_len=raw["seq_len"],
        text_range=text,
        audio_range=audio,
        video_range=video,
        video_shape=(raw["latent_t"], height // 32, width // 32),
        audio_t=(audio[1] - audio[0]) // 2,
        reference_ranges=refs,
        segments=list(raw["segments"]),
    )


def snapshot(layout, device):
    return RuntimeSnapshot(
        request_id=0,
        step_index=10,
        total_steps=20,
        sigma=0.5,
        branch=(0,),
        layout=layout,
        layout_signature=(layout.seq_len, tuple(layout.segments)),
        compute_dtype=torch.bfloat16,
        device=device,
    )


def fused_qkv(sequence, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    q, k, v = torch.randn(
        sequence,
        FUSED_STRIDE,
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    ).split(INNER, dim=-1)
    return (
        q.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        k.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
        v.view(sequence, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0),
    )


def run_once(backend, sequence, runtime, device, seed):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)
    q, k, v = fused_qkv(sequence, device, seed)
    options = {RUNTIME_KEY: runtime}
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    prepared = backend.prepare(
        q, k, v, layer_index=10, transformer_options=options
    )
    del q, k, v
    out = backend.execute(prepared)
    del prepared
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    peak = (torch.cuda.max_memory_allocated(device) - base) / GIB
    output = out.cpu()
    del out
    return elapsed, peak, output


def measure(backend, sequence, runtime, device, repeats):
    times, peaks = [], []
    last = None
    for index in range(repeats):
        elapsed, peak, output = run_once(
            backend, sequence, runtime, device, BASE_SEED + index
        )
        times.append(elapsed)
        peaks.append(peak)
        if last is not None:
            del last
        last = output
    return statistics.median(times), statistics.median(peaks), last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gate-heads", type=int, default=4)
    parser.add_argument("--density-heads", type=int, default=4)
    parser.add_argument("--sink-mode", default="prefix")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    environment = RuntimeEnvironment.detect()
    layout = token_layout(args.frames, args.width, args.height, args.text_len)
    runtime = snapshot(layout, device)

    dense = resolve_attention(
        ATTENTION_AUTO,
        FALLBACK_ERROR,
        environment=environment,
    ).backend
    sol = resolve_attention(
        ATTENTION_SOL,
        FALLBACK_ERROR,
        environment=environment,
        adapter_options={
            "tau": args.tau,
            "thresh_type": "diag",
            "dense_steps": 0,
            "dense_layers": 0,
            "sink_mode": args.sink_mode,
            "correctness_gate": True,
            "strict": True,
            "kv_splits": 1,
            "gate_heads": args.gate_heads,
            "density_heads": args.density_heads,
            "max_sink_fraction": 1.0,
        },
    ).backend

    # Warm compilation outside the measurements.
    warm_layout = token_layout(5, 256, 256, 64)
    warm_runtime = snapshot(warm_layout, device)
    run_once(dense, warm_layout.seq_len, warm_runtime, device, BASE_SEED - 1)
    run_once(sol, warm_layout.seq_len, warm_runtime, device, BASE_SEED - 1)

    dense_ms, dense_peak, dense_out = measure(
        dense, layout.seq_len, runtime, device, args.repeats
    )
    sol_ms, sol_peak, sol_out = measure(
        sol, layout.seq_len, runtime, device, args.repeats
    )
    error = tensor_error_metrics(sol_out, dense_out)
    result = {
        "frames": args.frames,
        "sequence": layout.seq_len,
        "architecture": environment.architecture,
        "dense_backend": getattr(dense, "name", type(dense).__name__),
        "dense_ms": dense_ms,
        "sol_ms": sol_ms,
        "sol_speedup": dense_ms / sol_ms,
        "dense_peak_gib": dense_peak,
        "sol_peak_gib": sol_peak,
        "sol_minus_dense_peak_gib": sol_peak - dense_peak,
        "error_vs_dense_sage": error,
        "sol_status": sol.as_status(),
    }
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for key, value in result.items():
            print("%-28s %s" % (key, value))


if __name__ == "__main__":
    with torch.no_grad():
        main()
