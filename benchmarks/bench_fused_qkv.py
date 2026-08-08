"""Benchmark H3's BF16 QKV path against the Sparse Sage-native projection.

This is an explicit CUDA benchmark. It loads only one attention block's QKV
projection and norm weights from the selected safetensors checkpoint.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention.forward import project_qkv, to_hnd  # noqa: E402
from h3_attention.hybrid.fused_qkv import (  # noqa: E402
    HEAD_DIM,
    PreparedFusedQKV,
    run_fused_qkv,
)
from h3_attention.hybrid.router import KV_TILE, Q_TILE, SparseTileRouter  # noqa: E402
from h3_attention.hybrid.sparse_quant import _run as quantize_blocks  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    SparseSageExecutor,
    load_sparse_sage_api,
)


def resolve_checkpoint(value):
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.suffix.lower() != ".safetensors" or not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return str(candidate.resolve())

    import folder_paths

    resolved = Path(folder_paths.get_full_path_or_raise("diffusion_models", value))
    if resolved.suffix.lower() != ".safetensors" or not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return str(resolved.resolve())


def _prefixes(block_index):
    stem = "blocks.%d.attn." % int(block_index)
    return ("model.diffusion_model." + stem, "diffusion_model." + stem, stem)


def load_attention_tensors(checkpoint, block_index):
    from safetensors import safe_open

    required = ("qkv_proj.weight", "q_norm.weight", "k_norm.weight")
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        prefix = next(
            (item for item in _prefixes(block_index)
             if all(item + suffix in keys for suffix in required)),
            None,
        )
        if prefix is None:
            raise KeyError("checkpoint has no complete blocks.%d.attn QKV state" % block_index)
        state = {
            key[len(prefix):]: handle.get_tensor(key)
            for key in keys
            if key.startswith(prefix)
            and key[len(prefix):].startswith(("qkv_proj.", "q_norm.", "k_norm."))
        }
    return prefix, state


def build_attention(checkpoint, block_index, epsilon):
    import comfy.ops

    prefix, state = load_attention_tensors(checkpoint, block_index)
    weight = state["qkv_proj.weight"]
    if weight.ndim != 2 or int(weight.shape[0]) % (3 * HEAD_DIM):
        raise ValueError("QKV storage shape is not H3-compatible: %s" % (tuple(weight.shape),))
    hidden = int(weight.shape[1])
    heads = int(weight.shape[0]) // (3 * HEAD_DIM)
    ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16)
    qkv_proj = ops.Linear(hidden, heads * HEAD_DIM * 3, bias=False)
    qkv_state = {
        key[len("qkv_proj."):]: value
        for key, value in state.items()
        if key.startswith("qkv_proj.")
    }
    qkv_proj.load_state_dict(qkv_state, strict=True)
    q_norm = SimpleNamespace(weight=state["q_norm.weight"], eps=float(epsilon))
    k_norm = SimpleNamespace(weight=state["k_norm.weight"], eps=float(epsilon))
    return SimpleNamespace(
        qkv_proj=qkv_proj,
        q_norm=q_norm,
        k_norm=k_norm,
        heads=heads,
        head_dim=HEAD_DIM,
    ), hidden, prefix


def make_rope(sequence, device):
    angles = torch.arange(sequence * 48, device=device, dtype=torch.float32).reshape(sequence, 48)
    angles = angles * (1.0 / 8192.0)
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack((c, -s, s, c), dim=-1).reshape(
        1, sequence, 1, 48, 2, 2
    ).to(torch.bfloat16)


def baseline(module, x, rope, block_index):
    q, k, v = project_qkv(module, x, rope)
    q, k, v = to_hnd(q, k, v)
    q_summary = SparseTileRouter._mean_pool(q, Q_TILE).contiguous()
    k_summary = SparseTileRouter._mean_pool(k, KV_TILE).contiguous()
    q_int8, q_scale = quantize_blocks(q, Q_TILE)
    k_int8, k_scale = quantize_blocks(k, KV_TILE)
    return PreparedFusedQKV(
        q_int8=q_int8,
        q_scale=q_scale,
        k_int8=k_int8,
        k_scale=k_scale,
        v=v.contiguous(),
        q_summary=q_summary,
        k_summary=k_summary,
        output_dtype=x.dtype,
        sequence=int(x.shape[0]),
        heads=int(module.heads),
        head_dim=HEAD_DIM,
        layer_index=int(block_index),
        smooth_k=False,
    )


def benchmark_case(fn, warmup, iterations):
    for _ in range(warmup):
        result = fn()
        del result
    torch.cuda.synchronize()
    samples = []
    peaks = []
    for _ in range(iterations):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
        peaks.append(torch.cuda.max_memory_allocated() - before)
        del result
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "peak_bytes": max(peaks),
    }


def max_abs(a, b):
    return float((a.float() - b.float()).abs().max().item())


def relative_rmse(actual, reference):
    return float((
        (actual.float() - reference.float()).square().mean().sqrt()
        / reference.float().square().mean().sqrt().clamp_min(1e-8)
    ).item())


def verify_attention(module, x, rope, block_index):
    sequence = int(x.shape[0])
    q_blocks = (sequence + Q_TILE - 1) // Q_TILE
    k_blocks = (sequence + KV_TILE - 1) // KV_TILE
    dense = torch.arange(k_blocks, dtype=torch.int32, device=x.device)
    delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
    lut = delta.view(1, 1, 1, -1).expand(
        1, module.heads, q_blocks, k_blocks
    ).contiguous()
    valid = torch.full(
        (1, module.heads, q_blocks),
        k_blocks,
        dtype=torch.int32,
        device=x.device,
    )
    executor = SparseSageExecutor(load_sparse_sage_api())

    q, k, v = project_qkv(module, x, rope)
    q, k, v = to_hnd(q, k, v)
    dense_output = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    established = executor.prepare(
        q,
        k,
        v,
        lut,
        valid,
        layer_index=block_index,
        metadata={"path": "established"},
    )
    established_output = executor.execute(established)
    del established, q, k, v

    projected = run_fused_qkv(module, x, rope, layer_index=block_index)
    fused = executor.prepare_projected(
        projected,
        lut,
        valid,
        metadata={"path": "fused"},
    )
    fused_output = executor.execute(fused)
    torch.cuda.synchronize()
    return {
        "fused_vs_established_relative_rmse": relative_rmse(
            fused_output, established_output
        ),
        "established_vs_dense_relative_rmse": relative_rmse(
            established_output, dense_output
        ),
        "fused_vs_dense_relative_rmse": relative_rmse(
            fused_output, dense_output
        ),
        "fused_vs_established_max_abs": max_abs(fused_output, established_output),
        "sequence": sequence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--sequence", type=int, default=54006)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verify-attention", action="store_true")
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    args = parser.parse_args()
    if not args.i_understand_this_uses_gpu:
        raise SystemExit("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("the fused H3 QKV experiment is SM89-only")
    if args.sequence <= 0 or args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("sequence/iteration arguments are invalid")

    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(checkpoint, args.block, args.epsilon)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn(
        (args.sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(args.sequence, device)

    baseline_fn = lambda: baseline(module, x, rope, args.block)
    fused_fn = lambda: run_fused_qkv(module, x, rope, layer_index=args.block)
    reference = baseline_fn()
    fused = fused_fn()
    torch.cuda.synchronize()
    comparisons = {
        "q_int8_equal_fraction": float((reference.q_int8 == fused.q_int8).float().mean().item()),
        "k_int8_equal_fraction": float((reference.k_int8 == fused.k_int8).float().mean().item()),
        "q_scale_max_abs": max_abs(reference.q_scale, fused.q_scale),
        "k_scale_max_abs": max_abs(reference.k_scale, fused.k_scale),
        "v_max_abs": max_abs(reference.v, fused.v),
        "q_summary_max_abs": max_abs(reference.q_summary, fused.q_summary),
        "k_summary_max_abs": max_abs(reference.k_summary, fused.k_summary),
    }
    del reference, fused

    baseline_result = benchmark_case(baseline_fn, args.warmup, args.iterations)
    fused_result = benchmark_case(fused_fn, args.warmup, args.iterations)
    result = {
        "checkpoint": checkpoint,
        "prefix": prefix,
        "block": args.block,
        "sequence": args.sequence,
        "hidden": hidden,
        "heads": module.heads,
        "device": torch.cuda.get_device_name(),
        "baseline": baseline_result,
        "fused": fused_result,
        "peak_reduction_bytes": baseline_result["peak_bytes"] - fused_result["peak_bytes"],
        "speedup": baseline_result["median_ms"] / fused_result["median_ms"],
        "comparisons": comparisons,
    }
    if args.verify_attention:
        result["attention"] = verify_attention(
            module,
            x,
            rope,
            args.block,
        )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("sequence=%d hidden=%d heads=%d" % (args.sequence, hidden, module.heads))
        print("baseline: %.3f ms, peak %.3f GiB" % (
            baseline_result["median_ms"], baseline_result["peak_bytes"] / 2**30))
        print("fused:    %.3f ms, peak %.3f GiB" % (
            fused_result["median_ms"], fused_result["peak_bytes"] / 2**30))
        print("speedup: %.3fx; peak reduction: %.3f GiB" % (
            result["speedup"], result["peak_reduction_bytes"] / 2**30))
        print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
