"""CUDA A/B benchmark for the H3 ConvRot MLP compile boundary.

The checkpoint loader and ConvRot invocation deliberately remain the exact
helpers used by ``benchmark_h3_activation_memory``.  Only the Python/DLPack
boundary is wrapped in a benchmark-local custom operator so that the per-slab
core can be captured by ``torch.compile``.
"""

import argparse
import itertools
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PACK_ROOT = os.path.dirname(_HERE)
if _PACK_ROOT not in sys.path:
    sys.path.insert(0, _PACK_ROOT)

from benchmarks import benchmark_h3_activation_memory as activation_bench  # noqa: E402
from h3_runtime.metrics import tensor_error_metrics  # noqa: E402


DEFAULT_SEQ = 63448
DEFAULT_CHUNK_ROWS = 2048
DEFAULT_FEATURE_TILE_WIDTH = 7168
DEFAULT_WARMUP = 1
DEFAULT_ITERATIONS = 3
DEFAULT_DTYPE = "bf16"
REQUIRED_SM = (8, 9)
COMPILE_KWARGS = {"fullgraph": True, "dynamic": False}
_OP_COUNTER = itertools.count()


def plan_slab_shapes(seq=DEFAULT_SEQ, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Return stable token-slab row counts, retaining a separate tail shape."""
    seq = int(seq)
    chunk_rows = int(chunk_rows)
    if seq <= 0 or chunk_rows <= 0:
        raise ValueError("seq and chunk_rows must be positive")
    full, tail = divmod(seq, chunk_rows)
    shapes = [chunk_rows] * full
    if tail:
        shapes.append(tail)
    return tuple(shapes)


def plan_slab_ranges(seq=DEFAULT_SEQ, chunk_rows=DEFAULT_CHUNK_ROWS):
    """Return ``(start, stop)`` ranges corresponding to ``plan_slab_shapes``."""
    offset = 0
    ranges = []
    for rows in plan_slab_shapes(seq, chunk_rows):
        ranges.append((offset, offset + rows))
        offset += rows
    return tuple(ranges)


def compile_kwargs():
    """The fixed-shape, no-graph-break policy used by this benchmark."""
    return dict(COMPILE_KWARGS)


def tensor_parity(actual, reference):
    return tensor_error_metrics(actual, reference)


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_path(fn, device, warmup=DEFAULT_WARMUP, iterations=DEFAULT_ITERATIONS):
    """Measure complete slab-loop calls; warmup and allocation are excluded."""
    if int(warmup) < 0 or int(iterations) <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    for _ in range(int(warmup)):
        result = fn()
        del result
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    samples = []
    result = None
    for _ in range(int(iterations)):
        started = time.perf_counter()
        result = fn()
        _synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
        del result
    peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else 0
    return {
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "samples_ms": samples,
        "peak_allocated_bytes": int(peak),
    }


def _fake_output(x, output_features):
    return torch.empty((x.shape[0], int(output_features)), device=x.device, dtype=torch.bfloat16)


def make_convrot_adapter(ck, weight, weight_scale, group_size, output_features, *, input_act=None, label="convrot"):
    """Create a custom op whose implementation is the unchanged CK call."""
    name = "h3_mlp_compile::%s_%d" % (label, next(_OP_COUNTER))

    def invoke(x: torch.Tensor, op_weight: torch.Tensor, op_scale: torch.Tensor) -> torch.Tensor:
        return activation_bench._convrot_output(
            ck, x, op_weight, op_scale, group_size, input_act=input_act
        )

    op = torch.library.custom_op(name, mutates_args=())(invoke)

    @op.register_fake
    def fake(x: torch.Tensor, op_weight: torch.Tensor, op_scale: torch.Tensor):
        del op_weight, op_scale
        return _fake_output(x, output_features)

    return op


def _move_weight(info, device):
    moved = dict(info)
    moved["weight"] = info["weight"].to(device=device)
    moved["weight_scale"] = info["weight_scale"].to(device=device, dtype=torch.float32)
    return moved


def _whole_direct(ck, activation, fc1, fc2, slab_shapes, chunk_rows):
    output = torch.empty((activation.shape[0], fc2["weight"].shape[0]), device=activation.device, dtype=torch.bfloat16)
    offset = 0
    for rows in slab_shapes:
        slab = activation[offset : offset + rows]
        expanded = activation_bench._convrot_output(ck, slab, fc1["weight"], fc1["weight_scale"], fc1["group_size"], input_act=None)
        output[offset : offset + rows] = activation_bench._convrot_output(
            ck, expanded, fc2["weight"], fc2["weight_scale"], fc2["group_size"], input_act="swiglu"
        )
        offset += rows
    return output


def _tiled_direct(ck, activation, fc1, fc2, tiles, slab_shapes):
    output = torch.empty((activation.shape[0], fc2["weight"].shape[0]), device=activation.device, dtype=torch.bfloat16)
    offset = 0
    for rows in slab_shapes:
        slab = activation[offset : offset + rows]
        partial_output = None
        for tile in tiles:
            expanded = activation_bench._convrot_output(ck, slab, tile["fc1_weight"], tile["fc1_scale"], fc1["group_size"], input_act=None)
            partial = activation_bench._convrot_output(ck, expanded, tile["fc2_weight"], fc2["weight_scale"], fc2["group_size"], input_act="swiglu")
            del expanded
            if partial_output is None:
                partial_output = partial
            else:
                partial_output.add_(partial)
                del partial
        output[offset : offset + rows] = partial_output
        offset += rows
    return output


def _make_whole_adapter_core(ck, fc1, fc2):
    fc1_op = make_convrot_adapter(ck, fc1["weight"], fc1["weight_scale"], fc1["group_size"], fc1["weight"].shape[0], label="whole_fc1")
    fc2_op = make_convrot_adapter(ck, fc2["weight"], fc2["weight_scale"], fc2["group_size"], fc2["weight"].shape[0], input_act="swiglu", label="whole_fc2")

    def core(x):
        return fc2_op(fc1_op(x, fc1["weight"], fc1["weight_scale"]), fc2["weight"], fc2["weight_scale"])

    return core


def _make_tiled_adapter_core(ck, fc1, fc2, tiles):
    adapters = tuple(
        (
            make_convrot_adapter(ck, tile["fc1_weight"], tile["fc1_scale"], fc1["group_size"], tile["fc1_weight"].shape[0], label="tile_fc1"),
            make_convrot_adapter(ck, tile["fc2_weight"], fc2["weight_scale"], fc2["group_size"], fc2["weight"].shape[0], input_act="swiglu", label="tile_fc2"),
        )
        for tile in tiles
    )

    def core(x):
        output = None
        for (fc1_op, fc2_op), tile in zip(adapters, tiles):
            expanded = fc1_op(x, tile["fc1_weight"], tile["fc1_scale"])
            partial = fc2_op(expanded, tile["fc2_weight"], fc2["weight_scale"])
            output = partial if output is None else output + partial
        return output

    return core


def _run_core_slabs(core_by_shape, activation, slab_shapes, output_features):
    output = torch.empty((activation.shape[0], int(output_features)), device=activation.device, dtype=torch.bfloat16)
    offset = 0
    for rows in slab_shapes:
        output[offset : offset + rows] = core_by_shape[rows](activation[offset : offset + rows])
        offset += rows
    return output


def _compile_shape_cores(core, activation, slab_shapes, device):
    unique_shapes = tuple(dict.fromkeys(slab_shapes))
    compiled = {}
    started = time.perf_counter()
    for rows in unique_shapes:
        compiled[rows] = torch.compile(core, **compile_kwargs())
        # Compiling each fixed shape separately makes dynamic=False tail cost
        # explicit and ensures no first-call work enters measured repeats.
        warm = activation[:rows]
        result = compiled[rows](warm)
        del result
        _synchronize(device)
    return compiled, (time.perf_counter() - started) * 1000.0


def benchmark_mode(mode, ck, activation, fc1, fc2, tiles, chunk_rows, warmup, iterations, device):
    slab_shapes = plan_slab_shapes(activation.shape[0], chunk_rows)
    if mode == "whole_feature_native":
        direct = lambda: _whole_direct(ck, activation, fc1, fc2, slab_shapes, chunk_rows)
        adapter_core = _make_whole_adapter_core(ck, fc1, fc2)
    elif mode == "feature_tiled":
        direct = lambda: _tiled_direct(ck, activation, fc1, fc2, tiles, slab_shapes)
        adapter_core = _make_tiled_adapter_core(ck, fc1, fc2, tiles)
    else:
        raise ValueError("unknown MLP mode: %s" % mode)
    adapter = lambda: _run_core_slabs({rows: adapter_core for rows in set(slab_shapes)}, activation, slab_shapes, fc2["weight"].shape[0])
    compiled_cores, compile_ms = _compile_shape_cores(adapter_core, activation, slab_shapes, device)
    compiled = lambda: _run_core_slabs(compiled_cores, activation, slab_shapes, fc2["weight"].shape[0])
    direct_metrics = measure_path(direct, device, warmup, iterations)
    adapter_metrics = measure_path(adapter, device, warmup, iterations)
    compiled_metrics = measure_path(compiled, device, warmup, iterations)
    direct_output = direct().detach().cpu()
    adapter_output = adapter().detach().cpu()
    direct_vs_adapter = tensor_parity(adapter_output, direct_output)
    del direct_output
    compiled_output = compiled().detach().cpu()
    adapter_vs_compiled = tensor_parity(compiled_output, adapter_output)
    del adapter_output, compiled_output
    adapter_ms = adapter_metrics["median_ms"] - direct_metrics["median_ms"]
    result = {
        "mode": mode,
        "slabs": {"count": len(slab_shapes), "shapes": list(dict.fromkeys(slab_shapes)), "chunk_rows": int(chunk_rows)},
        "direct_eager": direct_metrics,
        "adapter_eager": adapter_metrics,
        "compiled_adapter": compiled_metrics,
        "adapter_overhead_ms": adapter_ms,
        "adapter_overhead_ratio": adapter_metrics["median_ms"] / max(direct_metrics["median_ms"], 1e-12),
        "compiled_speedup_vs_adapter_eager": adapter_metrics["median_ms"] / max(compiled_metrics["median_ms"], 1e-12),
        "compiled_speedup_vs_direct_eager": direct_metrics["median_ms"] / max(compiled_metrics["median_ms"], 1e-12),
        "compile": {"kwargs": compile_kwargs(), "warmup_ms": compile_ms, "shape_count": len(compiled_cores), "graph_count": len(compiled_cores)},
        "parity": {
            "direct_vs_adapter": direct_vs_adapter,
            "adapter_vs_compiled": adapter_vs_compiled,
        },
    }
    if mode == "feature_tiled":
        result["feature_tile_count"] = len(tiles)
        result["feature_tile_ranges"] = [[int(tile["start"]), int(tile["stop"])] for tile in tiles]
        result["prepared_tile_bytes"] = activation_bench.prepared_tile_bytes(tiles)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    parser.add_argument("--chunk-rows", type=int, default=DEFAULT_CHUNK_ROWS)
    parser.add_argument("--feature-tile-width", type=int, default=DEFAULT_FEATURE_TILE_WIDTH)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    return parser


def run(args):
    if not args.i_understand_this_uses_gpu:
        raise RuntimeError("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != REQUIRED_SM:
        raise RuntimeError("this H3 MLP compile benchmark is SM89-only")
    if not args.checkpoint.is_file() or args.checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("--checkpoint must be an existing .safetensors file")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if args.seq <= 0 or args.chunk_rows <= 0:
        raise ValueError("seq and chunk_rows must be positive")
    loaded = activation_bench.load_block_mlp_tensors(str(args.checkpoint), args.block_index)
    convrot = activation_bench.load_convrot_mlp(loaded)
    device = torch.device("cuda")
    fc1, fc2 = _move_weight(convrot["fc1"], device), _move_weight(convrot["fc2"], device)
    tiles_cpu = activation_bench.prepare_convrot_tiles(convrot["fc1"], convrot["fc2"], args.feature_tile_width)
    tiles = tuple({**tile, "fc1_weight": tile["fc1_weight"].to(device), "fc1_scale": tile["fc1_scale"].to(device=device, dtype=torch.float32), "fc2_weight": tile["fc2_weight"].to(device)} for tile in tiles_cpu)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    activation = torch.randn((args.seq, convrot["hidden_width"]), generator=generator, device=device, dtype=torch.bfloat16)
    ck = _load_ck()
    results = [benchmark_mode(mode, ck, activation, fc1, fc2, tiles if mode == "feature_tiled" else (), args.chunk_rows, args.warmup, args.iterations, device) for mode in ("whole_feature_native", "feature_tiled")]
    payload = {"checkpoint": {"path": str(args.checkpoint), "block_index": int(args.block_index), "weight_prefix": loaded["prefix"]}, "shape": {"seq": int(args.seq), "hidden": convrot["hidden_width"], "ffn": convrot["ffn_width"], "dtype": str(torch.bfloat16), "chunk_rows": int(args.chunk_rows), "feature_tile_width": int(args.feature_tile_width)}, "device": {"name": torch.cuda.get_device_name(), "capability": list(capability), "capability_label": "SM%d%d" % capability}, "results": results}
    return payload


def _load_ck():
    return activation_bench._load_comfy_kitchen()


def serialize_result(payload):
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        payload = run(args)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc))
    encoded = serialize_result(payload)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
