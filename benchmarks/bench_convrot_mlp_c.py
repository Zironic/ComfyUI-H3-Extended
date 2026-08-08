"""Benchmark the complete H3 ConvRot MLP paths on one real checkpoint block.

This is an explicit CUDA benchmark. Checkpoint parsing and validation remain
CPU-safe so focused contract tests do not need Comfy Kitchen or a model file.
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_EXPANDED_WIDTH = 28672
DEFAULT_FFN_WIDTH = 14336
DEFAULT_HIDDEN_WIDTH = 5376
DEFAULT_ROWS = (2048, 8192)
DEFAULT_FEATURE_TILES = (7168, 3584)


def max_absolute_error(actual, reference):
    return float((actual.float() - reference.float()).abs().max().item())


def relative_l2(actual, reference):
    delta = (actual.float() - reference.float()).reshape(-1)
    denominator = reference.float().reshape(-1).norm().clamp_min(1e-8)
    return float((delta.norm() / denominator).item())


def _run_timed(fn):
    result = fn()
    torch.cuda.synchronize()
    return result


def benchmark_case(fn, warmup, iterations):
    for _ in range(warmup):
        result = _run_timed(fn)
        del result
    torch.cuda.synchronize()
    samples = []
    peaks = []
    for _ in range(iterations):
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        stop.record()
        stop.synchronize()
        samples.append(float(start.elapsed_time(stop)))
        peaks.append(int(torch.cuda.max_memory_allocated() - before))
        del result
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "mean_ms": statistics.mean(samples),
        "peak_allocated_delta_bytes": max(peaks),
    }


def _decode_quant_config(value):
    return json.loads(value.detach().cpu().numpy().tobytes())


def load_convrot_layer(checkpoint, layer, block_index=0, safe_open_fn=None):
    """Load one H3 MLP TensorWise-INT8 ConvRot layer, fail-closed."""
    if safe_open_fn is None:
        from safetensors import safe_open as safe_open_fn

    layer = str(layer)
    if layer not in ("fc1", "fc2"):
        raise ValueError("layer must be fc1 or fc2")
    suffix = "blocks.%d.mlp.%s." % (int(block_index), layer)
    prefixes = ("model.diffusion_model.", "diffusion_model.", "")
    with safe_open_fn(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        prefix = next(
            (candidate + suffix for candidate in prefixes if candidate + suffix + "weight" in keys),
            None,
        )
        if prefix is None:
            raise KeyError("checkpoint has no blocks.%d.mlp.%s weight" % (block_index, layer))
        required = (prefix + "weight", prefix + "weight_scale", prefix + "comfy_quant")
        missing = [key for key in required if key not in keys]
        if missing:
            raise KeyError("%s checkpoint tensors are missing: %s" % (layer, ", ".join(missing)))
        weight = handle.get_tensor(required[0])
        weight_scale = handle.get_tensor(required[1])
        quant = _decode_quant_config(handle.get_tensor(required[2]))

    if quant.get("format") != "int8_tensorwise" or quant.get("convrot") is not True:
        raise ValueError("%s must be TensorWise-INT8 with ConvRot enabled" % layer)
    if quant.get("transposed", False):
        raise ValueError("transposed %s ConvRot weights are unsupported" % layer)
    group_size = int(quant.get("convrot_groupsize", 256))
    if group_size <= 0:
        raise ValueError("%s ConvRot group size must be positive" % layer)
    if weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError("ConvRot %s weight must be a rank-2 INT8 tensor" % layer)
    if weight_scale.numel() not in (1, weight.shape[0]):
        raise ValueError("ConvRot %s scale must be scalar or per-output-channel" % layer)
    return {
        "layer": layer,
        "weight": weight,
        "weight_scale": weight_scale,
        "group_size": group_size,
        "prefix": prefix,
        "quant": quant,
    }


def _load_comfy_kitchen():
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError("checkpoint comparison requires comfy-kitchen") from exc
    return ck


def _convrot_output(ck, x, weight, weight_scale, group_size, input_act="swiglu"):
    kwargs = {
        "bias": None,
        "out_dtype": torch.bfloat16,
        "convrot": True,
        "convrot_groupsize": group_size,
    }
    if input_act is not None:
        kwargs["input_act"] = input_act
    return ck.int8_linear(x, weight, weight_scale, **kwargs)


def _profile(fn, profile_path):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(profile_path))


def serialize_result(result):
    return json.dumps(_json_safe(result), indent=2, sort_keys=True)


def parse_rows(value):
    rows = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be a comma-separated list of positive integers")
    return rows


def parse_feature_tiles(value):
    tiles = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not tiles or any(item <= 0 or item % 256 for item in tiles):
        raise argparse.ArgumentTypeError(
            "feature tiles must be positive multiples of 256"
        )
    return tiles


def feature_ranges(ffn_width, tile_width):
    if tile_width <= 0 or tile_width % 256:
        raise ValueError("feature tile width must be a positive multiple of 256")
    if ffn_width % 256:
        raise ValueError("FFN width must be divisible by 256")
    return tuple(
        (start, min(start + tile_width, ffn_width))
        for start in range(0, ffn_width, tile_width)
    )


def validate_dimensions(expanded_width, ffn_width, hidden_width, rows):
    values = {
        "expanded_width": int(expanded_width),
        "ffn_width": int(ffn_width),
        "hidden_width": int(hidden_width),
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("model dimensions must be positive")
    if values["expanded_width"] != 2 * values["ffn_width"]:
        raise ValueError("expanded_width must be exactly twice ffn_width for SwiGLU")
    rows = tuple(int(item) for item in rows)
    if not rows or any(item <= 0 for item in rows):
        raise ValueError("rows must contain positive integers")
    return values["expanded_width"], values["ffn_width"], values["hidden_width"], rows


def load_convrot_mlp(checkpoint, block_index=0, safe_open_fn=None):
    """Load both exact TensorWise-INT8 ConvRot MLP projections."""
    fc1 = load_convrot_layer(checkpoint, "fc1", block_index, safe_open_fn)
    fc2 = load_convrot_layer(checkpoint, "fc2", block_index, safe_open_fn)
    if tuple(fc1["weight"].shape)[1] != tuple(fc2["weight"].shape)[0]:
        raise ValueError("fc1 input and fc2 output dimensions do not match")
    if tuple(fc1["weight"].shape)[0] != 2 * tuple(fc2["weight"].shape)[1]:
        raise ValueError("fc1/fc2 dimensions are not a SwiGLU pair")
    if fc1["weight"].shape[1] % fc1["group_size"]:
        raise ValueError("fc1 ConvRot input width is not group-size aligned")
    if fc2["weight"].shape[1] % fc2["group_size"]:
        raise ValueError("fc2 ConvRot input width is not group-size aligned")
    return {
        "fc1": fc1,
        "fc2": fc2,
        "block_index": int(block_index),
        "checkpoint": str(checkpoint),
        "hidden_width": int(fc1["weight"].shape[1]),
        "ffn_width": int(fc2["weight"].shape[1]),
        "expanded_width": int(fc1["weight"].shape[0]),
    }


def _dequantize_weight(weight_info, device):
    weight = weight_info["weight"].to(device=device)
    scale = weight_info["weight_scale"].to(device=device, dtype=torch.float32)
    return torch.ops.comfy_kitchen.dequantize_int8_convrot_weight_dtype(
        weight, scale, weight_info["group_size"], 2
    ).to(dtype=torch.bfloat16)


def _move_convrot_weight(weight_info, device):
    moved = dict(weight_info)
    moved["weight"] = weight_info["weight"].to(device=device)
    moved["weight_scale"] = weight_info["weight_scale"].to(
        device=device, dtype=torch.float32
    )
    return moved


def _bf16_mlp(x, fc1_weight, fc2_weight):
    expanded = F.linear(x, fc1_weight)
    gate, up = expanded.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, fc2_weight)


def _convrot_mlp(ck, x, fc1, fc2):
    expanded = _convrot_output(
        ck, x, fc1["weight"], fc1["weight_scale"], fc1["group_size"], input_act=None
    )
    return _convrot_output(
        ck, expanded, fc2["weight"], fc2["weight_scale"], fc2["group_size"]
    )


def _prepare_convrot_tiles(fc1, fc2, tile_width):
    ffn_width = int(fc2["weight"].shape[1])
    tiles = []
    for start, stop in feature_ranges(ffn_width, tile_width):
        width = stop - start
        if width % fc2["group_size"]:
            raise ValueError("every feature tile must preserve complete ConvRot groups")
        tiles.append({
            "start": start,
            "stop": stop,
            "fc1_weight": torch.cat((
                fc1["weight"][start:stop],
                fc1["weight"][ffn_width + start : ffn_width + stop],
            ), dim=0).contiguous(),
            "fc1_scale": torch.cat((
                fc1["weight_scale"][start:stop],
                fc1["weight_scale"][ffn_width + start : ffn_width + stop],
            ), dim=0).contiguous(),
            "fc2_weight": fc2["weight"][:, start:stop].contiguous(),
        })
    return tuple(tiles)


def _convrot_tiled_mlp(ck, x, fc1, fc2, tiles):
    output = None
    for tile in tiles:
        expanded = _convrot_output(
            ck, x, tile["fc1_weight"], tile["fc1_scale"],
            fc1["group_size"], input_act=None,
        )
        partial = _convrot_output(
            ck, expanded, tile["fc2_weight"], fc2["weight_scale"],
            fc2["group_size"],
        )
        del expanded
        if output is None:
            output = partial
        else:
            output.add_(partial)
            del partial
    return output


def _prepared_tile_bytes(tiles):
    names = ("fc1_weight", "fc1_scale", "fc2_weight")
    return sum(
        tile[name].numel() * tile[name].element_size()
        for tile in tiles
        for name in names
    )


def _profile_has_allocation(profile_path, allocation_bytes):
    with Path(profile_path).open(encoding="utf-8") as handle:
        events = json.load(handle).get("traceEvents", ())
    return any(
        event.get("name") == "[memory]"
        and int(event.get("args", {}).get("Bytes", 0)) == int(allocation_bytes)
        for event in events
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS, metavar="N[,N...]")
    parser.add_argument(
        "--feature-tiles",
        type=parse_feature_tiles,
        default=DEFAULT_FEATURE_TILES,
        metavar="N[,N...]",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--json", type=Path, metavar="PATH")
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    return parser


def _json_safe(value):
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def run(args):
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not args.i_understand_this_uses_gpu:
        raise RuntimeError("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for the ConvRot MLP benchmark")
    loaded = load_convrot_mlp(args.checkpoint, args.block_index)
    expanded_width, ffn_width, hidden_width, rows_values = validate_dimensions(
        loaded["expanded_width"], loaded["ffn_width"], loaded["hidden_width"], args.rows
    )
    ck = _load_comfy_kitchen()
    device = torch.device("cuda")
    fc1_convrot = _move_convrot_weight(loaded["fc1"], device)
    fc2_convrot = _move_convrot_weight(loaded["fc2"], device)
    fc1_weight = _dequantize_weight(fc1_convrot, device)
    fc2_weight = _dequantize_weight(fc2_convrot, device)
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    results = []
    for rows in rows_values:
        x = torch.randn((rows, hidden_width), generator=generator, dtype=torch.bfloat16, device=device)
        reference = _bf16_mlp(x, fc1_weight, fc2_weight)
        paths = (
            ("bf16_full", lambda: _bf16_mlp(x, fc1_weight, fc2_weight)),
            ("comfy_convrot_int8", lambda: _convrot_mlp(ck, x, fc1_convrot, fc2_convrot)),
        )
        for name, fn in paths:
            output = fn()
            case = {
                "rows": rows,
                "case": name,
                "relative_l2": relative_l2(output, reference),
                "max_absolute_error": max_absolute_error(output, reference),
                "timing": benchmark_case(fn, args.warmup, args.iterations),
            }
            if args.profile_dir:
                profile_path = args.profile_dir / ("%s_rows%d.json" % (name, rows))
                _profile(fn, profile_path)
                case["profile"] = str(profile_path)
            results.append(case)
            del output
        for tile_width in args.feature_tiles:
            tiles = _prepare_convrot_tiles(fc1_convrot, fc2_convrot, tile_width)
            fn = lambda tiles=tiles: _convrot_tiled_mlp(
                ck, x, fc1_convrot, fc2_convrot, tiles
            )
            output = fn()
            case = {
                "rows": rows,
                "case": "comfy_convrot_int8_feature_tiled",
                "feature_tile_width": int(tile_width),
                "feature_tile_count": len(tiles),
                "prepared_tile_weight_bytes": _prepared_tile_bytes(tiles),
                "relative_l2": relative_l2(output, reference),
                "max_absolute_error": max_absolute_error(output, reference),
                "timing": benchmark_case(fn, args.warmup, args.iterations),
            }
            if args.profile_dir:
                profile_path = args.profile_dir / (
                    "comfy_convrot_int8_feature_tiled_%d_rows%d.json"
                    % (tile_width, rows)
                )
                _profile(fn, profile_path)
                full_expansion_bytes = rows * expanded_width * torch.bfloat16.itemsize
                case["profile"] = str(profile_path)
                case["full_bf16_expansion_allocation_seen"] = _profile_has_allocation(
                    profile_path, full_expansion_bytes
                )
                if case["full_bf16_expansion_allocation_seen"]:
                    raise RuntimeError(
                        "feature-tiled path allocated the full BF16 fc1 expansion"
                    )
            results.append(case)
            del output, tiles
        del x, reference
    return {
        "versions": {"torch": torch.__version__},
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "capability_label": "SM%d%d" % capability,
        },
        "dimensions": {
            "expanded_width": expanded_width,
            "ffn_width": ffn_width,
            "hidden_width": hidden_width,
            "rows": list(rows_values),
        },
        "checkpoint": {
            "path": str(args.checkpoint),
            "block_index": int(args.block_index),
            "weight_prefix": loaded["fc1"]["prefix"].rsplit("fc1.", 1)[0],
            "fc1_quant": loaded["fc1"]["quant"],
            "fc2_quant": loaded["fc2"]["quant"],
        },
        "feature_tiles": list(args.feature_tiles),
        "reference": {
            "dtype": str(torch.bfloat16),
            "weights": "exact ConvRot INT8 dequantized to BF16",
        },
        "results": results,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (RuntimeError, ValueError, FileNotFoundError, KeyError) as exc:
        raise SystemExit(str(exc))
    encoded = serialize_result(_json_safe(result))
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
