"""Compare H3's BF16 down projection with Transformer Engine FP8."""

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F


DEFAULT_EXPANDED_WIDTH = 28672
DEFAULT_FFN_WIDTH = 14336
DEFAULT_HIDDEN_WIDTH = 5376
DEFAULT_ROWS = (2048, 8192)
DEFAULT_RECIPES = (
    "delayed_e4m3",
    "current_e4m3",
    "delayed_hybrid",
    "current_hybrid",
)
FP8_DIM_MULTIPLE = 16


def parse_rows(value):
    rows = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise argparse.ArgumentTypeError("rows must be a comma-separated list of positive integers")
    return rows


def parse_recipes(value):
    recipes = tuple(item.strip().lower() for item in str(value).split(",") if item.strip())
    unknown = sorted(set(recipes) - set(DEFAULT_RECIPES))
    if not recipes or unknown:
        raise argparse.ArgumentTypeError(
            "recipes must contain delayed/current e4m3/hybrid names; unknown: %s" % ",".join(unknown)
        )
    return recipes


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
    if any(value % FP8_DIM_MULTIPLE for value in values.values()):
        raise ValueError("expanded, ffn, and hidden dimensions must be multiples of 16 for FP8")
    rows = tuple(int(item) for item in rows)
    if not rows or any(item <= 0 for item in rows):
        raise ValueError("rows must contain positive integers")
    return values["expanded_width"], values["ffn_width"], values["hidden_width"], rows


def max_absolute_error(actual, reference):
    return float((actual.float() - reference.float()).abs().max().item())


def relative_l2(actual, reference):
    delta = (actual.float() - reference.float()).reshape(-1)
    denominator = reference.float().reshape(-1).norm().clamp_min(1e-8)
    return float((delta.norm() / denominator).item())


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


def _run_timed(fn):
    result = fn()
    torch.cuda.synchronize()
    return result


def _load_te():
    import transformer_engine
    import transformer_engine.pytorch as te

    from transformer_engine.common import recipe

    from transformer_engine.pytorch.tensor import QuantizedTensor

    return transformer_engine, te, recipe, QuantizedTensor


def _recipe_for(te_recipe, recipe_name):
    scaling, format_name = recipe_name.split("_", 1)
    try:
        fmt = getattr(te_recipe.Format, format_name.upper())
    except AttributeError as exc:
        raise RuntimeError("installed Transformer Engine has no %s format" % format_name) from exc
    if scaling == "delayed":
        return te_recipe.DelayedScaling(fp8_format=fmt)
    current = getattr(te_recipe, "Float8CurrentScaling", None)
    if current is None:
        current = getattr(te_recipe, "CurrentScaling", None)
    if current is None:
        raise RuntimeError("installed Transformer Engine has no current-scaling recipe API")
    return current(fp8_format=fmt)


def _fp8_available(te):
    try:
        available = te.is_fp8_available(return_reason=True)
    except TypeError:
        available = te.is_fp8_available()
    if isinstance(available, tuple):
        return bool(available[0]), str(available[1]) if len(available) > 1 else ""
    return bool(available), ""


def _te_version(te):
    return str(getattr(te, "__version__", "unknown"))


def _build_case(te, recipe, ffn_width, hidden_width, weight):
    with te.quantized_model_init(enabled=True, recipe=recipe):
        linear = te.ops.Linear(
            ffn_width,
            hidden_width,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        )
        chain = te.ops.Sequential(
            te.ops.SwiGLU(),
            linear,
        )
    with torch.no_grad():
        linear.weight.copy_(weight)
    return chain


def _eager_output(x, weight):
    gate, up = x.chunk(2, dim=-1)
    return F.linear(F.silu(gate) * up, weight)


def _te_output(te, chain, x, recipe):
    with te.autocast(enabled=True, recipe=recipe):
        return chain(x)


def _carrier_is_te_quantized(carrier, quantized_tensor_type):
    return isinstance(carrier, quantized_tensor_type)


def inspect_carrier(carrier, quantized_tensor_type):
    if not _carrier_is_te_quantized(carrier, quantized_tensor_type):
        raise RuntimeError("TE Quantize probe did not return a Transformer Engine quantized tensor")
    shape = tuple(int(item) for item in carrier.shape)
    dtype = str(carrier.dtype)
    backing = getattr(carrier, "_data", None)
    if not isinstance(backing, torch.Tensor):
        backing = getattr(carrier, "data", None)
    details = {
        "carrier_type": "%s.%s" % (type(carrier).__module__, type(carrier).__qualname__),
        "logical_dtype": dtype,
        "logical_shape": shape,
    }
    if isinstance(backing, torch.Tensor):
        details["backing_data_dtype"] = str(backing.dtype)
        details["backing_data_shape"] = tuple(int(item) for item in backing.shape)
        details["backing_data_bytes"] = int(backing.numel() * backing.element_size())
    return details


def probe_carrier(te, x, recipe, quantized_tensor_type):
    probe = te.ops.Sequential(te.ops.SwiGLU(), te.ops.Quantize())
    with te.autocast(enabled=True, recipe=recipe):
        carrier = probe(x)
    return inspect_carrier(carrier, quantized_tensor_type)


def _profile(fn, profile_path):
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, profile_memory=True, record_shapes=True) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(profile_path))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded-width", type=int, default=DEFAULT_EXPANDED_WIDTH)
    parser.add_argument("--ffn-width", type=int, default=DEFAULT_FFN_WIDTH)
    parser.add_argument("--hidden-width", type=int, default=DEFAULT_HIDDEN_WIDTH)
    parser.add_argument("--rows", type=parse_rows, default=DEFAULT_ROWS, metavar="N[,N...]")
    parser.add_argument("--recipes", type=parse_recipes, default=DEFAULT_RECIPES, metavar="NAME[,NAME...]")
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


def serialize_result(result):
    return json.dumps(_json_safe(result), indent=2, sort_keys=True)


def run(args):
    validate_dimensions(args.expanded_width, args.ffn_width, args.hidden_width, args.rows)
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if not args.i_understand_this_uses_gpu:
        raise RuntimeError("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 9):
        raise RuntimeError("this H3 FP8 experiment is SM89-only")
    try:
        transformer_engine, te, te_recipe, quantized_tensor_type = _load_te()
    except ImportError as exc:
        raise RuntimeError("Transformer Engine is required for the FP8 benchmark") from exc
    fp8_available, fp8_reason = _fp8_available(te)
    if not fp8_available:
        suffix = ": " + fp8_reason if fp8_reason else ""
        raise RuntimeError("Transformer Engine reports that FP8 is unavailable%s" % suffix)

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    weight = torch.randn((args.hidden_width, args.ffn_width), generator=generator, dtype=torch.bfloat16, device=device)
    results = []
    carrier_details = []
    if args.profile_dir:
        args.profile_dir.mkdir(parents=True, exist_ok=True)
    for rows in args.rows:
        x = torch.randn((rows, args.expanded_width), generator=generator, dtype=torch.bfloat16, device=device)
        reference = _eager_output(x, weight)
        baseline_timing = benchmark_case(
            lambda: _eager_output(x, weight), args.warmup, args.iterations
        )
        results.append({
            "rows": rows,
            "case": "bf16_eager",
            "recipe": None,
            "timing": baseline_timing,
        })
        for recipe_name in args.recipes:
            recipe = _recipe_for(te_recipe, recipe_name)
            chain = _build_case(te, recipe, args.ffn_width, args.hidden_width, weight)
            output = _te_output(te, chain, x, recipe)
            torch.cuda.synchronize()
            case = {
                "rows": rows,
                "case": "te_fp8",
                "recipe": recipe_name,
                "relative_l2": relative_l2(output, reference),
                "max_absolute_error": max_absolute_error(output, reference),
                "timing": benchmark_case(lambda: _te_output(te, chain, x, recipe), args.warmup, args.iterations),
            }
            if args.profile_dir:
                profile_path = args.profile_dir / ("te_fp8_mlp_rows%d_%s.json" % (rows, recipe_name))
                _profile(lambda: _te_output(te, chain, x, recipe), profile_path)
                case["profile"] = str(profile_path)
            results.append(case)
            carrier_details.append({"rows": rows, "recipe": recipe_name, "details": probe_carrier(te, x, recipe, quantized_tensor_type)})
            del chain, output
        del x, reference
    result = {
        "versions": {"torch": torch.__version__, "transformer_engine": _te_version(transformer_engine)},
        "device": {
            "name": torch.cuda.get_device_name(),
            "capability": list(capability),
            "capability_label": "SM%d%d" % capability,
        },
        "dimensions": {
            "expanded_width": args.expanded_width,
            "ffn_width": args.ffn_width,
            "hidden_width": args.hidden_width,
            "rows": list(args.rows),
        },
        "recipes": list(args.recipes),
        "kernel_source_expectation": "Transformer Engine FP8 SwiGLU + FP8-weight Linear",
        "results": results,
        "carrier_details": carrier_details,
    }
    return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc))
    encoded = serialize_result(result)
    print(encoded)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
