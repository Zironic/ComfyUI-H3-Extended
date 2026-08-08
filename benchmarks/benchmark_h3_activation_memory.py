"""Synthetic and opt-in real-checkpoint H3 MLP memory/runtime sweeps.

The synthetic mode intentionally has no ComfyUI or safetensors dependency.  A
checkpoint is only opened when ``--checkpoint`` is supplied, and only one
block's two MLP projections are read in that mode.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

CHECKSUM_SAMPLES = 4096
DEFAULT_CHUNKS = (2048, 4096, 8192, 16384)
DEFAULT_SWIGLU_MODES = ("bf16", "native")
ACTUAL_SWIGLU_MODES = DEFAULT_SWIGLU_MODES + ("tiled_convrot",)
DEFAULT_HELD_MODES = ("off", "on")
DEFAULT_FEATURE_TILE_WIDTH = 3584


def dtype_from_name(name):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def parse_chunks(value):
    values = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("chunks must contain positive integers")
    return values


def parse_modes(value, allowed, option):
    values = tuple(item.strip().lower() for item in str(value).split(",") if item.strip())
    if not values or any(item not in allowed for item in values):
        raise ValueError("%s must contain only %s" % (option, ", ".join(allowed)))
    return values


def parse_feature_tile_width(value):
    try:
        width = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("feature tile width must be a positive multiple of 256") from exc
    if width <= 0 or width % 256:
        raise argparse.ArgumentTypeError("feature tile width must be a positive multiple of 256")
    return width


def feature_ranges(ffn_width, tile_width):
    ffn_width = int(ffn_width)
    tile_width = int(tile_width)
    if ffn_width <= 0 or ffn_width % 256:
        raise ValueError("FFN width must be a positive multiple of 256")
    if tile_width <= 0 or tile_width % 256:
        raise ValueError("feature tile width must be a positive multiple of 256")
    return tuple(
        (start, min(start + tile_width, ffn_width))
        for start in range(0, ffn_width, tile_width)
    )


def iter_cases(chunks=DEFAULT_CHUNKS, swiglu_modes=DEFAULT_SWIGLU_MODES, held_modes=DEFAULT_HELD_MODES):
    """Return the requested chunk x SwiGLU x held matrix in stable order."""
    cases = []
    for chunk_rows in chunks:
        for swiglu_mode in swiglu_modes:
            if swiglu_mode == "tiled_convrot":
                cases.append({
                    "chunk_rows": chunk_rows,
                    "swiglu_mode": swiglu_mode,
                    "held_mode": "prepacked",
                })
            else:
                cases.extend(
                    {
                        "chunk_rows": chunk_rows,
                        "swiglu_mode": swiglu_mode,
                        "held_mode": held_mode,
                    }
                    for held_mode in held_modes
                )
    return tuple(cases)


def resolve_checkpoint(value):
    """Validate an absolute safetensors path or resolve a diffusion-model name."""
    if not value:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.suffix.lower() != ".safetensors":
            raise ValueError("absolute checkpoint must end in .safetensors")
        if not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return str(candidate.resolve())

    # folder_paths performs the registered-folder containment check.  Do not
    # turn a relative value into a local path before asking it to resolve.
    import folder_paths

    resolved = Path(folder_paths.get_full_path_or_raise("diffusion_models", value))
    if resolved.suffix.lower() != ".safetensors":
        raise ValueError("resolved checkpoint must end in .safetensors")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return str(resolved.resolve())


def _block_prefixes(block_index):
    stem = "blocks.%d.mlp." % block_index
    return ("model.diffusion_model." + stem, "diffusion_model." + stem, stem)


def load_block_mlp_tensors(checkpoint, block_index=0, safe_open_fn=None):
    """Read just fc1/fc2 tensors and metadata for one MLP block."""
    if safe_open_fn is None:
        from safetensors import safe_open as safe_open_fn

    prefixes = _block_prefixes(block_index)
    with safe_open_fn(checkpoint, framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        selected_prefix = next(
            (prefix for prefix in prefixes if prefix + "fc1.weight" in available and prefix + "fc2.weight" in available),
            None,
        )
        if selected_prefix is None:
            raise KeyError("checkpoint has no fc1/fc2 weights for blocks.%d.mlp" % block_index)

        state = {}
        full_prefix = selected_prefix
        for full_key in sorted(available):
            if not full_key.startswith(full_prefix):
                continue
            local_key = full_key[len(full_prefix) :]
            if not (local_key.startswith("fc1.") or local_key.startswith("fc2.")):
                continue
            state[local_key] = handle.get_tensor(full_key)

    return {
        "state_dict": state,
        "prefix": selected_prefix,
        "block_index": int(block_index),
        "checkpoint": str(checkpoint),
    }


def _decode_quant_config(value):
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return json.loads(value)
    try:
        raw = value.detach().cpu().numpy().tobytes()
    except AttributeError as exc:
        raise ValueError("comfy_quant metadata must be JSON bytes") from exc
    return json.loads(raw)


def _convrot_info(state_dict, layer):
    prefix = "%s." % layer
    required = tuple(prefix + name for name in ("weight", "weight_scale", "comfy_quant"))
    missing = [name for name in required if name not in state_dict]
    if missing:
        raise ValueError("%s ConvRot metadata is missing: %s" % (layer, ", ".join(missing)))
    weight = state_dict[required[0]]
    scale = state_dict[required[1]]
    quant = _decode_quant_config(state_dict[required[2]])
    if quant.get("format") != "int8_tensorwise" or quant.get("convrot") is not True:
        raise ValueError("%s must be non-transposed TensorWise-INT8 ConvRot" % layer)
    if quant.get("transposed", False):
        raise ValueError("transposed %s ConvRot weights are unsupported" % layer)
    group_size = int(quant.get("convrot_groupsize", 256))
    if group_size <= 0:
        raise ValueError("%s ConvRot group size must be positive" % layer)
    if weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError("ConvRot %s weight must be a rank-2 INT8 tensor" % layer)
    if scale.numel() not in (1, int(weight.shape[0])):
        raise ValueError("ConvRot %s scale must be scalar or per-output-channel" % layer)
    return {
        "layer": layer,
        "weight": weight,
        "weight_scale": scale,
        "group_size": group_size,
        "quant": quant,
    }


def load_convrot_mlp(loaded, hidden=None, ffn=None):
    """Validate and return both raw, non-transposed TensorWise ConvRot layers."""
    state = loaded["state_dict"]
    fc1 = _convrot_info(state, "fc1")
    fc2 = _convrot_info(state, "fc2")
    if tuple(fc1["weight"].shape)[1] != tuple(fc2["weight"].shape)[0]:
        raise ValueError("fc1 input and fc2 output dimensions do not match")
    if tuple(fc1["weight"].shape)[0] != 2 * tuple(fc2["weight"].shape)[1]:
        raise ValueError("fc1/fc2 dimensions are not a SwiGLU pair")
    hidden_width = int(fc1["weight"].shape[1])
    ffn_width = int(fc2["weight"].shape[1])
    if hidden_width % fc1["group_size"]:
        raise ValueError("fc1 ConvRot input width is not group-size aligned")
    if ffn_width % fc2["group_size"]:
        raise ValueError("fc2 ConvRot input width is not group-size aligned")
    if hidden is not None and int(hidden) != hidden_width:
        raise ValueError("explicit hidden dimension conflicts with checkpoint")
    if ffn is not None and int(ffn) != ffn_width:
        raise ValueError("explicit ffn dimension conflicts with checkpoint")
    return {
        "fc1": fc1,
        "fc2": fc2,
        "hidden_width": hidden_width,
        "ffn_width": ffn_width,
        "expanded_width": 2 * ffn_width,
    }


def prepare_convrot_tiles(fc1, fc2, tile_width):
    """Pack feature-major INT8 tiles before the measured region."""
    tile_width = int(tile_width)
    if tile_width <= 0 or tile_width % 256:
        raise ValueError("feature tile width must be a positive multiple of 256")
    ffn_width = int(fc2["weight"].shape[1])
    if int(fc1["weight"].shape[0]) != 2 * ffn_width:
        raise ValueError("fc1/fc2 dimensions are not a SwiGLU pair")
    if tile_width % int(fc2["group_size"]):
        raise ValueError("feature tile width must preserve complete fc2 ConvRot groups")
    if ffn_width % int(fc2["group_size"]):
        raise ValueError("fc2 ConvRot input width is not group-size aligned")
    tiles = []
    fc1_scale = fc1["weight_scale"]
    for start, stop in feature_ranges(ffn_width, tile_width):
        width = stop - start
        if width % int(fc2["group_size"]):
            raise ValueError("every feature tile must preserve complete fc2 ConvRot groups")
        if fc1_scale.numel() == 1:
            packed_scale = fc1_scale.contiguous()
        else:
            packed_scale = torch.cat((fc1_scale[start:stop], fc1_scale[ffn_width + start : ffn_width + stop]), dim=0).contiguous()
        tiles.append({
            "start": start,
            "stop": stop,
            "fc1_weight": torch.cat((
                fc1["weight"][start:stop],
                fc1["weight"][ffn_width + start : ffn_width + stop],
            ), dim=0).contiguous(),
            "fc1_scale": packed_scale,
            "fc2_weight": fc2["weight"][:, start:stop].contiguous(),
        })
    return tuple(tiles)


def prepared_tile_bytes(tiles):
    names = ("fc1_weight", "fc1_scale", "fc2_weight")
    return sum(
        int(tile[name].numel() * tile[name].element_size())
        for tile in tiles
        for name in names
    )


def derive_mlp_dimensions(state_dict, hidden=None, ffn=None):
    fc1 = state_dict["fc1.weight"]
    fc2 = state_dict["fc2.weight"]
    if fc1.ndim != 2 or fc2.ndim != 2:
        raise ValueError("fc1 and fc2 weights must be matrices")
    actual_hidden = int(fc2.shape[0])
    if int(fc1.shape[0]) % 2:
        raise ValueError("fc1 output dimension must be divisible by two")
    actual_ffn = int(fc1.shape[0]) // 2
    # Packed ConvRot/NVFP4 layouts expose a storage width smaller than the
    # logical FFN width; their comfy_quant metadata lets Linear reconstruct it.
    if int(fc2.shape[1]) != actual_ffn and "fc2.comfy_quant" not in state_dict:
        raise ValueError("fc1/fc2 FFN dimensions conflict")
    if hidden is not None and int(hidden) != actual_hidden:
        raise ValueError("explicit hidden dimension conflicts with checkpoint")
    if ffn is not None and int(ffn) != actual_ffn:
        raise ValueError("explicit ffn dimension conflicts with checkpoint")
    return actual_hidden, actual_ffn


def build_checkpoint_mlp(loaded, dtype, hidden=None, ffn=None, ops_factory=None):
    """Construct real mixed-precision Linear modules while keeping weights offloaded."""
    state = dict(loaded["state_dict"])
    hidden, ffn = derive_mlp_dimensions(state, hidden=hidden, ffn=ffn)
    if ops_factory is None:
        import comfy.ops

        ops_factory = comfy.ops.mixed_precision_ops
    ops = ops_factory(compute_dtype=dtype)
    has_fc1_bias = "fc1.bias" in state
    has_fc2_bias = "fc2.bias" in state
    fc1 = ops.Linear(hidden, ffn * 2, bias=has_fc1_bias)
    fc2 = ops.Linear(ffn, hidden, bias=has_fc2_bias)
    fc1_state = {key[4:]: value for key, value in state.items() if key.startswith("fc1.")}
    fc2_state = {key[4:]: value for key, value in state.items() if key.startswith("fc2.")}
    fc1.load_state_dict(fc1_state, strict=True)
    fc2.load_state_dict(fc2_state, strict=True)
    return SimpleNamespace(fc1=fc1, fc2=fc2), hidden, ffn


def sampled_checksum(output):
    flat = output.reshape(-1)
    stride = max(1, (flat.numel() + CHECKSUM_SAMPLES - 1) // CHECKSUM_SAMPLES)
    return float(flat[::stride].detach().float().mean())


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
        output = fn()
        checksum = sampled_checksum(output)
        del output
    synchronize(device)
    elapsed = (time.perf_counter() - started) / iterations
    peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    return {"seconds": elapsed, "peak_bytes": peak, "peak_reserved_bytes": reserved, "checksum": checksum}


def _native_weight_supported(weight):
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        return False
    from comfy.quant_ops import QuantizedTensor

    return isinstance(weight, QuantizedTensor) and not getattr(
        getattr(weight, "_params", None), "transposed", False
    )


def _convrot_output(ck, x, weight, weight_scale, group_size, input_act=None):
    kwargs = {
        "bias": None,
        "out_dtype": torch.bfloat16,
        "convrot": True,
        "convrot_groupsize": int(group_size),
    }
    if input_act is not None:
        kwargs["input_act"] = input_act
    return ck.int8_linear(x, weight, weight_scale, **kwargs)


def _load_comfy_kitchen():
    try:
        import comfy_kitchen as ck
    except ImportError as exc:
        raise RuntimeError("tiled ConvRot mode requires comfy-kitchen") from exc
    return ck


def run_tiled_convrot_case(ck, activation, chunk_rows, fc1, fc2, tiles, device, convrot_fn=None):
    """Execute all activation chunks without materializing a full fc1 expansion."""
    output = torch.empty((activation.shape[0], fc2["weight"].shape[0]), device=activation.device, dtype=torch.bfloat16)
    fc1_ms = 0.0
    fc2_ms = 0.0
    fc1_events = []
    fc2_events = []
    convrot = _convrot_output if convrot_fn is None else convrot_fn
    for start in range(0, activation.shape[0], chunk_rows):
        stop = min(activation.shape[0], start + chunk_rows)
        chunk = activation[start:stop]
        for tile in tiles:
            if device.type == "cuda":
                fc1_start = torch.cuda.Event(enable_timing=True)
                fc1_end = torch.cuda.Event(enable_timing=True)
                fc1_start.record()
                expanded = convrot(ck, chunk, tile["fc1_weight"], tile["fc1_scale"], fc1["group_size"], input_act=None)
                fc1_end.record()
                fc1_events.append((fc1_start, fc1_end))
                fc2_start = torch.cuda.Event(enable_timing=True)
                fc2_end = torch.cuda.Event(enable_timing=True)
                fc2_start.record()
                partial = convrot(ck, expanded, tile["fc2_weight"], fc2["weight_scale"], fc2["group_size"], input_act="swiglu")
                if tile is tiles[0]:
                    output[start:stop] = partial
                else:
                    output[start:stop].add_(partial)
                fc2_end.record()
                fc2_events.append((fc2_start, fc2_end))
                del expanded, partial
            else:
                started = time.perf_counter()
                expanded = convrot(ck, chunk, tile["fc1_weight"], tile["fc1_scale"], fc1["group_size"], input_act=None)
                fc1_ms += (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                partial = convrot(ck, expanded, tile["fc2_weight"], fc2["weight_scale"], fc2["group_size"], input_act="swiglu")
                if tile is tiles[0]:
                    output[start:stop] = partial
                else:
                    output[start:stop].add_(partial)
                fc2_ms += (time.perf_counter() - started) * 1000
                del expanded, partial
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        fc1_ms = sum(start.elapsed_time(end) for start, end in fc1_events)
        fc2_ms = sum(start.elapsed_time(end) for start, end in fc2_events)
    return output, "tiled_convrot", fc1_ms, fc2_ms


def run_actual_case(mlp, activation, chunk_rows, swiglu_mode, held_mode, device):
    """Run one actual-weight case; returns output, path, and stage timings."""
    from h3_activation_memory.linear import HeldMLP, module_fc1, module_swiglu_fc2

    native = swiglu_mode == "native"
    if native and not _native_weight_supported(mlp.fc2.weight):
        raise RuntimeError("native SwiGLU requested but fc2 has no TensorWiseINT8 native path")
    output = torch.empty_like(activation)
    paths = []
    fc1_ms = 0.0
    fc2_ms = 0.0
    fc1_events = []
    fc2_events = []
    held = held_mode == "on"
    if held:
        with HeldMLP(mlp, activation[:1]) as session:
            for start in range(0, activation.shape[0], chunk_rows):
                stop = min(activation.shape[0], start + chunk_rows)
                started = time.perf_counter()
                fc1_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                fc1_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if fc1_start is not None:
                    fc1_start.record()
                expanded = session.fc1(activation[start:stop])
                if fc1_end is not None:
                    fc1_end.record()
                    fc1_events.append((fc1_start, fc1_end))
                else:
                    fc1_ms += (time.perf_counter() - started) * 1000
                started = time.perf_counter()
                fc2_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                fc2_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if fc2_start is not None:
                    fc2_start.record()
                out, path = session.fc2_swiglu(expanded, native=native)
                if fc2_end is not None:
                    fc2_end.record()
                    fc2_events.append((fc2_start, fc2_end))
                else:
                    fc2_ms += (time.perf_counter() - started) * 1000
                if native and "native" not in path:
                    raise RuntimeError("native SwiGLU silently fell back to %s" % path)
                output[start:stop] = out
                paths.append(path)
    else:
        for start in range(0, activation.shape[0], chunk_rows):
            stop = min(activation.shape[0], start + chunk_rows)
            started = time.perf_counter()
            fc1_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            fc1_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if fc1_start is not None:
                fc1_start.record()
            expanded = module_fc1(mlp, activation[start:stop])
            if fc1_end is not None:
                fc1_end.record()
                fc1_events.append((fc1_start, fc1_end))
            else:
                fc1_ms += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            fc2_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            fc2_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if fc2_start is not None:
                fc2_start.record()
            out, path = module_swiglu_fc2(mlp, expanded, native=native)
            if fc2_end is not None:
                fc2_end.record()
                fc2_events.append((fc2_start, fc2_end))
            else:
                fc2_ms += (time.perf_counter() - started) * 1000
            if native and "native" not in path:
                raise RuntimeError("native SwiGLU silently fell back to %s" % path)
            output[start:stop] = out
            paths.append(path)
    if not paths:
        raise RuntimeError("actual case produced no chunks")
    if any(path != paths[0] for path in paths):
        raise RuntimeError("MLP execution path changed between chunks")
    if device.type == "cuda":
        # One synchronization after the complete slab loop; no chunk-level
        # synchronization is introduced by the event timing.
        torch.cuda.synchronize(device)
        fc1_ms = sum(start.elapsed_time(end) for start, end in fc1_events)
        fc2_ms = sum(start.elapsed_time(end) for start, end in fc2_events)
    return output, paths[0], fc1_ms, fc2_ms


def run_actual(loaded, args, device, dtype):
    swiglu_modes = parse_modes(args.swiglu_modes, ACTUAL_SWIGLU_MODES, "--swiglu-modes")
    if "tiled_convrot" in swiglu_modes and dtype != torch.bfloat16:
        raise ValueError("tiled_convrot requires --dtype bf16")
    mlp, hidden, ffn = build_checkpoint_mlp(
        loaded, dtype, hidden=args.hidden, ffn=args.ffn
    )
    torch.manual_seed(0)
    activation = torch.randn(args.seq, hidden, device=device, dtype=dtype)
    tiled_ck = None
    tiled_fc1 = None
    tiled_fc2 = None
    tiled_tiles = None
    if "tiled_convrot" in swiglu_modes:
        tiled = load_convrot_mlp(loaded, hidden=hidden, ffn=ffn)
        tiled_ck = _load_comfy_kitchen()
        tiled_fc1 = dict(tiled["fc1"])
        tiled_fc2 = dict(tiled["fc2"])
        tiled_fc1["weight"] = tiled_fc1["weight"].to(device=device)
        tiled_fc1["weight_scale"] = tiled_fc1["weight_scale"].to(device=device, dtype=torch.float32)
        tiled_fc2["weight"] = tiled_fc2["weight"].to(device=device)
        tiled_fc2["weight_scale"] = tiled_fc2["weight_scale"].to(device=device, dtype=torch.float32)
        tiled_tiles = prepare_convrot_tiles(
            tiled_fc1, tiled_fc2, getattr(args, "feature_tile_width", DEFAULT_FEATURE_TILE_WIDTH)
        )
    rows = []
    for case in iter_cases(
        parse_chunks(args.chunks),
        swiglu_modes,
        parse_modes(args.held_modes, DEFAULT_HELD_MODES, "--held-modes"),
    ):
        samples = []
        stage_fc1 = []
        stage_fc2 = []
        paths = []
        measured_output = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_alloc = torch.cuda.memory_allocated(device)
        else:
            baseline_alloc = 0
        for _ in range(args.warmup):
            if case["swiglu_mode"] == "tiled_convrot":
                out, path, _, _ = run_tiled_convrot_case(
                    tiled_ck, activation, case["chunk_rows"], tiled_fc1, tiled_fc2, tiled_tiles, device
                )
            else:
                out, path, _, _ = run_actual_case(mlp, activation, case["chunk_rows"], case["swiglu_mode"], case["held_mode"], device)
            del out
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline_alloc = torch.cuda.memory_allocated(device)
        for _ in range(args.iterations):
            started = time.perf_counter()
            if case["swiglu_mode"] == "tiled_convrot":
                out, path, fc1_ms, fc2_ms = run_tiled_convrot_case(
                    tiled_ck, activation, case["chunk_rows"], tiled_fc1, tiled_fc2, tiled_tiles, device
                )
            else:
                out, path, fc1_ms, fc2_ms = run_actual_case(mlp, activation, case["chunk_rows"], case["swiglu_mode"], case["held_mode"], device)
            samples.append((time.perf_counter() - started) * 1000)
            stage_fc1.append(fc1_ms)
            stage_fc2.append(fc2_ms)
            paths.append(path)
            checksum = sampled_checksum(out)
            if case["swiglu_mode"] == "tiled_convrot" and _ == args.iterations - 1:
                measured_output = out
            else:
                del out
        synchronize(device)
        if any(path != paths[0] for path in paths):
            raise RuntimeError("MLP execution path changed between iterations")
        peak_allocated = torch.cuda.max_memory_allocated(device) - baseline_alloc if device.type == "cuda" else 0
        peak_reserved = torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
        error_metrics = {}
        if case["swiglu_mode"] == "tiled_convrot":
            # Capture the measured peak first; reference/error work is outside
            # timing and must not contaminate the transient allocation report.
            if measured_output is None:
                raise RuntimeError("tiled benchmark did not retain a measured output")
            delta_sq = 0.0
            reference_sq = 0.0
            max_abs = 0.0
            for start in range(0, activation.shape[0], case["chunk_rows"]):
                stop = min(activation.shape[0], start + case["chunk_rows"])
                reference_expanded = _convrot_output(
                    tiled_ck,
                    activation[start:stop],
                    tiled_fc1["weight"],
                    tiled_fc1["weight_scale"],
                    tiled_fc1["group_size"],
                    input_act=None,
                )
                reference = _convrot_output(
                    tiled_ck,
                    reference_expanded,
                    tiled_fc2["weight"],
                    tiled_fc2["weight_scale"],
                    tiled_fc2["group_size"],
                    input_act="swiglu",
                )
                reference_float = reference.float()
                difference = measured_output[start:stop].float().sub_(reference_float)
                delta_sq += float(torch.sum(difference * difference).item())
                reference_sq += float(torch.sum(reference_float * reference_float).item())
                max_abs = max(max_abs, float(difference.abs().max().item()))
                del reference_expanded, reference, difference, reference_float
            error_metrics = {
                "relative_l2_vs_current_convrot": (delta_sq / max(reference_sq, 1e-8)) ** 0.5,
                "max_abs_vs_current_convrot": max_abs,
            }
            del measured_output
        rows.append({
            **case,
            "total_mlp_ms_mean": statistics.mean(samples),
            "total_mlp_ms_median": statistics.median(samples),
            "mlp_fc1_ms": statistics.mean(stage_fc1),
            "mlp_swiglu_fc2_ms": statistics.mean(stage_fc2),
            "peak_allocated_bytes": peak_allocated,
            "transient_peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "checksum": checksum,
            "chunk_count": (args.seq + case["chunk_rows"] - 1) // case["chunk_rows"],
            "execution_path": paths[0],
            **({
                "feature_tile_width": int(getattr(args, "feature_tile_width", DEFAULT_FEATURE_TILE_WIDTH)),
                "feature_tile_count": len(tiled_tiles),
                "prepared_tile_bytes": prepared_tile_bytes(tiled_tiles),
                **error_metrics,
            } if case["swiglu_mode"] == "tiled_convrot" else {}),
        })
    return {
        "mode": "actual",
        "checkpoint": loaded["checkpoint"],
        "block_index": loaded["block_index"],
        "weight_prefix": loaded["prefix"],
        "shape": {"seq": args.seq, "hidden": hidden, "ffn": ffn, "dtype": str(dtype), "device": str(device)},
        "weight_layout": {"fc1": str(getattr(mlp.fc1, "layout_type", None)), "fc2": str(getattr(mlp.fc2, "layout_type", None))},
        "results": rows,
    }


def run_synthetic(args, device, dtype):
    def swiglu_eager(x):
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate).mul_(up)

    hidden = args.hidden if args.hidden is not None else 512
    ffn = args.ffn if args.ffn is not None else 1024
    torch.manual_seed(0)
    x = torch.randn(args.seq, hidden, device=device, dtype=dtype)
    fc1 = torch.nn.Linear(hidden, ffn * 2, bias=False, device=device, dtype=dtype)
    fc2 = torch.nn.Linear(ffn, hidden, bias=False, device=device, dtype=dtype)
    for parameter in (*fc1.parameters(), *fc2.parameters()):
        parameter.requires_grad_(False)

    def stock():
        return fc2(swiglu_eager(fc1(x)))

    baseline = measure(stock, device, args.warmup, args.iterations)
    rows = [{"mode": "full", "chunk_rows": args.seq, **baseline}]
    for chunk_rows in parse_chunks(args.chunks):
        def chunked(chunk_rows=chunk_rows):
            output = torch.empty_like(x)
            for start in range(0, args.seq, chunk_rows):
                stop = min(args.seq, start + chunk_rows)
                output[start:stop] = fc2(swiglu_eager(fc1(x[start:stop])))
            return output

        result = measure(chunked, device, args.warmup, args.iterations)
        got = chunked()
        want = stock()
        result["max_abs_vs_full"] = float((got - want).abs().max().detach())
        del got, want
        rows.append({"mode": "chunked", "chunk_rows": chunk_rows, **result})
    return {"mode": "synthetic", "shape": {"seq": args.seq, "hidden": hidden, "ffn": ffn, "dtype": str(dtype), "device": str(device)}, "results": rows}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--chunks", default=",".join(str(item) for item in DEFAULT_CHUNKS))
    parser.add_argument("--swiglu-modes", default=",".join(DEFAULT_SWIGLU_MODES))
    parser.add_argument("--held-modes", default=",".join(DEFAULT_HELD_MODES))
    parser.add_argument("--feature-tile-width", type=parse_feature_tile_width, default=DEFAULT_FEATURE_TILE_WIDTH)
    parser.add_argument("--checkpoint")
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if args.checkpoint:
        if device.type != "cuda":
            raise ValueError("actual checkpoint mode requires --device cuda")
        if dtype == torch.float32:
            raise ValueError("actual checkpoint mode requires --dtype bf16 or --dtype fp16")
        checkpoint = resolve_checkpoint(args.checkpoint)
        loaded = load_block_mlp_tensors(checkpoint, args.block_index)
        payload = run_actual(loaded, args, device, dtype)
    else:
        payload = run_synthetic(args, device, dtype)
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_path:
        destination = Path(args.json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
