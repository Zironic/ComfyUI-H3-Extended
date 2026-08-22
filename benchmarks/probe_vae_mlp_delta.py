"""Measure MiniMax H3 VAE MLP deltas across overlapping decoder windows.

This is a quality probe. It uses dense PyTorch operations for every analysis
arm and does not implement or benchmark a sparse kernel.
"""

import argparse
from dataclasses import asdict, dataclass
import gc
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F


_HERE = Path(__file__).resolve().parent
_PACK_ROOT = _HERE.parent
_COMFY_ROOT = _PACK_ROOT.parent.parent
sys.path.insert(0, str(_PACK_ROOT))
sys.path.insert(0, str(_COMFY_ROOT))

DEFAULT_LAYERS = (0, 5, 11, 17, 23, 29, 35)
DEFAULT_FRACTIONS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.0)
DEFAULT_GROUP_SIZES = (16, 32, 64, 128, 192)
SELECTORS = (
    "oracle_token",
    "oracle_voxel",
    "proxy_gate",
    "proxy_value",
    "proxy_gate_value",
    "proxy_swiglu",
    "proxy_voxel",
)


@dataclass(frozen=True)
class Window:
    t: int
    y: int
    x: int
    size_t: int = 7
    size_y: int = 16
    size_x: int = 16


@dataclass(frozen=True)
class Pair:
    name: str
    a: Window
    b: Window
    match_global: bool = True


def parse_ints(value):
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def parse_fractions(value):
    values = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not values or any(not 0.0 < fraction <= 1.0 for fraction in values):
        raise ValueError("fractions must be comma-separated values in (0, 1]")
    return values


def parse_selectors(value):
    values = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = tuple(selector for selector in values if selector not in SELECTORS)
    if not values or unknown:
        raise ValueError("selectors must contain only %s" % ", ".join(SELECTORS))
    return values


def split_tile_starts(latent_length, tile_pixels=256, overlap_pixels=64, ratio=16):
    input_pixels = latent_length * ratio
    if tile_pixels >= input_pixels:
        return (0,)

    count = math.ceil(input_pixels / tile_pixels)
    while tile_pixels * count - overlap_pixels * (count - 1) < input_pixels:
        count += 1

    overlaps = [overlap_pixels] * (count - 1)
    remaining = tile_pixels * count - sum(overlaps) - input_pixels
    for index in range(remaining // ratio):
        overlaps[index % len(overlaps)] += ratio

    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + (tile_pixels - overlap) // ratio)
    return tuple(starts)


def temporal_starts(latent_length, window=7, stride=5):
    return tuple(range(0, latent_length - window + 1, stride))


def choose_pair_plan(latent_shape):
    if len(latent_shape) != 5 or latent_shape[0] != 1 or latent_shape[1] != 24:
        raise ValueError("expected a MiniMax H3 video latent with shape [1, 24, T, H, W]")

    _, _, latent_t, latent_h, latent_w = latent_shape
    ts = temporal_starts(latent_t)
    ys = split_tile_starts(latent_h)
    xs = split_tile_starts(latent_w)
    if len(ts) < 4 or len(ys) < 2 or len(xs) < 2:
        raise ValueError("latent is too small for all temporal and spatial overlap controls")

    ti = min(1, len(ts) - 3)
    yi = min((len(ys) - 1) // 2, len(ys) - 2)
    xi = min((len(xs) - 1) // 2, len(xs) - 2)
    base = Window(ts[ti], ys[yi], xs[xi])
    repeat = Window(base.t, base.y, base.x)
    horizontal = Window(base.t, base.y, xs[xi + 1])
    vertical = Window(base.t, ys[yi + 1], base.x)
    temporal = Window(ts[ti + 1], base.y, base.x)
    combined = Window(ts[ti + 1], ys[yi + 1], xs[xi + 1])
    unrelated = Window(ts[ti + 2], ys[-1], xs[-1])
    return (
        Pair("same_window_twice", base, repeat),
        Pair("adjacent_horizontal", base, horizontal),
        Pair("adjacent_vertical", base, vertical),
        Pair("consecutive_temporal", base, temporal),
        Pair("spatial_temporal", base, combined),
        Pair("unrelated_windows", base, unrelated, match_global=False),
    )


def flat_index(window, t, y, x):
    return ((t - window.t) * window.size_y + (y - window.y)) * window.size_x + (x - window.x)


def pair_indices(pair):
    if not pair.match_global:
        count = min(
            pair.a.size_t * pair.a.size_y * pair.a.size_x,
            pair.b.size_t * pair.b.size_y * pair.b.size_x,
        )
        coords = torch.stack((
            torch.arange(count) // (pair.a.size_y * pair.a.size_x),
            torch.arange(count) // pair.a.size_x % pair.a.size_y,
            torch.arange(count) % pair.a.size_x,
        ), dim=1)
        indices = torch.arange(count, dtype=torch.long)
        return indices, indices.clone(), coords, 0

    starts = (
        max(pair.a.t, pair.b.t),
        max(pair.a.y, pair.b.y),
        max(pair.a.x, pair.b.x),
    )
    ends = (
        min(pair.a.t + pair.a.size_t, pair.b.t + pair.b.size_t),
        min(pair.a.y + pair.a.size_y, pair.b.y + pair.b.size_y),
        min(pair.a.x + pair.a.size_x, pair.b.x + pair.b.size_x),
    )
    if any(start >= end for start, end in zip(starts, ends)):
        raise ValueError("pair %s has no shared global tokens" % pair.name)

    coordinates = []
    indices_a = []
    indices_b = []
    for t in range(starts[0], ends[0]):
        for y in range(starts[1], ends[1]):
            for x in range(starts[2], ends[2]):
                coordinates.append((t, y, x))
                indices_a.append(flat_index(pair.a, t, y, x))
                indices_b.append(flat_index(pair.b, t, y, x))
    coords = torch.tensor(coordinates, dtype=torch.long)
    return (
        torch.tensor(indices_a, dtype=torch.long),
        torch.tensor(indices_b, dtype=torch.long),
        coords,
        len(coordinates),
    )


def choose_voxel_shape(target_size, extents):
    best = None
    for size_t in range(1, min(extents[0], target_size) + 1):
        for size_y in range(1, min(extents[1], target_size) + 1):
            for size_x in range(1, min(extents[2], target_size) + 1):
                volume = size_t * size_y * size_x
                distance = abs(volume - target_size)
                logs = [math.log2(size) for size in (size_t, size_y, size_x)]
                compactness = max(logs) - min(logs)
                score = (distance, compactness, -volume)
                if best is None or score < best[0]:
                    best = (score, (size_t, size_y, size_x))
    return best[1]


def voxel_groups(coords, target_size):
    minima = coords.amin(dim=0)
    extents = tuple(int(value) for value in (coords.amax(dim=0) - minima + 1))
    voxel_shape = choose_voxel_shape(target_size, extents)
    local = coords - minima
    bins = torch.stack(tuple(local[:, axis] // voxel_shape[axis] for axis in range(3)), dim=1)
    _, group_ids = torch.unique(bins, dim=0, return_inverse=True)
    counts = torch.bincount(group_ids)
    return group_ids, voxel_shape, counts


def tensor_bytes(value):
    return value.numel() * value.element_size()


def captured_bytes(capture):
    return sum(tensor_bytes(tensor) for layer in capture.values() for tensor in layer.values())


def capture_window(model, normalized_latent, window, layers, device, dtype):
    num_patches = window.size_t * window.size_y * window.size_x
    capture = {}
    handles = []

    for layer_index in layers:
        block = model.decoder.transformer_blocks[layer_index]

        def w1_hook(_module, inputs, output, layer_index=layer_index):
            layer = capture.setdefault(layer_index, {})
            x = inputs[0][:, :num_patches].detach()
            gate, value = output[:, :num_patches].detach().chunk(2, dim=-1)
            activation = F.silu(gate) * value
            layer["x"] = x.squeeze(0).to("cpu").contiguous()
            layer["a"] = activation.squeeze(0).to("cpu").contiguous()

        def w2_hook(_module, _inputs, output, layer_index=layer_index):
            capture[layer_index]["y"] = output[:, :num_patches].detach().squeeze(0).to("cpu").contiguous()

        handles.append(block.ff.w1.register_forward_hook(w1_hook))
        handles.append(block.ff.w2.register_forward_hook(w2_hook))

    latent_window = normalized_latent[
        :, :,
        window.t:window.t + window.size_t,
        window.y:window.y + window.size_y,
        window.x:window.x + window.size_x,
    ].to(device=device, dtype=dtype)
    mean = model.latents_mean.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    std = model.latents_std.view(1, -1, 1, 1, 1).to(device=device, dtype=dtype)
    latent_window = latent_window * std + mean

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        output = model._decode_pixels(latent_window)
        checksum = float(output.float().mean().to("cpu"))
        torch.cuda.synchronize(device)
    finally:
        for handle in handles:
            handle.remove()
    elapsed = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del output, latent_window, mean, std
    return capture, {
        "seconds": elapsed,
        "output_mean": checksum,
        "capture_bytes": captured_bytes(capture),
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }


def metric_values(target, prediction):
    error = target - prediction
    target_centered = target - target.mean()
    sse = float(error.square().sum())
    sst = float(target_centered.square().sum())
    target_energy = float(target.square().sum())
    if sst == 0.0:
        r2 = None
    else:
        r2 = 1.0 - sse / sst
    if target_energy == 0.0:
        relative_error = 0.0 if sse == 0.0 else None
    else:
        relative_error = math.sqrt(sse / target_energy)
    return {
        "r2": r2,
        "relative_delta_error": relative_error,
        "delta_rms": float(target.square().mean().sqrt()),
        "max_abs_error": float(error.abs().max()),
        "sse": sse,
        "sst": sst,
        "target_energy": target_energy,
    }


def token_oracle_mask(delta_a, column_norm, fraction):
    if fraction >= 1.0:
        return delta_a
    keep = max(1, min(delta_a.shape[1], round(delta_a.shape[1] * fraction)))
    score = delta_a.abs() * column_norm
    indices = score.topk(keep, dim=1, sorted=False).indices
    masked = torch.zeros_like(delta_a)
    masked.scatter_(1, indices, delta_a.gather(1, indices))
    return masked


def proxy_scores(x_a, x_b, rows, selector, column_norm, w1):
    mean_a = x_a.index_select(0, rows).mean(dim=0, keepdim=True).to(w1.weight.dtype)
    mean_b = x_b.index_select(0, rows).mean(dim=0, keepdim=True).to(w1.weight.dtype)
    raw_a = F.linear(mean_a, w1.weight, w1.bias)
    raw_b = F.linear(mean_b, w1.weight, w1.bias)
    gate_a, value_a = raw_a.chunk(2, dim=-1)
    gate_b, value_b = raw_b.chunk(2, dim=-1)
    if selector == "proxy_gate":
        score = (gate_b - gate_a).abs()
    elif selector == "proxy_value":
        score = (value_b - value_a).abs()
    elif selector == "proxy_gate_value":
        score = (gate_b - gate_a).abs() + (value_b - value_a).abs()
    else:
        score = (F.silu(gate_b) * value_b - F.silu(gate_a) * value_a).abs()
        if selector == "proxy_voxel":
            score = score.float() * column_norm
    return score.float().squeeze(0)


def group_masks(delta_a, x_a, x_b, group_ids, fractions, selector, column_norm, w1):
    masks = {fraction: torch.zeros_like(delta_a) for fraction in fractions}
    unique_groups = torch.unique(group_ids)
    for group_id in unique_groups:
        rows = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        group_delta = delta_a.index_select(0, rows)
        if selector == "oracle_voxel":
            score = group_delta.square().mean(dim=0).sqrt() * column_norm
        else:
            score = proxy_scores(x_a, x_b, rows, selector, column_norm, w1)
        for fraction in fractions:
            if fraction >= 1.0:
                masked_group = group_delta
            else:
                keep = max(1, min(delta_a.shape[1], round(delta_a.shape[1] * fraction)))
                indices = score.topk(keep, sorted=False).indices
                masked_group = torch.zeros_like(group_delta)
                expanded = indices.unsqueeze(0).expand(group_delta.shape[0], -1)
                masked_group.scatter_(1, expanded, group_delta.gather(1, expanded))
            masks[fraction].index_copy_(0, rows, masked_group)
    return masks


def analyze_layer(pair, layer_index, capture_a, capture_b, indices_a, indices_b, coords,
                  shared_count, fractions, selectors, group_sizes, model, device):
    layer_a = capture_a[layer_index]
    layer_b = capture_b[layer_index]
    a_a = layer_a["a"].index_select(0, indices_a).to(device=device, dtype=torch.float32)
    a_b = layer_b["a"].index_select(0, indices_b).to(device=device, dtype=torch.float32)
    y_a = layer_a["y"].index_select(0, indices_a).to(device=device, dtype=torch.float32)
    y_b = layer_b["y"].index_select(0, indices_b).to(device=device, dtype=torch.float32)
    x_a = layer_a["x"].index_select(0, indices_a).to(device=device, dtype=torch.float32)
    x_b = layer_b["x"].index_select(0, indices_b).to(device=device, dtype=torch.float32)

    delta_a = a_b - a_a
    target = y_b - y_a
    block = model.decoder.transformer_blocks[layer_index]
    w2 = block.ff.w2.weight.float()
    column_norm = torch.linalg.vector_norm(w2, dim=0)
    results = []

    def add_result(selector, group_size, fraction, masked, voxel_shape=None, group_counts=None):
        prediction = F.linear(masked, w2, bias=None)
        metrics = metric_values(target, prediction)
        row = {
            "layer": layer_index,
            "overlap_type": pair.name,
            "shared_token_count": shared_count,
            "compared_token_count": int(target.shape[0]),
            "selector": selector,
            "active_fraction_requested": fraction,
            "active_channels": int(round(delta_a.shape[1] * fraction)) if fraction < 1.0 else delta_a.shape[1],
            "active_fraction_actual": (int(round(delta_a.shape[1] * fraction)) if fraction < 1.0 else delta_a.shape[1]) / delta_a.shape[1],
            "token_group_size_requested": group_size,
            **metrics,
        }
        if voxel_shape is not None:
            row["voxel_shape"] = list(voxel_shape)
            row["token_group_size_min"] = int(group_counts.min())
            row["token_group_size_mean"] = float(group_counts.float().mean())
            row["token_group_size_max"] = int(group_counts.max())
        results.append(row)

    if "oracle_token" in selectors:
        for fraction in fractions:
            add_result("oracle_token", 1, fraction, token_oracle_mask(delta_a, column_norm, fraction))

    for group_size in group_sizes:
        group_ids, voxel_shape, group_counts = voxel_groups(coords, group_size)
        group_ids = group_ids.to(device)
        for selector in SELECTORS[1:]:
            if selector not in selectors:
                continue
            masks = group_masks(delta_a, x_a, x_b, group_ids, fractions, selector, column_norm, block.ff.w1)
            for fraction in fractions:
                add_result(selector, group_size, fraction, masks[fraction], voxel_shape, group_counts)
            del masks
    del a_a, a_b, y_a, y_b, x_a, x_b, delta_a, target, w2, column_norm
    return results


def pooled_gate(rows, selector="oracle_token", target_fraction=0.25):
    candidates = [
        row for row in rows
        if row["selector"] == selector
        and row["overlap_type"] in {
            "adjacent_horizontal", "adjacent_vertical", "consecutive_temporal", "spatial_temporal"
        }
    ]
    if not candidates:
        return None
    fraction = min({row["active_fraction_requested"] for row in candidates}, key=lambda value: abs(value - target_fraction))
    selected = [row for row in candidates if row["active_fraction_requested"] == fraction and row["sst"] > 0.0]
    if not selected:
        return None
    sse = sum(row["sse"] for row in selected)
    sst = sum(row["sst"] for row in selected)
    r2 = 1.0 - sse / sst
    if r2 < 0.4:
        verdict = "stop"
    elif r2 >= 0.9:
        verdict = "strong"
    elif r2 >= 0.65:
        verdict = "chipmunk_like"
    else:
        verdict = "mixed"
    return {
        "selector": selector,
        "active_fraction": fraction,
        "pooled_r2": r2,
        "verdict": verdict,
        "row_count": len(selected),
    }


def load_latent(path, key):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, torch.Tensor):
        latent = payload
    elif isinstance(payload, dict) and key in payload:
        latent = payload[key]
    else:
        raise ValueError("latent artifact does not contain key %r" % key)
    if not isinstance(latent, torch.Tensor):
        raise TypeError("latent value must be a tensor")
    return latent.contiguous()


def load_decoder(checkpoint, device, dtype, attention_backend):
    from safetensors.torch import load_file
    from comfy.cli_args import args as comfy_args

    if attention_backend == "sage":
        comfy_args.use_sage_attention = True
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    state = load_file(str(checkpoint), device="cpu")
    model = MiniMaxH3VideoVAE(operations=comfy.ops.disable_weight_init)
    missing, unexpected = model.load_state_dict(state, strict=False, assign=True)
    if missing or unexpected:
        raise RuntimeError("checkpoint mismatch: missing=%r unexpected=%r" % (missing, unexpected))
    del state
    gc.collect()

    model.eval()
    model.post_quant_conv.to(device=device, dtype=dtype)
    model.decoder.to(device=device, dtype=dtype)
    del model.encoder, model.quant_conv
    gc.collect()
    return model


def unique_windows(pairs):
    windows = []
    for pair in pairs:
        for window in (pair.a, pair.b):
            if window not in windows:
                windows.append(window)
    return tuple(windows)


def serialize_plan(latent, pairs):
    pair_rows = []
    for pair in pairs:
        indices_a, _, coords, shared = pair_indices(pair)
        pair_rows.append({
            "name": pair.name,
            "a": asdict(pair.a),
            "b": asdict(pair.b),
            "match_global": pair.match_global,
            "shared_token_count": shared,
            "compared_token_count": int(indices_a.numel()),
            "coordinate_min": coords.amin(dim=0).tolist(),
            "coordinate_max": coords.amax(dim=0).tolist(),
        })
    return {
        "latent_shape": list(latent.shape),
        "spatial_y_starts": list(split_tile_starts(latent.shape[-2])),
        "spatial_x_starts": list(split_tile_starts(latent.shape[-1])),
        "temporal_starts": list(temporal_starts(latent.shape[-3])),
        "pairs": pair_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--latent", required=True)
    parser.add_argument("--latent-key", default="video")
    parser.add_argument("--layers", default=",".join(str(layer) for layer in DEFAULT_LAYERS))
    parser.add_argument("--fractions", default=",".join(str(value) for value in DEFAULT_FRACTIONS))
    parser.add_argument("--selectors", default="oracle_token")
    parser.add_argument("--group-sizes", default=",".join(str(value) for value in DEFAULT_GROUP_SIZES))
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--attention-backend", choices=("default", "sage"), default="default")
    parser.add_argument("--json", default="")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    latent_path = Path(args.latent).resolve()
    if not checkpoint.is_file() or checkpoint.suffix.lower() != ".safetensors":
        raise ValueError("checkpoint must be an existing safetensors file")
    if not latent_path.is_file():
        raise ValueError("latent must be an existing file")

    layers = parse_ints(args.layers)
    if any(layer < 0 or layer >= 36 for layer in layers):
        raise ValueError("layers must be in [0, 35]")
    fractions = parse_fractions(args.fractions)
    selectors = parse_selectors(args.selectors)
    group_sizes = parse_ints(args.group_sizes)
    if any(size <= 0 for size in group_sizes):
        raise ValueError("group sizes must be positive")

    latent = load_latent(latent_path, args.latent_key)
    pairs = choose_pair_plan(tuple(latent.shape))
    if args.pairs != "all":
        requested = {name.strip() for name in args.pairs.split(",") if name.strip()}
        pairs = tuple(pair for pair in pairs if pair.name in requested)
        missing = requested - {pair.name for pair in pairs}
        if missing:
            raise ValueError("unknown pairs: %s" % ", ".join(sorted(missing)))

    plan = serialize_plan(latent, pairs)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real VAE probe")
    device = torch.device("cuda")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False

    started = time.perf_counter()
    print("loading H3 VAE decoder", file=sys.stderr, flush=True)
    model = load_decoder(checkpoint, device, dtype, args.attention_backend)
    w1_shape = tuple(model.decoder.transformer_blocks[0].ff.w1.weight.shape)
    w2_shape = tuple(model.decoder.transformer_blocks[0].ff.w2.weight.shape)
    if w1_shape != (16384, 2048) or w2_shape != (2048, 8192):
        raise RuntimeError("unexpected H3 VAE MLP shapes: w1=%r w2=%r" % (w1_shape, w2_shape))

    captures = {}
    capture_stats = {}
    rows = []
    base = pairs[0].a
    print("capturing base window %s" % (base,), file=sys.stderr, flush=True)
    captures[base], capture_stats[str(asdict(base))] = capture_window(model, latent, base, layers, device, dtype)

    for pair in pairs:
        if pair.b == base and pair.name != "same_window_twice":
            capture_b = captures[base]
        else:
            print("capturing %s window %s" % (pair.name, pair.b), file=sys.stderr, flush=True)
            capture_b, stats_b = capture_window(model, latent, pair.b, layers, device, dtype)
            capture_stats[pair.name + ":b"] = stats_b
        indices_a, indices_b, coords, shared = pair_indices(pair)
        print("analyzing %s (%d compared tokens)" % (pair.name, indices_a.numel()), file=sys.stderr, flush=True)
        for layer_index in layers:
            rows.extend(analyze_layer(
                pair, layer_index, captures[base], capture_b, indices_a, indices_b, coords,
                shared, fractions, selectors, group_sizes, model, device,
            ))
        if capture_b is not captures[base]:
            del capture_b
        gc.collect()

    payload = {
        "contract": {
            "question": "Can selected 8192-wide SwiGLU channel deltas reproduce H3 VAE MLP output deltas across overlapping decoder windows?",
            "quality_only": True,
            "sparse_kernel": False,
            "coordinate_mapping": "global latent (t,y,x); decoder RoPE remains window-local",
            "w1_shape": list(w1_shape),
            "w2_shape": list(w2_shape),
            "dtype": args.dtype,
            "attention_backend": args.attention_backend,
        },
        "checkpoint": str(checkpoint),
        "latent": str(latent_path),
        "latent_key": args.latent_key,
        "plan": plan,
        "layers": list(layers),
        "fractions": list(fractions),
        "selectors": list(selectors),
        "group_sizes": list(group_sizes),
        "capture_stats": capture_stats,
        "gate": pooled_gate(rows),
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json:
        output = Path(args.json)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
