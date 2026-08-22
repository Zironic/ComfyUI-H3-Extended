"""Kill test for paired-domain CUTLASS FC1 geometry on the exact H3 shape.

This is benchmark-only code. It does not install a model provider or change the
shipping two-slice ConvRot path. The custom extension changes only CUTLASS tile,
warp, and stage geometry while retaining Kitchen's INT8 mainloop and BF16
dequantization boundary. Kitchen's existing fused SwiGLU + ConvRot-256 + INT8
quantizer then measures the exact carrier epilogue separately.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
CUDA_SOURCE = Path(__file__).with_name("cuda") / "convrot_fc1_cutlass_geometry.cu"
DEFAULT_BUILD_DIRECTORY = PACK_ROOT / ".agent" / "tmp" / "convrot_fc1_cutlass_build"
HIDDEN = 5376
FFN = 14336
EXPANDED = FFN * 2
DOMAIN = 256

sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(COMFY_ROOT))

try:
    from benchmarks import benchmark_h3_activation_memory as base
    from benchmarks import bench_convrot_epilogue_kernel_quality as quality
except ImportError:  # Direct execution puts benchmarks/ on sys.path.
    import benchmark_h3_activation_memory as base
    import bench_convrot_epilogue_kernel_quality as quality

from h3_activation_memory.convrot_epilogue import _quantize_convrot_input


CONFIGS = (
    {
        "name": "copied_128x256x64_s3",
        "tile": [128, 256, 64],
        "warp": [64, 64, 64],
        "stages": 3,
    },
    {
        "name": "wide_64x512x64_s2",
        "tile": [64, 512, 64],
        "warp": [64, 64, 64],
        "stages": 2,
    },
    {
        "name": "wide_64x512x64_s3",
        "tile": [64, 512, 64],
        "warp": [64, 64, 64],
        "stages": 3,
    },
    {
        "name": "wide_64x512x64_w32x128_s2",
        "tile": [64, 512, 64],
        "warp": [32, 128, 64],
        "stages": 2,
    },
    {
        "name": "fallback_32x512x64_s2",
        "tile": [32, 512, 64],
        "warp": [32, 64, 64],
        "stages": 2,
        "alignment_ab": 8,
        "compile_rejection": "CUTLASS default A-tile thread map has zero iterations for 256 threads",
    },
    {
        "name": "negative_128x512x64_w64x128_s2",
        "tile": [128, 512, 64],
        "warp": [64, 128, 64],
        "stages": 2,
    },
    {
        "name": "compile_rejected_64x512x32_s3",
        "tile": [64, 512, 32],
        "warp": [64, 64, 32],
        "stages": 3,
        "compile_rejection": "CUTLASS SM80 multistage requires at least two warp-level K iterations",
    },
    {
        "name": "compile_rejected_64x512x32_s4",
        "tile": [64, 512, 32],
        "warp": [64, 64, 32],
        "stages": 4,
        "compile_rejection": "CUTLASS SM80 multistage requires at least two warp-level K iterations",
    },
)
CONFIG_BY_NAME = {item["name"]: index for index, item in enumerate(CONFIGS)}


def parse_rows(value):
    rows = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise ValueError("rows must contain positive integers")
    return rows


def parse_configs(value):
    names = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not names:
        raise ValueError("at least one CUTLASS configuration is required")
    unknown = tuple(name for name in names if name not in CONFIG_BY_NAME)
    if unknown:
        raise ValueError("unknown CUTLASS configurations: %s" % (unknown,))
    if len(set(names)) != len(names):
        raise ValueError("CUTLASS configurations must be unique")
    return names


def config_contract(config):
    tile_m, tile_n, _tile_k = config["tile"]
    warp_m, warp_n, _warp_k = config["warp"]
    warps = (tile_m // warp_m) * (tile_n // warp_n)
    accumulator_outputs = tile_m * tile_n
    return {
        **config,
        "warps_per_cta": warps,
        "threads_per_cta": warps * 32,
        "raw_accumulator_outputs_per_cta": accumulator_outputs,
        "raw_accumulator_outputs_per_thread": accumulator_outputs // (warps * 32),
    }


def carrier_contract(rows):
    rows = int(rows)
    groups = FFN // DOMAIN
    return {
        "raw_fc1_output": {
            "shape": [rows, EXPANDED],
            "dtype": "bfloat16",
            "bytes": rows * EXPANDED * 2,
        },
        "interleaved_view": [rows * groups, DOMAIN * 2],
        "carrier": {
            "shape": [rows, FFN],
            "dtype": "int8",
            "bytes": rows * FFN,
        },
        "scales": {
            "shape": [rows, groups],
            "dtype": "float32",
            "bytes": rows * groups * 4,
        },
        "bf16_round_trip_bytes_removed_by_real_fusion": rows * EXPANDED * 4,
    }


def interleave_gate_up(tensor, domain=DOMAIN):
    if tensor.ndim < 1 or tensor.shape[0] != EXPANDED:
        raise ValueError("FC1 tensor must have 28672 gate/up rows")
    trailing = tuple(tensor.shape[1:])
    gate = tensor[:FFN].reshape(FFN // domain, domain, *trailing)
    up = tensor[FFN:].reshape(FFN // domain, domain, *trailing)
    return torch.stack((gate, up), dim=1).reshape(EXPANDED, *trailing).contiguous()


def interleave_gate_up_output(output, domain=DOMAIN):
    if output.ndim != 2 or output.shape[1] != EXPANDED:
        raise ValueError("FC1 output must have shape [rows, 28672]")
    rows = int(output.shape[0])
    gate = output[:, :FFN].reshape(rows, FFN // domain, domain)
    up = output[:, FFN:].reshape(rows, FFN // domain, domain)
    return torch.stack((gate, up), dim=2).reshape(rows, EXPANDED).contiguous()


def deinterleave_gate_up_output(output, domain=DOMAIN):
    if output.ndim != 2 or output.shape[1] != EXPANDED:
        raise ValueError("interleaved FC1 output must have shape [rows, 28672]")
    rows = int(output.shape[0])
    paired = output.reshape(rows, FFN // domain, 2, domain)
    return torch.cat(
        (paired[:, :, 0].reshape(rows, FFN), paired[:, :, 1].reshape(rows, FFN)),
        dim=1,
    ).contiguous()


def resolve_cutlass_root(value):
    candidates = []
    if value:
        candidates.append(Path(value))
    if os.environ.get("CUTLASS_ROOT"):
        candidates.append(Path(os.environ["CUTLASS_ROOT"]))
    candidates.append(COMFY_ROOT.parent / "comfy-kitchen-chunked-qkv" / "third_party" / "cutlass")
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "include" / "cutlass" / "cutlass.h").is_file():
            return candidate
        nested = candidate / "third_party" / "cutlass"
        if (nested / "include" / "cutlass" / "cutlass.h").is_file():
            return nested.resolve()
    raise FileNotFoundError("CUTLASS headers were not found; pass --cutlass-root")


def load_extension(cutlass_root, build_directory, verbose=False):
    from torch.utils.cpp_extension import load

    if torch.cuda.get_device_capability() != (8, 9):
        raise RuntimeError("the FC1 geometry kill test currently targets SM89 only")
    build_directory = Path(build_directory).resolve()
    build_directory.mkdir(parents=True, exist_ok=True)
    old_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    try:
        return load(
            name="h3_convrot_fc1_cutlass_geometry_sm89",
            sources=[str(CUDA_SOURCE)],
            extra_include_paths=[str(Path(cutlass_root) / "include")],
            extra_cflags=["/O2"],
            extra_cuda_cflags=["-O3", "-lineinfo", "--ptxas-options=-v"],
            build_directory=str(build_directory),
            with_cuda=True,
            verbose=verbose,
        )
    finally:
        if old_arch_list is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = old_arch_list


def extension_attributes(extension, config_index):
    values = extension.kernel_attributes(config_index)
    keys = (
        "registers_per_thread",
        "static_shared_memory_bytes",
        "maximum_dynamic_shared_memory_bytes",
        "local_memory_bytes_per_thread",
        "maximum_threads_per_block",
        "binary_version",
        "ptx_version",
        "cutlass_shared_storage_bytes",
        "active_ctas_per_sm",
    )
    if len(values) != len(keys):
        raise RuntimeError("the CUTLASS extension returned an invalid attribute vector")
    return dict(zip(keys, (int(value) for value in values)))


def run_extension(extension, config_index, x, weight, x_scale, weight_scale, output):
    used = extension.run(
        x,
        weight,
        x_scale.reshape(-1),
        weight_scale.reshape(-1),
        output,
        int(config_index),
        int(torch.cuda.current_stream(x.device).cuda_stream),
    )
    if not used:
        raise RuntimeError("custom CUTLASS configuration declined the exact H3 FC1 shape")
    return output


def block_quantize_256_(expanded_interleaved, qdata, scales):
    from comfy_kitchen.backends import cuda as cuda_backend

    rows = int(expanded_interleaved.shape[0])
    groups = FFN // DOMAIN
    raw = expanded_interleaved.reshape(rows * groups, DOMAIN * 2)
    qview = qdata.reshape(rows * groups, DOMAIN)
    scale_view = scales.reshape(rows * groups, 1)
    cuda_backend._C.quantize_int8_rowwise_convrot64(
        cuda_backend._wrap_for_dlpack(raw),
        cuda_backend._wrap_for_dlpack(qview),
        cuda_backend._wrap_for_dlpack(scale_view),
        DOMAIN,
        False,
        cuda_backend._input_act_code("swiglu"),
        0,
        torch.cuda.current_stream(expanded_interleaved.device).cuda_stream,
    )
    return qdata, scales


def segmented_output_error(reference, actual):
    groups = FFN // DOMAIN
    return quality.tensor_error_segments(
        tuple(reference[:, index * DOMAIN * 2 : (index + 1) * DOMAIN * 2] for index in range(groups)),
        tuple(actual[:, index * DOMAIN * 2 : (index + 1) * DOMAIN * 2] for index in range(groups)),
    )


def segmented_carrier_error(reference, actual):
    if reference.shape != actual.shape:
        raise ValueError("carrier shapes differ")
    mismatches = 0
    absolute_steps = 0
    maximum = 0
    for start in range(0, FFN, DOMAIN):
        difference = actual[:, start : start + DOMAIN].to(torch.int16) - reference[:, start : start + DOMAIN].to(torch.int16)
        mismatches += int(difference.ne(0).sum().item())
        absolute_steps += int(difference.abs().sum().item())
        maximum = max(maximum, int(difference.abs().max().item()))
    elements = reference.numel()
    return {
        "exact": mismatches == 0,
        "mismatch_fraction": mismatches / elements,
        "mean_abs_int8_steps": absolute_steps / elements,
        "max_abs_int8_steps": maximum,
    }


def classify_candidate(timing, kitchen_timing, attributes, correct=True):
    ratio = timing["median_ms"] / kitchen_timing["median_ms"]
    if not correct:
        result = "INVALID-CORRECTNESS"
    elif attributes["local_memory_bytes_per_thread"]:
        result = "SPILL-LIMITED"
    elif ratio > 1.30:
        result = "KERNEL-LIMITED"
    elif ratio > 1.10:
        result = "MARGINAL"
    else:
        result = "MAINLOOP-COMPETITIVE"
    return {"classification": result, "custom_over_kitchen": ratio}


def run_rows(args, extension, loaded, rows, config_names):
    device = torch.device("cuda")
    fc1 = dict(loaded["fc1"])
    fc1["weight"] = fc1["weight"].to(device=device)
    fc1["weight_scale"] = fc1["weight_scale"].to(device=device, dtype=torch.float32)
    generator = torch.Generator(device=device).manual_seed(args.seed + rows)
    activation = torch.randn((rows, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator)
    x, x_scale = _quantize_convrot_input(activation)
    weight = fc1["weight"]
    weight_scale = fc1["weight_scale"].reshape(-1)
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.expand(EXPANDED).contiguous()
    paired_weight = interleave_gate_up(weight)
    paired_weight_scale = interleave_gate_up(weight_scale)

    kitchen_output = torch.empty((rows, EXPANDED), dtype=torch.bfloat16, device=device)
    kitchen_timing = quality.measure_kernel(
        lambda: quality.kitchen_cutlass_gemm_(x, weight, x_scale, weight_scale, kitchen_output),
        args.warmup,
        args.iterations,
        device,
    )
    quality.kitchen_cutlass_gemm_(x, weight, x_scale, weight_scale, kitchen_output)
    kitchen_paired = interleave_gate_up_output(kitchen_output)

    gate_output = torch.empty((rows, FFN), dtype=torch.bfloat16, device=device)
    up_output = torch.empty_like(gate_output)

    def sequential_control():
        quality.kitchen_cutlass_gemm_(x, weight[:FFN], x_scale, weight_scale[:FFN], gate_output)
        quality.kitchen_cutlass_gemm_(x, weight[FFN:], x_scale, weight_scale[FFN:], up_output)

    sequential_timing = quality.measure_kernel(
        sequential_control, args.warmup, args.iterations, device
    )

    reference_qdata = torch.empty((rows, FFN), dtype=torch.int8, device=device)
    reference_scales = torch.empty((rows, FFN // DOMAIN), dtype=torch.float32, device=device)
    epilogue_timing = quality.measure_kernel(
        lambda: block_quantize_256_(kitchen_paired, reference_qdata, reference_scales),
        args.warmup,
        args.iterations,
        device,
    )
    block_quantize_256_(kitchen_paired, reference_qdata, reference_scales)

    output = torch.empty_like(kitchen_output)
    candidate_qdata = torch.empty_like(reference_qdata)
    candidate_scales = torch.empty_like(reference_scales)
    candidates = []
    for name in config_names:
        config_index = CONFIG_BY_NAME[name]
        config = CONFIGS[config_index]
        contract = config_contract(config)
        if "compile_rejection" in config:
            candidates.append(
                {
                    **contract,
                    "supported": False,
                    "rejected_at": "compile-time feasibility probe",
                    "error": config["compile_rejection"],
                }
            )
            continue
        try:
            attributes = extension_attributes(extension, config_index)
            fn = lambda: run_extension(
                extension,
                config_index,
                x,
                paired_weight,
                x_scale,
                paired_weight_scale,
                output,
            )
            timing = quality.measure_kernel(fn, args.warmup, args.iterations, device)
            fn()
            block_quantize_256_(output, candidate_qdata, candidate_scales)
            torch.cuda.synchronize(device)
            raw_error = segmented_output_error(kitchen_paired, output)
            carrier_error = segmented_carrier_error(reference_qdata, candidate_qdata)
            scale_error = quality.tensor_error(reference_scales, candidate_scales)
            correct = raw_error["exact"] and carrier_error["exact"] and scale_error["exact"]
            assessment = classify_candidate(timing, kitchen_timing, attributes, correct)
            candidates.append(
                {
                    **contract,
                    "supported": True,
                    "timing": {
                        **timing,
                        "effective_tops": quality.effective_tops(rows, EXPANDED, HIDDEN, timing["median_ms"]),
                    },
                    "resources": attributes,
                    "correct": correct,
                    **assessment,
                    "raw_bf16_error_vs_kitchen": raw_error,
                    "carrier_error_vs_kitchen": carrier_error,
                    "scale_error_vs_kitchen": scale_error,
                    "unfused_mainloop_plus_epilogue_ms": timing["median_ms"] + epilogue_timing["median_ms"],
                }
            )
        except RuntimeError as exc:
            torch.cuda.synchronize(device)
            candidates.append({**contract, "supported": False, "error": str(exc)})

    operations = 2 * rows * EXPANDED * HIDDEN
    return {
        "rows": rows,
        "shape": {"m": rows, "n": EXPANDED, "k": HIDDEN},
        "kitchen_cutlass": {
            **kitchen_timing,
            "effective_tops": operations / (kitchen_timing["median_ms"] * 1e9),
        },
        "kitchen_sequential_gate_up_two_launch_control": {
            **sequential_timing,
            "effective_tops": operations / (sequential_timing["median_ms"] * 1e9),
            "over_full_width_kitchen": sequential_timing["median_ms"] / kitchen_timing["median_ms"],
            "same_cta": False,
        },
        "standalone_exact_256_domain_epilogue": {
            **epilogue_timing,
            "launches": 1,
            "input_shape": [rows * (FFN // DOMAIN), DOMAIN * 2],
            "output_shape": [rows * (FFN // DOMAIN), DOMAIN],
            "includes": ["bf16_read", "swiglu", "convrot256", "absmax", "int8_store", "fp32_scale_store"],
        },
        "carrier_contract": carrier_contract(rows),
        "candidates": candidates,
    }


def run(args):
    if not args.i_understand_this_uses_gpu:
        raise RuntimeError("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    cutlass_root = resolve_cutlass_root(args.cutlass_root)
    checkpoint = base.resolve_checkpoint(args.checkpoint)
    raw = base.load_block_mlp_tensors(checkpoint, args.block_index)
    loaded = base.load_convrot_mlp(raw)
    if (loaded["hidden_width"], loaded["ffn_width"], loaded["expanded_width"]) != (HIDDEN, FFN, EXPANDED):
        raise RuntimeError("checkpoint is not the exact H3 MLP shape")
    extension = load_extension(cutlass_root, args.build_directory, args.verbose_build)
    config_names = parse_configs(args.configs)
    results = [run_rows(args, extension, loaded, rows, config_names) for rows in parse_rows(args.rows)]
    return {
        "benchmark": "h3_convrot_fc1_cutlass_geometry",
        "scope": "benchmark-only; shipping implementation unchanged",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "block_index": args.block_index,
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cutlass_root": str(cutlass_root),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "evidence_boundary": (
            "CUTLASS mainloop geometry and the exact carrier epilogue are measured separately; "
            "their sum is not labeled as a fused-kernel measurement"
        ),
        "results": results,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--rows", default="4096")
    parser.add_argument("--configs", default=",".join(CONFIG_BY_NAME))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cutlass-root")
    parser.add_argument("--build-directory", default=str(DEFAULT_BUILD_DIRECTORY))
    parser.add_argument("--output")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    result = run(args)
    rendered = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    if args.json or not args.output:
        print(rendered)


if __name__ == "__main__":
    main()
