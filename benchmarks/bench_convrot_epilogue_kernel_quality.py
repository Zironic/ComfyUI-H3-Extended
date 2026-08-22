"""Measure H3 ConvRot MLP fusion separately from custom GEMM quality.

This benchmark deliberately does not install the epilogue prototype as a
shipping provider.  It compares the established two-slice Kitchen path with
the existing August-12 Triton epilogue prototype, then compares a minimal
Triton INT8 GEMM with Kitchen CUTLASS using identical quantized carriers. It
also compares two sliced Kitchen GEMMs with one full-width Kitchen GEMM and
locates numerical differences at the FC1, activation-quantization, FC2, and
residual-accumulation boundaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch


PACK_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACK_ROOT))
sys.path.insert(0, str(COMFY_ROOT))

try:
    from benchmarks import benchmark_h3_activation_memory as base
except ImportError:  # Direct script execution puts benchmarks/ on sys.path.
    import benchmark_h3_activation_memory as base
from h3_activation_memory.convrot_epilogue import (
    TRITON_AVAILABLE,
    ConvRotEpilogueMLP,
    _quantize_convrot_input,
    convrot_epilogue_launch_policy,
    convrot_fc1_swiglu_tensor_core,
    convrot_fc2_gated_residual_tensor_core_,
)
from h3_activation_memory.linear import swiglu_eager

if TRITON_AVAILABLE:
    import triton
    import triton.language as tl
else:  # pragma: no cover - explicit runtime failure in main
    triton = None
    tl = None


HIDDEN = 5376
FFN = 14336
EXPANDED = FFN * 2
DEFAULT_ROWS = (2048, 4096, 8192)


def parse_rows(value):
    rows = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not rows or any(item <= 0 for item in rows):
        raise ValueError("rows must contain positive integers")
    return rows


def tensor_error(reference, actual):
    delta = actual.float() - reference.float()
    reference_rms = reference.float().square().mean().sqrt()
    return {
        "exact": bool(torch.equal(reference, actual)),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(
            delta.square().mean().sqrt().div(reference_rms.clamp_min(1e-12)).item()
        ),
    }


def tensor_error_segments(reference_segments, actual_segments):
    if len(reference_segments) != len(actual_segments):
        raise ValueError("segmented tensor sizes differ")
    squared_error = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    exact = True
    for expected, actual in zip(reference_segments, actual_segments):
        if tuple(expected.shape) != tuple(actual.shape):
            raise ValueError("segmented tensor shapes differ")
        delta = actual.float() - expected.float()
        exact = exact and bool(torch.equal(expected, actual))
        max_abs = max(max_abs, float(delta.abs().max().item()))
        squared_error += float(delta.square().sum().item())
        squared_reference += float(expected.float().square().sum().item())
    return {
        "exact": exact,
        "max_abs": max_abs,
        "relative_l2": (squared_error / max(squared_reference, 1e-24)) ** 0.5,
    }


def carrier_error(reference, actual):
    if tuple(reference.shape) != tuple(actual.shape):
        raise ValueError("quantized carrier shapes differ")
    difference = actual.to(torch.int16) - reference.to(torch.int16)
    return {
        "exact": bool(torch.equal(reference, actual)),
        "mismatch_fraction": float(difference.ne(0).float().mean().item()),
        "mean_abs_int8_steps": float(difference.abs().float().mean().item()),
        "max_abs_int8_steps": int(difference.abs().max().item()),
    }


def tensor_bytes(shape, dtype):
    elements = 1
    for dimension in shape:
        elements *= int(dimension)
    return elements * torch.empty((), dtype=dtype).element_size()


def temporary_contract(rows):
    rows = int(rows)
    return {
        "input_bf16": {
            "shape": [rows, HIDDEN],
            "bytes": tensor_bytes((rows, HIDDEN), torch.bfloat16),
        },
        "production_fc1_tile_bf16": {
            "shape": [rows, FFN],
            "bytes": tensor_bytes((rows, FFN), torch.bfloat16),
            "carriers_per_mlp": 2,
        },
        "production_fc2_partial_bf16": {
            "shape": [rows, HIDDEN],
            "bytes": tensor_bytes((rows, HIDDEN), torch.bfloat16),
            "carriers_per_mlp": 2,
        },
        "fused_fc1_activated_tile_bf16": {
            "shape": [rows, FFN // 2],
            "bytes": tensor_bytes((rows, FFN // 2), torch.bfloat16),
            "carriers_per_mlp": 2,
        },
        "fc1_input_int8": {
            "shape": [rows, HIDDEN],
            "bytes": tensor_bytes((rows, HIDDEN), torch.int8),
            "production_carriers": 2,
            "fused_carriers": 1,
        },
        "fc2_input_int8": {
            "shape": [rows, FFN // 2],
            "bytes": tensor_bytes((rows, FFN // 2), torch.int8),
            "carriers_per_mlp": 2,
        },
    }


def gemm_contract(rows):
    rows = int(rows)
    return {
        "production": {
            "gemm_kernel_launches": 4,
            "fc1": {"count": 2, "m": rows, "n": FFN, "k": HIDDEN},
            "fc2": {"count": 2, "m": rows, "n": HIDDEN, "k": FFN // 2},
            "fc1_input_quantizations": 2,
            "swiglu_fc2_input_quantizations": 2,
        },
        "fused": {
            "gemm_kernel_launches": 4,
            "fc1": {
                "count": 2,
                "m": rows,
                "output_n": FFN // 2,
                "k": HIDDEN,
                "dot_streams_per_launch": 2,
            },
            "fc2": {"count": 2, "m": rows, "n": HIDDEN, "k": FFN // 2},
            "fc1_input_quantizations": 1,
            "fc2_input_quantizations": 2,
        },
    }


def full_width_gemm_contract(rows):
    rows = int(rows)
    return {
        "kitchen_control": {
            "gemm_kernel_launches": 2,
            "fc1": {"count": 1, "m": rows, "n": EXPANDED, "k": HIDDEN},
            "fc2": {"count": 1, "m": rows, "n": HIDDEN, "k": FFN},
            "fc1_input_quantizations": 1,
            "swiglu_fc2_input_quantizations": 1,
            "fc1_output": {
                "shape": [rows, EXPANDED],
                "dtype": "bfloat16",
            },
        },
        "candidate_kernel_contract": {
            "gemm_kernel_launches": 2,
            "fc1": {
                "input": {"shape": [rows, HIDDEN], "dtype": "bfloat16"},
                "weight": {"shape": [EXPANDED, HIDDEN], "dtype": "int8"},
                "operation": "ConvRot-256 INT8 GEMM + BF16 dequant boundary + SwiGLU",
                "output": {"shape": [rows, FFN], "dtype": "bfloat16"},
            },
            "fc2": {
                "input": {"shape": [rows, FFN], "dtype": "bfloat16"},
                "weight": {"shape": [HIDDEN, FFN], "dtype": "int8"},
                "operation": "ConvRot-256 INT8 GEMM + BF16 dequant boundary + one gate/residual update",
                "output": {"shape": [rows, HIDDEN], "dtype": "bfloat16"},
            },
        },
    }


def merge_fc1_tiles(tiles):
    if len(tiles) != 2:
        raise ValueError("full-width FC1 reconstruction requires two tiles")
    weights = [tile["fc1_weight"] for tile in tiles]
    scales = [tile["fc1_scale"].reshape(-1) for tile in tiles]
    if weights[0].shape != weights[1].shape or weights[0].shape[0] % 2:
        raise ValueError("FC1 tile shapes differ")
    tile_width = int(weights[0].shape[0]) // 2
    weight = torch.cat(
        (
            weights[0][:tile_width],
            weights[1][:tile_width],
            weights[0][tile_width:],
            weights[1][tile_width:],
        ),
        dim=0,
    ).contiguous()
    if all(scale.numel() == 1 for scale in scales):
        if not torch.equal(scales[0], scales[1]):
            raise ValueError("scalar FC1 tile scales differ")
        scale = scales[0].contiguous().clone()
    elif all(scale.numel() == weights[0].shape[0] for scale in scales):
        scale = torch.cat(
            (
                scales[0][:tile_width],
                scales[1][:tile_width],
                scales[0][tile_width:],
                scales[1][tile_width:],
            )
        ).contiguous()
    else:
        raise ValueError("FC1 tile scales are not scalar or per-output")
    return weight, scale


def merge_fc2_tiles(tiles):
    if len(tiles) != 2:
        raise ValueError("full-width FC2 reconstruction requires two tiles")
    weights = [tile["fc2_weight"] for tile in tiles]
    scales = [tile["fc2_scale"].reshape(-1) for tile in tiles]
    if weights[0].shape != weights[1].shape:
        raise ValueError("FC2 tile shapes differ")
    if not torch.equal(scales[0], scales[1]):
        raise ValueError("FC2 tile scales differ")
    return torch.cat(weights, dim=1).contiguous(), scales[0].contiguous().clone()


def intermediate_traffic_model(rows):
    tensors = temporary_contract(rows)
    production_expansion = (
        tensors["production_fc1_tile_bf16"]["bytes"]
        * tensors["production_fc1_tile_bf16"]["carriers_per_mlp"]
    )
    fused_expansion = (
        tensors["fused_fc1_activated_tile_bf16"]["bytes"]
        * tensors["fused_fc1_activated_tile_bf16"]["carriers_per_mlp"]
    )
    fc2_outputs = (
        tensors["production_fc2_partial_bf16"]["bytes"]
        * tensors["production_fc2_partial_bf16"]["carriers_per_mlp"]
    )
    return {
        "mode": "compulsory carrier traffic; cache effects and residual rereads excluded",
        "fc1_bf16_round_trip_saved_bytes": 2
        * (production_expansion - fused_expansion),
        "fc2_bf16_output_write_bytes_eliminated": fc2_outputs,
        "fc1_int8_carrier_write_bytes_eliminated": tensors["fc1_input_int8"][
            "bytes"
        ],
        "full_2f_bf16_intermediate_eliminated": True,
        "activated_bf16_intermediate_remains": True,
    }


def kernel_normalized_projection(
    production_timing,
    fused_timing,
    fused_stages,
    fc1_control,
    fc2_control,
):
    fc1_ratio = fc1_control["custom_over_kitchen"]
    fc2_ratio = fc2_control["custom_over_kitchen"]
    projected_ms = (
        fused_stages["fc1_input_quant"]["total_ms"]
        + fused_stages["fc1_gemm_swiglu"]["total_ms"] / fc1_ratio
        + fused_stages["swiglu_activation_quant"]["total_ms"]
        + fused_stages["fc2_gemm_gate_residual"]["total_ms"] / fc2_ratio
    )
    production_ms = production_timing["median_ms"]
    actual_ratio = fused_timing["median_ms"] / production_ms
    return {
        "classification": (
            "KERNEL-LIMITED"
            if actual_ratio > 1.0 and min(fc1_ratio, fc2_ratio) >= 1.25
            else "INCONCLUSIVE"
        ),
        "method": (
            "divide fused fc1/fc2 kernel stages by the measured same-carrier "
            "minimal-Triton-over-Kitchen ratios; keep measured quantization"
        ),
        "projected_fused_ms_at_kitchen_gemm_throughput": projected_ms,
        "projected_over_production": projected_ms / production_ms,
        "actual_over_production": actual_ratio,
        "fc1_custom_over_kitchen": fc1_ratio,
        "fc2_custom_over_kitchen": fc2_ratio,
    }


class StageEvents:
    def __init__(self):
        self.events = {}

    def call(self, name, fn, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        self.events.setdefault(name, []).append((start, end))
        return result

    def summary(self):
        torch.cuda.synchronize()
        return {
            name: {
                "calls": len(events),
                "total_ms": sum(start.elapsed_time(end) for start, end in events),
            }
            for name, events in self.events.items()
        }


def measure_case(fn, reset, warmup, iterations, device):
    for _ in range(warmup):
        reset()
        fn()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = torch.cuda.memory_allocated(device)
    events = []
    for _ in range(iterations):
        reset()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in events]
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "peak_allocated_delta_bytes": (
            torch.cuda.max_memory_allocated(device) - baseline_allocated
        ),
    }


def measure_kernel(fn, warmup, iterations, device):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize(device)
    events = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        events.append((start, end))
    torch.cuda.synchronize(device)
    samples = [start.elapsed_time(end) for start, end in events]
    return {
        "samples_ms": samples,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
    }


if TRITON_AVAILABLE:

    @triton.jit
    def _minimal_int8_gemm_dequant_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SCALE_SCALAR: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, K, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < K
            x = tl.load(
                x_ptr
                + rows[:, None].to(tl.int64) * stride_xm
                + inner[None, :].to(tl.int64) * stride_xk,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            weight = tl.load(
                weight_ptr
                + cols[:, None].to(tl.int64) * stride_wn
                + inner[None, :].to(tl.int64) * stride_wk,
                mask=col_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            accumulator += tl.dot(x, tl.trans(weight), out_dtype=tl.int32)

        row_scale = tl.load(x_scale_ptr + rows, mask=row_mask, other=0.0).to(
            tl.float32
        )
        value = accumulator.to(tl.float32)
        if SCALE_SCALAR:
            value *= row_scale[:, None] * tl.load(weight_scale_ptr).to(tl.float32)
        else:
            column_scale = tl.load(
                weight_scale_ptr + cols, mask=col_mask, other=0.0
            ).to(tl.float32)
            value *= row_scale[:, None] * column_scale[None, :]
        tl.store(
            output_ptr
            + rows[:, None].to(tl.int64) * stride_om
            + cols[None, :].to(tl.int64) * stride_on,
            value,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def minimal_int8_gemm_dequant_(x, weight, x_scale, weight_scale, output):
    if not TRITON_AVAILABLE:
        raise RuntimeError("minimal INT8 GEMM control requires Triton")
    m, k = (int(value) for value in x.shape)
    n = int(weight.shape[0])
    if tuple(weight.shape) != (n, k) or tuple(output.shape) != (m, n):
        raise ValueError("minimal INT8 GEMM control dimensions differ")
    launch = convrot_epilogue_launch_policy(m)
    grid = (
        triton.cdiv(m, launch["BLOCK_M"])
        * triton.cdiv(n, launch["BLOCK_N"]),
    )
    _minimal_int8_gemm_dequant_kernel[grid](
        x,
        weight,
        x_scale,
        weight_scale,
        output,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        M=m,
        N=n,
        K=k,
        SCALE_SCALAR=weight_scale.numel() == 1,
        **launch,
    )
    return output


def kitchen_cutlass_gemm_(x, weight, x_scale, weight_scale, output):
    from comfy_kitchen.backends import cuda as cuda_backend

    weight_scale = weight_scale.reshape(-1)
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.expand(weight.shape[0]).contiguous()
    empty_bias = torch.empty(0, dtype=torch.float32, device=x.device)
    used = cuda_backend._C.cutlass_int8_dequant(
        cuda_backend._wrap_for_dlpack(x),
        cuda_backend._wrap_for_dlpack(weight),
        cuda_backend._wrap_for_dlpack(x_scale.reshape(-1)),
        cuda_backend._wrap_for_dlpack(weight_scale),
        cuda_backend._wrap_for_dlpack(empty_bias),
        cuda_backend._wrap_for_dlpack(output),
        cuda_backend.DTYPE_TO_CODE[output.dtype],
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if not used:
        raise RuntimeError("Kitchen CUTLASS declined the same-shape GEMM control")
    return output


def effective_tops(m, n, k, milliseconds):
    return 2.0 * int(m) * int(n) * int(k) / (float(milliseconds) * 1e9)


def raw_gemm_control(name, x, weight, x_scale, weight_scale, warmup, iterations):
    device = x.device
    kitchen_output = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=device
    )
    custom_output = torch.empty_like(kitchen_output)
    kitchen = measure_kernel(
        lambda: kitchen_cutlass_gemm_(
            x, weight, x_scale, weight_scale, kitchen_output
        ),
        warmup,
        iterations,
        device,
    )
    custom = measure_kernel(
        lambda: minimal_int8_gemm_dequant_(
            x, weight, x_scale, weight_scale, custom_output
        ),
        warmup,
        iterations,
        device,
    )
    m, k = x.shape
    n = weight.shape[0]
    return {
        "name": name,
        "shape": {"m": int(m), "n": int(n), "k": int(k)},
        "kitchen_cutlass": {
            **kitchen,
            "effective_tops": effective_tops(m, n, k, kitchen["median_ms"]),
        },
        "custom_minimal_triton": {
            **custom,
            "effective_tops": effective_tops(m, n, k, custom["median_ms"]),
        },
        "custom_over_kitchen": custom["median_ms"] / kitchen["median_ms"],
        "output_error": tensor_error(kitchen_output, custom_output),
    }


def kitchen_same_carrier_shape_controls(
    activation,
    tiles,
    fc1_weight,
    fc1_scale,
    fc2_weight,
    fc2_scale,
    warmup,
    iterations,
):
    device = activation.device
    rows = int(activation.shape[0])
    fc1_input, fc1_input_scale = _quantize_convrot_input(activation)
    fc1_full_output = torch.empty(
        (rows, EXPANDED), dtype=torch.bfloat16, device=device
    )
    fc1_tile_outputs = [
        torch.empty((rows, FFN), dtype=torch.bfloat16, device=device)
        for _ in tiles
    ]

    def fc1_split():
        for tile, output in zip(tiles, fc1_tile_outputs):
            kitchen_cutlass_gemm_(
                fc1_input,
                tile["fc1_weight"],
                fc1_input_scale,
                tile["fc1_scale"],
                output,
            )

    def fc1_full():
        kitchen_cutlass_gemm_(
            fc1_input,
            fc1_weight,
            fc1_input_scale,
            fc1_scale,
            fc1_full_output,
        )

    fc1_split_timing = measure_kernel(fc1_split, warmup, iterations, device)
    fc1_full_timing = measure_kernel(fc1_full, warmup, iterations, device)
    fc1_split()
    fc1_full()
    torch.cuda.synchronize(device)
    tile_width = FFN // 2
    fc1_reference_segments = (
        fc1_full_output[:, :tile_width],
        fc1_full_output[:, tile_width:FFN],
        fc1_full_output[:, FFN : FFN + tile_width],
        fc1_full_output[:, FFN + tile_width :],
    )
    fc1_actual_segments = (
        fc1_tile_outputs[0][:, :tile_width],
        fc1_tile_outputs[1][:, :tile_width],
        fc1_tile_outputs[0][:, tile_width:],
        fc1_tile_outputs[1][:, tile_width:],
    )
    fc1_error = tensor_error_segments(
        fc1_reference_segments, fc1_actual_segments
    )

    activated = swiglu_eager(fc1_full_output)
    fc2_input, fc2_input_scale = _quantize_convrot_input(activated)
    fc2_inputs = tuple(
        fc2_input[:, start:stop].contiguous()
        for start, stop in ((0, tile_width), (tile_width, FFN))
    )
    fc2_full_output = torch.empty(
        (rows, HIDDEN), dtype=torch.bfloat16, device=device
    )
    fc2_tile_outputs = [torch.empty_like(fc2_full_output) for _ in tiles]

    def fc2_split():
        for x, tile, output in zip(fc2_inputs, tiles, fc2_tile_outputs):
            kitchen_cutlass_gemm_(
                x,
                tile["fc2_weight"],
                fc2_input_scale,
                tile["fc2_scale"],
                output,
            )

    def fc2_full():
        kitchen_cutlass_gemm_(
            fc2_input,
            fc2_weight,
            fc2_input_scale,
            fc2_scale,
            fc2_full_output,
        )

    fc2_split_timing = measure_kernel(fc2_split, warmup, iterations, device)
    fc2_full_timing = measure_kernel(fc2_full, warmup, iterations, device)
    fc2_split()
    fc2_full()
    torch.cuda.synchronize(device)
    fc2_split_output = fc2_tile_outputs[0] + fc2_tile_outputs[1]

    fc1_ops = 2 * rows * EXPANDED * HIDDEN
    fc2_ops = 2 * rows * HIDDEN * FFN

    def summarize(split, full, operations, error):
        return {
            "split_two_launches": {
                **split,
                "effective_tops": operations / (split["median_ms"] * 1e9),
            },
            "full_one_launch": {
                **full,
                "effective_tops": operations / (full["median_ms"] * 1e9),
            },
            "full_over_split": full["median_ms"] / split["median_ms"],
            "split_over_full": split["median_ms"] / full["median_ms"],
            "output_error_full_vs_split": error,
        }

    return {
        "method": (
            "prequantize one full carrier, then compare two sequential Kitchen "
            "CUTLASS GEMMs with one full-width Kitchen CUTLASS GEMM at equal FLOPs"
        ),
        "fc1": summarize(
            fc1_split_timing, fc1_full_timing, fc1_ops, fc1_error
        ),
        "fc2": summarize(
            fc2_split_timing,
            fc2_full_timing,
            fc2_ops,
            tensor_error(fc2_full_output, fc2_split_output),
        ),
    }


def production_run(ck, session, activation, residual, gate):
    output = None
    for tile in session.tiles:
        expanded = base._convrot_output(
            ck,
            activation,
            tile["fc1_weight"],
            tile["fc1_scale"],
            256,
            input_act=None,
        )
        partial = base._convrot_output(
            ck,
            expanded,
            tile["fc2_weight"],
            tile["fc2_scale"],
            256,
            input_act="swiglu",
        )
        if output is None:
            output = partial
        else:
            output.add_(partial)
        del expanded, partial
    residual.addcmul_(output, gate)
    return residual


def kitchen_full_width_run(
    ck,
    activation,
    fc1_weight,
    fc1_scale,
    fc2_weight,
    fc2_scale,
    residual,
    gate,
):
    expanded = base._convrot_output(
        ck, activation, fc1_weight, fc1_scale, 256, input_act=None
    )
    output = base._convrot_output(
        ck, expanded, fc2_weight, fc2_scale, 256, input_act="swiglu"
    )
    residual.addcmul_(output, gate)
    del expanded, output
    return residual


def production_stage_trace(ck, session, activation, residual, gate):
    gate_start = torch.cuda.Event(enable_timing=True)
    gate_end = torch.cuda.Event(enable_timing=True)

    def run():
        output = None
        for tile in session.tiles:
            expanded = base._convrot_output(
                ck,
                activation,
                tile["fc1_weight"],
                tile["fc1_scale"],
                256,
                input_act=None,
            )
            partial = base._convrot_output(
                ck,
                expanded,
                tile["fc2_weight"],
                tile["fc2_scale"],
                256,
                input_act="swiglu",
            )
            if output is None:
                output = partial
            else:
                output.add_(partial)
            del expanded, partial
        gate_start.record()
        residual.addcmul_(output, gate)
        gate_end.record()

    with base.NativeConvRotStageTrace(activation.device) as trace:
        run()
    summary = trace.summary(expected_chunks=2)
    torch.cuda.synchronize(activation.device)
    summary["gate_residual_ms"] = gate_start.elapsed_time(gate_end)
    return summary


def full_width_stage_trace(
    ck,
    activation,
    fc1_weight,
    fc1_scale,
    fc2_weight,
    fc2_scale,
    residual,
    gate,
):
    gate_start = torch.cuda.Event(enable_timing=True)
    gate_end = torch.cuda.Event(enable_timing=True)

    def run():
        expanded = base._convrot_output(
            ck, activation, fc1_weight, fc1_scale, 256, input_act=None
        )
        output = base._convrot_output(
            ck, expanded, fc2_weight, fc2_scale, 256, input_act="swiglu"
        )
        gate_start.record()
        residual.addcmul_(output, gate)
        gate_end.record()
        del expanded, output

    with base.NativeConvRotStageTrace(activation.device) as trace:
        run()
    summary = trace.summary(expected_chunks=1)
    torch.cuda.synchronize(activation.device)
    summary["gate_residual_ms"] = gate_start.elapsed_time(gate_end)
    return summary


def fused_stage_trace(session, activation, residual, gate):
    events = StageEvents()
    original_quantize = session.fc1_quantize
    original_fc1 = session.fc1_swiglu_tensor
    original_fc2 = session.fc2_gated_residual

    def fc1_quantize(x):
        return events.call("fc1_input_quant", _quantize_convrot_input, x)

    def fc1_tensor(x, weight, x_scale, weight_scale):
        return events.call(
            "fc1_gemm_swiglu",
            convrot_fc1_swiglu_tensor_core,
            x,
            weight,
            x_scale,
            weight_scale,
        )

    def fc2(x, weight, weight_scale, target, gate_value):
        x_int8, x_scale = events.call(
            "swiglu_activation_quant", _quantize_convrot_input, x
        )
        return events.call(
            "fc2_gemm_gate_residual",
            convrot_fc2_gated_residual_tensor_core_,
            x_int8,
            weight,
            x_scale,
            weight_scale,
            target,
            gate_value,
        )

    session.fc1_quantize = fc1_quantize
    session.fc1_swiglu_tensor = fc1_tensor
    session.fc2_gated_residual = fc2
    try:
        route = session.fc1_swiglu_fc2_gated_(activation, residual, gate)
        if route != "held_convrot_epilogue_prototype":
            raise RuntimeError("fused stage trace reached %s" % route)
        summary = events.summary()
    finally:
        session.fc1_quantize = original_quantize
        session.fc1_swiglu_tensor = original_fc1
        session.fc2_gated_residual = original_fc2
    expected = {
        "fc1_input_quant": 1,
        "fc1_gemm_swiglu": 2,
        "swiglu_activation_quant": 2,
        "fc2_gemm_gate_residual": 2,
    }
    actual = {name: details["calls"] for name, details in summary.items()}
    if actual != expected:
        raise RuntimeError("fused stage launches changed: %s" % actual)
    return summary


def numerical_boundary_diagnostics(
    ck,
    session,
    activation,
    residual_source,
    gate,
    production_output,
    fused_output,
):
    from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64

    fc1_input, fc1_input_scale = _quantize_convrot_input(activation)
    reference_partial_sum = None
    distributed_kitchen = residual_source.clone()
    same_carrier_custom = residual_source.clone()
    reconstructed_fused = residual_source.clone()
    unit_gate = torch.ones_like(gate)
    tiles = []

    for index, tile in enumerate(session.tiles):
        expanded = base._convrot_output(
            ck,
            activation,
            tile["fc1_weight"],
            tile["fc1_scale"],
            256,
            input_act=None,
        )
        eager_activated = swiglu_eager(expanded)
        fused_activated = convrot_fc1_swiglu_tensor_core(
            fc1_input,
            tile["fc1_weight"],
            fc1_input_scale,
            tile["fc1_scale"],
        )
        production_q, production_q_scale = quantize_int8_rowwise_convrot64(
            expanded, 256, input_act="swiglu"
        )
        eager_q, eager_q_scale = _quantize_convrot_input(eager_activated)
        fused_q, fused_q_scale = _quantize_convrot_input(fused_activated)

        kitchen_partial = torch.empty(
            (activation.shape[0], HIDDEN),
            dtype=torch.bfloat16,
            device=activation.device,
        )
        kitchen_cutlass_gemm_(
            production_q,
            tile["fc2_weight"],
            production_q_scale,
            tile["fc2_scale"],
            kitchen_partial,
        )
        kitchen_fused_carrier_partial = torch.empty_like(kitchen_partial)
        kitchen_cutlass_gemm_(
            fused_q,
            tile["fc2_weight"],
            fused_q_scale,
            tile["fc2_scale"],
            kitchen_fused_carrier_partial,
        )
        custom_same_carrier_value = torch.zeros_like(kitchen_partial)
        convrot_fc2_gated_residual_tensor_core_(
            production_q,
            tile["fc2_weight"],
            production_q_scale,
            tile["fc2_scale"],
            custom_same_carrier_value,
            unit_gate,
        )
        custom_fused_carrier_value = torch.zeros_like(kitchen_partial)
        convrot_fc2_gated_residual_tensor_core_(
            fused_q,
            tile["fc2_weight"],
            fused_q_scale,
            tile["fc2_scale"],
            custom_fused_carrier_value,
            unit_gate,
        )
        torch.cuda.synchronize(activation.device)

        if reference_partial_sum is None:
            reference_partial_sum = kitchen_partial.clone()
        else:
            reference_partial_sum.add_(kitchen_partial)
        distributed_kitchen.addcmul_(kitchen_partial, gate)
        convrot_fc2_gated_residual_tensor_core_(
            production_q,
            tile["fc2_weight"],
            production_q_scale,
            tile["fc2_scale"],
            same_carrier_custom,
            gate,
        )
        convrot_fc2_gated_residual_tensor_core_(
            fused_q,
            tile["fc2_weight"],
            fused_q_scale,
            tile["fc2_scale"],
            reconstructed_fused,
            gate,
        )
        tiles.append(
            {
                "tile": index,
                "fc1_activated_bf16_custom_vs_kitchen_eager": tensor_error(
                    eager_activated, fused_activated
                ),
                "activation_quant_eager_bf16_vs_kitchen_fused": {
                    "carrier": carrier_error(production_q, eager_q),
                    "scale": tensor_error(production_q_scale, eager_q_scale),
                },
                "activation_quant_custom_bf16_vs_kitchen_fused": {
                    "carrier": carrier_error(production_q, fused_q),
                    "scale": tensor_error(production_q_scale, fused_q_scale),
                },
                "fc2_custom_vs_kitchen_same_carrier": tensor_error(
                    kitchen_partial, custom_same_carrier_value
                ),
                "fc2_kitchen_custom_carrier_vs_production_carrier": tensor_error(
                    kitchen_partial, kitchen_fused_carrier_partial
                ),
                "fc2_custom_value_end_to_end_vs_production": tensor_error(
                    kitchen_partial, custom_fused_carrier_value
                ),
            }
        )
        del (
            expanded,
            eager_activated,
            fused_activated,
            production_q,
            production_q_scale,
            eager_q,
            eager_q_scale,
            fused_q,
            fused_q_scale,
            kitchen_partial,
            kitchen_fused_carrier_partial,
            custom_same_carrier_value,
            custom_fused_carrier_value,
        )

    reference_final = residual_source.clone()
    reference_final.addcmul_(reference_partial_sum, gate)
    torch.cuda.synchronize(activation.device)
    return {
        "method": (
            "compare each BF16/INT8 boundary; zero residual plus unit gate isolates "
            "FC2, while separate final reconstructions isolate accumulation order"
        ),
        "tiles": tiles,
        "accumulation_order_only_distributed_vs_sum_then_gate": tensor_error(
            reference_final, distributed_kitchen
        ),
        "same_carrier_custom_fc2_and_distributed_accumulation": tensor_error(
            reference_final, same_carrier_custom
        ),
        "all_custom_boundaries_reconstructed_vs_production": tensor_error(
            reference_final, reconstructed_fused
        ),
        "diagnostic_reference_vs_measured_production": tensor_error(
            production_output, reference_final
        ),
        "diagnostic_reconstruction_vs_measured_fused": tensor_error(
            fused_output, reconstructed_fused
        ),
    }


def benchmark_rows(loaded, rows, args, device):
    mlp, hidden, ffn = base.build_checkpoint_mlp(
        loaded, torch.bfloat16, hidden=HIDDEN, ffn=FFN
    )
    if (hidden, ffn) != (HIDDEN, FFN):
        raise RuntimeError("checkpoint is not the exact H3 MLP shape")
    generator = torch.Generator(device=device).manual_seed(args.seed + rows)
    activation = torch.randn(
        (rows, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    )
    residual_source = torch.randn(
        (rows, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    )
    gate = torch.randn(
        (HIDDEN,), generator=generator, dtype=torch.bfloat16, device=device
    )
    production_residual = torch.empty_like(residual_source)
    full_width_residual = torch.empty_like(residual_source)
    fused_residual = torch.empty_like(residual_source)
    ck = base._load_comfy_kitchen()
    session = ConvRotEpilogueMLP(mlp, activation[:1])
    session.__enter__()
    try:
        if len(session.tiles) != 2:
            raise RuntimeError("production H3 path did not prepare two feature slices")

        production_timing = measure_case(
            lambda: production_run(
                ck, session, activation, production_residual, gate
            ),
            lambda: production_residual.copy_(residual_source),
            args.warmup,
            args.iterations,
            device,
        )
        production_residual.copy_(residual_source)
        production_output = production_run(
            ck, session, activation, production_residual, gate
        ).clone()
        production_residual.copy_(residual_source)
        production_stages = production_stage_trace(
            ck, session, activation, production_residual, gate
        )

        fc1_weight, fc1_scale = merge_fc1_tiles(session.tiles)
        fc2_weight, fc2_scale = merge_fc2_tiles(session.tiles)

        def full_width_run():
            return kitchen_full_width_run(
                ck,
                activation,
                fc1_weight,
                fc1_scale,
                fc2_weight,
                fc2_scale,
                full_width_residual,
                gate,
            )

        full_width_timing = measure_case(
            full_width_run,
            lambda: full_width_residual.copy_(residual_source),
            args.warmup,
            args.iterations,
            device,
        )
        full_width_residual.copy_(residual_source)
        full_width_output = full_width_run().clone()
        full_width_residual.copy_(residual_source)
        full_width_stages = full_width_stage_trace(
            ck,
            activation,
            fc1_weight,
            fc1_scale,
            fc2_weight,
            fc2_scale,
            full_width_residual,
            gate,
        )

        def fused_run():
            route = session.fc1_swiglu_fc2_gated_(
                activation, fused_residual, gate
            )
            if route != "held_convrot_epilogue_prototype":
                raise RuntimeError("fused benchmark reached %s" % route)
            return fused_residual

        fused_timing = measure_case(
            fused_run,
            lambda: fused_residual.copy_(residual_source),
            args.warmup,
            args.iterations,
            device,
        )
        fused_residual.copy_(residual_source)
        fused_output = fused_run().clone()
        fused_residual.copy_(residual_source)
        fused_stages = fused_stage_trace(
            session, activation, fused_residual, gate
        )

        tile = session.tiles[0]
        fc1_input, fc1_input_scale = _quantize_convrot_input(activation)
        fc1_control = raw_gemm_control(
            "fc1_feature_tile",
            fc1_input,
            tile["fc1_weight"],
            fc1_input_scale,
            tile["fc1_scale"],
            args.control_warmup,
            args.control_iterations,
        )
        expanded = base._convrot_output(
            ck,
            activation,
            tile["fc1_weight"],
            tile["fc1_scale"],
            256,
            input_act=None,
        )
        activated = swiglu_eager(expanded)
        fc2_input, fc2_input_scale = _quantize_convrot_input(activated)
        fc2_control = raw_gemm_control(
            "fc2_feature_tile",
            fc2_input,
            tile["fc2_weight"],
            fc2_input_scale,
            tile["fc2_scale"],
            args.control_warmup,
            args.control_iterations,
        )
        kitchen_shape_controls = kitchen_same_carrier_shape_controls(
            activation,
            session.tiles,
            fc1_weight,
            fc1_scale,
            fc2_weight,
            fc2_scale,
            args.control_warmup,
            args.control_iterations,
        )

        numerical_diagnostics = None
        if rows == args.diagnostic_rows:
            numerical_diagnostics = numerical_boundary_diagnostics(
                ck,
                session,
                activation,
                residual_source,
                gate,
                production_output,
                fused_output,
            )

        normalized = kernel_normalized_projection(
            production_timing,
            fused_timing,
            fused_stages,
            fc1_control,
            fc2_control,
        )
        peak_reduction = 1.0 - (
            fused_timing["peak_allocated_delta_bytes"]
            / production_timing["peak_allocated_delta_bytes"]
        )

        return {
            "rows": int(rows),
            "dimensions": {
                "hidden": HIDDEN,
                "ffn": FFN,
                "fc1_output": EXPANDED,
                "fc2_output": HIDDEN,
            },
            "production_two_slice": {
                "route": "Kitchen ConvRot two-slice",
                "timing": production_timing,
                "stages": production_stages,
            },
            "kitchen_full_width": {
                "route": "Kitchen ConvRot full-width",
                "timing": full_width_timing,
                "stages": full_width_stages,
                "output_error_vs_two_slice": tensor_error(
                    production_output, full_width_output
                ),
                "full_width_over_two_slice": (
                    full_width_timing["median_ms"]
                    / production_timing["median_ms"]
                ),
                "peak_allocated_over_two_slice": (
                    full_width_timing["peak_allocated_delta_bytes"]
                    / production_timing["peak_allocated_delta_bytes"]
                ),
            },
            "fused_epilogue": {
                "route": "held_convrot_epilogue_prototype",
                "timing": fused_timing,
                "stages": fused_stages,
                "output_error_vs_production": tensor_error(
                    production_output, fused_output
                ),
            },
            "fused_over_production": (
                fused_timing["median_ms"] / production_timing["median_ms"]
            ),
            "peak_allocated_reduction_fraction": peak_reduction,
            "kernel_normalized_projection": normalized,
            "raw_gemm_controls": [fc1_control, fc2_control],
            "kitchen_same_carrier_shape_controls": kitchen_shape_controls,
            "numerical_boundary_diagnostics": numerical_diagnostics,
            "temporary_tensors": temporary_contract(rows),
            "intermediate_traffic_model": intermediate_traffic_model(rows),
            "launch_contract": gemm_contract(rows),
            "full_width_launch_and_candidate_contract": full_width_gemm_contract(
                rows
            ),
            "custom_launch_policy": convrot_epilogue_launch_policy(rows),
        }
    finally:
        session.__exit__(*sys.exc_info())


def build_parser():
    parser = argparse.ArgumentParser(
        description="Separate H3 ConvRot MLP fusion benefit from Triton GEMM quality."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument(
        "--rows", default=",".join(str(value) for value in DEFAULT_ROWS)
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--control-warmup", type=int, default=3)
    parser.add_argument("--control-iterations", type=int, default=15)
    parser.add_argument(
        "--diagnostic-rows",
        type=int,
        default=4096,
        help="row count for the memory-heavier numerical boundary trace; use 0 to disable",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not args.i_understand_this_uses_gpu:
        raise SystemExit(
            "pass --i-understand-this-uses-gpu after the required GPU preflight"
        )
    if not TRITON_AVAILABLE:
        raise SystemExit("the existing ConvRot epilogue prototype requires Triton")
    rows = parse_rows(args.rows)
    if 4096 not in rows:
        raise SystemExit("--rows must include the production 4096-row chunk")
    if min(args.warmup, args.iterations, args.control_warmup, args.control_iterations) < 0:
        raise SystemExit("benchmark iteration counts must be non-negative")
    if args.iterations == 0 or args.control_iterations == 0:
        raise SystemExit("measured iteration counts must be positive")
    if args.diagnostic_rows < 0:
        raise SystemExit("--diagnostic-rows must be non-negative")
    if args.diagnostic_rows and args.diagnostic_rows not in rows:
        raise SystemExit("--diagnostic-rows must be present in --rows or set to 0")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    checkpoint = base.resolve_checkpoint(args.checkpoint)
    loaded = base.load_block_mlp_tensors(checkpoint, args.block_index)
    device = torch.device("cuda")
    results = [benchmark_rows(loaded, row_count, args, device) for row_count in rows]
    report = {
        "classification_rule": (
            "A slower fused total is KERNEL-LIMITED when same-carrier minimal "
            "Triton GEMMs are materially slower than Kitchen CUTLASS."
        ),
        "interpretation_boundary": (
            "Kitchen full-width measures shape and launch effects but still writes "
            "the [M,28672] FC1 carrier; it is not a fused-CUTLASS latency prediction."
        ),
        "checkpoint": {
            "path": checkpoint,
            "block_index": int(args.block_index),
            "prefix": loaded["prefix"],
            "weight_format": "ConvRot-256 TensorWise INT8",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
