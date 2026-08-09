"""Benchmark H3's BF16 QKV path against the Sparse Sage-native projection.

This is an explicit CUDA benchmark. It loads only one attention block's QKV
projection and norm weights from the selected safetensors checkpoint. Without
``--frames`` it measures projection preparation alone; geometry mode measures
both production routed-attention paths from projection through the Sparse Sage
kernel.
"""

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch


_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

# Keep the projection-forward import lazy so ``--help`` and argument validation
# remain available on CPU-only developer environments without Comfy's optional
# acceleration wheels.
project_qkv = to_hnd = None
from h3_attention.hybrid.backend import HybridSparseBackend  # noqa: E402
from h3_attention.hybrid.config import (  # noqa: E402
    HybridSparseConfig,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
)
from h3_attention.hybrid.fused_qkv import (  # noqa: E402
    HEAD_DIM,
    FusedQKVProjector,
    PreparedFusedQKV,
    fused_qkv_tensor_core,
    run_fused_qkv,
)
from h3_attention.hybrid.router import KV_TILE, Q_TILE, SparseTileRouter  # noqa: E402
from h3_attention.hybrid.sparse_quant import _run as quantize_blocks  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    SparseSageExecutor,
    load_sparse_sage_spec,
    preflight_sparse_sage,
)
from h3_probe.layout import TokenLayout  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402
from h3_runtime.metrics import tensor_error_metrics  # noqa: E402
from h3_runtime.timing import publish_timing, timed_stage  # noqa: E402
from h3_attention.hybrid.stats import TIMING_STAGES  # noqa: E402


def _ensure_forward_imports():
    global project_qkv, to_hnd
    if project_qkv is None:
        from h3_attention.forward import project_qkv as _project_qkv, to_hnd as _to_hnd
        project_qkv, to_hnd = _project_qkv, _to_hnd


def compile_fused_qkv_core(torch_module, core):
    """Compile a fixed-shape tensor-only fused projection core."""
    return torch_module.compile(core, fullgraph=True, dynamic=False)


def compile_sparse_sage_kernel_core(torch_module, adapter):
    """Compile the fixed-shape tensor-only Sparse Sage kernel adapter."""
    return torch_module.compile(adapter, fullgraph=True, dynamic=False)


def make_sparse_sage_kernel_adapter(kernel, output_shape, output_dtype):
    """Bind one selected low-level op to the executor's tensor-only ABI."""
    output_shape = tuple(int(value) for value in output_shape)

    def adapter(q_int8, k_int8, v_fp8, lut, valid_block_num,
                pv_threshold, q_scale, k_scale, v_scale):
        output = torch.empty(
            output_shape, dtype=output_dtype, device=q_int8.device,
        )
        # The SM89 op mutates output and its return value is intentionally not
        # part of SparseSageExecutor.execute's contract.
        kernel(
            q_int8, k_int8, v_fp8, output,
            lut, valid_block_num, pv_threshold,
            q_scale, k_scale, v_scale,
            1, 0, 1, 128 ** -0.5, 0,
        )
        return output

    return adapter


def sparse_sage_carrier_tensors(prepared):
    """Return a prepared carrier in the exact order consumed by the kernel."""
    return (
        prepared.q_int8,
        prepared.k_int8,
        prepared.v_carrier,
        prepared.lut,
        prepared.valid_block_num,
        prepared.pv_threshold,
        prepared.q_scale,
        prepared.k_scale,
        prepared.v_scale,
    )


def sparse_sage_op_identity(kernel):
    module = getattr(kernel, "__module__", None)
    name = getattr(kernel, "__qualname__", None) or getattr(kernel, "__name__", None)
    if module and name:
        return "%s.%s" % (module, name)
    return type(kernel).__name__


def validate_compile_sage_request(enabled, layout):
    """Reject kernel-only compilation unless geometry supplied a carrier."""
    if enabled and layout is None:
        raise ValueError("--compile-sage requires --frames geometry mode")


def _prepare_sparse_sage_carrier(backend, module, x, rope, runtime, layer_index):
    """Prepare one real production fused-QKV Sparse Sage carrier."""
    _ensure_forward_imports()
    options = {RUNTIME_KEY: runtime}
    publish_timing(options, backend.timing)
    projected = backend.projector.project(
        module, x, rope, layer_index=layer_index,
        transformer_options=options,
    )
    prepared = backend.prepare_projected(
        projected, layer_index=layer_index, transformer_options=options,
    )
    del projected
    return prepared.sparse


def benchmark_sparse_sage_compile(prepared, executor, warmup, iterations, device):
    """A/B one prepared Sparse Sage carrier, excluding all preparation stages."""
    kernel = executor.low_level_selector(prepared.q_int8)
    adapter = make_sparse_sage_kernel_adapter(
        kernel, prepared.output_shape, prepared.output_dtype,
    )
    carrier = sparse_sage_carrier_tensors(prepared)
    eager_fn = lambda: adapter(*carrier)
    eager = benchmark_case(eager_fn, warmup, iterations)
    compiled = compile_sparse_sage_kernel_core(torch, adapter)
    compiled_fn = lambda: compiled(*carrier)
    compiled_result = benchmark_compiled_case(
        compiled_fn, warmup, iterations, device,
    )

    eager_output = eager_fn()
    torch.cuda.synchronize(device)
    eager_output = eager_output.detach().cpu()
    compiled_output = compiled_fn()
    torch.cuda.synchronize(device)
    compiled_output = compiled_output.detach().cpu()
    exact = bool(torch.equal(eager_output, compiled_output))
    parity = {
        "exact": exact,
        "exact_equal": exact,
        "max_abs": max_abs(eager_output, compiled_output),
        "relative_rmse": relative_rmse(eager_output, compiled_output),
    }
    del eager_output, compiled_output
    return {
        "boundary": {
            "input": "PreparedSparseSage carrier tensors",
            "included_stages": ["sparse_sage_low_level_kernel"],
            "excluded_stages": [
                "fused_qkv_projection",
                "direct_lut_construction",
                "q_k_int8_quantization",
                "v_preparation",
            ],
        },
        "excluded_stages": [
            "fused_qkv_projection",
            "direct_lut_construction",
            "q_k_int8_quantization",
            "v_preparation",
        ],
        "op_identity": sparse_sage_op_identity(kernel),
        "eager": eager,
        "compiled": compiled_result,
        "speedup": eager["median_ms"] / compiled_result["median_ms"],
        "parity": parity,
    }


def _compile_warmup(fn, device):
    """Run the first compiled call outside measured samples."""
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    del result
    return elapsed


def benchmark_compiled_case(fn, warmup, iterations, device):
    """Compile once outside the ordinary projection measurements."""
    compile_ms = _compile_warmup(fn, device)
    result = benchmark_case(fn, warmup, iterations)
    result["compile_warmup_ms"] = compile_ms
    return result


def warmup_compiled_routed_case(backend, module, x, rope, runtime, layer_index):
    """Compile one routed call, then discard its deferred timing samples."""
    device = x.device
    request_runtime = replace(runtime, request_id=3000)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    output, _metadata = _routed_call(
        backend, module, x, rope, request_runtime, layer_index, True,
    )
    backend.timing.resolve()
    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - started) * 1000.0
    del output
    return elapsed


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
    module = torch.nn.Module()
    module.qkv_proj = qkv_proj
    module.q_norm = q_norm
    module.k_norm = k_norm
    module.heads = heads
    module.head_dim = HEAD_DIM
    return module, hidden, prefix


def make_rope(sequence, device):
    angles = torch.arange(sequence * 48, device=device, dtype=torch.float32).reshape(sequence, 48)
    angles = angles * (1.0 / 8192.0)
    c = torch.cos(angles)
    s = torch.sin(angles)
    return torch.stack((c, -s, s, c), dim=-1).reshape(
        1, sequence, 1, 48, 2, 2
    ).to(torch.bfloat16)


def build_geometry_layout(frames, width, height, text_len):
    """Resolve the exact packed layout used by the MiniMax H3 forward."""
    if int(frames) <= 0:
        raise ValueError("frames must be positive")
    if int(frames) % 17 != 5:
        raise ValueError("frames must satisfy frames % 17 == 5")
    if int(width) <= 0 or int(height) <= 0 or int(width) % 32 or int(height) % 32:
        raise ValueError("width and height must be positive multiples of 32")
    if int(text_len) < 0:
        raise ValueError("text-len cannot be negative")
    import _minimax_vram_probe_base as probe

    raw = probe.build_layout(int(frames), int(width), int(height), int(text_len))
    text_range = audio_range = video_range = None
    references = []
    for start, stop, kind in raw["segments"]:
        value = (int(start), int(stop))
        if kind == "text":
            text_range = value
        elif kind == "audio":
            audio_range = value
        elif kind == "video":
            video_range = value
        else:
            references.append((kind, *value))
    if text_range is None or audio_range is None or video_range is None:
        raise ValueError("packed layout is missing text/audio/video segments")
    return TokenLayout(
        seq_len=int(raw["seq_len"]),
        text_range=text_range,
        audio_range=audio_range,
        video_range=video_range,
        video_shape=(int(raw["latent_t"]), int(height) // 32, int(width) // 32),
        audio_t=(audio_range[1] - audio_range[0]) // 2,
        reference_ranges=references,
        segments=[(int(a), int(b), str(kind)) for a, b, kind in raw["segments"]],
    )


def resolve_sequence(sequence, frames, width, height, text_len):
    """Resolve legacy sequence-only geometry and reject conflicting requests."""
    if frames is None:
        resolved = 54006 if sequence is None else int(sequence)
        if resolved <= 0:
            raise ValueError("sequence must be positive")
        return resolved, None
    layout = build_geometry_layout(frames, width, height, text_len)
    if sequence is not None and int(sequence) != layout.seq_len:
        raise ValueError(
            "--sequence conflicts with --frames geometry: requested %d, packed layout is %d"
            % (int(sequence), layout.seq_len)
        )
    return layout.seq_len, layout


def _runtime_snapshot(layout, device, request_id=0):
    return RuntimeSnapshot(
        request_id=int(request_id),
        step_index=0,
        total_steps=1,
        sigma=0.5,
        branch=(0,),
        layout=layout,
        layout_signature=(layout.seq_len, tuple(layout.segments)),
        compute_dtype=torch.bfloat16,
        device=device,
    )


def _routed_call(backend, module, x, rope, runtime, layer_index, fused):
    _ensure_forward_imports()
    options = {RUNTIME_KEY: runtime}
    publish_timing(options, backend.timing)
    if fused:
        with timed_stage(options, "fused_qkv_projection"):
            projected = backend.projector.project(
                module, x, rope, layer_index=layer_index,
                transformer_options=options,
            )
        prepared = backend.prepare_projected(
            projected, layer_index=layer_index, transformer_options=options,
        )
        del projected
    else:
        q, k, v = project_qkv(module, x, rope, options)
        q, k, v = to_hnd(q, k, v)
        prepared = backend.prepare(
            q, k, v, layer_index=layer_index, transformer_options=options,
        )
        del q, k, v
    output = backend.execute(prepared)
    metadata = dict(prepared.sparse.metadata)
    del prepared
    return output, metadata


def benchmark_routed_case(backend, module, x, rope, runtime, layer_index,
                          fused, warmup, iterations):
    """Measure one production routed path, including projection through kernel."""
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    samples, peaks, stage_samples = [], [], {stage: [] for stage in TIMING_STAGES}
    metadata = {}
    device = x.device
    for index in range(int(warmup) + int(iterations)):
        request_id = 1000 + index
        backend.timing.begin_request(request_id, cuda=True)
        request_runtime = replace(runtime, request_id=request_id)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        output, metadata = _routed_call(
            backend, module, x, rope, request_runtime, layer_index, fused,
        )
        timing_summary = backend.timing.resolve()
        elapsed = (time.perf_counter() - started) * 1000.0
        peak = torch.cuda.max_memory_allocated(device) - before
        if index >= warmup:
            samples.append(elapsed)
            peaks.append(peak)
            for stage, details in timing_summary["stages"].items():
                if details["count"]:
                    stage_samples[stage].append(float(details["mean_ms"]))
        del output
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "median_peak_allocated_bytes": int(statistics.median(peaks)),
        "stage_medians_ms": {
            stage: statistics.median(values)
            for stage, values in stage_samples.items() if values
        },
        "route_metadata": metadata,
    }


def baseline(module, x, rope, block_index):
    _ensure_forward_imports()
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
    _ensure_forward_imports()
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
    executor = SparseSageExecutor(load_sparse_sage_spec())

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


def run_routed_geometry(args, checkpoint, module, hidden, prefix, sequence, layout):
    if args.verify_attention:
        raise ValueError("--verify-attention is unsafe with --frames geometry mode")
    api = preflight_sparse_sage()
    router = SparseTileRouter()
    device = torch.device("cuda")
    runtime = _runtime_snapshot(layout, device)
    established = HybridSparseBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128,
            video_budget=args.video_budget,
            timing=True,
            run_tag="bench_fused_qkv_established",
        ),
        kernel_spec=api,
        router=router,
    )
    fused = HybridSparseBackend(
        HybridSparseConfig(
            mode=MODE_SAGE128_FUSED_QKV,
            video_budget=args.video_budget,
            timing=True,
            run_tag="bench_fused_qkv_fused",
        ),
        kernel_spec=api,
        router=router,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed)
    x = torch.randn((sequence, hidden), generator=generator,
                    dtype=torch.bfloat16, device=device)
    rope = make_rope(sequence, device)
    established_result = benchmark_routed_case(
        established, module, x, rope, runtime, args.block, False,
        args.warmup, args.iterations,
    )
    fused_result = benchmark_routed_case(
        fused, module, x, rope, runtime, args.block, True,
        args.warmup, args.iterations,
    )
    sparse_sage_compile = None
    if getattr(args, "compile_sage", False):
        carrier_backend = HybridSparseBackend(
            HybridSparseConfig(
                mode=MODE_SAGE128_FUSED_QKV,
                video_budget=args.video_budget,
                timing=False,
                run_tag="bench_fused_qkv_sparse_sage_compile",
            ),
            kernel_spec=api,
            router=router,
        )
        carrier = _prepare_sparse_sage_carrier(
            carrier_backend, module, x, rope, runtime, args.block,
        )
        sparse_sage_compile = benchmark_sparse_sage_compile(
            carrier,
            carrier_backend.executor,
            args.warmup,
            args.iterations,
            device,
        )
        del carrier
    compiled = compiled_core = compiled_result = None
    if getattr(args, "compile_fused", False):
        compiled_core = compile_fused_qkv_core(torch, fused_qkv_tensor_core)
        compiled = HybridSparseBackend(
            HybridSparseConfig(
                mode=MODE_SAGE128_FUSED_QKV,
                video_budget=args.video_budget,
                timing=True,
                run_tag="bench_fused_qkv_compiled",
            ),
            kernel_spec=api,
            router=router,
        )
        compiled.projector = FusedQKVProjector(compiled_core)
        compile_ms = warmup_compiled_routed_case(
            compiled, module, x, rope, runtime, args.block,
        )
        compiled_result = benchmark_routed_case(
            compiled, module, x, rope, runtime, args.block, True,
            args.warmup, args.iterations,
        )
        compiled_result["compile_warmup_ms"] = compile_ms

    # One untimed comparison uses the same real production backend contracts.
    compare_established = HybridSparseBackend(
        HybridSparseConfig(mode=MODE_SAGE128, video_budget=args.video_budget, timing=False),
        kernel_spec=api, router=router,
    )
    compare_fused = HybridSparseBackend(
        HybridSparseConfig(mode=MODE_SAGE128_FUSED_QKV,
                           video_budget=args.video_budget, timing=False),
        kernel_spec=api, router=router,
    )
    compare_established.timing.begin_request(2001, cuda=True)
    compare_fused.timing.begin_request(2002, cuda=True)
    established_output, established_metadata = _routed_call(
        compare_established, module, x, rope,
        replace(runtime, request_id=2001), args.block, False,
    )
    established_output = established_output.to("cpu")
    fused_output, fused_metadata = _routed_call(
        compare_fused, module, x, rope,
        replace(runtime, request_id=2002), args.block, True,
    )
    fused_output = fused_output.to("cpu")
    comparison = tensor_error_metrics(fused_output, established_output)
    del established_output
    compiled_comparison = None
    if compiled is not None:
        compare_compiled = HybridSparseBackend(
            HybridSparseConfig(
                mode=MODE_SAGE128_FUSED_QKV,
                video_budget=args.video_budget,
                timing=False,
            ),
            kernel_spec=api,
            router=router,
            projector=FusedQKVProjector(compiled_core),
        )
        compare_compiled.timing.begin_request(2003, cuda=True)
        compiled_output, _compiled_metadata = _routed_call(
            compare_compiled, module, x, rope,
            replace(runtime, request_id=2003), args.block, True,
        )
        compiled_output = compiled_output.to("cpu")
        compiled_comparison = tensor_error_metrics(compiled_output, fused_output)
        del compiled_output
    del fused_output
    established_route = {
        key: value for key, value in established_metadata.items()
        if key not in (
            "request_id", "step", "branch", "layer", "qkv_projection", "smooth_k",
        )
    }
    fused_route = {
        key: value for key, value in fused_metadata.items()
        if key not in (
            "request_id", "step", "branch", "layer", "qkv_projection", "smooth_k",
        )
    }
    result = {
        "benchmark": "routed_end_to_end",
        "checkpoint": checkpoint,
        "prefix": prefix,
        "block": args.block,
        "sequence": sequence,
        "hidden": hidden,
        "heads": module.heads,
        "device": torch.cuda.get_device_name(),
        "execution_mode": "production_hybrid_sparse",
        "status": {
            "established": established.as_status(),
            "fused": fused.as_status(),
            "sparge_api": api.version,
        },
        "layout": {
            **layout.as_dict(),
            "frames": int(args.frames),
            "width": int(args.width),
            "height": int(args.height),
            "text_len": int(args.text_len),
            "video_budget": float(args.video_budget),
        },
        "established": established_result,
        "fused": fused_result,
        "speedup": established_result["median_ms"] / fused_result["median_ms"],
        "peak_reduction_bytes": (
            established_result["median_peak_allocated_bytes"]
            - fused_result["median_peak_allocated_bytes"]
        ),
        "route_metadata": {
            "identical_selection": established_route == fused_route,
            "established": established_route,
            "fused": fused_route,
        },
        "comparisons": comparison,
    }
    if compiled_result is not None:
        result["status"]["compiled_fused"] = compiled.as_status()
        result["compiled_fused"] = compiled_result
        result["compiled_comparisons"] = compiled_comparison
        result["eager_vs_compiled_speedup"] = (
            fused_result["median_ms"] / compiled_result["median_ms"]
        )
        result["eager_vs_compiled_peak_reduction_bytes"] = (
            fused_result["median_peak_allocated_bytes"]
            - compiled_result["median_peak_allocated_bytes"]
        )
    if sparse_sage_compile is not None:
        result["sparse_sage_compile"] = sparse_sage_compile
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument(
        "--sequence", type=int, default=None,
        help="packed sequence for the legacy projection microbenchmark (default: 54006)",
    )
    parser.add_argument(
        "--frames", type=int, default=None,
        help="opt in to the geometry-driven routed-attention benchmark",
    )
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument(
        "--video-budget", type=float, default=0.5,
        help="fraction of pure target-video KV tiles retained by routing",
    )
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verify-attention", action="store_true")
    parser.add_argument(
        "--compile-fused", action="store_true",
        help="also benchmark a fixed-shape torch.compile fused projection (CUDA)",
    )
    parser.add_argument(
        "--compile-sage", action="store_true",
        help="also benchmark eager vs fixed-shape torch.compile Sparse Sage kernel (geometry only)",
    )
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("sequence/iteration arguments are invalid")
    try:
        sequence, layout = resolve_sequence(
            args.sequence, args.frames, args.width, args.height, args.text_len,
        )
    except ValueError as exc:
        raise SystemExit(str(exc))
    if sequence <= 0:
        raise SystemExit("sequence/iteration arguments are invalid")
    try:
        validate_compile_sage_request(args.compile_sage, layout)
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not 0.0 < float(args.video_budget) <= 1.0:
        raise SystemExit("video-budget must be in (0, 1]")
    if args.frames is not None and args.verify_attention:
        raise SystemExit("--verify-attention cannot be combined with --frames geometry mode")
    if not args.i_understand_this_uses_gpu:
        raise SystemExit("pass --i-understand-this-uses-gpu after the required idle-GPU preflight")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.get_device_capability() != (8, 9):
        raise SystemExit("the fused H3 QKV experiment is SM89-only")

    checkpoint = resolve_checkpoint(args.checkpoint)
    module, hidden, prefix = build_attention(checkpoint, args.block, args.epsilon)
    # Production patching binds every fused-QKV module before its first call.
    # The standalone benchmark owns its synthetic module and must do the same.
    FusedQKVProjector().bind(module)
    _ensure_forward_imports()
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    if layout is not None:
        result = run_routed_geometry(
            args, checkpoint, module, hidden, prefix, sequence, layout,
        )
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print("geometry=%s" % layout.describe())
            print("established: %.3f ms, peak %.3f GiB" % (
                result["established"]["median_ms"],
                result["established"]["median_peak_allocated_bytes"] / 2**30,
            ))
            print("fused:      %.3f ms, peak %.3f GiB" % (
                result["fused"]["median_ms"],
                result["fused"]["median_peak_allocated_bytes"] / 2**30,
            ))
            print("speedup: %.3fx; peak reduction: %.3f GiB" % (
                result["speedup"], result["peak_reduction_bytes"] / 2**30,
            ))
            if "compiled_fused" in result:
                print("compiled:   %.3f ms, peak %.3f GiB, compile warmup %.3f ms" % (
                    result["compiled_fused"]["median_ms"],
                    result["compiled_fused"]["median_peak_allocated_bytes"] / 2**30,
                    result["compiled_fused"]["compile_warmup_ms"],
                ))
                print("eager/compiled speedup: %.3fx; peak reduction: %.3f GiB" % (
                    result["eager_vs_compiled_speedup"],
                    result["eager_vs_compiled_peak_reduction_bytes"] / 2**30,
                ))
                print(json.dumps(
                    result["compiled_comparisons"], indent=2, sort_keys=True,
                ))
            if "sparse_sage_compile" in result:
                sage = result["sparse_sage_compile"]
                print("sparse sage eager: %.3f ms, peak %.3f GiB" % (
                    sage["eager"]["median_ms"],
                    sage["eager"]["peak_bytes"] / 2**30,
                ))
                print("sparse sage compiled: %.3f ms, peak %.3f GiB, compile warmup %.3f ms" % (
                    sage["compiled"]["median_ms"],
                    sage["compiled"]["peak_bytes"] / 2**30,
                    sage["compiled"]["compile_warmup_ms"],
                ))
                print("sparse sage eager/compiled speedup: %.3fx; exact parity: %s" % (
                    sage["speedup"], sage["parity"]["exact"],
                ))
            print(json.dumps(result["comparisons"], indent=2, sort_keys=True))
        return

    x = torch.randn(
        (sequence, hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope = make_rope(sequence, device)

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
        "benchmark": "projection_microbench",
        "sequence": sequence,
        "hidden": hidden,
        "heads": module.heads,
        "device": torch.cuda.get_device_name(),
        "baseline": baseline_result,
        "fused": fused_result,
        "peak_reduction_bytes": baseline_result["peak_bytes"] - fused_result["peak_bytes"],
        "speedup": baseline_result["median_ms"] / fused_result["median_ms"],
        "comparisons": comparisons,
    }
    if getattr(args, "compile_fused", False):
        compiled_core = compile_fused_qkv_core(torch, fused_qkv_tensor_core)
        compiled_fn = lambda: run_fused_qkv(
            module, x, rope, layer_index=args.block, tensor_core=compiled_core,
        )
        compiled_result = benchmark_compiled_case(
            compiled_fn, args.warmup, args.iterations, device,
        )
        result["compiled_fused"] = compiled_result
        result["eager_vs_compiled_speedup"] = (
            fused_result["median_ms"] / compiled_result["median_ms"]
        )
        result["eager_vs_compiled_peak_reduction_bytes"] = (
            fused_result["peak_bytes"] - compiled_result["peak_bytes"]
        )
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
        print("sequence=%d hidden=%d heads=%d" % (sequence, hidden, module.heads))
        print("baseline: %.3f ms, peak %.3f GiB" % (
            baseline_result["median_ms"], baseline_result["peak_bytes"] / 2**30))
        print("fused:    %.3f ms, peak %.3f GiB" % (
            fused_result["median_ms"], fused_result["peak_bytes"] / 2**30))
        print("speedup: %.3fx; peak reduction: %.3f GiB" % (
            result["speedup"], result["peak_reduction_bytes"] / 2**30))
        if "compiled_fused" in result:
            print("compiled: %.3f ms, peak %.3f GiB, compile warmup %.3f ms" % (
                result["compiled_fused"]["median_ms"],
                result["compiled_fused"]["peak_bytes"] / 2**30,
                result["compiled_fused"]["compile_warmup_ms"],
            ))
            print("eager/compiled speedup: %.3fx; peak reduction: %.3f GiB" % (
                result["eager_vs_compiled_speedup"],
                result["eager_vs_compiled_peak_reduction_bytes"] / 2**30,
            ))
        print(json.dumps(comparisons, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
