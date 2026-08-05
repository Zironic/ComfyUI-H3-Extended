"""Runtime patching and measurement for the activation-memory VRAM A/B probe."""

import statistics
import time

import torch

import _minimax_vram_probe_base as base


def build_forwards(block, args):
    # Imported only after base.select_attention(True) has selected Sage.
    from h3_attention.forward import make_forward as make_attention_forward
    from h3_attention.sage_mem_eff import SM89SageMemoryEfficientBackend
    from h3_activation_memory.config import ActivationMemoryConfig
    from h3_activation_memory.forward import make_forward as make_activation_forward

    stock_block_forward = block.forward
    backend = SM89SageMemoryEfficientBackend()
    block.attn.forward = make_attention_forward(
        block.attn, layer_index=0, backend=backend
    )
    config = ActivationMemoryConfig(
        mode=args.activation_mode,
        chunk_rows=args.activation_chunk_rows,
        alignment=args.activation_alignment,
        strict=not args.activation_nonstrict,
        prefer_held_weights=not args.no_held_weights,
    )
    activation_forward = make_activation_forward(
        block,
        layer_index=0,
        config=config,
        original_forward=stock_block_forward,
    )
    return stock_block_forward, activation_forward, backend, config


def measure_forward(forward_fn, layout, arch, dtype, device, *, warmup, iterations, seed):
    """Report max transient and median time with legacy-probe allocation scope."""
    from comfy.ldm.minimax.model import rope_rotation_table

    torch.manual_seed(seed)
    seq_len = layout["seq_len"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    allocation_base = torch.cuda.memory_allocated(device)

    x = torch.randn(seq_len, arch["hidden_size"], dtype=dtype, device=device)
    rope = rope_rotation_table(
        torch.randn(
            seq_len,
            arch["rope_inv_freq_len"] * 6,
            dtype=torch.float32,
            device=device,
        ),
        dtype,
    )
    t_emb = torch.randn(
        layout["n_t_classes"],
        arch["time_embed_dim"],
        dtype=torch.float16,
        device=device,
    )

    def invoke():
        return forward_fn(
            x,
            t_emb,
            layout["mod_segments"],
            rope,
            transformer_options={},
        )

    with torch.no_grad():
        for _ in range(max(0, warmup)):
            invoke()
    torch.cuda.synchronize(device)

    times, peaks = [], []
    for _ in range(max(1, iterations)):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        with torch.no_grad():
            y = invoke()
        torch.cuda.synchronize(device)
        times.append((time.perf_counter() - started) * 1000.0)
        peaks.append(torch.cuda.max_memory_allocated(device) - allocation_base)
        del y

    del x, rope, t_emb
    torch.cuda.empty_cache()
    return max(peaks), statistics.median(times)


def safe_measure(*args, **kwargs):
    try:
        return measure_forward(*args, **kwargs), None
    except base.OOM_ERRORS as exc:
        if not base.is_oom(exc):
            raise
        torch.cuda.empty_cache()
        return None, str(exc)


def update_resident_fit(points, sequence, ms, spill_ratio):
    coefficient = base.fit_ms(points)
    predicted = (
        coefficient[0] * sequence + coefficient[1] * sequence * sequence
        if coefficient
        else None
    )
    if predicted and ms > predicted * spill_ratio:
        return True, ms / predicted
    points.append((sequence, ms))
    return False, (ms / predicted if predicted else None)


def fmt_gib(value):
    return "   OOM" if value is None else f"{value / base.GB:>6.3f}G"


def fmt_ms(value):
    return "    OOM" if value is None else f"{value:>7.1f}"
