"""H3-owned DiT block forward with bounded token-chunked MLP activations."""

import logging

import torch

import comfy.model_management

from .chunks import iter_mod_chunks, validate_mod_segments
from .linear import (
    ConvRotTwoSliceMLP,
    HeldMLP,
    UnsafeHeldWeights,
    bind_convrot_mlp,
    module_fc1,
    module_swiglu_fc2,
)
from .observer import notify_activation
from .stats import get_stats

try:
    from ..h3_runtime.timing import timed_stage
except ImportError:
    from h3_runtime.timing import timed_stage

LOG_PREFIX = "[H3 activation memory]"


def _scale_shift(h, shift, scale):
    return h.mul_(1.0 + scale.to(h.dtype)).add_(shift.to(h.dtype))


def _gate_add(x, other, gate):
    return x.addcmul_(other, gate.to(x.dtype))


def make_forward(block, layer_index, config, original_forward=None):
    """Build an unbound replacement for one ``DiTBlock.forward``.

    ``ModelPatcher.add_object_patch`` installs this function on the instance, so
    the block is closed over rather than received as ``self``.
    """
    original_forward = original_forward or block.forward
    if config.convrot_2slice and isinstance(block.mlp, torch.nn.Module):
        bind_convrot_mlp(block.mlp)

    def _forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if comfy.model_management.in_training:
            raise RuntimeError(
                "h3_activation_memory is inference-only; training requires "
                "core's original block forward"
            )
        with timed_stage(transformer_options, "adaln_proj"):
            shifts = block.adaln_proj(t_emb)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = shifts
        segments = validate_mod_segments(
            mod_segments, x.shape[0], mod_rows=shift_msa.shape[0]
        )
        stats = None if torch.compiler.is_compiling() else get_stats(
            transformer_options, config
        )
        if stats is not None:
            stats.blocks += 1

        notify_activation(
            "block_enter",
            layer_index,
            transformer_options,
            seq_len=int(x.shape[0]),
            dtype=str(x.dtype),
        )

        with timed_stage(transformer_options, "norm1_modulation"):
            h = block.norm1(x)
            for start, stop, row in segments:
                _scale_shift(h[start:stop], shift_msa[row], scale_msa[row])
        notify_activation(
            "attention_norm_ready",
            layer_index,
            transformer_options,
            shape=tuple(h.shape),
        )
        attn_out = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        notify_activation(
            "attention_returned",
            layer_index,
            transformer_options,
            shape=tuple(attn_out.shape),
        )
        with timed_stage(transformer_options, "attention_residual_gate"):
            for start, stop, row in segments:
                _gate_add(x[start:stop], attn_out[start:stop], gate_msa[row])
        del h, attn_out
        notify_activation(
            "attention_gated", layer_index, transformer_options
        )

        chunks = tuple(
            iter_mod_chunks(
                segments,
                x.shape[0],
                config.chunk_rows,
                alignment=config.alignment,
                mod_rows=shift_mlp.shape[0],
            )
        )

        held = None
        held_error = None
        if config.convrot_2slice:
            held = ConvRotTwoSliceMLP(block.mlp, x[:1])
            held.__enter__()
            if stats is not None:
                stats.held_sessions += 1
        elif config.prefer_held_weights:
            try:
                held = HeldMLP(block.mlp, x[:1])
                held.__enter__()
                if stats is not None:
                    stats.held_sessions += 1
            except UnsafeHeldWeights as exc:
                held_error = str(exc)
                held = None
            except Exception as exc:
                held = None
                if config.strict:
                    raise
                held_error = "%s: %s" % (type(exc).__name__, exc)

        if held_error is not None:
            if stats is not None:
                stats.record_fallback(held_error)
            logging.warning(
                "%s block %d using ordinary module calls: %s",
                LOG_PREFIX,
                layer_index,
                held_error,
            )

        try:
            for chunk_index, chunk in enumerate(chunks):
                if stats is not None:
                    stats.record_chunk(chunk.rows)
                notify_activation(
                    "mlp_chunk_enter",
                    layer_index,
                    transformer_options,
                    chunk_index=chunk_index,
                    start=chunk.start,
                    stop=chunk.stop,
                    mod_row=chunk.mod_row,
                )
                with timed_stage(transformer_options, "norm2_modulation"):
                    h = block.norm2(x[chunk.start : chunk.stop])
                    _scale_shift(
                        h, shift_mlp[chunk.mod_row], scale_mlp[chunk.mod_row]
                    )

                expanded = None
                if config.convrot_2slice:
                    out, path = held.fc1_fc2(
                        h,
                        stage_factory=lambda name: timed_stage(
                            transformer_options, name
                        ),
                    )
                elif held is not None:
                    with timed_stage(transformer_options, "mlp_fc1"):
                        expanded = held.fc1(h)
                    with timed_stage(transformer_options, "mlp_swiglu_fc2"):
                        out, path = held.fc2_swiglu(
                            expanded, native=config.native_swiglu
                        )
                else:
                    with timed_stage(transformer_options, "mlp_fc1"):
                        expanded = module_fc1(block.mlp, h)
                    with timed_stage(transformer_options, "mlp_swiglu_fc2"):
                        out, path = module_swiglu_fc2(
                            block.mlp, expanded, native=config.native_swiglu
                        )

                if stats is not None:
                    stats.record_path(path)
                notify_activation(
                    "mlp_fc2_ready",
                    layer_index,
                    transformer_options,
                    chunk_index=chunk_index,
                    path=path,
                    output_shape=tuple(out.shape),
                )
                with timed_stage(transformer_options, "final_mlp_gate"):
                    _gate_add(
                        x[chunk.start : chunk.stop],
                        out,
                        gate_mlp[chunk.mod_row],
                    )
                del h, out, expanded
                notify_activation(
                    "mlp_chunk_gated",
                    layer_index,
                    transformer_options,
                    chunk_index=chunk_index,
                )
        finally:
            if held is not None:
                held.__exit__(None, None, None)

        notify_activation(
            "block_exit",
            layer_index,
            transformer_options,
            chunks=len(chunks),
        )
        return x

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        with timed_stage(transformer_options, "total_dit_block"):
            return _forward(
                x, t_emb, mod_segments, rope_freqs,
                transformer_options=transformer_options,
            )

    forward._h3_activation_memory = True
    forward._h3_activation_config = config.signature
    forward._h3_activation_layer_index = layer_index
    forward._h3_activation_original = original_forward
    return forward
