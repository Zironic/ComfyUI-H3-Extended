from __future__ import annotations

import torch
import comfy.model_management

from ..h3_activation_memory.chunks import iter_mod_chunks, validate_mod_segments
from ..h3_activation_memory.linear import ConvRotTwoSliceMLP
from ..h3_runtime.context import get_runtime_snapshot
from ..h3_runtime.timing import timed_stage
from .executor import run_chipmunk_chunk
from .selector import logical_swiglu


def _scale_shift(h, shift, scale):
    return h.mul_(1.0 + scale.to(h.dtype)).add_(shift.to(h.dtype))


def _gate_add(x, other, gate):
    return x.addcmul_(other, gate.to(x.dtype))


def make_forward(block, layer_index, config, session, original_forward=None):
    original_forward = original_forward or block.forward

    def _forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if comfy.model_management.in_training:
            raise RuntimeError("H3 Chipmunk is inference-only")
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            if config.strict:
                raise RuntimeError("H3 Chipmunk requires the H3 runtime layout wrapper")
            return original_forward(x, t_emb, mod_segments, rope_freqs, transformer_options)

        with timed_stage(transformer_options, "adaln_proj"):
            shifts = block.adaln_proj(t_emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = shifts
        segments = validate_mod_segments(mod_segments, x.shape[0], mod_rows=shift_msa.shape[0])

        with timed_stage(transformer_options, "norm1_modulation"):
            h = block.norm1(x)
            for start, stop, row in segments:
                _scale_shift(h[start:stop], shift_msa[row], scale_msa[row])
        attn_out = block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
        with timed_stage(transformer_options, "attention_residual_gate"):
            for start, stop, row in segments:
                _gate_add(x[start:stop], attn_out[start:stop], gate_msa[row])
        del h, attn_out

        chunk_rows = int(config.effective_chunk_rows)
        chunks = tuple(iter_mod_chunks(
            segments, x.shape[0], chunk_rows,
            alignment=min(256, chunk_rows), mod_rows=shift_mlp.shape[0],
        ))

        held = ConvRotTwoSliceMLP(block.mlp, x[:1])
        held.__enter__()
        try:
            def measure_activation_runner(value):
                """Compute logical SwiGLU activations from already-held fc1 tiles."""
                pieces = []
                for tile in held.tiles:
                    expanded = held.convrot_linear(
                        value,
                        tile["fc1_weight"],
                        tile["fc1_scale"],
                        input_act=None,
                    )
                    pieces.append(logical_swiglu(expanded))
                return torch.cat(pieces, dim=-1)

            def shadow_fc2_runner(selected_activation, logical_indices):
                """Project selected logical features using already-held fc2 tiles.

                Unselected logical features are represented as zeros. This avoids
                reacquiring/staging fc2 for every sampled shadow window while
                preserving complete ConvRot-256 groups and the same two-slice
                weight representation used by the exact dense path.
                """
                tile_widths = [int(tile["fc2_weight"].shape[1]) for tile in held.tiles]
                ffn = sum(tile_widths)
                full = selected_activation.new_zeros((selected_activation.shape[0], ffn))
                full.index_copy_(1, logical_indices.long(), selected_activation)
                result = None
                offset = 0
                for tile, width in zip(held.tiles, tile_widths):
                    partial = held.convrot_linear(
                        full[:, offset:offset + width].contiguous(),
                        tile["fc2_weight"],
                        tile["fc2_scale"],
                        input_act=None,
                    )
                    if result is None:
                        result = partial
                    else:
                        result.add_(partial)
                    offset += width
                return result

            for chunk_index, chunk in enumerate(chunks):
                with timed_stage(transformer_options, "norm2_modulation"):
                    h = block.norm2(x[chunk.start:chunk.stop])
                    _scale_shift(h, shift_mlp[chunk.mod_row], scale_mlp[chunk.mod_row])

                def dense_runner(value):
                    out, _path = held.fc1_fc2(
                        value,
                        stage_factory=lambda name: timed_stage(transformer_options, name),
                    )
                    return out

                # Chipmunk-specific work remains inside total_dit_block rather
                # than adding private stage names to Hybrid Sparse's shared timer.
                out, path = run_chipmunk_chunk(
                    block=block,
                    h=h,
                    layer_index=layer_index,
                    chunk_index=chunk_index,
                    chunk_start=chunk.start,
                    chunk_stop=chunk.stop,
                    snapshot=snapshot,
                    session=session,
                    config=config,
                    dense_runner=dense_runner,
                    measure_activation_runner=measure_activation_runner,
                    shadow_fc2_runner=shadow_fc2_runner,
                    # At this point x is the exact post-attention residual and
                    # has not yet received the MLP gate. shadow_validate uses it
                    # only to score post-block error, never to alter execution.
                    residual_base=x[chunk.start:chunk.stop],
                    mlp_gate=gate_mlp[chunk.mod_row],
                )
                with timed_stage(transformer_options, "final_mlp_gate"):
                    _gate_add(x[chunk.start:chunk.stop], out, gate_mlp[chunk.mod_row])
                del h, out
        finally:
            held.__exit__(None, None, None)
        return x

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        with timed_stage(transformer_options, "total_dit_block"):
            return _forward(x, t_emb, mod_segments, rope_freqs, transformer_options)

    forward._h3_chipmunk = True
    forward._h3_chipmunk_config = config.signature
    forward._h3_chipmunk_layer_index = int(layer_index)
    forward._h3_chipmunk_original = original_forward
    return forward
