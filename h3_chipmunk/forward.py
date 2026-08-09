from __future__ import annotations

import torch
import comfy.model_management

from ..h3_activation_memory.chunks import iter_mod_chunks, validate_mod_segments
from ..h3_activation_memory.linear import ConvRotTwoSliceMLP
from ..h3_runtime.context import get_runtime_snapshot
from ..h3_runtime.timing import timed_stage
from .executor import run_chipmunk_chunk


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

        chunks = tuple(iter_mod_chunks(
            segments, x.shape[0], config.chunk_rows,
            alignment=min(256, config.chunk_rows), mod_rows=shift_mlp.shape[0],
        ))

        held = ConvRotTwoSliceMLP(block.mlp, x[:1])
        held.__enter__()
        try:
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

                # Do not publish a Chipmunk-only timing stage through the shared
                # request timer yet. Hybrid Sparse intentionally validates a
                # closed timing-stage schema, and the full Chipmunk selector /
                # delta work is already included by total_dit_block. The exact
                # dense MLP sub-operations continue to use mlp_fc1 and
                # mlp_swiglu_fc2 above. Add dedicated shared timing stages only
                # when the production sparse kernels have a stable breakdown.
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
