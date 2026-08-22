from __future__ import annotations

import torch
import comfy.model_management

from ..h3_activation_memory.chunks import iter_mod_chunks, validate_mod_segments
from ..h3_activation_memory.linear import ConvRotTwoSliceMLP
from ..h3_runtime.context import get_runtime_snapshot
from ..h3_runtime.timing import timed_stage
from .executor import run_chipmunk_chunk, prefetch_chipmunk_chunk
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
        segments = validate_mod_segments(
            mod_segments,
            x.shape[0],
            mod_rows=shift_msa.shape[0],
        )

        with timed_stage(transformer_options, "norm1_modulation"):
            h = block.norm1(x)
            for start, stop, row in segments:
                _scale_shift(h[start:stop], shift_msa[row], scale_msa[row])
        attn_out = block.attn(
            h,
            rope_freqs=rope_freqs,
            transformer_options=transformer_options,
        )
        with timed_stage(transformer_options, "attention_residual_gate"):
            for start, stop, row in segments:
                _gate_add(x[start:stop], attn_out[start:stop], gate_msa[row])
        del h, attn_out

        chunk_rows = int(config.effective_chunk_rows)
        chunks = tuple(
            iter_mod_chunks(
                segments,
                x.shape[0],
                chunk_rows,
                alignment=min(256, chunk_rows),
                mod_rows=shift_mlp.shape[0],
            )
        )

        held = ConvRotTwoSliceMLP(block.mlp, x[:1])
        held.__enter__()
        try:
            if held.tiles is None or len(held.tiles) != 2:
                raise RuntimeError("Chipmunk requires exactly two held ConvRot MLP tiles")
            tile_ffn = int(held.tiles[0]["fc2_weight"].shape[1])
            if any(int(tile["fc2_weight"].shape[1]) != tile_ffn for tile in held.tiles):
                raise RuntimeError("Chipmunk requires equal-width ConvRot fc2 tiles")

            def full_activation_runner(value):
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

            def selected_activation_runner(value, logical_indices):
                # The balanced selector emits equal group counts from the two
                # 7168-wide logical feature tiles. Split by static tensor shape,
                # never by a CUDA scalar.
                half = int(logical_indices.shape[0]) // 2
                local_indices = (
                    logical_indices[:half].long(),
                    logical_indices[half:].long() - tile_ffn,
                )
                pieces = []
                for tile, local in zip(held.tiles, local_indices):
                    rows = torch.cat((local, local + tile_ffn), dim=0)
                    q = tile["fc1_weight"].index_select(0, rows).contiguous()
                    scale = tile["fc1_scale"]
                    if scale.numel() != 1:
                        scale = scale.index_select(0, rows).contiguous()
                    expanded = held.convrot_linear(
                        value,
                        q,
                        scale,
                        input_act=None,
                    )
                    pieces.append(logical_swiglu(expanded))
                return torch.cat(pieces, dim=-1)

            def selected_fc2_runner(selected_activation, logical_indices):
                half_indices = int(logical_indices.shape[0]) // 2
                half_activation = int(selected_activation.shape[-1]) // 2
                local_indices = (
                    logical_indices[:half_indices].long(),
                    logical_indices[half_indices:].long() - tile_ffn,
                )
                activations = (
                    selected_activation[:, :half_activation],
                    selected_activation[:, half_activation:],
                )
                result = None
                for tile, activation, local in zip(
                    held.tiles,
                    activations,
                    local_indices,
                ):
                    q = tile["fc2_weight"].index_select(1, local).contiguous()
                    partial = held.convrot_linear(
                        activation,
                        q,
                        tile["fc2_scale"],
                        input_act=None,
                    )
                    if result is None:
                        result = partial
                    else:
                        result.add_(partial)
                return result

            def queue_prefetch(index):
                if index < 0 or index >= len(chunks):
                    return
                chunk = chunks[index]
                prefetch_chipmunk_chunk(
                    block=block,
                    layer_index=layer_index,
                    chunk_index=index,
                    chunk_start=chunk.start,
                    chunk_stop=chunk.stop,
                    snapshot=snapshot,
                    session=session,
                    config=config,
                    device=x.device,
                )

            # Two staging slots let the transfer stream prepare the next chunk
            # while the current chunk is running. Nothing here waits on the CPU.
            queue_prefetch(0)
            queue_prefetch(1)

            for chunk_index, chunk in enumerate(chunks):
                with timed_stage(transformer_options, "norm2_modulation"):
                    h = block.norm2(x[chunk.start:chunk.stop])
                    _scale_shift(
                        h,
                        shift_mlp[chunk.mod_row],
                        scale_mlp[chunk.mod_row],
                    )

                def dense_runner(value):
                    out, _path = held.fc1_fc2(
                        value,
                        stage_factory=lambda name: timed_stage(
                            transformer_options,
                            name,
                        ),
                    )
                    return out

                out, _path = run_chipmunk_chunk(
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
                    full_activation_runner=full_activation_runner,
                    selected_activation_runner=selected_activation_runner,
                    selected_fc2_runner=selected_fc2_runner,
                )
                with timed_stage(transformer_options, "final_mlp_gate"):
                    _gate_add(
                        x[chunk.start:chunk.stop],
                        out,
                        gate_mlp[chunk.mod_row],
                    )
                del h, out

                # Refill the slot released by the current chunk two chunks ahead.
                queue_prefetch(chunk_index + 2)
        finally:
            held.__exit__(None, None, None)
        return x

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        with timed_stage(transformer_options, "total_dit_block"):
            return _forward(
                x,
                t_emb,
                mod_segments,
                rope_freqs,
                transformer_options,
            )

    forward._h3_chipmunk = True
    forward._h3_chipmunk_config = config.signature
    forward._h3_chipmunk_layer_index = int(layer_index)
    forward._h3_chipmunk_original = original_forward
    return forward
