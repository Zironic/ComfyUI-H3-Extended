"""Eager H3 layer dispatch for the shared compiled tensor block."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

import torch

import comfy.ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

try:
    from ..h3_activation_memory.chunks import iter_mod_chunks, validate_mod_segments
    from ..h3_activation_memory.config import MODE_CONVROT_2SLICE
    from ..h3_activation_memory.linear import acquire_linear
    from ..h3_activation_memory.observer import OBSERVER_KEY as ACTIVATION_OBSERVER_KEY
    from ..h3_activation_memory.stats import get_stats
    from ..h3_attention.hybrid.config import MODE_SAGE128_FUSED_QKV
    from ..h3_attention.observer import OBSERVER_KEY as ATTENTION_OBSERVER_KEY
except ImportError:
    from h3_activation_memory.chunks import iter_mod_chunks, validate_mod_segments
    from h3_activation_memory.config import MODE_CONVROT_2SLICE
    from h3_activation_memory.linear import acquire_linear
    from h3_activation_memory.observer import OBSERVER_KEY as ACTIVATION_OBSERVER_KEY
    from h3_activation_memory.stats import get_stats
    from h3_attention.hybrid.config import MODE_SAGE128_FUSED_QKV
    from h3_attention.observer import OBSERVER_KEY as ATTENTION_OBSERVER_KEY

from .block_compile import (
    BlockCarriers,
    BlockTopology,
    H3BlockError,
    make_compiled_block,
)
from .context import get_runtime_snapshot
from .timing import timed_stage


LOG_PREFIX = "[H3 compile]"
DISPATCHER_KEY = "minimax_h3_shared_block_dispatcher"
BLOCKS_ATTR = "diffusion_model.blocks"


def _tensor_meta(tensor):
    return (
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        tensor.dtype,
    )


def _quantized_parts(weight, name):
    if not isinstance(weight, QuantizedTensor):
        raise H3BlockError("%s must be a TensorWiseINT8 quantized weight" % name)
    if getattr(weight, "_layout_cls", None) != "TensorWiseINT8Layout":
        raise H3BlockError("%s must use TensorWiseINT8Layout" % name)
    params = getattr(weight, "_params", None)
    if getattr(params, "transposed", False):
        raise H3BlockError("%s must not be transposed" % name)
    if not getattr(params, "convrot", False):
        raise H3BlockError("%s must use ConvRot" % name)
    if int(getattr(params, "convrot_groupsize", 0)) != 256:
        raise H3BlockError("%s must use ConvRot group size 256" % name)
    qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
    if qdata.dtype != torch.int8 or scale.dtype != torch.float32:
        raise H3BlockError("%s has invalid ConvRot carrier dtypes" % name)
    if not qdata.is_contiguous() or not scale.is_contiguous():
        raise H3BlockError("%s ConvRot carriers must be contiguous" % name)
    return qdata, scale


def _plain_weight(weight, name):
    if not torch.is_tensor(weight) or isinstance(weight, QuantizedTensor):
        raise H3BlockError("%s must be a plain tensor weight" % name)
    if not weight.is_contiguous():
        raise H3BlockError("%s must be contiguous" % name)
    return weight


def block_execution_signature(block):
    """Return the layer-independent module/weight topology of one raw block."""

    adaln = block.adaln_proj
    if int(adaln.expand) != 6 or int(adaln.modalities) != 3:
        raise H3BlockError("H3 shared compilation requires 6x3 AdaLN")
    if float(block.norm1.eps) != float(block.norm2.eps):
        raise H3BlockError("H3 shared compilation requires matching block norm eps")
    if float(block.attn.q_norm.eps) != float(block.attn.k_norm.eps):
        raise H3BlockError("H3 shared compilation requires matching Q/K norm eps")

    qkv = _quantized_parts(block.attn.qkv_proj.weight, "qkv_proj")
    out = _quantized_parts(block.attn.out_proj.weight, "out_proj")
    fc1 = _quantized_parts(block.mlp.fc1.weight, "mlp.fc1")
    fc2 = _quantized_parts(block.mlp.fc2.weight, "mlp.fc2")
    plain = (
        _plain_weight(adaln.linear.weight, "adaln_proj.linear"),
        _plain_weight(adaln.linear.bias, "adaln_proj.linear.bias"),
        _plain_weight(block.norm1.weight, "norm1"),
        _plain_weight(block.norm2.weight, "norm2"),
        _plain_weight(block.attn.q_norm.weight, "q_norm"),
        _plain_weight(block.attn.k_norm.weight, "k_norm"),
    )
    return (
        int(block.attn.heads),
        int(block.attn.head_dim),
        float(block.norm1.eps),
        float(block.attn.q_norm.eps),
        int(adaln.expand),
        int(adaln.modalities),
        int(adaln.hidden),
        bool(adaln.apply_silu),
        int(adaln.linear.in_features),
        int(adaln.linear.out_features),
        int(block.mlp.fc2.in_features),
        tuple(_tensor_meta(value) for value in plain),
        tuple(_tensor_meta(value) for pair in (qkv, out, fc1, fc2) for value in pair),
        ("TensorWiseINT8Layout", False, True, 256),
    )


def validate_identical_blocks(blocks):
    if not blocks:
        raise H3BlockError("H3 model has no main blocks")
    expected = block_execution_signature(blocks[0])
    for index, block in enumerate(blocks[1:], 1):
        actual = block_execution_signature(block)
        if actual != expected:
            raise H3BlockError(
                "H3 block %d does not match the shared block signature" % index
            )
    return expected


@dataclass
class _Lease:
    block: object
    x: torch.Tensor
    t_emb: torch.Tensor

    def __post_init__(self):
        self._held = []
        self.carriers = None

    def _acquire(self, module, sample, name, *, bias):
        acquired = acquire_linear(module, sample)
        self._held.append(acquired)
        if bias and acquired.bias is None:
            raise H3BlockError("%s requires a bias" % name)
        if not bias and acquired.bias is not None:
            raise H3BlockError("%s must not have a bias" % name)
        return acquired

    def __enter__(self):
        comfy.ops.run_every_op()
        try:
            block = self.block
            adaln = self._acquire(
                block.adaln_proj.linear, self.t_emb, "adaln_proj.linear", bias=True
            )
            norm1 = self._acquire(block.norm1, self.x, "norm1", bias=False)
            norm2 = self._acquire(block.norm2, self.x, "norm2", bias=False)
            qkv = self._acquire(block.attn.qkv_proj, self.x, "qkv_proj", bias=False)
            q_norm = self._acquire(block.attn.q_norm, self.x, "q_norm", bias=False)
            k_norm = self._acquire(block.attn.k_norm, self.x, "k_norm", bias=False)
            attention_inner = int(block.attn.heads) * int(block.attn.head_dim)
            out_sample = self.x.new_empty((1, attention_inner))
            out = self._acquire(block.attn.out_proj, out_sample, "out_proj", bias=False)
            fc1 = self._acquire(block.mlp.fc1, self.x, "mlp.fc1", bias=False)
            fc2_sample = self.x.new_empty((1, int(block.mlp.fc2.in_features)))
            fc2 = self._acquire(block.mlp.fc2, fc2_sample, "mlp.fc2", bias=False)

            qkv_qdata, qkv_scale = _quantized_parts(qkv.weight, "qkv_proj")
            out_qdata, out_scale = _quantized_parts(out.weight, "out_proj")
            fc1_qdata, fc1_scale = _quantized_parts(fc1.weight, "mlp.fc1")
            fc2_qdata, fc2_scale = _quantized_parts(fc2.weight, "mlp.fc2")
            self.carriers = BlockCarriers(
                _plain_weight(adaln.weight, "adaln_proj.linear"),
                _plain_weight(adaln.bias, "adaln_proj.linear.bias"),
                _plain_weight(norm1.weight, "norm1"),
                _plain_weight(norm2.weight, "norm2"),
                qkv_qdata,
                qkv_scale,
                _plain_weight(q_norm.weight, "q_norm"),
                _plain_weight(k_norm.weight, "k_norm"),
                out_qdata,
                out_scale,
                fc1_qdata,
                fc1_scale,
                fc2_qdata,
                fc2_scale,
            )
            return self
        except Exception:
            self.release()
            raise

    def release(self, guard=None):
        while self._held:
            self._held.pop().release(guard=guard)
        self.carriers = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


class SharedBlockDispatcher:
    def __init__(self, blocks, backend, activation_config, compile_backend):
        self.blocks = tuple(blocks)
        self.backend = backend
        self.activation_config = activation_config
        self.compile_backend = compile_backend
        self.block_signature = validate_identical_blocks(self.blocks)
        self._variants = {}
        self._compile_counts = {}

    def _topology(self, block, x, t_emb, mod_segments, rope_freqs, layout):
        if rope_freqs is None:
            raise H3BlockError("shared H3 compilation requires RoPE")
        mod_rows = int(t_emb.shape[0]) * int(block.adaln_proj.modalities)
        segments = tuple(
            tuple(int(value) for value in segment)
            for segment in validate_mod_segments(
                mod_segments, int(x.shape[0]), mod_rows=mod_rows
            )
        )
        chunks = tuple(
            (int(chunk.start), int(chunk.stop), int(chunk.mod_row))
            for chunk in iter_mod_chunks(
                segments,
                int(x.shape[0]),
                self.activation_config.chunk_rows,
                alignment=self.activation_config.alignment,
                mod_rows=mod_rows,
            )
        )
        geometry = self.backend.router.geometry(layout)
        return BlockTopology(
            hidden_size=int(x.shape[1]),
            ffn_size=int(block.mlp.fc2.in_features),
            timestep_dim=int(t_emb.shape[-1]),
            heads=int(block.attn.heads),
            head_dim=int(block.attn.head_dim),
            norm_eps=float(block.norm1.eps),
            qk_norm_eps=float(block.attn.q_norm.eps),
            adaln_apply_silu=bool(block.adaln_proj.apply_silu),
            adaln_out_features=int(block.adaln_proj.linear.out_features),
            adaln_expand=int(block.adaln_proj.expand),
            adaln_modalities=int(block.adaln_proj.modalities),
            rope_strides=(
                int(rope_freqs.stride(1)),
                int(rope_freqs.stride(3)),
                int(rope_freqs.stride(4)),
                int(rope_freqs.stride(5)),
            ),
            has_rope=True,
            router_geometry=geometry,
            video_budget=float(self.backend.config.video_budget),
            mod_segments=segments,
            mlp_chunks=chunks,
            backend_config=self.backend.config,
            activation_config=self.activation_config,
        )

    def _variant(self, topology, carriers):
        key = topology.signature
        variant = self._variants.get(key)
        if variant is not None:
            return variant
        counter = {"count": 0}

        def backend(graph_module, example_inputs):
            counter["count"] += 1
            self._compile_counts[key] = counter["count"]
            if counter["count"] > 1:
                raise H3BlockError(
                    "shared H3 block recompiled for one static topology"
                )
            return self.compile_backend(graph_module, example_inputs)

        variant = make_compiled_block(topology, carriers, backend=backend)
        self._variants[key] = variant
        return variant

    def _record(self, layer_index, topology, snapshot):
        collector = self.backend.collector
        if collector is None:
            return
        geometry = topology.router_geometry
        retained = min(
            geometry.pure_video_kv_tiles,
            max(
                1,
                math.ceil(
                    float(topology.video_budget) * geometry.pure_video_kv_tiles
                ),
            ),
        )
        metadata = self.backend.router._metadata(
            geometry, topology.video_budget, retained
        ).as_dict()
        metadata.update({
            "request_id": int(snapshot.request_id),
            "step": int(snapshot.step_index),
            "total_steps": int(snapshot.total_steps),
            "branch": [int(value) for value in snapshot.branch],
            "layer": int(layer_index),
            "dense_sage_heads": 0,
            "sparse_sage_heads": int(topology.heads),
            "sol_heads": 0,
            "flex_fallback_tiles": 0,
            "total_128q_video_tiles": (
                int(geometry.pure_video_q_tiles) * int(topology.heads)
            ),
            "qkv_projection": "fused_int8",
            "smooth_k": False,
        })
        collector.record(metadata)

    def __call__(
        self,
        layer_index,
        block,
        x,
        t_emb,
        mod_segments,
        rope_freqs,
        transformer_options,
    ):
        if transformer_options.get(ACTIVATION_OBSERVER_KEY):
            raise H3BlockError(
                "activation observers are incompatible with shared H3 compilation"
            )
        if transformer_options.get(ATTENTION_OBSERVER_KEY):
            raise H3BlockError(
                "attention observers are incompatible with shared H3 compilation"
            )
        blocks_replace = transformer_options.get("patches_replace", {}).get("dit", {})
        if ("double_block", int(layer_index)) in blocks_replace:
            raise H3BlockError(
                "per-layer block replacement is incompatible with shared H3 compilation"
            )
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None or not snapshot.valid_layout:
            raise H3BlockError("shared H3 compilation requires a valid runtime layout")
        topology = self._topology(
            block, x, t_emb, mod_segments, rope_freqs, snapshot.layout
        )
        result = None
        with timed_stage(transformer_options, "total_dit_block"):
            lease = _Lease(block, x, t_emb)
            lease.__enter__()
            try:
                variant = self._variant(topology, lease.carriers)
                result = variant(x, t_emb, rope_freqs, lease.carriers)
            finally:
                lease.release(guard=result)

        stats = get_stats(transformer_options, self.activation_config)
        stats.blocks += 1
        stats.held_sessions += 1
        for start, stop, _row in topology.mlp_chunks:
            stats.record_chunk(stop - start)
            stats.record_path("held_convrot_2slice")
        self._record(layer_index, topology, snapshot)
        return result


def make_shared_forward(block, layer_index, dispatcher, original_forward):
    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        return dispatcher(
            layer_index,
            block,
            x,
            t_emb,
            mod_segments,
            rope_freqs,
            transformer_options,
        )

    forward._h3_activation_memory = True
    forward._h3_activation_config = dispatcher.activation_config.signature
    forward._h3_activation_layer_index = int(layer_index)
    forward._h3_activation_original = original_forward
    forward._h3_shared_block_compile = True
    forward._h3_shared_dispatcher = dispatcher
    return forward


def install_shared_block_dispatch(
    model_patcher,
    backend,
    activation_config,
    compile_backend,
):
    if getattr(backend, "name", None) != "hybrid_sparse":
        raise H3BlockError("shared H3 compilation requires Hybrid Sparse attention")
    if backend.config.mode != MODE_SAGE128_FUSED_QKV:
        raise H3BlockError("shared H3 compilation requires sage128_fused_qkv")
    if getattr(backend, "projector", None) is None:
        raise H3BlockError("shared H3 compilation requires fused QKV projection")
    if not getattr(getattr(backend, "executor", None), "_use_sparse_sage_op", False):
        raise H3BlockError("shared H3 compilation requires the Sparse Sage custom op")
    if activation_config is None or activation_config.mode != MODE_CONVROT_2SLICE:
        raise H3BlockError(
            "shared H3 compilation requires mlp_chunked_convrot_2slice"
        )
    blocks = tuple(model_patcher.get_model_object(BLOCKS_ATTR))
    if len(blocks) != 50:
        raise H3BlockError(
            "shared H3 compilation expected 50 main blocks; got %d" % len(blocks)
        )
    existing = model_patcher.object_patches
    for index in range(len(blocks)):
        block_key = "%s.%d.forward" % (BLOCKS_ATTR, index)
        attention_key = "%s.%d.attn.forward" % (BLOCKS_ATTR, index)
        block_forward = existing.get(block_key)
        attention_forward = existing.get(attention_key)
        if not getattr(block_forward, "_h3_activation_memory", False):
            raise H3BlockError("block %d has no H3 activation-memory patch" % index)
        if not getattr(attention_forward, "_h3_attention", False):
            raise H3BlockError("block %d has no H3 attention patch" % index)
        if getattr(block_forward, "_h3_activation_config", None) != activation_config.signature:
            raise H3BlockError("block %d activation topology differs" % index)
        if getattr(attention_forward, "_h3_backend", None) != "hybrid_sparse":
            raise H3BlockError("block %d attention topology differs" % index)
        if getattr(attention_forward, "_h3_projector", None) != "h3_fused_qkv":
            raise H3BlockError("block %d fused QKV topology differs" % index)

    dispatcher = SharedBlockDispatcher(
        blocks, backend, activation_config, compile_backend
    )
    for index, block in enumerate(blocks):
        key = "%s.%d.forward" % (BLOCKS_ATTR, index)
        original = existing[key]
        model_patcher.add_object_patch(
            key, make_shared_forward(block, index, dispatcher, original)
        )
    model_patcher.model_options[DISPATCHER_KEY] = dispatcher
    logging.info(
        "%s installed one shared tensor program for %d H3 main blocks",
        LOG_PREFIX,
        len(blocks),
    )
    return dispatcher


__all__ = [
    "H3BlockError",
    "block_execution_signature",
    "validate_identical_blocks",
    "SharedBlockDispatcher",
    "install_shared_block_dispatch",
]
