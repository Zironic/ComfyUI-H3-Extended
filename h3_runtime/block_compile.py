"""Shared, module-free tensor body for the MiniMax H3 main block.

The callable in this module deliberately receives only tensors.  A frozen
``BlockTopology`` supplies the shape/layout facts which are common to all 50
DiT block bindings; weights are passed as a stable ``BlockCarriers`` tuple.
This keeps layer identity and Comfy/AIMDO objects outside the Dynamo graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

import comfy.model_management

try:
    from ..h3_activation_memory.config import ActivationMemoryConfig, MODE_CONVROT_2SLICE
    from ..h3_activation_memory.linear import (
        _convrot_fc1_tiles_op,
        _convrot_fc2_tiles_op,
        _convrot_linear_op,
    )
    from ..h3_attention.hybrid.config import HybridSparseConfig, MODE_SAGE128_FUSED_QKV
    from ..h3_attention.hybrid.fused_qkv import fused_qkv_op
    from ..h3_attention.hybrid.router import SparseTileGeometry, sort_selected_indices
    from ..h3_attention.hybrid.sparse_sage import (
        prepare_sparse_sage_v_op,
        sparse_sage_attention_op,
    )
except ImportError:
    from h3_activation_memory.config import ActivationMemoryConfig, MODE_CONVROT_2SLICE
    from h3_activation_memory.linear import (
        _convrot_fc1_tiles_op,
        _convrot_fc2_tiles_op,
        _convrot_linear_op,
    )
    from h3_attention.hybrid.config import HybridSparseConfig, MODE_SAGE128_FUSED_QKV
    from h3_attention.hybrid.fused_qkv import fused_qkv_op
    from h3_attention.hybrid.router import SparseTileGeometry, sort_selected_indices
    from h3_attention.hybrid.sparse_sage import (
        prepare_sparse_sage_v_op,
        sparse_sage_attention_op,
    )


class H3BlockError(ValueError):
    """Invalid H3 block topology or carrier metadata."""


class BlockCarriers(NamedTuple):
    """Stable tensor order consumed by :func:`make_compiled_block`.

    No module, layer id, registry token, runtime options, collector, or timing
    value belongs in this tuple.  The order is part of the integration
    contract and is intentionally explicit rather than inferred from a state
    dict.
    """

    adaln_weight: torch.Tensor
    adaln_bias: torch.Tensor
    norm1_weight: torch.Tensor
    norm2_weight: torch.Tensor
    qkv_qdata: torch.Tensor
    qkv_scale: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    out_qdata: torch.Tensor
    out_scale: torch.Tensor
    fc1_qdata: torch.Tensor
    fc1_scale: torch.Tensor
    fc2_qdata: torch.Tensor
    fc2_scale: torch.Tensor


def carrier_tuple(carriers) -> tuple[torch.Tensor, ...]:
    """Normalize a carrier object while preserving the public tuple order."""

    if isinstance(carriers, BlockCarriers):
        values = tuple(carriers)
    elif isinstance(carriers, (tuple, list)):
        values = tuple(carriers)
    else:
        raise H3BlockError("block carriers must be a 14-tensor tuple")
    if len(values) != len(BlockCarriers._fields):
        raise H3BlockError(
            "block carriers must contain exactly 14 tensors; got %d" % len(values)
        )
    if not all(torch.is_tensor(value) for value in values):
        raise H3BlockError("block carriers may contain tensors only")
    return values


@dataclass(frozen=True)
class BlockTopology:
    """Static facts shared by every H3 main block binding."""

    hidden_size: int
    ffn_size: int
    timestep_dim: int
    heads: int
    head_dim: int
    norm_eps: float
    qk_norm_eps: float
    adaln_apply_silu: bool
    adaln_out_features: int
    adaln_expand: int = 6
    adaln_modalities: int = 3
    convrot_layout: str = "TensorWiseINT8Layout"
    convrot_groupsize: int = 256
    convrot_transposed: bool = False
    rope_strides: tuple[int, int, int, int] = (0, 0, 0, 0)
    has_rope: bool = True
    router_geometry: SparseTileGeometry | None = None
    video_budget: float = 0.5
    mod_segments: tuple[tuple[int, int, int], ...] = ()
    mlp_chunks: tuple[tuple[int, int, int], ...] = ()
    backend_config: HybridSparseConfig = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV, video_budget=0.5
    )
    activation_config: ActivationMemoryConfig = ActivationMemoryConfig(
        mode=MODE_CONVROT_2SLICE
    )

    def __post_init__(self):
        ints = {
            "hidden_size": self.hidden_size,
            "ffn_size": self.ffn_size,
            "timestep_dim": self.timestep_dim,
            "heads": self.heads,
            "head_dim": self.head_dim,
            "adaln_out_features": self.adaln_out_features,
        }
        if any(int(value) <= 0 for value in ints.values()):
            raise H3BlockError("H3 block dimensions must be positive: %s" % ints)
        if int(self.adaln_expand) != 6 or int(self.adaln_modalities) != 3:
            raise H3BlockError("H3 AdaLN requires expand=6 and modalities=3")
        if int(self.adaln_out_features) != self.adaln_expand * self.hidden_size * self.adaln_modalities:
            raise H3BlockError("AdaLN projection has an invalid output width")
        if self.convrot_layout != "TensorWiseINT8Layout" or int(self.convrot_groupsize) != 256:
            raise H3BlockError("H3 ConvRot requires TensorWiseINT8Layout with group size 256")
        if self.convrot_transposed:
            raise H3BlockError("H3 ConvRot carriers must use non-transposed weights")
        if len(tuple(self.rope_strides)) != 4:
            raise H3BlockError("rope_strides must contain four integers")
        if self.backend_config.mode != MODE_SAGE128_FUSED_QKV:
            raise H3BlockError("shared block requires sage128_fused_qkv backend")
        if self.activation_config.mode != MODE_CONVROT_2SLICE:
            raise H3BlockError("shared block requires mlp_chunked_convrot_2slice")
        if float(self.video_budget) != float(self.backend_config.video_budget):
            raise H3BlockError("video_budget disagrees with backend configuration")
        if self.router_geometry is None:
            raise H3BlockError("router_geometry is required")
        _validate_ranges(self.mod_segments, self.router_geometry.sequence)
        _validate_ranges(self.mlp_chunks, self.router_geometry.sequence)

    @property
    def signature(self) -> tuple:
        geometry = self.router_geometry
        return (
            int(self.hidden_size), int(self.ffn_size), int(self.timestep_dim),
            int(self.heads), int(self.head_dim), float(self.norm_eps),
            float(self.qk_norm_eps), bool(self.adaln_apply_silu),
            int(self.adaln_out_features), int(self.adaln_expand), int(self.adaln_modalities),
            str(self.convrot_layout), int(self.convrot_groupsize), bool(self.convrot_transposed),
            tuple(int(x) for x in self.rope_strides),
            bool(self.has_rope), geometry.signature, int(geometry.sequence),
            int(geometry.q_tiles), int(geometry.kv_tiles),
            int(geometry.pure_video_q_start), int(geometry.pure_video_kv_start),
            float(self.video_budget), tuple(self.mod_segments), tuple(self.mlp_chunks),
            self.backend_config.signature if hasattr(self.backend_config, "signature") else (
                self.backend_config.mode, float(self.backend_config.video_budget),
                bool(self.backend_config.strict),
            ),
            self.activation_config.signature,
        )


def _validate_ranges(ranges, sequence):
    previous = 0
    for item in ranges:
        if len(item) != 3:
            raise H3BlockError("modulation/chunk ranges must be (start, stop, row)")
        start, stop, row = (int(value) for value in item)
        if start != previous or not (start < stop <= int(sequence)) or row < 0:
            raise H3BlockError("ranges must be contiguous and cover the packed sequence")
        previous = stop
    if previous != int(sequence):
        raise H3BlockError("ranges must cover the packed sequence")


def _contiguous_stride(shape):
    stride = 1
    result = []
    for size in reversed(tuple(int(x) for x in shape)):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


def _meta(tensor):
    return (
        tuple(int(x) for x in tensor.shape),
        tuple(int(x) for x in tensor.stride()),
        tensor.dtype,
        str(tensor.device),
    )


@dataclass(frozen=True)
class BlockSignature:
    topology: tuple
    carriers: tuple


@dataclass(frozen=True)
class RuntimeSignature:
    x: tuple
    t_emb: tuple
    rope: tuple


def _expect(name, tensor, shape, dtypes=None):
    actual = tuple(int(x) for x in tensor.shape)
    if actual != tuple(int(x) for x in shape):
        raise H3BlockError("%s shape mismatch: expected %s, got %s" % (name, shape, actual))
    if tuple(int(x) for x in tensor.stride()) != _contiguous_stride(shape):
        raise H3BlockError("%s must be contiguous" % name)
    if dtypes is not None and tensor.dtype not in dtypes:
        raise H3BlockError("%s dtype mismatch: got %s" % (name, tensor.dtype))


def _scale_shape_ok(name, tensor, rows, *, allow_scalar):
    if tensor.dtype != torch.float32:
        raise H3BlockError("%s dtype mismatch: expected torch.float32, got %s" % (name, tensor.dtype))
    shape = tuple(int(x) for x in tensor.shape)
    accepted = shape == (int(rows),) or (allow_scalar and shape == (1,))
    if not accepted:
        if allow_scalar:
            raise H3BlockError("%s must have exactly 1 or %d values" % (name, rows))
        raise H3BlockError("%s must have exactly %d values" % (name, rows))
    if not tensor.is_contiguous():
        raise H3BlockError("%s must be contiguous" % name)


def build_block_signature(carriers, topology: BlockTopology) -> BlockSignature:
    """Validate all carrier metadata before any compilation is requested."""

    if not isinstance(topology, BlockTopology):
        raise H3BlockError("topology must be BlockTopology")
    values = carrier_tuple(carriers)
    hidden, ffn, tdim = topology.hidden_size, topology.ffn_size, topology.timestep_dim
    inner = topology.heads * topology.head_dim
    adaln_width = topology.adaln_expand * hidden * topology.adaln_modalities
    _expect("adaln_weight", values[0], (adaln_width, tdim), {torch.bfloat16})
    _expect("adaln_bias", values[1], (adaln_width,), {torch.bfloat16})
    _expect("norm1_weight", values[2], (hidden,), {torch.bfloat16})
    _expect("norm2_weight", values[3], (hidden,), {torch.bfloat16})
    _expect("qkv_qdata", values[4], (3 * inner, hidden), {torch.int8})
    _scale_shape_ok("qkv_scale", values[5], 3 * inner, allow_scalar=False)
    _expect("q_norm_weight", values[6], (topology.head_dim,), {torch.bfloat16})
    _expect("k_norm_weight", values[7], (topology.head_dim,), {torch.bfloat16})
    _expect("out_qdata", values[8], (hidden, inner), {torch.int8})
    _scale_shape_ok("out_scale", values[9], hidden, allow_scalar=True)
    _expect("fc1_qdata", values[10], (2 * ffn, hidden), {torch.int8})
    _scale_shape_ok("fc1_scale", values[11], 2 * ffn, allow_scalar=True)
    _expect("fc2_qdata", values[12], (hidden, ffn), {torch.int8})
    _scale_shape_ok("fc2_scale", values[13], hidden, allow_scalar=True)
    if inner % topology.convrot_groupsize or ffn % (2 * topology.convrot_groupsize) or hidden % topology.convrot_groupsize:
        raise H3BlockError("ConvRot dimensions must be group-aligned (hidden=256, ffn=512)")
    return BlockSignature(topology.signature, tuple(_meta(value) for value in values))


def validate_block_signature(signature, carriers, topology: BlockTopology) -> BlockSignature:
    current = build_block_signature(carriers, topology)
    if not isinstance(signature, BlockSignature):
        raise H3BlockError("expected BlockSignature")
    if current.topology != signature.topology:
        raise H3BlockError("H3 block topology signature mismatch")
    if current.carriers != signature.carriers:
        raise H3BlockError("H3 block carrier metadata signature mismatch")
    return current


def validate_runtime_signature(topology: BlockTopology, x, t_emb, rope) -> RuntimeSignature:
    if not all(torch.is_tensor(value) for value in (x, t_emb, rope)):
        raise H3BlockError("shared H3 block runtime inputs must be tensors")
    if (x.ndim != 2 or tuple(int(value) for value in x.shape) != (
            int(topology.router_geometry.sequence), int(topology.hidden_size))):
        raise H3BlockError("shared H3 block requires [sequence, hidden] input")
    if x.dtype != torch.bfloat16 or x.device.type not in ("cuda", "meta"):
        raise H3BlockError("shared H3 block requires a rank-2 CUDA BF16 input")
    if not x.is_contiguous():
        raise H3BlockError("shared H3 block input must be contiguous")
    if (t_emb.ndim != 2 or int(t_emb.shape[0]) <= 0
            or int(t_emb.shape[1]) != int(topology.timestep_dim)):
        raise H3BlockError("shared H3 block timestep input has an invalid shape")
    if t_emb.dtype != torch.bfloat16 or t_emb.device != x.device or not t_emb.is_contiguous():
        raise H3BlockError("shared H3 block timestep input must be contiguous CUDA BF16")
    expected_rope = (1, int(x.shape[0]), 1, 48, 2, 2)
    if (topology.head_dim != 128 or tuple(int(value) for value in rope.shape) != expected_rope
            or rope.dtype != torch.bfloat16 or rope.device != x.device):
        raise H3BlockError("shared H3 block requires H3's 96-wide split-half RoPE")
    rope_strides = (
        int(rope.stride(1)), int(rope.stride(3)),
        int(rope.stride(4)), int(rope.stride(5)),
    )
    if rope_strides != tuple(int(value) for value in topology.rope_strides):
        raise H3BlockError("shared H3 block RoPE strides differ from its topology")
    if x.device.type == "cuda" and comfy.model_management.in_training:
        raise H3BlockError("shared H3 block compilation is inference-only")
    return RuntimeSignature(_meta(x), _meta(t_emb), _meta(rope))


def _rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


def _route_lut(q_summary, k_summary, geometry, video_budget):
    batch, heads = q_summary.shape[:2]
    dense = torch.arange(geometry.kv_tiles, device=q_summary.device, dtype=torch.int32)
    delta = torch.cat((dense[:1], dense[1:] - dense[:-1]))
    lut = delta.view(1, 1, 1, -1).expand(batch, heads, geometry.q_tiles, -1).clone()
    valid = torch.full((batch, heads, geometry.q_tiles), geometry.kv_tiles,
                       dtype=torch.int32, device=q_summary.device)
    retained = min(geometry.pure_video_kv_tiles,
                   max(1, math.ceil(float(video_budget) * geometry.pure_video_kv_tiles)))
    if retained < geometry.pure_video_kv_tiles:
        scores = torch.matmul(
            q_summary[..., geometry.pure_video_q_start:, :],
            k_summary[..., geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )
        selected = sort_selected_indices(torch.topk(scores, retained, dim=-1).indices).to(torch.int32)
        selected = selected + geometry.pure_video_kv_start
        previous = dense[geometry.pure_video_kv_start - 1] if geometry.pure_video_kv_start else 0
        sparse_rows = torch.cat((
            delta[:geometry.pure_video_kv_start].view(1, 1, 1, -1).expand(
                batch, heads, geometry.pure_video_q_tiles, -1
            ),
            torch.cat((selected[..., :, :1] - previous,
                       selected[..., :, 1:] - selected[..., :, :-1]), dim=-1),
        ), dim=-1)
        lut[..., geometry.pure_video_q_start:, :sparse_rows.shape[-1]].copy_(sparse_rows)
        valid[..., geometry.pure_video_q_start:] = geometry.pure_video_kv_start + retained
    return lut.contiguous(), valid.contiguous()


def _tensor_block(topology: BlockTopology, x, t_emb, rope, carriers):
    (
        adaln_weight, adaln_bias, norm1_weight, norm2_weight,
        qkv_qdata, qkv_scale, q_norm_weight, k_norm_weight,
        out_qdata, out_scale, fc1_qdata, fc1_scale, fc2_qdata, fc2_scale,
    ) = carriers
    if topology.adaln_apply_silu:
        t_emb = F.silu(t_emb)
    mods = F.linear(t_emb, adaln_weight, adaln_bias)
    mods = mods.reshape(t_emb.shape[0] * topology.adaln_modalities,
                        topology.adaln_expand * topology.hidden_size)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mods.chunk(6, dim=-1)
    h = _rms_norm(x, norm1_weight, topology.norm_eps)
    normed = h.clone()
    for start, stop, row in topology.mod_segments:
        normed[start:stop].mul_(1.0 + scale_msa[row].to(normed.dtype)).add_(
            shift_msa[row].to(normed.dtype)
        )
    q, q_scale, k, k_scale, v, q_summary, k_summary = fused_qkv_op(
        normed, qkv_qdata, qkv_scale, q_norm_weight, k_norm_weight, rope,
        topology.heads, topology.qk_norm_eps, topology.has_rope,
        list(topology.rope_strides),
    )
    lut, valid = _route_lut(q_summary, k_summary, topology.router_geometry, topology.video_budget)
    v_fp8, v_scale = prepare_sparse_sage_v_op(v)
    threshold = torch.full((topology.heads,), 50.0, device=x.device, dtype=torch.float32)
    attn = sparse_sage_attention_op(q, k, v_fp8, lut, valid, threshold,
                                    q_scale, k_scale, v_scale, x.dtype)
    attention_inner = topology.heads * topology.head_dim
    attn = attn.transpose(1, 2).reshape(attn.shape[0], attn.shape[2], attention_inner).squeeze(0)
    attn = _convrot_linear_op(attn, out_qdata, out_scale, None)
    for start, stop, row in topology.mod_segments:
        x[start:stop].addcmul_(attn[start:stop], gate_msa[row].to(x.dtype))
    fc1_0, fc1_s0, fc1_1, fc1_s1 = _convrot_fc1_tiles_op(fc1_qdata, fc1_scale)
    fc2_0, fc2_1, fc2_scale_tiles = _convrot_fc2_tiles_op(fc2_qdata, fc2_scale)
    for start, stop, row in topology.mlp_chunks:
        chunk = _rms_norm(x[start:stop], norm2_weight, topology.norm_eps)
        chunk.mul_(1.0 + scale_mlp[row].to(chunk.dtype)).add_(shift_mlp[row].to(chunk.dtype))
        expanded0 = _convrot_linear_op(chunk, fc1_0, fc1_s0, None)
        expanded1 = _convrot_linear_op(chunk, fc1_1, fc1_s1, None)
        out0 = _convrot_linear_op(expanded0, fc2_0, fc2_scale_tiles, "swiglu")
        out1 = _convrot_linear_op(expanded1, fc2_1, fc2_scale_tiles, "swiglu")
        out = out0 + out1
        x[start:stop].addcmul_(out, gate_mlp[row].to(x.dtype))
    return x


class CompiledBlock:
    def __init__(self, topology, signature, runtime_signature, compiled):
        self.topology = topology
        self.signature = signature
        self.runtime_signature = runtime_signature
        self._compiled = compiled

    def __call__(self, x, t_emb, rope, carriers):
        runtime_signature = validate_runtime_signature(self.topology, x, t_emb, rope)
        if self.runtime_signature is None:
            self.runtime_signature = runtime_signature
        elif self.runtime_signature != runtime_signature:
            raise H3BlockError("H3 block runtime tensor signature mismatch")
        signature = validate_block_signature(self.signature, carriers, self.topology)
        devices = {metadata[3] for metadata in signature.carriers}
        if devices != {runtime_signature.x[3]}:
            raise H3BlockError("H3 block carriers must share the runtime input device")
        return self._compiled(x, t_emb, rope, *carrier_tuple(carriers))


def make_compiled_block(topology: BlockTopology, carriers=None, *, compiler=torch.compile,
                        backend=None, runtime_tensors=None) -> CompiledBlock:
    """Create one fullgraph/dynamic=False callable for one static topology."""

    signature = build_block_signature(carriers, topology) if carriers is not None else None
    runtime_signature = (
        validate_runtime_signature(topology, *runtime_tensors)
        if runtime_tensors is not None else None
    )

    def kernel(x, t_emb, rope, *carrier_values):
        return _tensor_block(topology, x, t_emb, rope, carrier_values)

    kwargs = {"fullgraph": True, "dynamic": False}
    if backend is not None:
        kwargs["backend"] = backend
    compiled = compiler(kernel, **kwargs)

    if signature is None:
        # Signature is established by the first call, before Dynamo sees it.
        class _Lazy(CompiledBlock):
            def __call__(self, x, t_emb, rope, carriers):
                if self.signature is None:
                    self.signature = build_block_signature(carriers, self.topology)
                return super().__call__(x, t_emb, rope, carriers)
        return _Lazy(topology, None, runtime_signature, compiled)
    return CompiledBlock(topology, signature, runtime_signature, compiled)


compile_shared_block = make_compiled_block

__all__ = [
    "H3BlockError", "BlockCarriers", "BlockTopology", "BlockSignature", "RuntimeSignature",
    "carrier_tuple", "build_block_signature", "validate_block_signature",
    "validate_runtime_signature",
    "CompiledBlock", "make_compiled_block", "compile_shared_block",
]
