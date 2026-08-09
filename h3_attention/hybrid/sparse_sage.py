"""Prepared Sparse Sage executor for the installed SM89 low-level ABI."""

from dataclasses import dataclass
import importlib
import importlib.metadata

import torch

from .router import KV_TILE, Q_TILE


class SparseSageError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseSageAPI:
    version: str
    low_level_f16: object
    low_level_f32: object
    v_fused: object = None

    @property
    def signature(self):
        return (
            str(self.version),
            id(self.low_level_f16),
            id(self.low_level_f32),
            id(self.v_fused),
        )


@dataclass
class PreparedSparseSage:
    q_int8: torch.Tensor
    k_int8: torch.Tensor
    v_fp8: torch.Tensor
    q_scale: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    lut: torch.Tensor
    valid_block_num: torch.Tensor
    output_dtype: torch.dtype
    output_shape: tuple
    layer_index: int
    sequence: int
    heads: int
    metadata: dict
    pv_threshold: torch.Tensor
    timing: object = None


def load_sparse_sage_api():
    """Import Sparge's fused and SM89 low-level extensions, never its wrapper."""
    try:
        importlib.import_module("spas_sage_attn")
        importlib.import_module("spas_sage_attn._fused")
        sm89 = importlib.import_module("spas_sage_attn.sm89_compile")
        version = importlib.metadata.version("spas-sage-attn")
        ops = sm89._qattn_sm89
        low_f32 = getattr(
            ops,
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        )
        low_f16 = getattr(
            ops,
            "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        )
        fused = torch.ops.spas_sage_attn_fused
    except Exception as exc:
        raise SparseSageError(
            "Hybrid Sparse Attention requires SpargeAttention's compiled SM89 "
            "_qattn and _fused extensions"
        ) from exc
    return SparseSageAPI(version, low_f16, low_f32, fused)


def preflight_sparse_sage(api_loader=load_sparse_sage_api, cuda_available=None,
                          capability_getter=None):
    """Fail before model mutation unless the installed extension targets Ada."""
    api = api_loader()
    if cuda_available is None:
        cuda_available = torch.cuda.is_available
    if capability_getter is None:
        capability_getter = torch.cuda.get_device_capability
    if not cuda_available():
        raise SparseSageError("Hybrid Sparse Attention requires CUDA")
    capability = tuple(capability_getter())
    if capability != (8, 9):
        raise SparseSageError(
            "Hybrid Sparse Attention Phase A is SM89-only; device capability is %d.%d"
            % capability
        )
    return api


def _quantize_blocks(x, block, *, subtract_mean=False):
    """Quantize HND directly from its strided view, one block at a time."""
    if x.ndim != 4 or x.stride(-1) != 1:
        raise SparseSageError("Q/K quantization requires HND views with contiguous head dimension")
    b, h, sequence, dim = x.shape
    blocks = (sequence + block - 1) // block
    output = torch.empty((b, h, sequence, dim), dtype=torch.int8, device=x.device)
    scales = torch.empty((b, h, blocks), dtype=torch.float32, device=x.device)
    mean = x.mean(dim=-2, keepdim=True).float() if subtract_mean else None
    for index in range(blocks):
        start = index * block
        stop = min(start + block, sequence)
        value = x[..., start:stop, :].float()
        if mean is not None:
            value = value - mean
        scale = value.abs().amax(dim=(-2, -1)) / 127.0 + 1e-7
        quantized = value / scale[..., None, None]
        quantized = quantized + torch.where(quantized >= 0, 0.5, -0.5)
        output[..., start:stop, :].copy_(quantized.to(torch.int8))
        scales[..., index].copy_(scale)
    return output, scales


def quantize_qk(q, k):
    if q.is_cuda:
        try:
            from .sparse_quant import quantize_qk as quantize_qk_triton
        except Exception as exc:
            raise SparseSageError(
                "Sparse Sage Q/K quantization requires Triton on CUDA"
            ) from exc
        return quantize_qk_triton(q, k, Q_TILE, KV_TILE)
    q_int8, q_scale = _quantize_blocks(q, Q_TILE)
    k_int8, k_scale = _quantize_blocks(k, KV_TILE, subtract_mean=True)
    return q_int8, q_scale, k_int8, k_scale


def prepare_v_fp8(v, fused=None):
    """Transpose/pad and quantize HND V through Sparge's installed fused ops."""
    if v.ndim != 4 or v.stride(-1) != 1:
        raise SparseSageError("V preparation requires an HND view with contiguous head dimension")
    if not v.is_cuda:
        raise SparseSageError("SM89 V FP8 preparation requires CUDA")
    if fused is None:
        fused = torch.ops.spas_sage_attn_fused
    b, h, sequence, dim = v.shape
    padded = (sequence + 127) // 128 * 128
    transposed = torch.empty((b, h, dim, padded), dtype=v.dtype, device=v.device)
    fused.transpose_pad_permute_cuda(v, transposed, 1)
    try:
        v_fp8 = torch.empty_like(transposed, dtype=torch.float8_e4m3fn)
        v_scale = torch.empty((b, h, dim), dtype=torch.float32, device=v.device)
        fused.scale_fuse_quant_cuda(transposed, v_fp8, v_scale, sequence, 2.25, 1)
    finally:
        del transposed
    return v_fp8, v_scale


def _cuda_version():
    parts = (torch.version.cuda or "0.0").split(".")
    return int(parts[0]), int(parts[1])


@torch.library.custom_op(
    "minimax_h3::prepare_sparse_sage_v",
    mutates_args=(),
    device_types="cuda",
)
def prepare_sparse_sage_v_op(
    v: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    api = load_sparse_sage_api()
    return prepare_v_fp8(v, api.v_fused)


@prepare_sparse_sage_v_op.register_fake
def _prepare_sparse_sage_v_fake(v):
    padded = (v.shape[-2] + 127) // 128 * 128
    return (
        v.new_empty((*v.shape[:-2], v.shape[-1], padded), dtype=torch.float8_e4m3fn),
        v.new_empty((*v.shape[:-2], v.shape[-1]), dtype=torch.float32),
    )


def prepare_sparse_sage_v(v, _fused=None):
    return prepare_sparse_sage_v_op(v)


@torch.library.custom_op(
    "minimax_h3::sparse_sage_attention",
    mutates_args=(),
    device_types="cuda",
)
def sparse_sage_attention_op(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    lut: torch.Tensor,
    valid_block_num: torch.Tensor,
    pv_threshold: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    api = load_sparse_sage_api()
    kernel = api.low_level_f16 if _cuda_version() >= (12, 8) else api.low_level_f32
    output = torch.empty(q_int8.shape, dtype=output_dtype, device=q_int8.device)
    kernel(
        q_int8, k_int8, v_fp8, output,
        lut, valid_block_num, pv_threshold,
        q_scale, k_scale, v_scale,
        1, 0, 1, 128 ** -0.5, 0,
    )
    return output


@sparse_sage_attention_op.register_fake
def _sparse_sage_attention_fake(
    q_int8,
    k_int8,
    v_fp8,
    lut,
    valid_block_num,
    pv_threshold,
    q_scale,
    k_scale,
    v_scale,
    output_dtype,
):
    return q_int8.new_empty(q_int8.shape, dtype=output_dtype)


class SparseSageExecutor:
    def __init__(self, api, *, allow_cpu_for_tests=False, qk_quantizer=None,
                 v_preparer=None, low_level_selector=None):
        if api is None:
            raise TypeError("SparseSageExecutor requires a preflighted API")
        self.api = api
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        self.qk_quantizer = qk_quantizer or quantize_qk
        self.v_preparer = v_preparer or (
            prepare_v_fp8 if self.allow_cpu_for_tests else prepare_sparse_sage_v
        )
        self.low_level_selector = low_level_selector or self._select_low_level
        self._use_sparse_sage_op = (
            low_level_selector is None and not self.allow_cpu_for_tests
        )

    def _validate(self, q, k, v, lut, valid):
        if q.shape != k.shape or q.shape != v.shape:
            raise SparseSageError(
                "Sparse Sage requires equal self-attention Q/K/V shapes; got %s %s %s"
                % (tuple(q.shape), tuple(k.shape), tuple(v.shape))
            )
        if q.ndim != 4:
            raise SparseSageError("Sparse Sage expects HND rank-4 tensors")
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise SparseSageError("released H3 expects attention batch 1; got %d" % batch)
        if head_dim != 128:
            raise SparseSageError("Sparse Sage requires head_dim 128; got %d" % head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise SparseSageError("Sparse Sage requires fp16 or bf16 Q/K/V")
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise SparseSageError("Sparse Sage Q/K/V dtypes differ")
        if q.device != k.device or q.device != v.device:
            raise SparseSageError("Sparse Sage Q/K/V devices differ")
        if any(t.stride(-1) != 1 for t in (q, k, v)):
            raise SparseSageError("Sparse Sage Q/K/V last dimension must be contiguous")
        expected_lut = (batch, heads, (sequence + Q_TILE - 1) // Q_TILE,
                        (sequence + KV_TILE - 1) // KV_TILE)
        if tuple(lut.shape) != expected_lut or tuple(valid.shape) != expected_lut[:-1]:
            raise SparseSageError("Sparse Sage LUT/valid shapes do not match H3 geometry")
        if lut.dtype != torch.int32 or valid.dtype != torch.int32:
            raise SparseSageError("Sparse Sage LUT and valid counts must be int32")
        if not lut.is_contiguous() or not valid.is_contiguous():
            raise SparseSageError("Sparse Sage LUT and valid counts must be contiguous")
        if lut.device != q.device or valid.device != q.device:
            raise SparseSageError("Sparse Sage LUT and Q/K/V devices differ")
        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise SparseSageError("Sparse Sage requires CUDA")
            capability = tuple(torch.cuda.get_device_capability(q.device))
            if capability != (8, 9):
                raise SparseSageError(
                    "Hybrid Sparse Attention Phase A is SM89-only; device capability is %d.%d"
                    % capability
                )
        return heads, sequence

    def _select_low_level(self, _q):
        name = "low_level_f16" if _cuda_version() >= (12, 8) else "low_level_f32"
        kernel = getattr(self.api, name, None)
        if kernel is None:
            raise SparseSageError("SpargeAttention SM89 low-level kernel is unavailable")
        return kernel

    def prepare(self, q, k, v, lut, valid_block_num, *, layer_index, metadata,
                timing=None):
        heads, sequence = self._validate(q, k, v, lut, valid_block_num)
        # Keep V's temporary transposed buffer out of the Q/K quantization peak.
        v_timing = timing.begin("v_fp8_preparation") if timing is not None else None
        try:
            v_fp8, v_scale = self.v_preparer(v, self.api.v_fused)
            expected_v_shape = (v.shape[0], v.shape[1], v.shape[3],
                                (sequence + 127) // 128 * 128)
            if (tuple(v_fp8.shape) != expected_v_shape
                    or v_fp8.dtype != torch.float8_e4m3fn
                    or v_fp8.device != v.device
                    or not v_fp8.is_contiguous()
                    or tuple(v_scale.shape) != (v.shape[0], v.shape[1], v.shape[3])
                    or v_scale.dtype != torch.float32
                    or v_scale.device != v.device
                    or not v_scale.is_contiguous()):
                raise SparseSageError("V preparer returned an invalid FP8 carrier")
        finally:
            if timing is not None:
                timing.end(v_timing)
        qk_timing = timing.begin("q_k_int8_quantization") if timing is not None else None
        try:
            q_int8, q_scale, k_int8, k_scale = self.qk_quantizer(q, k)
            q_blocks = (sequence + Q_TILE - 1) // Q_TILE
            k_blocks = (sequence + KV_TILE - 1) // KV_TILE
            if (tuple(q_int8.shape) != tuple(q.shape)
                    or tuple(k_int8.shape) != tuple(k.shape)
                    or q_int8.dtype != torch.int8 or k_int8.dtype != torch.int8
                    or q_int8.device != q.device or k_int8.device != q.device
                    or q_int8.stride(-1) != 1 or k_int8.stride(-1) != 1
                    or tuple(q_scale.shape) != (q.shape[0], heads, q_blocks)
                    or tuple(k_scale.shape) != (q.shape[0], heads, k_blocks)
                    or q_scale.dtype != torch.float32 or k_scale.dtype != torch.float32
                    or q_scale.device != q.device or k_scale.device != q.device
                    or not q_scale.is_contiguous() or not k_scale.is_contiguous()):
                raise SparseSageError("Q/K quantizer returned an invalid INT8 carrier")
        finally:
            if timing is not None:
                timing.end(qk_timing)
        pv_threshold = torch.full((heads,), 50.0, dtype=torch.float32, device=q.device)
        return PreparedSparseSage(
            q_int8=q_int8,
            k_int8=k_int8,
            v_fp8=v_fp8,
            q_scale=q_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            lut=lut,
            valid_block_num=valid_block_num,
            output_dtype=q.dtype,
            output_shape=tuple(q.shape),
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            metadata=dict(metadata),
            pv_threshold=pv_threshold,
            timing=timing,
        )

    def prepare_projected(self, projected, lut, valid_block_num, *, metadata,
                          timing=None):
        """Consume Q/K already emitted in Sparse Sage's native representation."""
        from .fused_qkv import validate_prepared_fused_qkv

        validate_prepared_fused_qkv(projected)
        heads = int(projected.heads)
        sequence = int(projected.sequence)
        expected_lut = (
            1,
            heads,
            (sequence + Q_TILE - 1) // Q_TILE,
            (sequence + KV_TILE - 1) // KV_TILE,
        )
        if tuple(lut.shape) != expected_lut or tuple(valid_block_num.shape) != expected_lut[:-1]:
            raise SparseSageError("Sparse Sage LUT/valid shapes do not match fused H3 QKV")
        if lut.dtype != torch.int32 or valid_block_num.dtype != torch.int32:
            raise SparseSageError("Sparse Sage LUT and valid counts must be int32")
        if not lut.is_contiguous() or not valid_block_num.is_contiguous():
            raise SparseSageError("Sparse Sage LUT and valid counts must be contiguous")
        if lut.device != projected.q_int8.device or valid_block_num.device != projected.q_int8.device:
            raise SparseSageError("Sparse Sage LUT and fused H3 QKV devices differ")
        if not self.allow_cpu_for_tests:
            if not projected.q_int8.is_cuda:
                raise SparseSageError("fused H3 Sparse Sage requires CUDA")
            capability = tuple(torch.cuda.get_device_capability(projected.q_int8.device))
            if capability != (8, 9):
                raise SparseSageError(
                    "fused H3 Sparse Sage is SM89-only; device capability is %d.%d"
                    % capability
                )

        v_timing = timing.begin("v_fp8_preparation") if timing is not None else None
        try:
            v_fp8, v_scale = self.v_preparer(projected.v, self.api.v_fused)
            padded = (sequence + 127) // 128 * 128
            if (tuple(v_fp8.shape) != (1, heads, projected.head_dim, padded)
                    or v_fp8.dtype != torch.float8_e4m3fn
                    or v_fp8.device != projected.v.device
                    or not v_fp8.is_contiguous()
                    or tuple(v_scale.shape) != (1, heads, projected.head_dim)
                    or v_scale.dtype != torch.float32
                    or v_scale.device != projected.v.device
                    or not v_scale.is_contiguous()):
                raise SparseSageError("V preparer returned an invalid FP8 carrier")
        finally:
            if timing is not None:
                timing.end(v_timing)

        projected_metadata = dict(metadata)
        projected_metadata.update({
            "qkv_projection": "fused_int8",
            "smooth_k": bool(projected.smooth_k),
        })
        pv_threshold = torch.full(
            (heads,), 50.0, dtype=torch.float32, device=projected.q_int8.device
        )
        return PreparedSparseSage(
            q_int8=projected.q_int8,
            k_int8=projected.k_int8,
            v_fp8=v_fp8,
            q_scale=projected.q_scale,
            k_scale=projected.k_scale,
            v_scale=v_scale,
            lut=lut,
            valid_block_num=valid_block_num,
            output_dtype=projected.output_dtype,
            output_shape=(1, heads, sequence, projected.head_dim),
            layer_index=int(projected.layer_index),
            sequence=sequence,
            heads=heads,
            metadata=projected_metadata,
            pv_threshold=pv_threshold,
            timing=timing,
        )

    def execute(self, prepared):
        kernel_timing = prepared.timing.begin("sparse_sage_low_level_kernel") if prepared.timing is not None else None
        try:
            if self._use_sparse_sage_op:
                output = sparse_sage_attention_op(
                    prepared.q_int8,
                    prepared.k_int8,
                    prepared.v_fp8,
                    prepared.lut,
                    prepared.valid_block_num,
                    prepared.pv_threshold,
                    prepared.q_scale,
                    prepared.k_scale,
                    prepared.v_scale,
                    prepared.output_dtype,
                )
            else:
                output = torch.empty(prepared.output_shape, dtype=prepared.output_dtype,
                                     device=prepared.q_int8.device)
                kernel = self.low_level_selector(prepared.q_int8)
                kernel(
                    prepared.q_int8, prepared.k_int8, prepared.v_fp8, output,
                    prepared.lut, prepared.valid_block_num, prepared.pv_threshold,
                    prepared.q_scale, prepared.k_scale, prepared.v_scale,
                    1, 0, 1, 128 ** -0.5, 0,
                )
        except Exception as exc:
            raise SparseSageError(
                "Sparse Sage kernel failed: layer=%d sequence=%d heads=%d "
                "dtype=%s SpargeAttention=%s"
                % (prepared.layer_index, prepared.sequence, prepared.heads,
                   prepared.output_dtype, self.api.version)
            ) from exc
        finally:
            if prepared.timing is not None:
                prepared.timing.end(kernel_timing)
        return output
