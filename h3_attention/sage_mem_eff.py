"""SM89-only memory-efficient SageAttention for MiniMax H3.

The backend reproduces the released H3 Sage path (HND, per-thread INT8 Q/K,
FP8 V, no K smoothing, FP32+FP16 accumulation) while separating preparation
from execution so the caller can release the fused BF16 QKV projection before
the attention kernel starts.
"""

from dataclasses import dataclass
import importlib.metadata
import logging

import torch

from . import stats
from .triton_i64 import per_thread_int8_i64

SUPPORTED_SAGE_PREFIXES = ("2.2.",)
V_OFFSET_LIMIT = (1 << 32) - 1


class EfficientSageError(RuntimeError):
    """The custom backend cannot run safely in the current environment."""


@dataclass(frozen=True)
class SageSM89API:
    version: str
    per_channel_fp8: object
    kernel: object
    kernel_name: str


@dataclass
class PreparedSM89:
    q_int8: torch.Tensor
    q_scale: torch.Tensor
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_fp8: torch.Tensor
    v_scale: torch.Tensor
    output_dtype: torch.dtype
    layer_index: int
    sequence: int
    heads: int
    head_dim: int
    softmax_scale: float
    kernel: object
    kernel_name: str


def _load_api():
    try:
        version = importlib.metadata.version("sageattention")
        import sageattention.core as core
    except Exception as exc:
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "sage_mem_eff requires SageAttention 2.2.x with the SM89 extension") from exc

    if not version.startswith(SUPPORTED_SAGE_PREFIXES):
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "sage_mem_eff was validated against SageAttention 2.2.x; installed version is %s"
            % version)
    if not getattr(core, "SM89_ENABLED", False):
        stats.increment("compatibility_errors")
        raise EfficientSageError("SageAttention's SM89 extension is unavailable")

    per_channel_fp8 = getattr(core, "per_channel_fp8", None)
    sm89_compile = getattr(core, "sm89_compile", None)
    kernel_name = "qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf"
    kernel = getattr(sm89_compile, kernel_name, None) if sm89_compile is not None else None
    if per_channel_fp8 is None or kernel is None:
        stats.increment("compatibility_errors")
        raise EfficientSageError(
            "SageAttention 2.2.x internal SM89 API changed: missing %s"
            % kernel_name)
    return SageSM89API(version, per_channel_fp8, kernel, kernel_name)


def max_linear_offset(tensor):
    return sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))


def guard_v_stride(v):
    """Copy only when SageAttention's unsigned-32-bit V preprocessor could wrap."""
    if max_linear_offset(v) > V_OFFSET_LIMIT:
        stats.increment("v_guard_copies")
        return v.contiguous()
    return v


def first_unsafe_v_length(heads=56, head_dim=128, sequence_stride=None):
    """First HND sequence length whose maximum element offset exceeds uint32."""
    sequence_stride = sequence_stride or heads * head_dim * 3
    non_sequence_tail = (heads - 1) * head_dim + (head_dim - 1)
    last_safe_row = (V_OFFSET_LIMIT - non_sequence_tail) // sequence_stride
    return last_safe_row + 2  # row index + one for sequence length


class SM89SageMemoryEfficientBackend:
    name = "sage_mem_eff"

    def __init__(self, api=None, quantizer=None, allow_cpu_for_tests=False):
        self.api = api if api is not None else _load_api()
        self.quantizer = quantizer if quantizer is not None else per_thread_int8_i64
        self.allow_cpu_for_tests = bool(allow_cpu_for_tests)
        stats.increment("configured")
        self._logged = False

    def _validate(self, q, k, v):
        if q.shape != k.shape or q.shape != v.shape:
            raise EfficientSageError(
                "sage_mem_eff requires equal self-attention Q/K/V shapes; got %s %s %s"
                % (tuple(q.shape), tuple(k.shape), tuple(v.shape)))
        if q.ndim != 4:
            raise EfficientSageError("sage_mem_eff expects HND rank-4 tensors")
        batch, heads, sequence, head_dim = q.shape
        if batch != 1:
            raise EfficientSageError("released H3 expects attention batch 1; got %d" % batch)
        if head_dim != 128:
            raise EfficientSageError("sage_mem_eff supports head_dim 128; got %d" % head_dim)
        if q.dtype not in (torch.float16, torch.bfloat16):
            raise EfficientSageError("sage_mem_eff requires fp16 or bf16; got %s" % q.dtype)
        if q.dtype != k.dtype or q.dtype != v.dtype:
            raise EfficientSageError("Q/K/V dtypes differ")
        if q.device != k.device or q.device != v.device:
            raise EfficientSageError("Q/K/V devices differ")
        if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
            raise EfficientSageError("Q/K/V last dimension must be contiguous")
        if not self.allow_cpu_for_tests:
            if not q.is_cuda:
                raise EfficientSageError("sage_mem_eff requires CUDA")
            capability = torch.cuda.get_device_capability(q.device)
            if capability != (8, 9):
                raise EfficientSageError(
                    "sage_mem_eff is SM89-only; device capability is %d.%d" % capability)
        return batch, heads, sequence, head_dim

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        batch, heads, sequence, head_dim = self._validate(q, k, v)
        if not self.allow_cpu_for_tests:
            torch.cuda.set_device(q.device)

        q_int8, q_scale, k_int8, k_scale = self.quantizer(
            q, k, None,
            BLKQ=128,
            WARPQ=32,
            BLKK=64,
            WARPK=64,
            tensor_layout="HND",
        )

        guarded_v = guard_v_stride(v)
        # Match stock SM89's fp32+fp16 path: scale_max 2.25, smooth_v False.
        v_fp8, v_scale, _ = self.api.per_channel_fp8(
            guarded_v,
            tensor_layout="HND",
            scale_max=2.25,
            smooth_v=False,
        )
        del guarded_v

        stats.observe_sequence(sequence)
        if not self._logged:
            logging.info(
                "[H3 attention] sage_mem_eff active: SageAttention %s, HND, "
                "per-thread int64 Q/K, stock FP8 V, kernel=%s",
                self.api.version,
                self.api.kernel_name,
            )
            self._logged = True

        return PreparedSM89(
            q_int8=q_int8,
            q_scale=q_scale,
            k_int8=k_int8,
            k_scale=k_scale,
            v_fp8=v_fp8,
            v_scale=v_scale,
            output_dtype=q.dtype,
            layer_index=int(layer_index),
            sequence=int(sequence),
            heads=int(heads),
            head_dim=int(head_dim),
            softmax_scale=head_dim ** -0.5,
            kernel=self.api.kernel,
            kernel_name=self.api.kernel_name,
        )

    def execute(self, prepared):
        output = torch.empty(
            prepared.q_int8.shape,
            dtype=prepared.output_dtype,
            device=prepared.q_int8.device,
        )
        try:
            prepared.kernel(
                prepared.q_int8,
                prepared.k_int8,
                prepared.v_fp8,
                output,
                prepared.q_scale,
                prepared.k_scale,
                prepared.v_scale,
                1,  # HND
                0,  # non-causal
                3,  # per-thread Q/K quantization
                prepared.softmax_scale,
                0,  # no LSE return
            )
        except Exception as exc:
            stats.increment("kernel_errors")
            device = prepared.q_int8.device
            gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
            raise EfficientSageError(
                "sage_mem_eff kernel failed: layer=%d sequence=%d heads=%d head_dim=%d "
                "dtype=%s device=%s kernel=%s SageAttention=%s"
                % (
                    prepared.layer_index,
                    prepared.sequence,
                    prepared.heads,
                    prepared.head_dim,
                    prepared.output_dtype,
                    gpu,
                    prepared.kernel_name,
                    self.api.version,
                )
            ) from exc
        stats.increment("executed")
        return output
