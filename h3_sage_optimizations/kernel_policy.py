"""Kernel-confidence policy for H3 Sage optimizations.

Production automation is deliberately limited to optimizations that keep every
large matrix multiplication on an existing validated Comfy, Comfy Kitchen,
SageAttention, or SpargeAttention kernel.  Attractive dataflows that require a
new GEMM mainloop or kernel ABI remain research-only until an implementation is
shown to match the established optimized path.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os


RESEARCH_KERNELS_ENV = "H3_SAGE_ENABLE_RESEARCH_KERNELS"


class KernelBucket(str, Enum):
    EXISTING_OPTIMIZED_KERNEL = "existing_optimized_kernel"
    REQUIRES_NEW_KERNEL = "requires_new_kernel"


@dataclass(frozen=True)
class OptimizationCandidate:
    candidate_id: str
    subsystem: str
    bucket: KernelBucket
    kernel_basis: str
    summary: str
    active: bool = False
    auto_eligible: bool = False

    @property
    def ab_benchmarkable(self) -> bool:
        return self.bucket == KernelBucket.EXISTING_OPTIMIZED_KERNEL

    def as_status(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "subsystem": self.subsystem,
            "bucket": self.bucket.value,
            "kernel_basis": self.kernel_basis,
            "summary": self.summary,
            "active": bool(self.active),
            "auto_eligible": bool(self.auto_eligible),
            "ab_benchmarkable": bool(self.ab_benchmarkable),
        }


STANDARD_QKV = "standard_qkv_dispatch"
FUSED_QKV_TRITON = "fused_qkv_custom_triton"
GENERIC_CHUNKED_MLP = "generic_chunked_mlp"
CONVROT_TWO_SLICE_MLP = "convrot_two_slice_mlp"
MLP_EPILOGUE_TRITON = "mlp_epilogue_custom_triton"
DENSE_SAGE = "dense_sage_attention"
SPARSE_SAGE = "sparse_sage_attention"
RMS_ADALN_THEN_QUANTIZE = "rms_adaln_then_existing_quantizer"
OUT_PROJ_GATED_RESIDUAL = "out_proj_gated_residual"
DIRECT_FP8_V = "direct_fp8_v_preparation"
ZERO_COPY_WEIGHT_SLICES = "zero_copy_weight_slices"
SHARED_INDUCTOR = "shared_inductor_custom_qkv_pipeline"


_CANDIDATES = {
    STANDARD_QKV: OptimizationCandidate(
        STANDARD_QKV,
        "qkv",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "Comfy quantized Linear/F.linear dispatch",
        "Preserve the checkpoint's native optimized QKV GEMM.",
        active=True,
        auto_eligible=True,
    ),
    FUSED_QKV_TRITON: OptimizationCandidate(
        FUSED_QKV_TRITON,
        "qkv",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "custom Triton tl.dot projection mainloop",
        (
            "Projection plus RMSNorm, RoPE, and Sage-carrier emission. The "
            "dataflow is attractive, but the custom GEMM has not matched the "
            "established optimized linear kernel."
        ),
        active=True,
    ),
    GENERIC_CHUNKED_MLP: OptimizationCandidate(
        GENERIC_CHUNKED_MLP,
        "mlp",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "Comfy quantized F.linear dispatch",
        (
            "Bound activation memory by token chunking while preserving the "
            "checkpoint's native optimized fc1/fc2 kernels."
        ),
        active=True,
        auto_eligible=True,
    ),
    CONVROT_TWO_SLICE_MLP: OptimizationCandidate(
        CONVROT_TWO_SLICE_MLP,
        "mlp",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "Comfy Kitchen ck.int8_linear with input_act='swiglu'",
        (
            "Two feature slices executed by the existing optimized ConvRot "
            "INT8 GEMM, including Kitchen's fused SwiGLU input preparation."
        ),
        active=True,
        auto_eligible=True,
    ),
    MLP_EPILOGUE_TRITON: OptimizationCandidate(
        MLP_EPILOGUE_TRITON,
        "mlp",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "custom Triton fc1/fc2 tl.dot mainloops",
        (
            "fc1+SwiGLU and fc2+gated-residual eliminate intermediates, but "
            "the prototype replaces the optimized Kitchen GEMMs."
        ),
        active=True,
    ),
    DENSE_SAGE: OptimizationCandidate(
        DENSE_SAGE,
        "attention",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "compiled SageAttention kernel",
        "Prepared dense Sage execution through the existing kernel.",
        active=True,
        auto_eligible=True,
    ),
    SPARSE_SAGE: OptimizationCandidate(
        SPARSE_SAGE,
        "attention",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "compiled SpargeAttention Sparse Sage kernel",
        "Sparse routing feeding the existing compiled attention kernel.",
        active=True,
        auto_eligible=True,
    ),
    RMS_ADALN_THEN_QUANTIZE: OptimizationCandidate(
        RMS_ADALN_THEN_QUANTIZE,
        "activation_input",
        KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        (
            "existing RMS/AdaLN operation plus existing rowwise or ConvRot "
            "quantizer feeding the existing GEMM"
        ),
        (
            "A benchmarkable composition that leaves the GEMM unchanged. It "
            "still needs an A/B test because an extra transform boundary may "
            "cost more than the saved allocation."
        ),
    ),
    OUT_PROJ_GATED_RESIDUAL: OptimizationCandidate(
        OUT_PROJ_GATED_RESIDUAL,
        "attention_output",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "new optimized GEMM output epilogue required",
        "Fuse out_proj or fc2 output with the AdaLN gate and residual add.",
    ),
    DIRECT_FP8_V: OptimizationCandidate(
        DIRECT_FP8_V,
        "attention_v",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "new direct HND-to-FP8 compiled V-preparation kernel",
        "Remove the transposed BF16 V scratch carrier.",
    ),
    ZERO_COPY_WEIGHT_SLICES: OptimizationCandidate(
        ZERO_COPY_WEIGHT_SLICES,
        "mlp",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "optimized GEMM needs packed-weight row/column offset support",
        "Address original packed fc1/fc2 storage without contiguous tile copies.",
    ),
    SHARED_INDUCTOR: OptimizationCandidate(
        SHARED_INDUCTOR,
        "compile",
        KernelBucket.REQUIRES_NEW_KERNEL,
        "shared graph currently depends on the custom fused-QKV Triton GEMM",
        (
            "Shared block compilation is not production-benchmarkable until its "
            "required QKV path uses a performance-equivalent optimized kernel."
        ),
        active=True,
    ),
}


_QKV_PROVIDER_CANDIDATES = {
    "standard_h3_qkv": STANDARD_QKV,
    "convrot_int8_dense_sage": FUSED_QKV_TRITON,
    "convrot_int8_sparse_sage": FUSED_QKV_TRITON,
}

_MLP_PROVIDER_CANDIDATES = {
    "generic_chunked_quantized": GENERIC_CHUNKED_MLP,
    "convrot_int8_two_slice": CONVROT_TWO_SLICE_MLP,
    "convrot_int8_epilogue": MLP_EPILOGUE_TRITON,
}


@dataclass(frozen=True)
class KernelPolicy:
    allow_research_kernels: bool = False
    source: str = "production"

    @classmethod
    def from_environment(cls, environ=None):
        environ = os.environ if environ is None else environ
        raw = str(environ.get(RESEARCH_KERNELS_ENV, "")).strip().lower()
        enabled = raw in {"1", "true", "yes", "on"}
        return cls(
            allow_research_kernels=enabled,
            source=(RESEARCH_KERNELS_ENV if enabled else "production"),
        )

    def require_research(self, candidate_id: str) -> OptimizationCandidate:
        candidate = candidate_by_id(candidate_id)
        if candidate.bucket != KernelBucket.REQUIRES_NEW_KERNEL:
            return candidate
        if not self.allow_research_kernels:
            raise RuntimeError(
                "%s is a research-kernel candidate and is disabled by the "
                "production kernel policy; set %s=1 only for explicit kernel "
                "development or characterization"
                % (candidate_id, RESEARCH_KERNELS_ENV)
            )
        return candidate

    def as_status(self) -> dict:
        return {
            "allow_research_kernels": bool(self.allow_research_kernels),
            "source": self.source,
            "research_environment_variable": RESEARCH_KERNELS_ENV,
        }


def candidate_by_id(candidate_id: str) -> OptimizationCandidate:
    try:
        return _CANDIDATES[candidate_id]
    except KeyError as exc:
        raise KeyError("unknown H3 optimization candidate %r" % candidate_id) from exc


def candidate_for_qkv_provider(provider_id: str) -> OptimizationCandidate:
    return candidate_by_id(_QKV_PROVIDER_CANDIDATES.get(provider_id, STANDARD_QKV))


def candidate_for_mlp_provider(provider_id: str):
    candidate_id = _MLP_PROVIDER_CANDIDATES.get(provider_id)
    return None if candidate_id is None else candidate_by_id(candidate_id)


def benchmarkable_candidates() -> tuple[OptimizationCandidate, ...]:
    return tuple(item for item in _CANDIDATES.values() if item.ab_benchmarkable)


def research_candidates() -> tuple[OptimizationCandidate, ...]:
    return tuple(
        item
        for item in _CANDIDATES.values()
        if item.bucket == KernelBucket.REQUIRES_NEW_KERNEL
    )
