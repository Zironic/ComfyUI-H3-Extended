"""Resolve validated QKV and MLP execution providers from model formats.

Production ``auto`` may select only candidates backed by existing optimized
kernels.  Custom Triton GEMM prototypes remain available for explicit kernel
development, but require the research-kernel policy gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..kernel_policy import (
    CONVROT_TWO_SLICE_MLP,
    FUSED_QKV_TRITON,
    GENERIC_CHUNKED_MLP,
    MLP_EPILOGUE_TRITON,
    STANDARD_QKV,
    KernelPolicy,
    candidate_by_id,
)
from ..plan import (
    FUSED_QKV_OFF,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_OFF,
    MLP_MEMORY_STRICT_AUTO,
    MLP_MEMORY_STRICT_BF16,
    MLP_MEMORY_STRICT_NATIVE,
)

QKV_STANDARD = "standard_h3_qkv"
QKV_DENSE_CONVROT_INT8 = "convrot_int8_dense_sage"
QKV_SPARSE_CONVROT_INT8 = "convrot_int8_sparse_sage"

MLP_OFF = "off"
MLP_GENERIC_CHUNKED = "generic_chunked_quantized"
MLP_CONVROT_INT8_TWO_SLICE = "convrot_int8_two_slice"
MLP_CONVROT_INT8_EPILOGUE = "convrot_int8_epilogue"


@dataclass(frozen=True)
class QKVProviderResolution:
    provider_id: str
    fused: bool
    reason: str
    candidate_id: str = STANDARD_QKV

    @property
    def candidate(self):
        return candidate_by_id(self.candidate_id)


@dataclass(frozen=True)
class MLPProviderResolution:
    provider_id: str
    activation_mode: str
    reason: str
    candidate_id: str | None = None

    @property
    def candidate(self):
        return (
            None
            if self.candidate_id is None
            else candidate_by_id(self.candidate_id)
        )


def _policy(policy):
    return KernelPolicy.from_environment() if policy is None else policy


def _standard_qkv(reason):
    return QKVProviderResolution(
        QKV_STANDARD,
        False,
        reason,
        candidate_id=STANDARD_QKV,
    )


def _required_or_standard(request, reason, policy):
    if request == FUSED_QKV_REQUIRED:
        policy.require_research(FUSED_QKV_TRITON)
        raise RuntimeError("required fused QKV is unavailable: %s" % reason)
    return _standard_qkv(reason)


def resolve_qkv_provider(
    inventory,
    *,
    request,
    backend_kind,
    capability,
    triton_available,
    sparse_spec=None,
    policy=None,
):
    policy = _policy(policy)
    if request == FUSED_QKV_OFF:
        return _standard_qkv("fused QKV was disabled")

    # The current fused QKV providers replace the established optimized Comfy
    # linear GEMM with custom Triton tl.dot mainloops. They therefore cannot be
    # selected by production auto even when their format preflight succeeds.
    if request != FUSED_QKV_REQUIRED:
        return _standard_qkv(
            "production auto keeps the existing optimized QKV GEMM; the current "
            "fused-QKV dataflow requires a new performance-equivalent kernel"
        )

    policy.require_research(FUSED_QKV_TRITON)

    if not inventory.qkv:
        return _required_or_standard(
            request, "the H3 model has no QKV projection inventory", policy
        )
    if not inventory.homogeneous("qkv"):
        return _required_or_standard(
            request, "H3 QKV layers use mixed weight formats", policy
        )
    if not inventory.qkv_convrot_int8_256:
        labels = ", ".join(sorted(set(inventory.labels("qkv"))))
        return _required_or_standard(
            request,
            "no research fused provider supports QKV format %s"
            % (labels or "unknown"),
            policy,
        )
    if tuple(capability or ()) != (8, 9):
        return _required_or_standard(
            request,
            "the current research fused QKV providers require SM89",
            policy,
        )
    if not triton_available:
        return _required_or_standard(request, "Triton is unavailable", policy)

    if backend_kind == "dense_sage_sm89":
        return QKVProviderResolution(
            QKV_DENSE_CONVROT_INT8,
            True,
            "research ConvRot-256 QKV into dense Sage per-thread carriers",
            candidate_id=FUSED_QKV_TRITON,
        )

    if backend_kind == "sparse_sage":
        if sparse_spec is None:
            return _required_or_standard(
                request, "Sparse Sage ABI was not resolved", policy
            )
        if (
            tuple(getattr(sparse_spec, "capability", ())) != (8, 9)
            or int(getattr(sparse_spec, "q_tile", 0)) != 128
            or int(getattr(sparse_spec, "kv_tile", 0)) != 64
        ):
            return _required_or_standard(
                request,
                "the selected Sparse Sage ABI is not SM89 128Q x 64KV",
                policy,
            )
        return QKVProviderResolution(
            QKV_SPARSE_CONVROT_INT8,
            True,
            "research ConvRot-256 QKV into Sparse Sage block carriers",
            candidate_id=FUSED_QKV_TRITON,
        )

    return _required_or_standard(
        request,
        "the resolved attention backend has no fused-QKV consumer",
        policy,
    )


def _convrot_compatible(inventory):
    return (
        bool(inventory.fc1)
        and bool(inventory.fc2)
        and inventory.homogeneous("fc1")
        and inventory.homogeneous("fc2")
        and inventory.mlp_convrot_int8_256
    )


def _generic_resolution(*, native, strict, reason):
    if native:
        mode = "mlp_chunked_native_strict" if strict else "mlp_chunked_native"
    else:
        mode = "mlp_chunked_bf16_strict" if strict else "mlp_chunked_bf16"
    return MLPProviderResolution(
        MLP_GENERIC_CHUNKED,
        mode,
        reason,
        candidate_id=GENERIC_CHUNKED_MLP,
    )


def _two_slice_resolution(mode, reason):
    return MLPProviderResolution(
        MLP_CONVROT_INT8_TWO_SLICE,
        mode,
        reason,
        candidate_id=CONVROT_TWO_SLICE_MLP,
    )


def resolve_mlp_provider(inventory, *, request, policy=None):
    policy = _policy(policy)
    if request == MLP_MEMORY_OFF:
        return MLPProviderResolution(
            MLP_OFF,
            "off",
            "MLP memory optimization was disabled",
            candidate_id=None,
        )

    if request == MLP_MEMORY_EPILOGUE:
        policy.require_research(MLP_EPILOGUE_TRITON)

    if not inventory.fc1 or not inventory.fc2:
        if request == MLP_MEMORY_LEGACY_CONVROT_REQUIRED:
            raise RuntimeError(
                "required ConvRot two-slice MLP is unavailable: "
                "the H3 model has no MLP inventory"
            )
        return MLPProviderResolution(
            MLP_OFF,
            "off",
            "the H3 model has no MLP inventory",
            candidate_id=None,
        )

    compatible = _convrot_compatible(inventory)

    if request == MLP_MEMORY_STRICT_AUTO:
        if compatible:
            return _two_slice_resolution(
                "mlp_chunked_convrot_2slice_strict",
                "strict automatic selection uses ConvRot two-slice execution",
            )
        labels = sorted(
            set(inventory.labels("fc1")) | set(inventory.labels("fc2"))
        )
        return _generic_resolution(
            native=True,
            strict=True,
            reason=(
                "strict automatic selection preserves the model's formats: %s"
                % (", ".join(labels) or "unknown")
            ),
        )

    if request in (MLP_MEMORY_LEGACY_BF16, MLP_MEMORY_STRICT_BF16):
        strict = request == MLP_MEMORY_STRICT_BF16
        return _generic_resolution(
            native=False,
            strict=strict,
            reason=(
                "explicit strict BF16 SwiGLU chunking"
                if strict
                else "explicit BF16 SwiGLU chunking"
            ),
        )

    if request in (MLP_MEMORY_LEGACY_NATIVE, MLP_MEMORY_STRICT_NATIVE):
        strict = request == MLP_MEMORY_STRICT_NATIVE
        return _generic_resolution(
            native=True,
            strict=strict,
            reason=(
                "explicit strict native chunked MLP"
                if strict
                else "explicit native chunked MLP"
            ),
        )

    if request == MLP_MEMORY_LEGACY_CONVROT_REQUIRED:
        if not compatible:
            labels = sorted(
                set(inventory.labels("fc1")) | set(inventory.labels("fc2"))
            )
            raise RuntimeError(
                "required ConvRot two-slice MLP is unavailable for %s"
                % (", ".join(labels) or "unknown formats")
            )
        return _two_slice_resolution(
            "mlp_chunked_convrot_2slice",
            "explicit required ConvRot two-slice execution",
        )

    if request == MLP_MEMORY_EPILOGUE:
        if not compatible:
            labels = sorted(
                set(inventory.labels("fc1")) | set(inventory.labels("fc2"))
            )
            raise RuntimeError(
                "MLP epilogue prototype requires homogeneous ConvRot-256 "
                "TensorWise INT8 fc1/fc2 weights; got %s"
                % (", ".join(labels) or "unknown")
            )
        return MLPProviderResolution(
            MLP_CONVROT_INT8_EPILOGUE,
            "mlp_chunked_convrot_epilogue",
            "research fc1+SwiGLU and fc2+gated-residual prototype",
            candidate_id=MLP_EPILOGUE_TRITON,
        )

    if compatible:
        return _two_slice_resolution(
            "mlp_chunked_convrot_2slice",
            "ConvRot-256 MLP uses token chunks and optimized Kitchen feature slices",
        )

    labels = sorted(
        set(inventory.labels("fc1")) | set(inventory.labels("fc2"))
    )
    return _generic_resolution(
        native=True,
        strict=False,
        reason=(
            "generic token chunking preserves the model's own linear formats: %s"
            % (", ".join(labels) or "unknown")
        ),
    )
