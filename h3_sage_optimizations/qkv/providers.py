"""Resolve validated QKV and MLP execution providers from model formats."""

from __future__ import annotations

from dataclasses import dataclass

from ..plan import (
    FUSED_QKV_OFF,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_OFF,
)

QKV_STANDARD = "standard_h3_qkv"
QKV_DENSE_CONVROT_INT8 = "convrot_int8_dense_sage"
QKV_SPARSE_CONVROT_INT8 = "convrot_int8_sparse_sage"

MLP_OFF = "off"
MLP_GENERIC_CHUNKED = "generic_chunked_quantized"
MLP_CONVROT_INT8_TWO_SLICE = "convrot_int8_two_slice"


@dataclass(frozen=True)
class QKVProviderResolution:
    provider_id: str
    fused: bool
    reason: str


@dataclass(frozen=True)
class MLPProviderResolution:
    provider_id: str
    activation_mode: str
    reason: str


def _required_or_standard(request, reason):
    if request == FUSED_QKV_REQUIRED:
        raise RuntimeError("required fused QKV is unavailable: %s" % reason)
    return QKVProviderResolution(QKV_STANDARD, False, reason)


def resolve_qkv_provider(
    inventory,
    *,
    request,
    backend_kind,
    capability,
    triton_available,
    sparse_spec=None,
):
    if request == FUSED_QKV_OFF:
        return QKVProviderResolution(
            QKV_STANDARD, False, "fused QKV was disabled"
        )

    if not inventory.qkv:
        return _required_or_standard(
            request, "the H3 model has no QKV projection inventory"
        )
    if not inventory.homogeneous("qkv"):
        return _required_or_standard(
            request, "H3 QKV layers use mixed weight formats"
        )
    if not inventory.qkv_convrot_int8_256:
        labels = ", ".join(sorted(set(inventory.labels("qkv"))))
        return _required_or_standard(
            request,
            "no fused provider supports QKV format %s" % (labels or "unknown"),
        )
    if tuple(capability or ()) != (8, 9):
        return _required_or_standard(
            request, "the current fused QKV providers require SM89"
        )
    if not triton_available:
        return _required_or_standard(request, "Triton is unavailable")

    if backend_kind == "dense_sage_sm89":
        return QKVProviderResolution(
            QKV_DENSE_CONVROT_INT8,
            True,
            "ConvRot-256 TensorWise INT8 QKV into dense Sage per-thread carriers",
        )

    if backend_kind == "sparse_sage":
        if sparse_spec is None:
            return _required_or_standard(
                request, "Sparse Sage ABI was not resolved"
            )
        if (
            tuple(getattr(sparse_spec, "capability", ())) != (8, 9)
            or int(getattr(sparse_spec, "q_tile", 0)) != 128
            or int(getattr(sparse_spec, "kv_tile", 0)) != 64
        ):
            return _required_or_standard(
                request,
                "the selected Sparse Sage ABI is not SM89 128Q x 64KV",
            )
        return QKVProviderResolution(
            QKV_SPARSE_CONVROT_INT8,
            True,
            "ConvRot-256 TensorWise INT8 QKV into Sparse Sage block carriers",
        )

    return _required_or_standard(
        request, "the resolved attention backend has no fused-QKV consumer"
    )


def resolve_mlp_provider(inventory, *, request):
    if request == MLP_MEMORY_OFF:
        return MLPProviderResolution(
            MLP_OFF, "off", "MLP memory optimization was disabled"
        )

    if not inventory.fc1 or not inventory.fc2:
        return MLPProviderResolution(
            MLP_OFF, "off", "the H3 model has no MLP inventory"
        )

    if (
        inventory.homogeneous("fc1")
        and inventory.homogeneous("fc2")
        and inventory.mlp_convrot_int8_256
    ):
        return MLPProviderResolution(
            MLP_CONVROT_INT8_TWO_SLICE,
            "mlp_chunked_convrot_2slice",
            "ConvRot-256 TensorWise INT8 MLP uses token chunks and two feature slices",
        )

    labels = sorted(
        set(inventory.labels("fc1")) | set(inventory.labels("fc2"))
    )
    return MLPProviderResolution(
        MLP_GENERIC_CHUNKED,
        "mlp_chunked_native",
        "generic token chunking preserves the model's own linear formats: %s"
        % (", ".join(labels) or "unknown"),
    )
