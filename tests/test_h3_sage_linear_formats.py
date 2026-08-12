"""Pure format inspection and production-provider contracts."""

import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.kernel_policy import KernelPolicy  # noqa: E402
from h3_sage_optimizations.plan import MLP_MEMORY_EPILOGUE  # noqa: E402
from h3_sage_optimizations.qkv.formats import (  # noqa: E402
    describe_linear,
    inspect_h3_linears,
)
from h3_sage_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_EPILOGUE,
    MLP_CONVROT_INT8_TWO_SLICE,
    MLP_GENERIC_CHUNKED,
    QKV_DENSE_CONVROT_INT8,
    QKV_STANDARD,
    resolve_mlp_provider,
    resolve_qkv_provider,
)


class FakeWeight:
    def __init__(
        self,
        *,
        layout=None,
        convrot=False,
        group=0,
        transposed=False,
        dtype="bf16",
        shape=(10, 10),
    ):
        self._layout_cls = layout
        self._params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=group,
            transposed=transposed,
        )
        self.dtype = dtype
        self.shape = shape


def linear(weight, bias=None):
    return SimpleNamespace(weight=weight, bias=bias)


def block(weight):
    return SimpleNamespace(
        attn=SimpleNamespace(qkv_proj=linear(weight)),
        mlp=SimpleNamespace(fc1=linear(weight), fc2=linear(weight)),
    )


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    production = KernelPolicy()
    research = KernelPolicy(allow_research_kernels=True, source="test")
    convrot = FakeWeight(
        layout="TensorWiseINT8Layout",
        convrot=True,
        group=256,
        dtype="int8",
    )
    plain = FakeWeight(layout=None, dtype="bfloat16")

    convrot_format = describe_linear(linear(convrot))
    plain_format = describe_linear(linear(plain))
    check(
        convrot_format.convrot_int8_256,
        "ConvRot-256 TensorWise INT8 is recognized",
    )
    check(
        not plain_format.convrot_int8_256,
        "plain BF16 is not mislabeled as fused-compatible",
    )

    convrot_inventory = inspect_h3_linears([block(convrot), block(convrot)])
    dense = resolve_qkv_provider(
        convrot_inventory,
        request="auto",
        backend_kind="dense_sage_sm89",
        capability=(8, 9),
        triton_available=True,
        policy=production,
    )
    check(
        dense.provider_id == QKV_STANDARD and not dense.fused,
        "production auto preserves the optimized QKV GEMM",
    )
    mlp = resolve_mlp_provider(
        convrot_inventory, request="auto", policy=production
    )
    check(
        mlp.provider_id == MLP_CONVROT_INT8_TWO_SLICE,
        "auto keeps the established Kitchen-backed two-slice MLP",
    )

    try:
        resolve_mlp_provider(
            convrot_inventory,
            request=MLP_MEMORY_EPILOGUE,
            policy=production,
        )
    except RuntimeError as exc:
        check(
            "research-kernel candidate" in str(exc),
            "epilogue prototype is blocked by production policy",
        )
    else:
        raise AssertionError("production policy accepted custom epilogue GEMMs")

    prototype = resolve_mlp_provider(
        convrot_inventory,
        request=MLP_MEMORY_EPILOGUE,
        policy=research,
    )
    check(
        prototype.provider_id == MLP_CONVROT_INT8_EPILOGUE,
        "research policy retains explicit epilogue characterization",
    )
    fused = resolve_qkv_provider(
        convrot_inventory,
        request="required",
        backend_kind="dense_sage_sm89",
        capability=(8, 9),
        triton_available=True,
        policy=research,
    )
    check(
        fused.provider_id == QKV_DENSE_CONVROT_INT8 and fused.fused,
        "research policy retains explicit fused-QKV characterization",
    )

    plain_inventory = inspect_h3_linears([block(plain), block(plain)])
    dense = resolve_qkv_provider(
        plain_inventory,
        request="auto",
        backend_kind="dense_sage_sm89",
        capability=(8, 9),
        triton_available=True,
        policy=production,
    )
    check(
        dense.provider_id == QKV_STANDARD and not dense.fused,
        "BF16 QKV stays on standard optimized dispatch",
    )
    mlp = resolve_mlp_provider(
        plain_inventory, request="auto", policy=production
    )
    check(
        mlp.provider_id == MLP_GENERIC_CHUNKED,
        "BF16 MLP preserves its optimized format through chunking",
    )
    print("\nall H3 Sage linear-format tests passed")


if __name__ == "__main__":
    main()
