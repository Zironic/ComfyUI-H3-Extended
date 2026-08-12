"""Kernel bucket and production-auto contracts."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.kernel_policy import (  # noqa: E402
    CONVROT_TWO_SLICE_MLP,
    FUSED_QKV_TRITON,
    MLP_EPILOGUE_TRITON,
    RESEARCH_KERNELS_ENV,
    SHARED_INDUCTOR,
    STANDARD_QKV,
    KernelBucket,
    KernelPolicy,
    benchmarkable_candidates,
    candidate_by_id,
    research_candidates,
)
from h3_sage_optimizations.plan import MLP_MEMORY_EPILOGUE  # noqa: E402
from h3_sage_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_EPILOGUE,
    MLP_CONVROT_INT8_TWO_SLICE,
    QKV_DENSE_CONVROT_INT8,
    QKV_STANDARD,
    resolve_mlp_provider,
    resolve_qkv_provider,
)


class Inventory:
    qkv = (object(),)
    fc1 = (object(),)
    fc2 = (object(),)
    qkv_convrot_int8_256 = True
    mlp_convrot_int8_256 = True

    @staticmethod
    def homogeneous(name):
        return True

    @staticmethod
    def labels(name):
        return ("TensorWiseINT8Layout+convrot256",)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def expect_runtime(fn, text):
    try:
        fn()
    except RuntimeError as exc:
        check(text in str(exc), text)
    else:
        raise AssertionError("expected RuntimeError containing %r" % text)


def main():
    print("H3 Sage kernel policy")
    production = KernelPolicy()
    research = KernelPolicy(allow_research_kernels=True, source="test")

    check(
        candidate_by_id(CONVROT_TWO_SLICE_MLP).bucket
        == KernelBucket.EXISTING_OPTIMIZED_KERNEL,
        "Kitchen-backed two-slice MLP is in bucket 1",
    )
    check(
        candidate_by_id(FUSED_QKV_TRITON).bucket
        == KernelBucket.REQUIRES_NEW_KERNEL
        and candidate_by_id(MLP_EPILOGUE_TRITON).bucket
        == KernelBucket.REQUIRES_NEW_KERNEL
        and candidate_by_id(SHARED_INDUCTOR).bucket
        == KernelBucket.REQUIRES_NEW_KERNEL,
        "custom-GEMM and dependent compile paths are in bucket 2",
    )
    check(
        all(item.ab_benchmarkable for item in benchmarkable_candidates())
        and all(not item.ab_benchmarkable for item in research_candidates()),
        "ordinary adoption A/B tests are restricted to bucket 1",
    )

    qkv = resolve_qkv_provider(
        Inventory(),
        request="auto",
        backend_kind="dense_sage_sm89",
        capability=(8, 9),
        triton_available=True,
        policy=production,
    )
    check(
        qkv.provider_id == QKV_STANDARD
        and not qkv.fused
        and qkv.candidate_id == STANDARD_QKV,
        "production QKV auto preserves the existing optimized GEMM",
    )

    expect_runtime(
        lambda: resolve_qkv_provider(
            Inventory(),
            request="required",
            backend_kind="dense_sage_sm89",
            capability=(8, 9),
            triton_available=True,
            policy=production,
        ),
        "research-kernel candidate",
    )
    qkv = resolve_qkv_provider(
        Inventory(),
        request="required",
        backend_kind="dense_sage_sm89",
        capability=(8, 9),
        triton_available=True,
        policy=research,
    )
    check(
        qkv.provider_id == QKV_DENSE_CONVROT_INT8
        and qkv.fused
        and qkv.candidate_id == FUSED_QKV_TRITON,
        "the explicit research policy can characterize fused QKV",
    )

    mlp = resolve_mlp_provider(
        Inventory(), request="auto", policy=production
    )
    check(
        mlp.provider_id == MLP_CONVROT_INT8_TWO_SLICE
        and mlp.candidate_id == CONVROT_TWO_SLICE_MLP,
        "production MLP auto retains the Kitchen GEMM path",
    )
    expect_runtime(
        lambda: resolve_mlp_provider(
            Inventory(), request=MLP_MEMORY_EPILOGUE, policy=production
        ),
        "research-kernel candidate",
    )
    mlp = resolve_mlp_provider(
        Inventory(), request=MLP_MEMORY_EPILOGUE, policy=research
    )
    check(
        mlp.provider_id == MLP_CONVROT_INT8_EPILOGUE
        and mlp.candidate_id == MLP_EPILOGUE_TRITON,
        "the explicit research policy can characterize MLP epilogues",
    )

    env_policy = KernelPolicy.from_environment({RESEARCH_KERNELS_ENV: "1"})
    check(
        env_policy.allow_research_kernels,
        "the research gate is explicit and environment-scoped",
    )
    print("\nall H3 Sage kernel-policy tests passed")


if __name__ == "__main__":
    main()
