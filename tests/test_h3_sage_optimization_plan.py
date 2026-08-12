"""Pure contracts for the two-node H3 Sage optimization plan."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.plan import (  # noqa: E402
    COMPILE_INDUCTOR,
    DENSITY_ADAPTIVE_BUDGET,
    FUSED_QKV_AUTO,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_EPILOGUE,
    H3SageOptimizationPlan,
    MemoryRequest,
    SparseRequest,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def expect_error(fn, text):
    try:
        fn()
    except (TypeError, ValueError) as exc:
        check(text in str(exc), text)
    else:
        raise AssertionError("expected an error containing %r" % text)


def main():
    print("H3 Sage optimization plan")
    memory = MemoryRequest()
    sparse = SparseRequest(video_budget=0.5)

    first = (
        H3SageOptimizationPlan()
        .with_memory(memory)
        .with_sparse(sparse)
    )
    second = (
        H3SageOptimizationPlan()
        .with_sparse(sparse)
        .with_memory(memory)
    )
    check(
        first == second and first.signature == second.signature,
        "Memory and Sparse nodes compose independently of order",
    )
    check(
        memory.fused_qkv == FUSED_QKV_AUTO
        and memory.mlp_memory == "auto"
        and memory.strict is False,
        "format-aware QKV and MLP selection default to safe fallback",
    )
    strict_memory = MemoryRequest(strict=True)
    check(
        strict_memory.fused_qkv == FUSED_QKV_REQUIRED
        and strict_memory.strict,
        "strict Memory mode canonicalizes fused auto into a required request",
    )
    check(
        MemoryRequest(fused_qkv="off", strict=True).fused_qkv == "off",
        "strict mode does not re-enable explicitly disabled QKV optimization",
    )
    prototype = MemoryRequest(mlp_memory=MLP_MEMORY_EPILOGUE)
    check(
        prototype.mlp_memory == "epilogue_prototype",
        "the epilogue prototype is an explicit plan request",
    )
    check(
        sparse.density_mode == "fixed"
        and sparse.max_video_density == 1.0
        and not sparse.reporting_enabled,
        "production Sparse defaults are fixed, unconstrained above, and quiet",
    )

    adaptive = SparseRequest(
        video_budget=0.4,
        density_mode=DENSITY_ADAPTIVE_BUDGET,
        min_video_density=0.1,
        max_video_density=0.9,
        adaptive_temperature=0.75,
        adaptive_target_mass=0.85,
        write_report=True,
        timing=True,
        run_tag="adaptive40",
    )
    check(
        adaptive.reporting_enabled
        and adaptive.signature != sparse.signature,
        "adaptive rails, diagnostics, and run tags participate in the plan",
    )
    check(
        first.with_memory(memory) == first,
        "reapplying the same Memory request is idempotent",
    )
    check(
        first.with_sparse(sparse) == first,
        "reapplying the same Sparse request is idempotent",
    )

    expect_error(
        lambda: first.with_memory(
            MemoryRequest(fused_qkv="off")
        ),
        "different H3 Sage Memory Optimizer",
    )
    expect_error(
        lambda: first.with_sparse(
            SparseRequest(video_budget=0.4)
        ),
        "different H3 Sparse Sage",
    )
    expect_error(
        lambda: SparseRequest(video_budget=0.0),
        "video_budget",
    )
    expect_error(
        lambda: SparseRequest(
            video_budget=0.4,
            density_mode=DENSITY_ADAPTIVE_BUDGET,
            min_video_density=0.5,
            max_video_density=0.9,
        ),
        "must lie between",
    )
    expect_error(
        lambda: SparseRequest(run_tag="bad tag"),
        "run_tag",
    )
    expect_error(
        lambda: SparseRequest(
            density_mode=DENSITY_ADAPTIVE_BUDGET,
            compile_backend=COMPILE_INDUCTOR,
        ),
        "requires fixed",
    )
    print("\nall H3 Sage optimization plan tests passed")


if __name__ == "__main__":
    main()
