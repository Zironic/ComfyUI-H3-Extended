"""Pure contracts for the two-node H3 Sage optimization plan."""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.plan import (  # noqa: E402
    FUSED_QKV_AUTO,
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
        and memory.mlp_memory == "auto",
        "format-aware QKV and MLP selection default to auto",
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
    print("\nall H3 Sage optimization plan tests passed")


if __name__ == "__main__":
    main()
