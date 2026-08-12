"""Contracts for the deprecated combined Hybrid Sparse adapter."""

import os
import sys
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

import h3_sparse_attention.nodes as legacy  # noqa: E402
from h3_sage_optimizations.plan import (  # noqa: E402
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    H3SageOptimizationPlan,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    print("deprecated Hybrid Sparse compatibility")
    schema = legacy.MiniMaxH3HybridSparseAttention.define_schema()
    ids = [item.id for item in schema.inputs]
    check(schema.is_deprecated, "legacy combined node is marked deprecated")
    check(
        ids == [
            "model",
            "enabled",
            "mode",
            "video_budget",
            "strict",
            "activation",
            "chunk_rows",
            "run_tag",
            "timing",
            "compile_backend",
            "density_mode",
            "min_video_density",
            "max_video_density",
            "adaptive_temperature",
            "adaptive_target_mass",
        ],
        "legacy widget order is preserved for saved workflows",
    )
    check(
        "legacy H3 sparse" in schema.search_aliases,
        "legacy node publishes migration search aliases",
    )

    marker = object()
    captured = {}
    patched = object()

    def apply_plan(model, plan):
        captured["model"] = model
        captured["plan"] = plan
        return patched

    with mock.patch.object(
        legacy,
        "read_plan",
        return_value=H3SageOptimizationPlan(),
    ), mock.patch.object(
        legacy,
        "apply_plan",
        side_effect=apply_plan,
    ), mock.patch.object(
        legacy,
        "format_legacy_status",
        return_value="compatibility status",
    ):
        result = legacy.MiniMaxH3HybridSparseAttention.execute(
            marker,
            mode="sage128_fused_qkv",
            video_budget=0.4,
            strict=True,
            activation="mlp_chunked_native",
            timing=True,
        )

    plan = captured["plan"]
    check(
        captured["model"] is marker and result.args[0] is patched,
        "legacy node delegates execution to the production apply path",
    )
    check(
        plan.memory.fused_qkv == FUSED_QKV_REQUIRED,
        "strict legacy fused mode becomes an immediate required request",
    )
    check(
        plan.memory.mlp_memory == MLP_MEMORY_LEGACY_NATIVE,
        "legacy native MLP mode is preserved internally",
    )
    check(
        plan.sparse.video_budget == 0.4,
        "legacy video budget becomes the production Sparse request",
    )
    check(
        result.ui.value == "compatibility status",
        "legacy adapter returns visible migration status",
    )

    cases = (
        ({"compile_backend": "bogus"}, "compile backend"),
        ({"compile_backend": "inductor"}, "sage128_fused_qkv"),
        (
            {
                "compile_backend": "inductor",
                "mode": "sage128_fused_qkv",
            },
            "convrot_2slice",
        ),
        ({"chunk_rows": 128}, "chunk_rows"),
    )
    for kwargs, expected in cases:
        try:
            legacy.MiniMaxH3HybridSparseAttention.execute(
                marker, **kwargs
            )
        except ValueError as exc:
            check(
                expected in str(exc),
                "legacy %s validation remains preflight-safe" % expected,
            )
        else:
            raise AssertionError(
                "legacy validation accepted %r" % (kwargs,)
            )

    try:
        legacy.MiniMaxH3HybridSparseAttention.execute(
            marker, density_mode="adaptive_budget"
        )
    except ValueError as exc:
        check(
            "fixed density only" in str(exc),
            "unsupported adaptive legacy workflows fail with migration guidance",
        )
    else:
        raise AssertionError("adaptive compatibility request was accepted")

    try:
        legacy.MiniMaxH3HybridSparseAttention.execute(
            marker,
            compile_backend="inductor",
            mode="sage128_fused_qkv",
            activation="mlp_chunked_convrot_2slice",
        )
    except ValueError as exc:
        check(
            "does not support shared Inductor" in str(exc),
            "fully valid legacy compile request reaches migration guidance",
        )
    else:
        raise AssertionError("compiled compatibility request was accepted")

    print("\nall deprecated Hybrid Sparse compatibility tests passed")


if __name__ == "__main__":
    main()
