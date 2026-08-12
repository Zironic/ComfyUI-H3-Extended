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
    COMPILE_INDUCTOR,
    DENSITY_ADAPTIVE_BUDGET,
    FUSED_QKV_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    H3SageOptimizationPlan,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def capture_execution(marker, **kwargs):
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
            marker, **kwargs
        )
    return captured, patched, result


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
    captured, patched, result = capture_execution(
        marker,
        mode="sage128_fused_qkv",
        video_budget=0.4,
        strict=True,
        activation="mlp_chunked_native",
        run_tag="legacy40",
        timing=True,
        density_mode=DENSITY_ADAPTIVE_BUDGET,
        min_video_density=0.1,
        max_video_density=0.9,
        adaptive_temperature=0.75,
        adaptive_target_mass=0.85,
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
        plan.sparse.video_budget == 0.4
        and plan.sparse.density_mode == DENSITY_ADAPTIVE_BUDGET
        and plan.sparse.min_video_density == 0.1
        and plan.sparse.max_video_density == 0.9,
        "legacy adaptive budget and density rails are preserved",
    )
    check(
        plan.sparse.adaptive_temperature == 0.75
        and plan.sparse.adaptive_target_mass == 0.85,
        "legacy adaptive scoring controls are preserved",
    )
    check(
        plan.sparse.write_report
        and plan.sparse.timing
        and plan.sparse.run_tag == "legacy40",
        "legacy structural reports, CUDA timing, and run tags are preserved",
    )
    check(
        result.ui.value == "compatibility status",
        "legacy adapter returns visible migration status",
    )

    compiled, _patched, _result = capture_execution(
        marker,
        mode="sage128_fused_qkv",
        activation="mlp_chunked_convrot_2slice",
        compile_backend=COMPILE_INDUCTOR,
        density_mode="fixed",
    )
    check(
        compiled["plan"].sparse.compile_backend == COMPILE_INDUCTOR,
        "fully valid legacy compile requests reach the production plan",
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
        (
            {
                "compile_backend": "inductor",
                "mode": "sage128_fused_qkv",
                "activation": "mlp_chunked_convrot_2slice",
                "density_mode": "adaptive_budget",
            },
            "fixed density_mode",
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

    print("\nall deprecated Hybrid Sparse compatibility tests passed")


if __name__ == "__main__":
    main()
