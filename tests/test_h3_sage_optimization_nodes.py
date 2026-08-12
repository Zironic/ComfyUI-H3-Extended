"""Public schema and UI contracts for the split H3 Sage nodes."""

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _ROOT)

sys.argv = [sys.argv[0], "--cpu"]
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from h3_sage_optimizations.nodes import (  # noqa: E402
    MiniMaxH3SageMemoryOptimizer,
    MiniMaxH3SparseSageAttention,
    _resolve_mlp_request,
)
from h3_sage_optimizations.plan import (  # noqa: E402
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def input_by_id(schema, input_id):
    return next(item for item in schema.inputs if item.id == input_id)


def main():
    print("split H3 Sage node schemas")
    memory = MiniMaxH3SageMemoryOptimizer.define_schema()
    sparse = MiniMaxH3SparseSageAttention.define_schema()

    memory_ids = [item.id for item in memory.inputs]
    sparse_ids = [item.id for item in sparse.inputs]
    check(
        memory.node_id == "MiniMaxH3SageMemoryOptimizerZi",
        "memory optimizer has a stable production node id",
    )
    check(
        memory_ids == [
            "model",
            "enabled",
            "attention",
            "fused_qkv",
            "mlp_memory",
            "mlp_execution",
            "strict",
            "chunk_rows",
            "prefer_held_weights",
        ],
        "memory optimizer exposes all meaningful execution controls",
    )
    check(
        {
            item.id
            for item in memory.inputs
            if getattr(item, "advanced", False)
        }
        == {
            "attention",
            "mlp_execution",
            "strict",
            "chunk_rows",
            "prefer_held_weights",
        },
        "explicit provider and fallback controls are advanced",
    )
    check(
        input_by_id(memory, "attention").display_name
        == "Dense attention when Sparse is absent",
        "dense attention has a user-facing label",
    )
    check(
        input_by_id(memory, "fused_qkv").display_name
        == "QKV projection optimization",
        "QKV control has a user-facing label",
    )
    check(
        input_by_id(memory, "mlp_memory").display_name
        == "MLP memory optimization",
        "MLP control has a user-facing label",
    )
    check(
        input_by_id(memory, "mlp_execution").options
        == ["auto", "chunked_bf16", "chunked_native", "convrot_two_slice"],
        "former explicit MLP execution choices are available under Advanced",
    )
    check(
        input_by_id(memory, "strict").default is False,
        "safe fallback remains the ordinary production default",
    )
    check(
        input_by_id(memory, "prefer_held_weights").display_name
        == "Hold weights across chunks",
        "held-weight policy is named descriptively",
    )
    check(
        "H3 VRAM" in memory.search_aliases
        and "H3 fused QKV" in memory.search_aliases,
        "memory node publishes search aliases",
    )

    fused = input_by_id(memory, "fused_qkv")
    mlp = input_by_id(memory, "mlp_memory")
    check(
        fused.default == "auto" and mlp.default == "auto",
        "safe format-aware selection is the production default",
    )
    check(
        _resolve_mlp_request("auto", "chunked_bf16")
        == MLP_MEMORY_LEGACY_BF16
        and _resolve_mlp_request("auto", "chunked_native")
        == MLP_MEMORY_LEGACY_NATIVE
        and _resolve_mlp_request("auto", "convrot_two_slice")
        == MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
        "advanced MLP overrides preserve the established execution modes",
    )

    check(
        sparse.node_id == "MiniMaxH3SparseSageAttentionZi",
        "Sparse Sage has a stable production node id",
    )
    check(
        sparse_ids == [
            "model",
            "enabled",
            "video_budget",
            "density_mode",
            "min_video_density",
            "max_video_density",
            "adaptive_temperature",
            "adaptive_target_mass",
            "strict",
            "write_report",
            "timing",
            "run_tag",
            "compile_backend",
        ],
        "Sparse Sage exposes every meaningful former sparse control",
    )
    check(
        {
            item.id
            for item in sparse.inputs
            if getattr(item, "advanced", False)
        }
        == set(sparse_ids[3:]),
        "all optional routing, validation, reporting, and compile controls are advanced",
    )
    budget = input_by_id(sparse, "video_budget")
    check(
        budget.display_name == "Video KV budget",
        "Sparse budget has a user-facing label",
    )
    check(
        "rounded up to a whole KV-tile count" in budget.tooltip
        and "mixed boundary tiles remain dense" in budget.tooltip
        and "1.0" in budget.tooltip,
        "Sparse budget tooltip explains quantization and dense context",
    )
    check(
        input_by_id(sparse, "density_mode").options
        == ["fixed", "adaptive_budget"]
        and input_by_id(sparse, "max_video_density").default == 1.0,
        "adaptive routing is available with room to redistribute by default",
    )
    check(
        input_by_id(sparse, "timing").default is False
        and input_by_id(sparse, "write_report").default is False
        and input_by_id(sparse, "compile_backend").default == "off",
        "diagnostics and shared compilation remain opt-in",
    )
    check(
        "Sparse Sage" in sparse.search_aliases
        and "H3 adaptive attention" in sparse.search_aliases,
        "Sparse node publishes fixed and adaptive search aliases",
    )

    marker = object()
    memory_out = MiniMaxH3SageMemoryOptimizer.execute(
        marker, enabled=False
    )
    sparse_out = MiniMaxH3SparseSageAttention.execute(
        marker, enabled=False
    )
    check(
        memory_out.args[0] is marker
        and "disabled" in memory_out.ui.value.lower(),
        "disabled memory optimizer is pass-through with visible status",
    )
    check(
        sparse_out.args[0] is marker
        and "disabled" in sparse_out.ui.value.lower(),
        "disabled Sparse Sage is pass-through with visible status",
    )
    print("\nall split H3 Sage node tests passed")


if __name__ == "__main__":
    main()
