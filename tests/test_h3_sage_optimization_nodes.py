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
            "chunk_rows",
            "prefer_held_weights",
        ],
        "memory optimizer preserves its serialized input ids",
    )
    check(
        {
            item.id
            for item in memory.inputs
            if getattr(item, "advanced", False)
        }
        == {"attention", "chunk_rows", "prefer_held_weights"},
        "implementation controls are advanced while QKV and MLP remain visible",
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
        sparse.node_id == "MiniMaxH3SparseSageAttentionZi",
        "Sparse Sage has a stable production node id",
    )
    check(
        sparse_ids == ["model", "enabled", "video_budget"],
        "Sparse Sage contains no QKV or MLP controls",
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
        "Sparse Sage" in sparse.search_aliases
        and "Sparge" in sparse.search_aliases,
        "Sparse node publishes search aliases",
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
