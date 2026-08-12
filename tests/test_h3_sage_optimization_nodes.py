"""Public schema contracts for the split H3 Sage optimization nodes."""

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
        "memory optimizer exposes format-neutral controls",
    )
    fused = next(
        item for item in memory.inputs
        if item.id == "fused_qkv"
    )
    mlp = next(
        item for item in memory.inputs
        if item.id == "mlp_memory"
    )
    check(
        fused.default == "auto" and mlp.default == "auto",
        "safe format-aware selection is the production default",
    )
    check(
        mlp.options == ["auto", "epilogue_prototype", "off"],
        "the prototype is explicit and does not replace auto",
    )
    check(
        sparse.node_id == "MiniMaxH3SparseSageAttentionZi",
        "Sparse Sage has a stable production node id",
    )
    check(
        sparse_ids == ["model", "enabled", "video_budget"],
        "Sparse Sage contains no QKV or MLP controls",
    )

    marker = object()
    check(
        MiniMaxH3SageMemoryOptimizer.execute(
            marker, enabled=False
        ).args[0]
        is marker,
        "disabled memory optimizer is an exact pass-through",
    )
    check(
        MiniMaxH3SparseSageAttention.execute(
            marker, enabled=False
        ).args[0]
        is marker,
        "disabled Sparse Sage is an exact pass-through",
    )
    print("\nall split H3 Sage node tests passed")


if __name__ == "__main__":
    main()
