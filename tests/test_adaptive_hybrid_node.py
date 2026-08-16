"""Node-surface contracts for adaptive H3 hybrid sparse attention."""

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

from h3_attention.hybrid import (  # noqa: E402
    DENSITY_ADAPTIVE_BUDGET,
    DENSITY_FIXED,
)
from h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttention  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def test_schema():
    print("adaptive node schema")
    schema = MiniMaxH3HybridSparseAttention.define_schema()
    ids = [item.id for item in schema.inputs]
    check(
        ids[:10] == [
            "model", "enabled", "mode", "video_budget", "strict",
            "activation", "chunk_rows", "run_tag", "timing",
            "compile_backend",
        ],
        "all established widgets retain their original order",
    )
    inputs = {item.id: item for item in schema.inputs}
    check(inputs["density_mode"].default == DENSITY_FIXED,
          "fixed density remains the backward-compatible default")
    check(inputs["density_mode"].options == [DENSITY_FIXED, DENSITY_ADAPTIVE_BUDGET],
          "node exposes fixed and adaptive-budget density modes")
    check(inputs["min_video_density"].default == 0.05,
          "node exposes the adaptive minimum rail")
    check(inputs["max_video_density"].default == 0.50,
          "node exposes the adaptive maximum rail")
    check(inputs["adaptive_temperature"].default == 1.0,
          "node exposes adaptive score temperature")
    check(inputs["adaptive_target_mass"].default == 0.80,
          "node exposes adaptive cumulative-mass target")


def test_compile_rejection_precedes_preflight():
    print("adaptive shared-compile guard")
    marker = object()
    with mock.patch(
        "h3_sparse_attention.nodes.RuntimeEnvironment.detect",
        side_effect=AssertionError("preflight must not run"),
    ):
        try:
            MiniMaxH3HybridSparseAttention.execute(
                marker,
                density_mode=DENSITY_ADAPTIVE_BUDGET,
                video_budget=0.20,
                min_video_density=0.05,
                max_video_density=0.50,
                compile_backend="inductor",
            )
        except ValueError as exc:
            check("fixed density_mode" in str(exc),
                  "adaptive compile request fails with an explicit boundary")
        else:
            raise AssertionError("adaptive shared compilation was accepted")


def main():
    test_schema()
    test_compile_rejection_precedes_preflight()
    print("\nall adaptive hybrid node tests passed")


if __name__ == "__main__":
    main()
