"""Pure UI status formatting contracts for H3 Sage optimizations."""

import os
import sys
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_sage_optimizations.plan import STATUS_KEY  # noqa: E402
from h3_sage_optimizations.status import (  # noqa: E402
    format_legacy_status,
    format_memory_status,
    format_sparse_status,
)


def model_with_status(status):
    return SimpleNamespace(
        model_options={"transformer_options": {STATUS_KEY: status}}
    )


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def main():
    print("H3 Sage UI status")
    skipped = SimpleNamespace(model_options={})
    check(
        "not MiniMax H3" in format_memory_status(skipped),
        "missing status is reported as non-H3 pass-through",
    )

    model = model_with_status(
        {
            "attention": {"selected": "dense_sage_sm89"},
            "fused_qkv": {
                "provider": "standard_h3_qkv",
                "reason": "no fused provider supports TensorCoreFP8Layout",
            },
            "mlp": {
                "provider": "generic_chunked_quantized",
                "reason": "preserves TensorCoreFP8Layout",
                "chunk_rows": 2048,
            },
            "sparse": {"video_budget": 0.5},
        }
    )
    memory = format_memory_status(model)
    check(
        "dense_sage_sm89" in memory
        and "standard_h3_qkv" in memory
        and "2048-row chunks" in memory,
        "memory status surfaces selected providers and chunk size",
    )
    sparse = format_sparse_status(model)
    check(
        "50.0%" in sparse
        and "rounded up to a whole KV-tile count" in sparse
        and "mixed boundary tiles remain dense" in sparse,
        "Sparse status explains requested versus effective density",
    )
    legacy = format_legacy_status(
        model, warnings=("Legacy timing is ignored.",)
    )
    check(
        "Deprecated compatibility node" in legacy
        and "Legacy timing is ignored." in legacy,
        "legacy status carries migration and ignored-option warnings",
    )
    print("\nall H3 Sage status tests passed")


if __name__ == "__main__":
    main()
