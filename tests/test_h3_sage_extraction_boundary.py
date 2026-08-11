"""Static guard against regressing into the monolithic optimizer layer."""

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = _HERE.parent / "h3_sage_optimizations"


def main():
    apply_source = (_PACKAGE / "apply.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "h3_memory_optimizer",
        "h3_adaln",
        "h3_block_cache",
        "h3_attention.sol",
    )
    found = [
        name for name in forbidden if name in apply_source
    ]
    if found:
        raise AssertionError(
            "production apply path imports unrelated orchestration: %s"
            % ", ".join(found)
        )
    print(
        "  ok: production apply path does not use the monolithic optimizer"
    )


if __name__ == "__main__":
    main()
