"""MiniMax-H3 VRAM probe dispatcher.

The legacy probe remains byte-for-byte in ``_minimax_vram_probe_base.py``.
Passing ``--ab-activation-memory`` dispatches to the efficient-Sage versus
activation-memory A/B probe; every other invocation preserves the original CLI
and behavior.
"""

import sys

# Preserve the module-level helper API for anything that imports this file.
from _minimax_vram_probe_base import *  # noqa: F401,F403
from _minimax_vram_probe_base import main as _legacy_main


def main():
    if "--ab-activation-memory" in sys.argv[1:]:
        from minimax_vram_probe_ab import main as ab_main
        return ab_main()
    return _legacy_main()


if __name__ == "__main__":
    main()
