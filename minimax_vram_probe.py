"""MiniMax-H3 VRAM probe dispatcher.

The legacy probe remains byte-for-byte in ``_minimax_vram_probe_base.py``.
Passing ``--ab-activation-memory`` dispatches to the efficient-Sage versus
activation-memory A/B probe; every other invocation preserves the original CLI
and behavior.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMFYUI_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

if not os.path.isdir(os.path.join(_COMFYUI_ROOT, "comfy")):
    raise RuntimeError(
        "Could not locate the ComfyUI root. Expected a 'comfy' package at: "
        + _COMFYUI_ROOT
    )

if _COMFYUI_ROOT not in sys.path:
    sys.path.insert(0, _COMFYUI_ROOT)

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
