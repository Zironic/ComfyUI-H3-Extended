"""H3 attention probe: token-layout metadata, selective instrumentation, reports.

Stage 1 of sparse-attention work. This measures what a mask could drop; it does
not implement one.
"""

from . import capture, layout, metrics, report
from . import q_mask_sharing_sweep
from . import router_characterization

__all__ = [
    "capture",
    "layout",
    "metrics",
    "report",
    "q_mask_sharing_sweep",
    "router_characterization",
]
