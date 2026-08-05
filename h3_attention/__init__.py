"""H3-owned attention observation, forward patching, and efficient Sage backend."""

from .observer import (
    OBSERVER_KEY,
    OBSERVED_KEY,
    notify_attention,
    observing,
    marked_observed,
)
from .sage_mem_eff import SM89SageMemoryEfficientBackend

__all__ = [
    "OBSERVER_KEY",
    "OBSERVED_KEY",
    "notify_attention",
    "observing",
    "marked_observed",
    "SM89SageMemoryEfficientBackend",
]
