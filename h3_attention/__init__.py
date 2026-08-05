"""H3-owned attention observation, forward patching, and efficient Sage backend."""

from .observer import (
    OBSERVER_KEY,
    OBSERVED_KEY,
    notify_attention,
    observing,
    marked_observed,
)
from . import sage_mem_eff as _sage_mem_eff
from . import sm89_compat as _sm89_compat  # installs hashed torch.ops discovery

SM89SageMemoryEfficientBackend = _sage_mem_eff.SM89SageMemoryEfficientBackend

__all__ = [
    "OBSERVER_KEY",
    "OBSERVED_KEY",
    "notify_attention",
    "observing",
    "marked_observed",
    "SM89SageMemoryEfficientBackend",
]
