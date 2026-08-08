"""MiniMax H3 sequence-scaled activation-memory controls.

The package owns DiT-block execution and token-slab MLP evaluation. Attention
representation and kernels remain in :mod:`h3_attention`.
"""

from .config import (
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MODE,
    ActivationMemoryConfig,
    IMPLEMENTED_MODES,
    MODES,
)
from .observer import OBSERVER_KEY, notify_activation, observing

__all__ = [
    "ActivationMemoryConfig",
    "DEFAULT_CHUNK_ROWS",
    "DEFAULT_MODE",
    "IMPLEMENTED_MODES",
    "MODES",
    "OBSERVER_KEY",
    "notify_activation",
    "observing",
]
