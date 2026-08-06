"""Approximate signal-driven block-stack cache for MiniMax H3."""

from .config import (
    FirstBlockCacheConfig,
    MODE_FIRST_BLOCK,
    MODE_OFF,
    MODES,
)
from .coordinator import FirstBlockCacheCoordinator
from .patch import install

__all__ = [
    "FirstBlockCacheConfig",
    "FirstBlockCacheCoordinator",
    "MODE_FIRST_BLOCK",
    "MODE_OFF",
    "MODES",
    "install",
]
