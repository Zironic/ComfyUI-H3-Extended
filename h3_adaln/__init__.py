"""Run-scoped MiniMax H3 AdaLN precomputation."""

from .config import (
    AdaLNPrecomputeConfig,
    MODE_AUTO,
    MODE_OFF,
    MODE_ON,
    MODES,
)
from .patch import install
from .provider import AdaLNProvider, AdaLNPrecomputeError

__all__ = [
    "AdaLNPrecomputeConfig",
    "AdaLNProvider",
    "AdaLNPrecomputeError",
    "MODE_AUTO",
    "MODE_OFF",
    "MODE_ON",
    "MODES",
    "install",
]
