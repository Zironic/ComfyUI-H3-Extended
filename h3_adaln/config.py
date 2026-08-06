"""Configuration for run-scoped MiniMax H3 AdaLN precomputation."""

from dataclasses import dataclass
import math

MODE_OFF = "off"
MODE_AUTO = "auto"
MODE_ON = "on"
MODES = (MODE_OFF, MODE_AUTO, MODE_ON)


@dataclass(frozen=True)
class AdaLNPrecomputeConfig:
    mode: str = MODE_OFF
    max_table_gib: float = 2.0
    strict: bool = False

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown AdaLN precompute mode %r" % self.mode)
        if not math.isfinite(float(self.max_table_gib)) or float(self.max_table_gib) <= 0:
            raise ValueError("adaln max_table_gib must be finite and positive")

    @property
    def enabled(self):
        return self.mode != MODE_OFF
