"""Configuration for approximate H3 FirstBlockCache."""

from dataclasses import dataclass
import math

MODE_OFF = "off"
MODE_FIRST_BLOCK = "first_block"
MODES = (MODE_OFF, MODE_FIRST_BLOCK)


@dataclass(frozen=True)
class FirstBlockCacheConfig:
    mode: str = MODE_OFF
    threshold: float = 0.08
    warmup_steps: int = 3
    strict: bool = False
    collective: bool = True

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown block-cache mode %r" % self.mode)
        if not math.isfinite(float(self.threshold)) or float(self.threshold) < 0:
            raise ValueError("FirstBlockCache threshold must be finite and non-negative")
        if int(self.warmup_steps) < 0:
            raise ValueError("FirstBlockCache warmup_steps must be non-negative")

    @property
    def enabled(self):
        return self.mode == MODE_FIRST_BLOCK
