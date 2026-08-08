"""Configuration for the incremental H3 hybrid sparse experiment."""

from dataclasses import dataclass
import math

MODE_SAGE128 = "sage128"
IMPLEMENTED_MODES = (MODE_SAGE128,)


@dataclass(frozen=True)
class HybridSparseConfig:
    mode: str = MODE_SAGE128
    video_budget: float = 0.5
    strict: bool = True
    run_tag: str = "hybrid50"

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                "hybrid sparse mode %r is unavailable; implemented modes: %s"
                % (self.mode, ", ".join(IMPLEMENTED_MODES))
            )
        budget = float(self.video_budget)
        if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("video_budget must be finite and in (0, 1]")
        if not str(self.run_tag).strip():
            raise ValueError("run_tag must not be empty")
