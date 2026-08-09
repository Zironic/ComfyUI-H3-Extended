"""Configuration for the incremental H3 hybrid sparse experiment."""

from dataclasses import dataclass
import math

MODE_SAGE128 = "sage128"
MODE_SAGE128_FUSED_QKV = "sage128_fused_qkv"
IMPLEMENTED_MODES = (MODE_SAGE128, MODE_SAGE128_FUSED_QKV)

DENSITY_FIXED = "fixed"
DENSITY_ADAPTIVE_BUDGET = "adaptive_budget"
DENSITY_MODES = (DENSITY_FIXED, DENSITY_ADAPTIVE_BUDGET)


@dataclass(frozen=True)
class HybridSparseConfig:
    mode: str = MODE_SAGE128
    video_budget: float = 0.5
    strict: bool = True
    run_tag: str = "hybrid50"
    timing: bool = False
    density_mode: str = DENSITY_FIXED
    min_video_density: float = 0.05
    max_video_density: float = 0.50
    adaptive_temperature: float = 1.0
    adaptive_target_mass: float = 0.80

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                "hybrid sparse mode %r is unavailable; implemented modes: %s"
                % (self.mode, ", ".join(IMPLEMENTED_MODES))
            )
        budget = float(self.video_budget)
        if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("video_budget must be finite and in (0, 1]")
        if self.density_mode not in DENSITY_MODES:
            raise ValueError(
                "density_mode %r is unavailable; implemented modes: %s"
                % (self.density_mode, ", ".join(DENSITY_MODES))
            )
        minimum = float(self.min_video_density)
        maximum = float(self.max_video_density)
        if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
            raise ValueError("min_video_density must be finite and in (0, 1]")
        if not math.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("max_video_density must be finite and in (0, 1]")
        if minimum > maximum:
            raise ValueError("min_video_density must not exceed max_video_density")
        if self.density_mode == DENSITY_ADAPTIVE_BUDGET:
            if budget < minimum or budget > maximum:
                raise ValueError(
                    "adaptive video_budget must lie between min_video_density "
                    "and max_video_density"
                )
        temperature = float(self.adaptive_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("adaptive_temperature must be finite and greater than zero")
        target_mass = float(self.adaptive_target_mass)
        if not math.isfinite(target_mass) or not 0.0 < target_mass <= 1.0:
            raise ValueError("adaptive_target_mass must be finite and in (0, 1]")
        if not str(self.run_tag).strip():
            raise ValueError("run_tag must not be empty")

    @property
    def signature(self):
        return (
            self.mode,
            float(self.video_budget),
            bool(self.strict),
            str(self.run_tag),
            bool(self.timing),
            str(self.density_mode),
            float(self.min_video_density),
            float(self.max_video_density),
            float(self.adaptive_temperature),
            float(self.adaptive_target_mass),
        )
