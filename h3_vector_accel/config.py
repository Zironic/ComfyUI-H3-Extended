"""Immutable, validated configuration for the fixed vector-accel study."""

from dataclasses import dataclass, field
import math

METHODS = ("native", "hold", "linear_velocity")
PROFILES = ("native_20", "conservative_12", "early_aggressive_13", "uniform_13", "late_aggressive_13")
DIAGNOSTICS = ("off", "summary", "full")
MASK_VERSION = "v1"
PREDICTOR_VERSION = "v1"
CURVATURE_RATIO = 0.5
MIN_DIRECTION_COSINE = 0.0
DEFAULT_MAX_EXTRAPOLATION_RATIO = 1.5

_MASKS = {
    "native_20": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    "conservative_12": (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19),
    "early_aggressive_13": (0, 1, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19),
    "uniform_13": (0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19),
    "late_aggressive_13": (0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 17, 18, 19),
}


def profile_mask(profile: str, logical_steps: int = 20) -> tuple[bool, ...]:
    """Return the immutable actual-evaluation mask for a named profile."""
    if profile not in PROFILES:
        raise ValueError(f"unknown evaluation profile: {profile}")
    if logical_steps != 20:
        raise ValueError(f"evaluation profile {profile} requires exactly 20 logical steps")
    mask = [False] * logical_steps
    for index in _MASKS[profile]:
        mask[index] = True
    return tuple(mask)


def actual_mask(profile: str, logical_steps: int = 20) -> tuple[int, ...]:
    return tuple(i for i, actual in enumerate(profile_mask(profile, logical_steps)) if actual)


@dataclass(frozen=True)
class SamplerConfig:
    method: str = "native"
    evaluation_profile: str = "native_20"
    diagnostics: str = "off"
    fallback_on_guard: bool = True
    max_extrapolation_ratio: float = DEFAULT_MAX_EXTRAPOLATION_RATIO
    curvature_ratio: float = CURVATURE_RATIO
    min_direction_cosine: float = MIN_DIRECTION_COSINE
    mask_version: str = MASK_VERSION
    predictor_version: str = PREDICTOR_VERSION
    quality_preset: str | None = None
    adaptive_profile_hash: str | None = None
    _mask: tuple[bool, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown vector acceleration method: {self.method}")
        if self.evaluation_profile not in PROFILES:
            raise ValueError(f"unknown evaluation profile: {self.evaluation_profile}")
        if self.diagnostics not in DIAGNOSTICS:
            raise ValueError(f"unknown diagnostics mode: {self.diagnostics}")
        for name, value in (("max_extrapolation_ratio", self.max_extrapolation_ratio),
                            ("curvature_ratio", self.curvature_ratio),
                            ("min_direction_cosine", self.min_direction_cosine)):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_extrapolation_ratio <= 0:
            raise ValueError("max_extrapolation_ratio must be positive")
        if self.curvature_ratio < 0:
            raise ValueError("curvature_ratio must be non-negative")
        if not -1 <= self.min_direction_cosine <= 1:
            raise ValueError("min_direction_cosine must be between -1 and 1")
        object.__setattr__(self, "_mask", profile_mask(self.evaluation_profile, 20))

    @property
    def mask(self) -> tuple[bool, ...]:
        return self._mask

    @property
    def actual_indices(self) -> tuple[int, ...]:
        return tuple(i for i, value in enumerate(self._mask) if value)

    def validate_schedule_length(self, logical_steps: int) -> None:
        if logical_steps < 1:
            raise ValueError("sigma schedule must contain at least one derivative interval")
        if self.method != "native" and logical_steps != 20:
            raise ValueError(f"{self.evaluation_profile} requires exactly 20 logical steps")
