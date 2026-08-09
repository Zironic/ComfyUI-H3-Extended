"""Immutable, validated configuration for H3 vector acceleration."""

from dataclasses import dataclass, field
import math

PREDICTOR_METHODS = frozenset(("hold", "linear_velocity", "vde"))
CORE_SOLVER_METHODS = frozenset(("euler", "res_multistep"))
METHODS = ("euler", "res_multistep", "hold", "linear_velocity", "vde")
PROFILES = (
    "full_20", "late_aggressive_13", "late_cautious_14",
    "late_aggressive_12", "late_max_11", "conservative_12",
    "early_aggressive_13", "uniform_13",
)
ADAPTIVE_PROFILES = frozenset(("adaptive_history_v1", "adaptive_history_v2", "adaptive_history_v3", "adaptive_embedded_res_v1"))
EVALUATION_PROFILES = PROFILES + ("adaptive_history_v1", "adaptive_history_v2", "adaptive_history_v3", "adaptive_embedded_res_v1")
DIAGNOSTICS = ("off", "summary", "full")
POLICIES = ("fixed", "adaptive_repair")
QUALITY_PRESETS = ("conservative", "balanced", "aggressive")
CONDITIONING_MODES = ("default", "text_to_av", "reference_to_av", "continuation")
MASK_VERSION = "v1"
PREDICTOR_VERSION = "v2"
CURVATURE_RATIO = 0.5
MIN_DIRECTION_COSINE = 0.0
DEFAULT_MAX_EXTRAPOLATION_RATIO = 1.5
DEFAULT_MAX_ADAPTIVE_STEP_SCALE = 3.0
DEFAULT_SAFETY_FACTOR = 1.25
DEFAULT_PROTECTED_PREFIX_STEPS = 6
DEFAULT_AUDIO_EMERGENCY_MULTIPLIER = 4.0
DEFAULT_EMBEDDED_VIDEO_TOLERANCE = 0.05
DEFAULT_ADAPTIVE_SAFETY_FACTOR = 0.8
DEFAULT_MAX_ADAPTIVE_GROWTH_RATIO = 2.0

_MASKS = {
    "full_20": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19),
    "conservative_12": (0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19),
    "early_aggressive_13": (0, 1, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19),
    "uniform_13": (0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19),
    "late_cautious_14": (0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 18, 19),
    "late_aggressive_13": (0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 17, 18, 19),
    "late_aggressive_12": (0, 1, 2, 3, 4, 5, 9, 13, 16, 17, 18, 19),
    "late_max_11": (0, 1, 2, 3, 4, 5, 9, 13, 17, 18, 19),
}


def profile_mask(profile: str, logical_steps: int = 20) -> tuple[bool, ...]:
    """Return the immutable actual-evaluation mask for a named profile."""
    if profile == "native_20":
        profile = "full_20"
    if profile in ADAPTIVE_PROFILES:
        raise ValueError(f"adaptive schedule {profile} does not have a fixed mask")
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
    method: str = "euler"
    evaluation_profile: str = "full_20"
    diagnostics: str = "off"
    fallback_on_guard: bool = True
    max_extrapolation_ratio: float = DEFAULT_MAX_EXTRAPOLATION_RATIO
    curvature_ratio: float = CURVATURE_RATIO
    min_direction_cosine: float = MIN_DIRECTION_COSINE
    policy: str = "fixed"
    quality_preset: str = "balanced"
    repairability_profile: str | None = None
    conditioning_mode: str = "default"
    safety_factor: float = DEFAULT_SAFETY_FACTOR
    recovery_actual_steps: int = 2
    max_consecutive_forecasts: int = 1
    mask_version: str = MASK_VERSION
    predictor_version: str = PREDICTOR_VERSION
    adaptive_profile_hash: str | None = None
    protected_prefix_steps: int = DEFAULT_PROTECTED_PREFIX_STEPS
    audio_emergency_multiplier: float = DEFAULT_AUDIO_EMERGENCY_MULTIPLIER
    max_adaptive_step_scale: float = DEFAULT_MAX_ADAPTIVE_STEP_SCALE
    embedded_video_tolerance: float = DEFAULT_EMBEDDED_VIDEO_TOLERANCE
    adaptive_safety_factor: float = DEFAULT_ADAPTIVE_SAFETY_FACTOR
    max_adaptive_growth_ratio: float = DEFAULT_MAX_ADAPTIVE_GROWTH_RATIO
    _mask: tuple[bool, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        legacy_native = self.method == "native"
        method = {
            "native": "euler",
            "sparse_euler": "euler",
            "sparse_res_multistep": "res_multistep",
        }.get(self.method, self.method)
        evaluation_profile = self.evaluation_profile
        if legacy_native or evaluation_profile == "native_20":
            evaluation_profile = "full_20"
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "evaluation_profile", evaluation_profile)
        if self.method not in METHODS:
            raise ValueError(f"unknown vector acceleration method: {self.method}")
        if self.evaluation_profile not in EVALUATION_PROFILES:
            raise ValueError(f"unknown evaluation profile: {self.evaluation_profile}")
        if self.diagnostics not in DIAGNOSTICS:
            raise ValueError(f"unknown diagnostics mode: {self.diagnostics}")
        if self.policy not in POLICIES:
            raise ValueError(f"unknown vector acceleration policy: {self.policy}")
        if self.quality_preset not in QUALITY_PRESETS:
            raise ValueError(f"unknown quality preset: {self.quality_preset}")
        if self.conditioning_mode not in CONDITIONING_MODES:
            raise ValueError(f"unknown conditioning mode: {self.conditioning_mode}")
        if self.method in CORE_SOLVER_METHODS and self.policy != "fixed":
            raise ValueError("core solver methods require the fixed policy")
        if self.evaluation_profile in ADAPTIVE_PROFILES and self.method != "res_multistep":
            raise ValueError("adaptive history schedule requires the res_multistep method")
        if self.policy == "adaptive_repair" and self.method not in PREDICTOR_METHODS:
            raise ValueError("adaptive repair policy requires a predictor method")
        if self.policy == "adaptive_repair" and not self.repairability_profile:
            raise ValueError("adaptive repair policy requires a repairability profile")
        if self.policy == "adaptive_repair" and self.evaluation_profile != "full_20":
            raise ValueError("adaptive repair policy uses the full_20 candidate grid")
        if self.repairability_profile is not None:
            if not isinstance(self.repairability_profile, str) or not self.repairability_profile.strip():
                raise ValueError("repairability_profile must be a non-empty profile filename")
        for name, value in (("max_extrapolation_ratio", self.max_extrapolation_ratio),
                            ("max_adaptive_step_scale", self.max_adaptive_step_scale),
                            ("curvature_ratio", self.curvature_ratio),
                            ("min_direction_cosine", self.min_direction_cosine),
                            ("safety_factor", self.safety_factor),
                            ("audio_emergency_multiplier", self.audio_emergency_multiplier)):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        for name, value in (("embedded_video_tolerance", self.embedded_video_tolerance),
                            ("adaptive_safety_factor", self.adaptive_safety_factor),
                            ("max_adaptive_growth_ratio", self.max_adaptive_growth_ratio)):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.max_extrapolation_ratio <= 0:
            raise ValueError("max_extrapolation_ratio must be positive")
        if self.max_adaptive_step_scale < 1:
            raise ValueError("max_adaptive_step_scale must be at least one")
        if self.curvature_ratio < 0:
            raise ValueError("curvature_ratio must be non-negative")
        if not -1 <= self.min_direction_cosine <= 1:
            raise ValueError("min_direction_cosine must be between -1 and 1")
        if self.safety_factor < 1:
            raise ValueError("safety_factor must be at least one")
        if not isinstance(self.recovery_actual_steps, int) or self.recovery_actual_steps < 0:
            raise ValueError("recovery_actual_steps must be a non-negative integer")
        if not isinstance(self.max_consecutive_forecasts, int) or self.max_consecutive_forecasts != 1:
            raise ValueError("the initial adaptive controller requires exactly one maximum consecutive forecast")
        if not isinstance(self.protected_prefix_steps, int) or self.protected_prefix_steps < 0:
            raise ValueError("protected_prefix_steps must be a non-negative integer")
        if self.audio_emergency_multiplier <= 0:
            raise ValueError("audio_emergency_multiplier must be positive")
        if self.embedded_video_tolerance <= 0:
            raise ValueError("embedded_video_tolerance must be positive")
        if not 0 < self.adaptive_safety_factor <= 1:
            raise ValueError("adaptive_safety_factor must be greater than zero and at most one")
        if self.max_adaptive_growth_ratio < 1:
            raise ValueError("max_adaptive_growth_ratio must be at least one")
        mask = tuple() if self.evaluation_profile in ADAPTIVE_PROFILES else profile_mask(self.evaluation_profile, 20)
        object.__setattr__(self, "_mask", mask)

    @property
    def mask(self) -> tuple[bool, ...]:
        if self.evaluation_profile in ADAPTIVE_PROFILES:
            raise ValueError(f"adaptive schedule {self.evaluation_profile} does not have a fixed mask")
        return self._mask

    @property
    def actual_indices(self) -> tuple[int, ...]:
        if self.evaluation_profile in ADAPTIVE_PROFILES:
            raise ValueError(f"adaptive schedule {self.evaluation_profile} does not have fixed actual indices")
        return tuple(i for i, value in enumerate(self._mask) if value)

    def validate_schedule_length(self, logical_steps: int) -> None:
        if logical_steps < 1:
            raise ValueError("sigma schedule must contain at least one derivative interval")
        if logical_steps != 20:
            raise ValueError(f"{self.evaluation_profile} requires exactly 20 logical steps")
