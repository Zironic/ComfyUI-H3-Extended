"""Fixed and repairability-weighted vector evaluation policies."""

from dataclasses import dataclass

from .config import SamplerConfig


@dataclass(frozen=True)
class Decision:
    is_forecast: bool
    reason: str
    risk: float | None = None
    video_risk: float | None = None
    audio_risk: float | None = None


class NativePolicy:
    def reset(self):
        pass

    def decide(self, step, **kwargs) -> Decision:
        return Decision(False, "native")

    def observe_actual(self, *args, **kwargs):
        pass

    def observe_step(self, forecast):
        pass


class FixedMaskPolicy:
    def __init__(self, mask):
        self.mask = tuple(bool(value) for value in mask)

    def reset(self):
        pass

    def decide(self, step, **kwargs) -> Decision:
        if step < 0 or step >= len(self.mask):
            return Decision(False, "mask_out_of_range")
        return Decision(not self.mask[step], "fixed_mask_forecast" if not self.mask[step] else "fixed_mask_actual")

    def observe_actual(self, *args, **kwargs):
        pass

    def observe_step(self, forecast):
        pass


class AdaptiveRepairPolicy:
    """Forecast when conservatively estimated final modal risk is acceptable."""

    def __init__(self, profile, tolerance, logical_steps, safety_factor=1.25,
                 recovery_actual_steps=2, max_consecutive_forecasts=1,
                 warmup_steps=2, tail_actual_steps=3):
        self.profile = profile
        self.tolerance = float(tolerance)
        self.logical_steps = int(logical_steps)
        self.safety_factor = float(safety_factor)
        self.recovery_actual_steps = int(recovery_actual_steps)
        self.max_consecutive_forecasts = int(max_consecutive_forecasts)
        self.warmup_steps = int(warmup_steps)
        self.tail_actual_steps = int(tail_actual_steps)
        self.reset()

    def reset(self):
        self._local_errors = {"video": [], "audio": []}
        self._recovery_remaining = 0
        self._consecutive_forecasts = 0

    def _progress(self, step):
        return float(step) / max(1, self.logical_steps - 1)

    def _risk(self, progress):
        survival = self.profile.survival(progress)
        risks = {}
        for modality in ("video", "audio"):
            recent = self._local_errors[modality][-2:]
            if len(recent) < 2:
                return None
            estimate = self.safety_factor * max(recent)
            risks[modality] = float(survival[modality]) * estimate
        return risks

    def decide(self, step, predictor_ready=False, **kwargs):
        if step < self.warmup_steps:
            return Decision(False, "adaptive_warmup")
        if step >= self.logical_steps - self.tail_actual_steps:
            return Decision(False, "adaptive_forced_tail")
        if self._recovery_remaining:
            self._recovery_remaining -= 1
            return Decision(False, "adaptive_recovery")
        if not predictor_ready:
            return Decision(False, "adaptive_predictor_not_ready")
        if self._consecutive_forecasts >= self.max_consecutive_forecasts:
            return Decision(False, "adaptive_consecutive_limit")
        risks = self._risk(self._progress(step))
        if risks is None:
            return Decision(False, "adaptive_insufficient_error_history")
        risk = max(risks.values())
        return Decision(
            risk <= self.tolerance,
            "adaptive_risk_forecast" if risk <= self.tolerance else "adaptive_risk_actual",
            risk=risk,
            video_risk=risks["video"],
            audio_risk=risks["audio"],
        )

    def observe_actual(self, step, prediction_metrics=None, **kwargs):
        if not prediction_metrics or "video" not in prediction_metrics or "audio" not in prediction_metrics:
            return
        current = {}
        for modality in ("video", "audio"):
            value = prediction_metrics[modality].get("integration_error_proxy")
            if value is None:
                return
            current[modality] = float(value)
        for modality, value in current.items():
            self._local_errors[modality].append(value)
            del self._local_errors[modality][:-2]
        risks = self._risk(self._progress(step))
        if risks is not None and max(risks.values()) > self.tolerance:
            self._recovery_remaining = max(self._recovery_remaining, self.recovery_actual_steps)
            self._consecutive_forecasts = 0

    def observe_step(self, forecast):
        if forecast:
            self._consecutive_forecasts += 1
        else:
            self._consecutive_forecasts = 0


def make_policy(config: SamplerConfig, profile=None, logical_steps=20):
    if config.method == "native":
        return NativePolicy()
    if config.policy == "adaptive_repair":
        if profile is None:
            raise ValueError("adaptive repair policy requires a loaded profile")
        return AdaptiveRepairPolicy(
            profile,
            profile.tolerance(config.quality_preset),
            logical_steps,
            safety_factor=config.safety_factor,
            recovery_actual_steps=config.recovery_actual_steps,
            max_consecutive_forecasts=config.max_consecutive_forecasts,
        )
    return FixedMaskPolicy(config.mask)
