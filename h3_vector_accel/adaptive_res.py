"""Stateful eta-zero RES and the causal history controller."""

from dataclasses import dataclass
import math

import torch


CONTROLLER_VERSION = "adaptive_history_v1"
CONTROLLER_CONSTANTS = {
    "protected_prefix": 6,
    "protected_tail": (17, 18, 19),
    "initial_step_scale": 1.0,
    "step_scale_min": 1.0,
    "step_scale_max": 3.0,
    "low_change_multiplier": 1.5,
    "high_change_multiplier": 0.7,
    "low_change_ratio": 0.75,
    "high_change_ratio": 1.5,
    "audio_emergency_multiplier": 4.0,
    "reference_intervals": 3,
    "time_coordinate": "negative_log_sigma",
    "change_signal": "mean_relative_velocity_x0_rate",
    "max_nfe": 20,
}
V2_CONTROLLER_VERSION = "adaptive_history_v2"
V2_CONTROLLER_CONSTANTS = dict(
    CONTROLLER_CONSTANTS,
    protected_prefix=0,
    bootstrap_anchors=3,
    reference_anchors=2,
    reference_intervals=1,
    protected_tail=(18, 19),
    low_change_streak=2,
)
V3_CONTROLLER_VERSION = "adaptive_history_v3"
V3_CONTROLLER_CONSTANTS = {
    "protected_prefix": 0,
    "bootstrap_anchors": 3,
    "prediction_start_anchor": 3,
    "protected_tail": (),
    "initial_step_scale": 1.0,
    "terminal_positive_index": 19,
    "initial_step_scale": 1.0,
    "step_scale_min": 1.0,
    "step_scale_max": 3.0,
    "grow_multiplier": 1.5,
    "shrink_multiplier": 0.7,
    "grow_below": 0.40,
    "hold_below": 0.70,
    "shrink_below": 1.00,
    "reset_below": 1.30,
    "critical_recovery_intervals": 1,
    "predictor": "linear_secant_negative_log_sigma",
    "residual": "symmetric_relative_l2",
    "video_control": "max_derivative_x0_error",
    "audio_control": "diagnostic_only",
    "max_nfe": 20,
}
EMBEDDED_RES_CONTROLLER_VERSION = "adaptive_embedded_res_v1"
EMBEDDED_RES_CONTROLLER_CONSTANTS = {
    "bootstrap_intervals": 1,
    "terminal_positive_index": 19,
    "defect": "eta_zero_incremental_res",
    "control": "video_only",
    "audio_control": "diagnostic_only",
    "bisection_iterations": 48,
    "epsilon": 1e-8,
    "max_nfe": 20,
    "protected_tail": (),
    "initial_step_scale": 1.0,
    "step_scale_min": 1.0,
    "step_scale_max": 3.0,
}


def controller_identity(version=CONTROLLER_VERSION, max_step_scale=None):
    constants = {
        CONTROLLER_VERSION: CONTROLLER_CONSTANTS,
        V2_CONTROLLER_VERSION: V2_CONTROLLER_CONSTANTS,
        V3_CONTROLLER_VERSION: V3_CONTROLLER_CONSTANTS,
        EMBEDDED_RES_CONTROLLER_VERSION: EMBEDDED_RES_CONTROLLER_CONSTANTS,
    }.get(version, CONTROLLER_CONSTANTS)
    if max_step_scale is not None:
        constants = dict(constants, step_scale_max=float(max_step_scale))
    return {"version": version, "constants": dict(constants)}


def _scalar(value):
    return float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)


def _t(sigma):
    value = _scalar(sigma)
    if value <= 0:
        return float("inf")
    return -math.log(value)


class IncrementalRES:
    """One eta-zero RES interval, retaining stock sampler history."""

    def __init__(self):
        self.old_sigma_down = None
        self.old_denoised = None
        self.previous_sigma = None

    @property
    def history(self):
        return self.old_denoised is not None

    def reset(self):
        self.old_sigma_down = self.old_denoised = self.previous_sigma = None

    def step(self, x, sigma, denoised, sigma_next):
        sigma_tensor = sigma if isinstance(sigma, torch.Tensor) else x.new_tensor(sigma)
        next_tensor = sigma_next if isinstance(sigma_next, torch.Tensor) else x.new_tensor(sigma_next)
        next_value = _scalar(next_tensor)
        # eta=0 means sigma_down is exactly the requested next sigma.
        sigma_down = next_value
        if sigma_down == 0.0 or self.old_denoised is None:
            derivative = (x - denoised) / sigma_tensor.reshape((1,) + (1,) * (x.ndim - 1))
            x_next = x + derivative * (next_tensor - sigma_tensor)
        else:
            t = sigma_tensor.log().neg()
            t_old = self.old_sigma_down.log().neg()
            t_next = next_tensor.log().neg()
            t_prev = self.previous_sigma.log().neg()
            h = t_next - t
            c2 = (t_prev - t_old) / h
            phi1 = torch.expm1(-h) / (-h)
            phi2 = (phi1 - 1.0) / (-h)
            b1 = torch.nan_to_num(phi1 - phi2 / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2 / c2, nan=0.0)
            x_next = torch.exp(-h) * x + h * (b1 * denoised + b2 * self.old_denoised)
        self.old_denoised = denoised
        self.old_sigma_down = next_tensor
        self.previous_sigma = sigma_tensor
        return x_next


def _relative(a, b, eps=1e-8):
    return float(torch.linalg.vector_norm((a - b).float()).item() /
                 (torch.linalg.vector_norm(a.float()).item() + eps))


def _cosine(a, b, eps=1e-8):
    af, bf = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.dot(af, bf).item() /
                 (torch.linalg.vector_norm(af).item() * torch.linalg.vector_norm(bf).item() + eps))


@dataclass
class AnchorObservation:
    sigma: float
    source_index: int | None
    video_change: float | None
    audio_change: float | None
    video_cosine: float | None
    audio_cosine: float | None
    video_rate: float | None
    audio_rate: float | None
    video_x0_change: float | None = None
    audio_x0_change: float | None = None
    video_velocity_rate: float | None = None
    video_x0_rate: float | None = None
    audio_velocity_rate: float | None = None
    audio_x0_rate: float | None = None
    previous_step_scale: float | None = None
    actual_delta_t: float | None = None
    residuals: dict | None = None
    established_reference: bool = False
    video_x0_difference_rms: float | None = None
    audio_x0_difference_rms: float | None = None
    video_normalization_scale: float | None = None
    audio_normalization_scale: float | None = None
    tolerance_solution_h: float | None = None
    safety_adjusted_h: float | None = None
    accepted_h: float | None = None
    previous_accepted_h: float | None = None
    clamp_selected: str | None = None

    @property
    def video_score(self):
        values = [v for v in (self.video_change, self.video_x0_change) if v is not None]
        return sum(values) / len(values) if values else None

    @property
    def audio_score(self):
        values = [v for v in (self.audio_change, self.audio_x0_change) if v is not None]
        return sum(values) / len(values) if values else None


class AdaptiveHistoryController:
    """Causal schedule controller; all proposals are accepted anchors."""

    version = CONTROLLER_VERSION
    constants = CONTROLLER_CONSTANTS

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20, max_step_scale=None):
        self.source = tuple(_scalar(v) for v in source_sigmas)
        if len(self.source) != 21:
            raise ValueError(f"{self.version} requires 20 source intervals")
        if not all(math.isfinite(value) for value in self.source):
            raise ValueError("source sigma schedule must be finite")
        if any(a <= b for a, b in zip(self.source, self.source[1:])):
            raise ValueError("source sigma schedule must be strictly descending")
        self.latent_shapes = latent_shapes
        self.max_nfe = min(int(max_nfe), self.constants["max_nfe"])
        self.max_step_scale = float(
            self.constants["step_scale_max"] if max_step_scale is None else max_step_scale
        )
        if self.max_step_scale < 1.0 or not math.isfinite(self.max_step_scale):
            raise ValueError("max_step_scale must be finite and at least one")
        self.constants = dict(self.constants, step_scale_max=self.max_step_scale)
        self.step_scale = self.constants["initial_step_scale"]
        self.anchors = []
        self.decisions = []
        self.reference_video_rate = None
        self.reference_audio_rate = None

    def _source_index(self, sigma):
        for index, value in enumerate(self.source):
            if abs(value - sigma) <= max(1e-9, abs(value) * 1e-8):
                return index
        return None

    def _containing_index(self, sigma):
        for index in range(len(self.source) - 1):
            if self.source[index] >= sigma > self.source[index + 1]:
                return index
        return 19

    def _modal_parts(self, value):
        if self.latent_shapes and len(self.latent_shapes) >= 2:
            parts = []
            offset = 0
            for shape in self.latent_shapes[:2]:
                size = math.prod(shape[1:])
                part = value[:, :, offset:offset + size]
                parts.append(part.reshape((part.shape[0],) + tuple(shape[1:])))
                offset += size
            return tuple(parts)
        return (value,)

    def observe(self, sigma, derivative, denoised, previous_derivative=None,
                previous_denoised=None, previous_sigma=None):
        video_change = audio_change = video_cosine = audio_cosine = None
        video_x0_change = audio_x0_change = None
        video_rate = audio_rate = None
        video_velocity_rate = video_x0_rate = audio_velocity_rate = audio_x0_rate = None
        delta_t = (
            abs(_t(sigma) - _t(previous_sigma))
            if previous_sigma is not None and sigma > 0 else None
        )
        if previous_derivative is not None:
            old_parts, new_parts = self._modal_parts(previous_derivative), self._modal_parts(derivative)
            video_change = _relative(old_parts[0], new_parts[0])
            video_cosine = _cosine(old_parts[0], new_parts[0])
            if len(old_parts) >= 2 and len(new_parts) >= 2:
                audio_change = _relative(old_parts[1], new_parts[1])
                audio_cosine = _cosine(old_parts[1], new_parts[1])
        if previous_denoised is not None:
            old_parts, new_parts = self._modal_parts(previous_denoised), self._modal_parts(denoised)
            video_x0_change = _relative(old_parts[0], new_parts[0])
            if len(old_parts) >= 2 and len(new_parts) >= 2:
                audio_x0_change = _relative(old_parts[1], new_parts[1])
        if delta_t is not None:
            if video_change is not None:
                video_velocity_rate = video_change / max(delta_t, 1e-8)
            if video_x0_change is not None:
                video_x0_rate = video_x0_change / max(delta_t, 1e-8)
            if audio_change is not None:
                audio_velocity_rate = audio_change / max(delta_t, 1e-8)
            if audio_x0_change is not None:
                audio_x0_rate = audio_x0_change / max(delta_t, 1e-8)
            video_values = [value for value in (video_change, video_x0_change) if value is not None]
            audio_values = [value for value in (audio_change, audio_x0_change) if value is not None]
            if video_values:
                video_rate = (sum(video_values) / len(video_values)) / max(delta_t, 1e-8)
            if audio_values:
                audio_rate = (sum(audio_values) / len(audio_values)) / max(delta_t, 1e-8)
        observation = AnchorObservation(_scalar(sigma), self._source_index(_scalar(sigma)),
                                        video_change, audio_change, video_cosine, audio_cosine,
                                        video_rate, audio_rate, video_x0_change, audio_x0_change,
                                        video_velocity_rate, video_x0_rate,
                                        audio_velocity_rate, audio_x0_rate)
        observation.actual_delta_t = delta_t
        self.anchors.append(observation)
        reference_anchors = self.constants.get(
            "reference_anchors",
            self.constants.get(
                "bootstrap_anchors", self.constants.get("protected_prefix", 0)
            ),
        )
        window = self.constants.get("reference_intervals")
        if window and len(self.anchors) == reference_anchors:
            values = [row.video_rate for row in self.anchors[-window:] if row.video_rate is not None]
            if values:
                self.reference_video_rate = sum(values) / len(values)
            audio_values = [row.audio_rate for row in self.anchors[-window:] if row.audio_rate is not None]
            if audio_values:
                self.reference_audio_rate = sum(audio_values) / len(audio_values)
        return observation

    def propose(self, sigma, observation=None):
        sigma = _scalar(sigma)
        source_index = self._source_index(sigma)
        prefix_count = self.constants["protected_prefix"]
        tail_start, tail_middle, tail_end = self.constants["protected_tail"]
        reason = "adaptive_base"
        emergency = False
        if source_index is not None and source_index < prefix_count - 1:
            next_sigma = self.source[source_index + 1]
            reason = "protected_prefix"
        elif source_index is not None and source_index >= tail_start:
            next_sigma = (
                self.source[tail_middle] if source_index == tail_start else
                self.source[tail_end] if source_index == tail_middle else 0.0
            )
            reason = "protected_tail" if source_index < tail_end else "terminal_zero"
        else:
            if (observation is not None and observation.video_rate is not None and
                    self.reference_video_rate is not None):
                ratio = observation.video_rate / max(self.reference_video_rate, 1e-8)
                if ratio <= self.constants["low_change_ratio"]:
                    self.step_scale = min(
                        self.max_step_scale,
                        self.step_scale * self.constants["low_change_multiplier"],
                    )
                    reason = "low_video_change_grow"
                elif ratio >= self.constants["high_change_ratio"]:
                    self.step_scale = max(
                        self.constants["step_scale_min"],
                        self.step_scale * self.constants["high_change_multiplier"],
                    )
                    reason = "high_video_change_shrink"
                audio_reference = self.reference_audio_rate
                if audio_reference is None:
                    audio_reference = self.reference_video_rate
                if (observation.audio_rate is not None and
                        observation.audio_rate > max(audio_reference, 1e-8) * self.constants["audio_emergency_multiplier"]):
                    self.step_scale = max(
                        self.constants["step_scale_min"],
                        self.step_scale * self.constants["high_change_multiplier"],
                    )
                    reason = "audio_emergency_shrink"
                    emergency = True
            left = source_index if source_index is not None else self._containing_index(sigma)
            left = max(prefix_count - 1, min(left, tail_start - 1))
            base = _t(self.source[left + 1]) - _t(self.source[left])
            target_t = _t(sigma) + base * self.step_scale
            tail_t = _t(self.source[tail_start])
            target_t = min(target_t, tail_t)
            next_sigma = math.exp(-target_t)
            if next_sigma >= sigma:
                next_sigma = math.nextafter(sigma, 0.0)
            if next_sigma <= self.source[tail_start]:
                next_sigma = self.source[tail_start]
        proposed_interval = None if next_sigma == 0 else _t(next_sigma) - _t(sigma)
        protected_region = (
            "prefix" if reason == "protected_prefix" else
            "tail" if reason in ("protected_tail", "terminal_zero") else None
        )
        self.decisions.append({"sigma": sigma, "next_sigma": next_sigma,
                               "step_scale": self.step_scale, "reason": reason,
                               "source_index": source_index, "audio_emergency": emergency,
                               "local_base_interval": locals().get("base"),
                               "proposed_interval_t": proposed_interval,
                               "protected_region": protected_region})
        return next_sigma, self.decisions[-1]

    def next_sigma(self, sigma, observation=None):
        return self.propose(sigma, observation)[0]


class AdaptiveHistoryControllerV2(AdaptiveHistoryController):
    """Four-anchor bootstrap controller with a two-sample low-change gate."""

    version = V2_CONTROLLER_VERSION
    constants = V2_CONTROLLER_CONSTANTS

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20, max_step_scale=None):
        super().__init__(source_sigmas, latent_shapes=latent_shapes, max_nfe=max_nfe,
                         max_step_scale=max_step_scale)
        self.low_change_streak = 0
        self.reference_video_rate = None
        self.reference_audio_rate = None

    def propose(self, sigma, observation=None):
        sigma = _scalar(sigma)
        source_index = self._source_index(sigma)
        bootstrap_anchors = self.constants["bootstrap_anchors"]
        tail_start, tail_end = self.constants["protected_tail"]
        reason, emergency = "adaptive_base", False
        if source_index is not None and source_index < bootstrap_anchors - 1:
            next_sigma, reason = self.source[source_index + 1], "bootstrap"
        elif source_index is not None and source_index >= tail_start:
            next_sigma = self.source[tail_end] if source_index == tail_start else 0.0
            reason = "protected_tail" if source_index == tail_start else "terminal_zero"
        else:
            ratio = None
            if observation is not None and observation.video_rate is not None and self.reference_video_rate:
                ratio = observation.video_rate / max(self.reference_video_rate, 1e-8)
                if ratio <= self.constants["low_change_ratio"]:
                    self.low_change_streak += 1
                    if self.low_change_streak >= self.constants["low_change_streak"]:
                        requested_scale = self.step_scale * self.constants["low_change_multiplier"]
                        self.step_scale = min(self.max_step_scale, requested_scale)
                        reason = (
                            "low_video_change_capped"
                            if requested_scale > self.max_step_scale else
                            "low_video_change_grow"
                        )
                    else:
                        reason = "low_video_change_wait"
                elif ratio >= self.constants["high_change_ratio"]:
                    self.low_change_streak = 0
                    self.step_scale = max(
                        self.constants["step_scale_min"],
                        self.step_scale * self.constants["high_change_multiplier"],
                    )
                    reason = "high_video_change_shrink"
                else:
                    self.low_change_streak = 0
            audio_reference = self.reference_audio_rate or self.reference_video_rate
            if (observation is not None and observation.audio_rate is not None and
                    audio_reference is not None and observation.audio_rate >
                    audio_reference * self.constants["audio_emergency_multiplier"]):
                self.low_change_streak = 0
                self.step_scale = max(
                    self.constants["step_scale_min"],
                    self.step_scale * self.constants["high_change_multiplier"],
                )
                reason, emergency = "audio_emergency_shrink", True
            left = self._containing_index(sigma) if source_index is None else source_index
            left = max(bootstrap_anchors - 1, min(left, tail_start - 1))
            base = _t(self.source[left + 1]) - _t(self.source[left])
            target_t = min(_t(sigma) + base * self.step_scale, _t(self.source[tail_start]))
            next_sigma = max(self.source[tail_start], math.exp(-target_t))
            if next_sigma >= sigma:
                next_sigma = math.nextafter(sigma, 0.0)
        proposed_interval = None if next_sigma == 0 else _t(next_sigma) - _t(sigma)
        anchor_region = (
            "bootstrap" if source_index is not None and source_index < bootstrap_anchors else
            "tail" if reason in ("protected_tail", "terminal_zero") else None
        )
        self.decisions.append({"sigma": sigma, "next_sigma": next_sigma,
                               "step_scale": self.step_scale, "reason": reason,
                               "source_index": source_index, "audio_emergency": emergency,
                               "local_base_interval": locals().get("base"),
                               "proposed_interval_t": proposed_interval,
                               "protected_region": anchor_region,
                               "low_change_streak": self.low_change_streak})
        return next_sigma, self.decisions[-1]


class AdaptiveHistoryControllerV3(AdaptiveHistoryController):
    """Local predictive-error controller; predictions never enter RES state."""

    version = V3_CONTROLLER_VERSION
    constants = V3_CONTROLLER_CONSTANTS

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20, max_step_scale=None):
        super().__init__(source_sigmas, latent_shapes=latent_shapes, max_nfe=max_nfe,
                         max_step_scale=max_step_scale)
        self.derivative_slope = None
        self.denoised_slope = None
        self.last_proposed_step_scale = self.constants["initial_step_scale"]
        self.recovery_remaining = 0
        self.reference_video_error = None
        self.reference_audio_error = None

    @staticmethod
    def _symmetric_error(actual, predicted, eps=1e-8):
        actual = actual.float()
        predicted = predicted.float()
        if not bool(torch.isfinite(actual).all().item()) or not bool(torch.isfinite(predicted).all().item()):
            return float("inf")
        denominator = 0.5 * (
            torch.linalg.vector_norm(actual) + torch.linalg.vector_norm(predicted)
        )
        return float((torch.linalg.vector_norm(actual - predicted) / (denominator + eps)).item())

    def _prediction_residuals(self, derivative, denoised, predicted_derivative, predicted_denoised):
        actual_d, predicted_d = self._modal_parts(derivative), self._modal_parts(predicted_derivative)
        actual_x0, predicted_x0 = self._modal_parts(denoised), self._modal_parts(predicted_denoised)
        residuals = {
            "video_v_error": self._symmetric_error(actual_d[0], predicted_d[0]),
            "video_x0_error": self._symmetric_error(actual_x0[0], predicted_x0[0]),
        }
        residuals["video_error"] = max(residuals["video_v_error"], residuals["video_x0_error"])
        if len(actual_d) >= 2:
            residuals["audio_v_error"] = self._symmetric_error(actual_d[1], predicted_d[1])
            residuals["audio_x0_error"] = self._symmetric_error(actual_x0[1], predicted_x0[1])
            residuals["audio_error"] = max(
                residuals["audio_v_error"], residuals["audio_x0_error"]
            )
        else:
            residuals.update(audio_v_error=None, audio_x0_error=None, audio_error=None)
        return residuals

    def observe(self, sigma, derivative, denoised, previous_derivative=None,
                previous_denoised=None, previous_sigma=None):
        observation = super().observe(
            sigma, derivative, denoised,
            previous_derivative, previous_denoised, previous_sigma,
        )
        delta_t = (
            _t(sigma) - _t(previous_sigma)
            if previous_sigma is not None and _scalar(sigma) > 0 else None
        )
        observation.previous_step_scale = self.last_proposed_step_scale
        observation.actual_delta_t = delta_t
        if (len(self.anchors) > self.constants["prediction_start_anchor"] and
                delta_t is not None and delta_t > 0 and
                self.derivative_slope is not None and self.denoised_slope is not None):
            predicted_derivative = previous_derivative + self.derivative_slope * delta_t
            predicted_denoised = previous_denoised + self.denoised_slope * delta_t
            observation.residuals = self._prediction_residuals(
                derivative, denoised, predicted_derivative, predicted_denoised
            )
            video_error = observation.residuals["video_error"]
            audio_error = observation.residuals["audio_error"]
            if self.reference_video_error is None and math.isfinite(video_error):
                self.reference_video_error = video_error
                observation.established_reference = True
            if self.reference_audio_error is None and audio_error is not None and math.isfinite(audio_error):
                self.reference_audio_error = audio_error
            observation.residuals["video_error_ratio"] = (
                None if self.reference_video_error is None else
                video_error / max(self.reference_video_error, 1e-8)
            )
            observation.residuals["audio_error_ratio"] = (
                None if self.reference_audio_error is None or audio_error is None else
                audio_error / max(self.reference_audio_error, 1e-8)
            )
        if (previous_derivative is not None and previous_denoised is not None and
                delta_t is not None and delta_t > 0):
            self.derivative_slope = ((derivative - previous_derivative) / delta_t).detach()
            self.denoised_slope = ((denoised - previous_denoised) / delta_t).detach()
        return observation

    def propose(self, sigma, observation=None):
        sigma = _scalar(sigma)
        source_index = self._source_index(sigma)
        terminal_index = self.constants["terminal_positive_index"]
        previous_scale = (
            observation.previous_step_scale
            if observation is not None and observation.previous_step_scale is not None
            else self.last_proposed_step_scale
        )
        reason, action, emergency = "prediction_unavailable", "hold", False
        base = None
        if source_index is not None and source_index < self.constants["bootstrap_anchors"] - 1:
            next_sigma = self.source[source_index + 1]
            self.step_scale = 1.0
            reason, action = "bootstrap", "bootstrap"
        elif (source_index == self.constants["bootstrap_anchors"] - 1 and
              (observation is None or observation.residuals is None)):
            next_sigma = self.source[source_index + 1]
            self.step_scale = 1.0
            reason, action = "bootstrap_predict", "bootstrap"
        elif source_index == terminal_index:
            next_sigma = 0.0
            reason, action = "terminal_zero", "terminal"
        else:
            residuals = observation.residuals if observation is not None else None
            video_error = None if residuals is None else residuals.get("video_error")
            video_ratio = None if residuals is None else residuals.get("video_error_ratio")
            if (video_ratio is None and video_error is not None and
                    self.reference_video_error is not None):
                video_ratio = video_error / max(self.reference_video_error, 1e-8)
                if residuals is not None:
                    residuals["video_error_ratio"] = video_ratio
            if residuals is not None and residuals.get("audio_error_ratio") is None:
                audio_error = residuals.get("audio_error")
                if audio_error is not None and self.reference_audio_error is not None:
                    residuals["audio_error_ratio"] = audio_error / max(self.reference_audio_error, 1e-8)
            nonfinite_video = video_error is not None and not math.isfinite(video_error)
            previous_accelerated = previous_scale > self.constants["step_scale_min"]
            if self.recovery_remaining > 0:
                self.recovery_remaining -= 1
                self.step_scale = 1.0
                reason, action = "forced_recovery", "forced_recovery"
            elif residuals is None:
                self.step_scale = 1.0
            elif observation.established_reference:
                self.step_scale = 1.0
                reason, action = "reference_calibration", "reference_calibration"
            elif nonfinite_video:
                if previous_accelerated:
                    self.step_scale = 1.0
                    self.recovery_remaining = self.constants["critical_recovery_intervals"]
                    reason, action, emergency = "nonfinite_video_recovery", "critical_recovery", True
                else:
                    self.step_scale = 1.0
                    reason, action = "minimum_step_hold", "minimum_step_hold"
            elif video_ratio is None:
                self.step_scale = 1.0
            elif video_ratio < self.constants["grow_below"]:
                requested = self.step_scale * self.constants["grow_multiplier"]
                self.step_scale = min(self.max_step_scale, requested)
                capped = requested > self.max_step_scale
                reason = "low_residual_capped" if capped else "low_residual_grow"
                action = "capped" if capped else "grow"
            elif video_ratio < self.constants["hold_below"]:
                reason, action = "moderate_residual_hold", "hold"
            elif video_ratio < self.constants["shrink_below"]:
                if previous_accelerated:
                    self.step_scale = max(
                        self.constants["step_scale_min"],
                        self.step_scale * self.constants["shrink_multiplier"],
                    )
                    reason, action = "high_residual_shrink", "shrink"
                else:
                    self.step_scale = 1.0
                    reason, action = "minimum_step_hold", "minimum_step_hold"
            elif video_ratio < self.constants["reset_below"]:
                if previous_accelerated:
                    self.step_scale = 1.0
                    reason, action = "very_high_residual_reset", "reset"
                else:
                    self.step_scale = 1.0
                    reason, action = "minimum_step_hold", "minimum_step_hold"
            else:
                if previous_accelerated:
                    self.step_scale = 1.0
                    self.recovery_remaining = self.constants["critical_recovery_intervals"]
                    reason, action = "critical_residual_recovery", "critical_recovery"
                else:
                    self.step_scale = 1.0
                    reason, action = "minimum_step_hold", "minimum_step_hold"
            left = self._containing_index(sigma) if source_index is None else source_index
            left = max(self.constants["bootstrap_anchors"] - 1, min(left, terminal_index - 1))
            base = _t(self.source[left + 1]) - _t(self.source[left])
            target_t = _t(sigma) + base * self.step_scale
            terminal_t = _t(self.source[terminal_index])
            next_sigma = 0.0 if target_t > terminal_t else math.exp(-target_t)
            if next_sigma >= sigma:
                next_sigma = math.nextafter(sigma, 0.0)
        proposed_interval = None if next_sigma == 0 else _t(next_sigma) - _t(sigma)
        protected_region = (
            "bootstrap" if action == "bootstrap" else
            None
        )
        decision = {
            "sigma": sigma, "next_sigma": next_sigma,
            "previous_step_scale": previous_scale,
            "actual_delta_t": None if observation is None else observation.actual_delta_t,
            "residuals": None if observation is None else observation.residuals,
            "reference_video_error": self.reference_video_error,
            "reference_audio_error": self.reference_audio_error,
            "video_error_ratio": None if observation is None else (observation.residuals or {}).get("video_error_ratio"),
            "audio_error_ratio": None if observation is None else (observation.residuals or {}).get("audio_error_ratio"),
            "step_scale": self.step_scale, "action": action, "reason": reason,
            "source_index": source_index, "audio_emergency": emergency,
            "local_base_interval": base, "proposed_interval_t": proposed_interval,
            "protected_region": protected_region,
            "recovery_remaining": self.recovery_remaining,
        }
        self.last_proposed_step_scale = self.step_scale
        self.decisions.append(decision)
        return next_sigma, decision


class AdaptiveEmbeddedRESController(AdaptiveHistoryController):
    version = EMBEDDED_RES_CONTROLLER_VERSION
    constants = EMBEDDED_RES_CONTROLLER_CONSTANTS

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20,
                 max_step_scale=None, video_tolerance=0.05,
                 safety_factor=0.8, max_growth_ratio=2.0):
        super().__init__(source_sigmas, latent_shapes=latent_shapes,
                         max_nfe=max_nfe, max_step_scale=max_step_scale)
        self.video_tolerance = float(video_tolerance)
        self.safety_factor = float(safety_factor)
        self.max_growth_ratio = float(max_growth_ratio)
        self.previous_accepted_h = None

    @staticmethod
    def _rms(value):
        return 0.0 if value is None or value.numel() == 0 else float(torch.sqrt(torch.mean(value.float() ** 2)).item())

    @staticmethod
    def embedded_defect(h, h_previous, x0_difference_rms, normalization_scale, epsilon=1e-8):
        """Return the eta-zero RES embedded defect estimate for transformed h."""
        h = float(h)
        h_previous = max(float(h_previous), epsilon)
        normalization_scale = max(float(normalization_scale), epsilon)
        if h <= 0:
            return 0.0
        value = ((h + math.expm1(-h)) / h_previous) * (
            float(x0_difference_rms) / normalization_scale
        )
        return value if math.isfinite(value) else float("inf")

    def observe(self, sigma, derivative, denoised, previous_derivative=None,
                previous_denoised=None, previous_sigma=None):
        observation = super().observe(sigma, derivative, denoised,
                                      previous_derivative, previous_denoised, previous_sigma)
        if previous_denoised is not None:
            current, previous = self._modal_parts(denoised), self._modal_parts(previous_denoised)
            observation.video_x0_difference_rms = self._rms(current[0] - previous[0])
            observation.audio_x0_difference_rms = self._rms(current[1] - previous[1]) if len(current) > 1 else None
        else:
            observation.video_x0_difference_rms = None
            observation.audio_x0_difference_rms = None
        observation.previous_accepted_h = self.previous_accepted_h
        return observation

    def propose(self, sigma, observation=None, current_x=None, current_x0=None):
        sigma = _scalar(sigma)
        source_index = self._source_index(sigma)
        terminal = self.constants["terminal_positive_index"]
        if source_index == 0:
            next_sigma = self.source[1]
            h = _t(next_sigma) - _t(sigma)
            self.previous_accepted_h = h
            decision = {"sigma": sigma, "next_sigma": next_sigma, "source_index": source_index,
                        "reason": "bootstrap", "action": "bootstrap", "step_scale": 1.0,
                        "local_base_interval": h, "proposed_interval_t": h,
                        "tolerance_solution_h": None, "safety_adjusted_h": None,
                        "accepted_h": h, "previous_accepted_h": None, "growth_ratio": None,
                        "defect_at_accepted_h": None, "video_x0_difference_rms": None,
                        "audio_x0_difference_rms": None, "video_normalization_scale": None,
                        "audio_normalization_scale": None, "clamp_selected": "bootstrap",
                        "video_tolerance": self.video_tolerance,
                        "protected_region": "bootstrap"}
            self.decisions.append(decision)
            self.step_scale = 1.0
            return next_sigma, decision
        if source_index == terminal:
            decision = {"sigma": sigma, "next_sigma": 0.0, "source_index": source_index,
                        "reason": "terminal_zero", "action": "terminal", "step_scale": 1.0,
                        "proposed_interval_t": None, "clamp_selected": "terminal_zero",
                        "protected_region": "terminal"}
            self.decisions.append(decision)
            self.step_scale = 1.0
            return 0.0, decision
        if observation is None or observation.video_x0_difference_rms is None:
            left = self._containing_index(sigma) if source_index is None else min(source_index, terminal - 1)
            next_sigma = self.source[left + 1]
            h = _t(next_sigma) - _t(sigma)
            self.previous_accepted_h = h
            decision = {"sigma": sigma, "next_sigma": next_sigma, "source_index": source_index,
                        "reason": "bootstrap_fallback", "action": "bootstrap", "step_scale": 1.0,
                        "local_base_interval": h, "proposed_interval_t": h,
                        "accepted_h": h, "clamp_selected": "bootstrap",
                        "video_tolerance": self.video_tolerance}
            self.decisions.append(decision)
            self.step_scale = 1.0
            return next_sigma, decision
        parts_x = self._modal_parts(current_x) if current_x is not None else ()
        parts_x0 = self._modal_parts(current_x0) if current_x0 is not None else ()
        video_scale = max(self._rms(parts_x[0]) if parts_x else 0.0,
                          self._rms(parts_x0[0]) if parts_x0 else 0.0,
                          self.constants["epsilon"])
        audio_scale = None
        if len(parts_x) > 1 and len(parts_x0) > 1:
            audio_scale = max(self._rms(parts_x[1]), self._rms(parts_x0[1]), self.constants["epsilon"])
        h_previous = max(self.previous_accepted_h or self.constants["epsilon"], self.constants["epsilon"])
        remaining = max(0.0, _t(self.source[terminal]) - _t(sigma))
        left = self._containing_index(sigma) if source_index is None else min(source_index, terminal - 1)
        local_base = _t(self.source[left + 1]) - _t(self.source[left])
        def defect(h):
            return self.embedded_defect(h, h_previous,
                                        observation.video_x0_difference_rms,
                                        video_scale)
        if observation.video_x0_difference_rms <= self.constants["epsilon"]:
            solved = None
            safety_h = None
        else:
            hi = max(1.0, remaining, local_base * self.max_step_scale,
                     h_previous * self.max_growth_ratio)
            for _ in range(64):
                if defect(hi) > self.video_tolerance:
                    break
                hi *= 2.0
            if defect(hi) <= self.video_tolerance:
                solved = None
                safety_h = None
            else:
                lo = 0.0
                for _ in range(self.constants["bisection_iterations"]):
                    mid = (lo + hi) * 0.5
                    if defect(mid) <= self.video_tolerance:
                        lo = mid
                    else:
                        hi = mid
                solved = lo
                safety_h = solved * self.safety_factor
        accepted, clamp = (float("inf"), "unbounded_defect") if safety_h is None else (safety_h, "tolerance")
        for limit, name in ((local_base * self.max_step_scale, "absolute"),
                            (h_previous * self.max_growth_ratio, "growth"),
                            (remaining, "terminal_floor")):
            if accepted > limit:
                accepted, clamp = limit, name
        accepted = max(0.0, min(remaining, accepted))
        next_sigma = self.source[terminal] if accepted >= remaining else math.exp(-(_t(sigma) + accepted))
        if next_sigma >= sigma:
            next_sigma = math.nextafter(sigma, 0.0)
            accepted = _t(next_sigma) - _t(sigma)
        decision = {"sigma": sigma, "next_sigma": next_sigma, "source_index": source_index,
                    "reason": "embedded_res", "action": "accept",
                    "step_scale": accepted / max(local_base, self.constants["epsilon"]),
                    "local_base_interval": local_base, "proposed_interval_t": accepted,
                    "tolerance_solution_h": solved, "safety_adjusted_h": safety_h,
                    "accepted_h": accepted, "previous_accepted_h": self.previous_accepted_h,
                    "growth_ratio": accepted / h_previous,
                    "defect_at_accepted_h": defect(accepted),
                    "audio_defect_at_accepted_h": (
                        self.embedded_defect(accepted, h_previous,
                                             observation.audio_x0_difference_rms,
                                             audio_scale)
                        if observation.audio_x0_difference_rms is not None and audio_scale else None
                    ),
                    "video_x0_difference_rms": observation.video_x0_difference_rms,
                    "audio_x0_difference_rms": observation.audio_x0_difference_rms,
                    "video_normalization_scale": video_scale, "audio_normalization_scale": audio_scale,
                    "video_tolerance": self.video_tolerance, "clamp_selected": clamp,
                    "protected_region": "terminal_floor" if clamp == "terminal_floor" else None}
        self.previous_accepted_h = accepted
        self.step_scale = decision["step_scale"]
        self.decisions.append(decision)
        return next_sigma, decision


AdaptiveHistoryControllerV1 = AdaptiveHistoryController
