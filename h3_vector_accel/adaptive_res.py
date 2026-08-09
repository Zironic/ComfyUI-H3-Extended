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
    bootstrap_anchors=4,
    protected_tail=(18, 19),
    low_change_streak=2,
)


def controller_identity(version=CONTROLLER_VERSION):
    constants = V2_CONTROLLER_CONSTANTS if version == V2_CONTROLLER_VERSION else CONTROLLER_CONSTANTS
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

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20):
        self.source = tuple(_scalar(v) for v in source_sigmas)
        if len(self.source) != 21:
            raise ValueError(f"{self.version} requires 20 source intervals")
        if not all(math.isfinite(value) for value in self.source):
            raise ValueError("source sigma schedule must be finite")
        if any(a <= b for a, b in zip(self.source, self.source[1:])):
            raise ValueError("source sigma schedule must be strictly descending")
        self.latent_shapes = latent_shapes
        self.max_nfe = min(int(max_nfe), self.constants["max_nfe"])
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
            video_values = [value for value in (video_change, video_x0_change) if value is not None]
            audio_values = [value for value in (audio_change, audio_x0_change) if value is not None]
            if video_values:
                video_rate = (sum(video_values) / len(video_values)) / max(delta_t, 1e-8)
            if audio_values:
                audio_rate = (sum(audio_values) / len(audio_values)) / max(delta_t, 1e-8)
        observation = AnchorObservation(_scalar(sigma), self._source_index(_scalar(sigma)),
                                        video_change, audio_change, video_cosine, audio_cosine,
                                        video_rate, audio_rate, video_x0_change, audio_x0_change)
        self.anchors.append(observation)
        reference_anchors = self.constants.get(
            "bootstrap_anchors", self.constants["protected_prefix"]
        )
        if len(self.anchors) == reference_anchors:
            window = self.constants["reference_intervals"]
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
                        self.constants["step_scale_max"],
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

    def __init__(self, source_sigmas, latent_shapes=None, max_nfe=20):
        super().__init__(source_sigmas, latent_shapes=latent_shapes, max_nfe=max_nfe)
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
                        self.step_scale = min(
                            self.constants["step_scale_max"],
                            self.step_scale * self.constants["low_change_multiplier"],
                        )
                        reason = "low_video_change_grow"
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


AdaptiveHistoryControllerV1 = AdaptiveHistoryController
