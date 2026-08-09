"""Deterministic derivative predictors used by the vector sampler."""

from dataclasses import dataclass, field
import math
from typing import Optional

import torch


def _finite(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value).all().item())


def _sigma_value(sigma) -> float:
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() != 1:
            raise ValueError("predictor sigmas must be scalar")
        return float(sigma.detach().float().item())
    return float(sigma)


@dataclass
class Prediction:
    derivative: Optional[torch.Tensor] = None
    slope: Optional[torch.Tensor] = None
    valid: bool = False
    failure_reason: Optional[str] = None
    diagnostic_scalars: dict = field(default_factory=dict)


class VectorPredictor:
    def reset(self) -> None:
        raise NotImplementedError

    def ready(self) -> bool:
        raise NotImplementedError


class _HistoryPredictor(VectorPredictor):
    def __init__(self):
        self._history: list[tuple[float, torch.Tensor]] = []

    def reset(self) -> None:
        self._history.clear()

    @property
    def history(self):
        return tuple(self._history)

    @property
    def last_actual_sigma(self):
        return self._history[-1][0] if self._history else None

    def snapshot(self):
        return tuple((sigma, derivative.clone()) for sigma, derivative in self._history)

    def restore(self, snapshot) -> None:
        restored = []
        for sigma, derivative in snapshot:
            value = float(sigma)
            tensor = derivative.detach().float().clone()
            if not math.isfinite(value) or not _finite(tensor):
                raise ValueError("predictor snapshot contains non-finite values")
            restored.append((value, tensor))
        self._history = restored[-2:]

    def observe_actual(self, x, sigma, derivative) -> None:
        value = _sigma_value(sigma)
        if not math.isfinite(value):
            return
        d = derivative.detach().float().clone()
        if not _finite(d):
            return
        self._history.append((value, d))
        if len(self._history) > 2:
            del self._history[:-2]

    def _invalid(self, reason):
        return Prediction(valid=False, failure_reason=reason)


class HoldPredictor(_HistoryPredictor):
    """Hold the most recent genuine derivative over forecast intervals."""

    def ready(self) -> bool:
        return bool(self._history)

    def predict(self, x, sigma) -> Prediction:
        if not self.ready():
            return self._invalid("insufficient_history")
        d = self._history[-1][1]
        if not _finite(d):
            return self._invalid("non_finite_derivative")
        return Prediction(derivative=d, valid=True)

    def integrate(self, x, sigma, sigma_next, prediction: Prediction) -> torch.Tensor:
        if not prediction.valid:
            raise ValueError("cannot integrate an invalid prediction")
        h = _sigma_value(sigma_next) - _sigma_value(sigma)
        return (x.float() + prediction.derivative.float() * h).to(dtype=x.dtype)


class LinearVelocityPredictor(_HistoryPredictor):
    """Linear extrapolation in sigma from the last two actual anchors."""

    def ready(self) -> bool:
        return len(self._history) >= 2

    def predict(self, x, sigma) -> Prediction:
        if not self.ready():
            return self._invalid("insufficient_history")
        sigma_b, d_b = self._history[-2]
        sigma_a, d_a = self._history[-1]
        delta = sigma_a - sigma_b
        if not math.isfinite(delta) or delta == 0:
            return self._invalid("duplicate_anchor_sigma")
        slope = (d_a - d_b) / delta
        if not _finite(slope):
            return self._invalid("non_finite_slope")
        current = _sigma_value(sigma)
        derivative = d_a + (current - sigma_a) * slope
        if not _finite(derivative):
            return self._invalid("non_finite_prediction")
        return Prediction(
            derivative=derivative,
            slope=slope,
            valid=True,
            diagnostic_scalars={"previous_actual_sigma": sigma_a, "anchor_sigma_delta": delta},
        )

    def integrate(self, x, sigma, sigma_next, prediction: Prediction) -> torch.Tensor:
        if not prediction.valid:
            raise ValueError("cannot integrate an invalid prediction")
        h = _sigma_value(sigma_next) - _sigma_value(sigma)
        derivative = prediction.derivative.float()
        slope = prediction.slope
        if slope is None:
            return x + prediction.derivative * h
        correction = 0.5 * (h * h) * slope
        result = x.float() + h * derivative + correction
        return result.to(dtype=x.dtype)


def _modality_ranges(value, latent_shapes):
    if latent_shapes and len(latent_shapes) >= 2 and value.ndim == 3:
        counts = [math.prod(tuple(shape)[1:]) for shape in latent_shapes[:2]]
        if sum(counts) <= value.shape[-1]:
            start = 0
            for count in counts:
                yield start, start + count
                start += count
            return
    yield 0, value.shape[-1]


def _decompose_velocity(x, derivative, latent_shapes, eps=1e-12):
    components = []
    for start, end in _modality_ranges(x, latent_shapes):
        state = x[..., start:end].detach().float()
        velocity = derivative[..., start:end].detach().float()
        dims = tuple(range(1, state.ndim))
        state_norm_sq = torch.sum(state * state, dim=dims, keepdim=True)
        state_norm = torch.sqrt(state_norm_sq)
        alpha = torch.sum(velocity * state, dim=dims, keepdim=True) / state_norm_sq.clamp_min(eps)
        residual = velocity - alpha * state
        residual_norm = torch.sqrt(torch.sum(residual * residual, dim=dims, keepdim=True))
        direction = residual / residual_norm.clamp_min(eps)
        direction = torch.where(residual_norm > eps, direction, torch.zeros_like(direction))
        beta = residual_norm / state_norm.clamp_min(eps)
        components.append((alpha, beta, direction))
    return tuple(components)


class VDEPredictor(_HistoryPredictor):
    """Velocity decomposition and estimation from two genuine anchors.

    Coefficients are extrapolated linearly in sigma and the most recent
    orthogonal direction is reused. Decomposition is performed independently
    for packed video and audio so the smaller audio stream cannot be hidden by
    the video norm.
    """

    def __init__(self, latent_shapes=None):
        super().__init__()
        self.latent_shapes = latent_shapes
        self._components = []

    def reset(self) -> None:
        super().reset()
        self._components.clear()

    def ready(self) -> bool:
        return len(self._history) >= 2 and len(self._components) >= 2

    def snapshot(self):
        components = tuple(
            tuple((alpha.clone(), beta.clone(), direction.clone()) for alpha, beta, direction in anchor)
            for anchor in self._components
        )
        return {"history": super().snapshot(), "components": components}

    def restore(self, snapshot) -> None:
        super().restore(snapshot["history"])
        components = []
        for anchor in snapshot["components"]:
            values = tuple(tuple(value.detach().float().clone() for value in part) for part in anchor)
            if not all(_finite(value) for part in values for value in part):
                raise ValueError("VDE snapshot contains non-finite values")
            components.append(values)
        self._components = components[-2:]
        if len(self._components) != len(self._history):
            raise ValueError("VDE snapshot history is inconsistent")

    def observe_actual(self, x, sigma, derivative) -> None:
        if x is None:
            return
        value = _sigma_value(sigma)
        state = x.detach().float()
        velocity = derivative.detach().float()
        if not math.isfinite(value) or not _finite(state) or not _finite(velocity):
            return
        components = _decompose_velocity(state, velocity, self.latent_shapes)
        if not all(_finite(value) for part in components for value in part):
            return
        super().observe_actual(state, value, velocity)
        self._components.append(components)
        if len(self._components) > 2:
            del self._components[:-2]

    def predict(self, x, sigma) -> Prediction:
        if not self.ready():
            return self._invalid("insufficient_history")
        sigma_old, _ = self._history[-2]
        sigma_new, _ = self._history[-1]
        delta = sigma_new - sigma_old
        if not math.isfinite(delta) or delta == 0:
            return self._invalid("duplicate_anchor_sigma")
        current = _sigma_value(sigma)
        factor = (current - sigma_new) / delta
        predicted = torch.empty_like(x, dtype=torch.float32)
        ranges = tuple(_modality_ranges(x, self.latent_shapes))
        if len(ranges) != len(self._components[-1]):
            return self._invalid("latent_shape_mismatch")
        direction_cosines = []
        for index, (start, end) in enumerate(ranges):
            old_alpha, old_beta, old_direction = self._components[-2][index]
            new_alpha, new_beta, new_direction = self._components[-1][index]
            alpha = new_alpha + (new_alpha - old_alpha) * factor
            beta = new_beta + (new_beta - old_beta) * factor
            state = x[..., start:end].float()
            dims = tuple(range(1, state.ndim))
            state_norm = torch.sqrt(torch.sum(state * state, dim=dims, keepdim=True))
            predicted[..., start:end] = alpha * state + beta * state_norm * new_direction
            direction_cosines.append(float(torch.sum(old_direction * new_direction).item() /
                                             (torch.linalg.vector_norm(old_direction).item() *
                                              torch.linalg.vector_norm(new_direction).item() + 1e-12)))
        if not _finite(predicted):
            return self._invalid("non_finite_prediction")
        return Prediction(
            derivative=predicted,
            valid=True,
            diagnostic_scalars={
                "previous_actual_sigma": sigma_new,
                "anchor_sigma_delta": delta,
                "orthogonal_direction_cosine_min": min(direction_cosines, default=1.0),
            },
        )

    def integrate(self, x, sigma, sigma_next, prediction: Prediction) -> torch.Tensor:
        if not prediction.valid:
            raise ValueError("cannot integrate an invalid prediction")
        h = _sigma_value(sigma_next) - _sigma_value(sigma)
        return (x.float() + h * prediction.derivative.float()).to(dtype=x.dtype)


def make_predictor(method: str, latent_shapes=None) -> VectorPredictor:
    if method == "hold":
        return HoldPredictor()
    if method == "linear_velocity":
        return LinearVelocityPredictor()
    if method == "vde":
        return VDEPredictor(latent_shapes=latent_shapes)
    raise ValueError(f"method {method!r} does not use a forecast predictor")
