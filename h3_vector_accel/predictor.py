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


def make_predictor(method: str) -> VectorPredictor:
    if method == "hold":
        return HoldPredictor()
    if method == "linear_velocity":
        return LinearVelocityPredictor()
    raise ValueError(f"method {method!r} does not use a forecast predictor")
