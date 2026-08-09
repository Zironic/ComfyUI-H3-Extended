"""Deterministic Euler-compatible H3 vector acceleration loop."""

from dataclasses import dataclass
import math
import time

import torch

from .config import SamplerConfig
from .diagnostics import RunDiagnostics, callback_metadata_scope
from .fingerprint import (
    configuration_fingerprint,
    configuration_payload,
    model_fingerprint,
    sigma_hash,
)
from .policy import make_policy
from .predictor import make_predictor


@dataclass(frozen=True)
class H3SamplingContext:
    sampling: object
    is_h3_flow_av: bool
    latent_shapes: object = None
    audio_scale: float | None = None
    model_fingerprint: str = "unknown"


def _iter_inner_models(model):
    current = model
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "inner_model", None)


def _sampling_for(node):
    value = getattr(node, "model_sampling", None)
    if value is not None:
        return value
    getter = getattr(node, "get_model_object", None)
    if getter is not None:
        try:
            value = getter("model_sampling")
        except (KeyError, AttributeError, TypeError):
            value = None
    return value


def _is_const(value):
    if value is None:
        return False
    try:
        import comfy.model_sampling as model_sampling
        if isinstance(value, model_sampling.CONST):
            return True
    except (ImportError, TypeError):
        pass
    return any(cls.__name__ == "CONST" for cls in type(value).__mro__)


def _is_av(value):
    if value is None:
        return False
    try:
        import comfy.model_sampling as model_sampling
        av = getattr(model_sampling, "ModelSamplingAV", None)
        if av is not None and isinstance(value, av):
            return True
    except (ImportError, TypeError):
        pass
    return any(cls.__name__ == "ModelSamplingAV" for cls in type(value).__mro__)


def _latent_shapes(model):
    for node in _iter_inner_models(model):
        value = getattr(node, "latent_shapes", None)
        if value is not None:
            return value
        options = getattr(node, "model_options", None)
        if isinstance(options, dict):
            transforms = options.get("transformer_options", {})
            if isinstance(transforms, dict) and transforms.get("latent_shapes") is not None:
                return transforms["latent_shapes"]
        inner_model = getattr(node, "model", None)
        value = getattr(inner_model, "latent_shapes", None)
        if value is not None:
            return value
    return None


def resolve_h3_sampling(model, latent_shapes=None) -> H3SamplingContext:
    """Resolve model sampling through the cycle-safe inner-model chain."""
    sampling = None
    sampling_owner = None
    for node in _iter_inner_models(model):
        sampling = _sampling_for(node)
        if sampling is not None:
            sampling_owner = node
            break
    if sampling is None:
        raise RuntimeError("unable to resolve model_sampling from the model chain")
    av = _is_av(sampling)
    is_flow_av = bool(av and _is_const(sampling))
    if latent_shapes is None:
        latent_shapes = _latent_shapes(model)
    audio_scale = getattr(sampling, "audio_scale", None)
    if audio_scale is not None:
        try:
            audio_scale = float(audio_scale)
        except (TypeError, ValueError):
            audio_scale = None
    return H3SamplingContext(
        sampling,
        is_flow_av,
        latent_shapes,
        audio_scale,
        model_fingerprint(sampling_owner),
    )


def _scalar_sigma(sigma):
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() != 1:
            raise ValueError("sigma schedule values must be scalar")
        return float(sigma.detach().float().item())
    return float(sigma)


def _sigma_broadcast(sigma, x):
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() == 1:
            return sigma.reshape((1,) + (1,) * (x.ndim - 1))
        return sigma.reshape((sigma.shape[0],) + (1,) * (x.ndim - 1))
    return x.new_tensor(sigma).reshape((1,) + (1,) * (x.ndim - 1))


def _to_d(x, sigma, denoised):
    return (x - denoised) / _sigma_broadcast(sigma, x)


def _rms(value):
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value.float() ** 2)).item())


def _finite(value):
    return bool(torch.isfinite(value).all().item())


def _guard_reason(config, x, sigma, sigma_next, prediction, predictor):
    if not prediction.valid:
        return prediction.failure_reason or "invalid_prediction"
    if not _finite(prediction.derivative):
        return "non_finite_prediction"
    history = getattr(predictor, "history", ())
    if not history:
        return "insufficient_history"
    actual = history[-1][1]
    predicted = prediction.derivative.float()
    denominator = _rms(actual) + 1e-8
    if _rms(predicted) / denominator > config.max_extrapolation_ratio:
        return "extrapolation_ratio"
    h = _scalar_sigma(sigma_next) - _scalar_sigma(sigma)
    if prediction.slope is not None:
        correction_ratio = _rms(0.5 * h * h * prediction.slope) / (_rms(h * predicted) + 1e-8)
        if not math.isfinite(correction_ratio) or correction_ratio > config.curvature_ratio:
            return "curvature_ratio"
        cosine = float(torch.dot(predicted.reshape(-1), actual.float().reshape(-1)).item() /
                       (torch.linalg.vector_norm(predicted).item() * torch.linalg.vector_norm(actual).item() + 1e-8))
        if not math.isfinite(cosine) or cosine < config.min_direction_cosine:
            return "direction_cosine"
    try:
        proposed = predictor.integrate(x, sigma, sigma_next, prediction)
    except (RuntimeError, ValueError, OverflowError):
        return "integration_error"
    if not _finite(proposed):
        return "non_finite_state"
    return None


def sample_vector_accel(model, x, sigmas, extra_args=None, callback=None, disable=None,
                        config=None, diagnostics=None, latent_shapes=None):
    """Run native Euler or a fixed forecast mask over a nominal sigma grid."""
    del disable  # kept for KSAMPLER compatibility
    extra_args = {} if extra_args is None else extra_args
    config = config if isinstance(config, SamplerConfig) else SamplerConfig(**(config or {}))
    sigma_values = torch.as_tensor(sigmas)
    if sigma_values.ndim != 1 or sigma_values.numel() < 2:
        raise ValueError("sigma schedule must contain at least two scalar values")
    differences = sigma_values[1:] - sigma_values[:-1]
    schedule_fallback_reason = None
    if not _finite(sigma_values):
        schedule_fallback_reason = "non_finite_sigma_sequence"
    elif bool((differences >= 0).any().item()):
        schedule_fallback_reason = "non_monotonic_sigma_sequence"
    logical_steps = int(sigma_values.numel() - 1)
    config.validate_schedule_length(logical_steps)
    context = resolve_h3_sampling(model, latent_shapes=latent_shapes or extra_args.get("latent_shapes"))
    if config.method == "native" and not _is_const(context.sampling):
        raise RuntimeError("native vector sampling requires CONST flow sampling")
    if config.method != "native":
        if not context.is_h3_flow_av:
            raise RuntimeError("forecast methods require ModelSamplingAV combined with CONST flow sampling")
        if not context.latent_shapes or len(context.latent_shapes) < 2:
            raise RuntimeError("forecast methods require video and audio latent_shapes")

    predictor = make_predictor(config.method) if config.method != "native" else None
    policy = make_policy(config)
    policy.reset()
    if predictor is not None:
        predictor.reset()
    diagnostics = diagnostics or RunDiagnostics(config=config, latent_shapes=context.latent_shapes,
                                                model_fingerprint=context.model_fingerprint)
    fingerprint = configuration_fingerprint(config, sigma_values, context.model_fingerprint)
    diagnostics.start_run(
        sigmas=sigma_values.tolist(),
        method=config.method,
        evaluation_profile=config.evaluation_profile,
        configuration_fingerprint=fingerprint,
        configuration=configuration_payload(
            config, sigma_values, context.model_fingerprint
        ),
        sigma_hash=sigma_hash(sigma_values),
    )
    s_in = x.new_ones([x.shape[0]])
    true_nfe = 0
    model_call_seconds = 0.0
    started = time.perf_counter()
    for i in range(logical_steps):
        sigma = sigma_values[i]
        sigma_next = sigma_values[i + 1]
        decision = policy.decide(i)
        counterfactual = None
        fallback_reason = None
        use_forecast = False
        if predictor is not None and predictor.ready():
            counterfactual = predictor.predict(x, sigma)
        if predictor is not None and decision.is_forecast:
            if schedule_fallback_reason is not None:
                fallback_reason = schedule_fallback_reason
            elif counterfactual is None:
                fallback_reason = "insufficient_history"
            else:
                fallback_reason = _guard_reason(
                    config, x, sigma, sigma_next, counterfactual, predictor
                )
            use_forecast = fallback_reason is None
            if not use_forecast and not config.fallback_on_guard:
                raise RuntimeError(f"unsafe vector forecast at step {i}: {fallback_reason}")
        if use_forecast:
            denoised = x - _sigma_broadcast(sigma, x) * counterfactual.derivative
            x_next = predictor.integrate(x, sigma, sigma_next, counterfactual)
            actual_anchor_index = len(predictor.history) - 1
        else:
            model_started = time.perf_counter()
            denoised = model(x, sigma * s_in, **extra_args)
            model_call_seconds += time.perf_counter() - model_started
            derivative = _to_d(x, sigma, denoised)
            if not _finite(derivative):
                raise RuntimeError(f"model returned a non-finite derivative at step {i}")
            true_nfe += 1
            if predictor is not None:
                diagnostics.observe_actual_anchor(i, _scalar_sigma(sigma), x=x,
                                                  actual_derivative=derivative,
                                                  counterfactual=counterfactual,
                                                  previous_actual_sigma=predictor.last_actual_sigma,
                                                  fallback_reason=fallback_reason)
                predictor.observe_actual(x, sigma, derivative)
                actual_anchor_index = len(predictor.history) - 1
            else:
                actual_anchor_index = i
            x_next = x + derivative * (sigma_next - sigma)
        metadata = {
            "x": x,
            "i": i,
            "sigma": sigma,
            "sigma_hat": sigma,
            "denoised": denoised,
            "h3_vector_forecast": bool(use_forecast),
            "h3_vector_true_nfe": true_nfe,
            "h3_vector_actual_anchor_index": actual_anchor_index,
            "h3_vector_method": config.method,
            "h3_vector_profile": config.evaluation_profile,
            "h3_vector_fallback_reason": fallback_reason,
        }
        if callback is not None:
            context_metadata = {
                key: value for key, value in metadata.items()
                if key.startswith("h3_vector_")
            }
            with callback_metadata_scope(context_metadata):
                callback(metadata)
        diagnostics.observe_step(i, _scalar_sigma(sigma), use_forecast, true_nfe,
                                 fallback_reason=fallback_reason,
                                 actual_anchor_index=actual_anchor_index,
                                 method=config.method, profile=config.evaluation_profile)
        x = x_next
    wall_seconds = time.perf_counter() - started
    diagnostics.finish_run(
        model_call_seconds=model_call_seconds,
        sampler_overhead_seconds=max(0.0, wall_seconds - model_call_seconds),
        wall_seconds=wall_seconds,
    )
    return x


def make_sampler(config: SamplerConfig | None = None):
    """Return a KSAMPLER-compatible function closure without importing nodes."""
    config = config or SamplerConfig()
    return lambda model, x, sigmas, extra_args=None, callback=None, disable=None: sample_vector_accel(
        model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, config=config
    )


sample = sample_vector_accel
