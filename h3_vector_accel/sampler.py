"""Deterministic Euler-compatible H3 vector acceleration loop."""

from dataclasses import dataclass, replace
import math
import time

import torch
from comfy.utils import model_trange as trange

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
from .repairability import ProfileCompatibility, RepairabilityProfile


@dataclass(frozen=True)
class H3SamplingContext:
    sampling: object
    is_h3_flow_av: bool
    latent_shapes: object = None
    audio_scale: float | None = None
    video_shift: float | None = None
    audio_shift: float | None = None
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
    video_shift = getattr(sampling, "shift", None)
    audio_shift = getattr(sampling, "audio_shift", None)
    try:
        video_shift = None if video_shift is None else float(video_shift)
        audio_shift = None if audio_shift is None else float(audio_shift)
    except (TypeError, ValueError):
        video_shift = audio_shift = None
    return H3SamplingContext(
        sampling,
        is_flow_av,
        latent_shapes,
        audio_scale,
        video_shift,
        audio_shift,
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


def _guard_reason(config, x, sigma, sigma_next, prediction, predictor, metrics=None):
    if metrics is None:
        metrics = {}
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
    predicted_ratio = _rms(predicted) / denominator
    metrics["predicted_derivative_ratio"] = predicted_ratio
    h = _scalar_sigma(sigma_next) - _scalar_sigma(sigma)
    cosine = None
    if predicted.numel() == actual.numel():
        actual_flat = actual.float().reshape(-1)
        predicted_flat = predicted.reshape(-1)
        cosine = float(torch.dot(predicted_flat, actual_flat).item() /
                       (torch.linalg.vector_norm(predicted_flat).item() *
                        torch.linalg.vector_norm(actual_flat).item() + 1e-8))
        metrics["anchor_direction_cosine"] = cosine
    if prediction.slope is not None:
        correction_ratio = _rms(0.5 * h * h * prediction.slope) / (_rms(h * predicted) + 1e-8)
        metrics["curvature_correction_ratio"] = correction_ratio
    if predicted_ratio > config.max_extrapolation_ratio:
        return "extrapolation_ratio"
    if prediction.slope is not None:
        if not math.isfinite(correction_ratio) or correction_ratio > config.curvature_ratio:
            return "curvature_ratio"
        if cosine is None or not math.isfinite(cosine) or cosine < config.min_direction_cosine:
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

    profile = None
    if config.policy == "adaptive_repair":
        profile = RepairabilityProfile.load(config.repairability_profile)
        profile.validate_compatibility(ProfileCompatibility(
            model_fingerprint=context.model_fingerprint,
            sigma_hash=sigma_hash(sigma_values),
            video_shift=context.video_shift,
            audio_shift=context.audio_shift,
            nominal_steps=logical_steps,
            predictor_method=config.method,
            conditioning_mode=config.conditioning_mode,
        ))
        config = replace(config, adaptive_profile_hash=profile.hash)

    predictor = make_predictor(config.method, latent_shapes=context.latent_shapes) if config.method != "native" else None
    policy = make_policy(config, profile=profile, logical_steps=logical_steps)
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
    for i in trange(logical_steps, disable=disable):
        sigma = sigma_values[i]
        sigma_next = sigma_values[i + 1]
        decision = policy.decide(
            i,
            predictor_ready=bool(predictor is not None and predictor.ready()),
            diagnostics_state=diagnostics.policy_state(),
        )
        counterfactual = None
        fallback_reason = None
        guard_metrics = {}
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
                    config, x, sigma, sigma_next, counterfactual, predictor,
                    metrics=guard_metrics,
                )
            use_forecast = fallback_reason is None
            if not use_forecast and not config.fallback_on_guard:
                raise RuntimeError(f"unsafe vector forecast at step {i}: {fallback_reason}")
        if use_forecast:
            denoised = x - _sigma_broadcast(sigma, x) * counterfactual.derivative
            x_next = predictor.integrate(x, sigma, sigma_next, counterfactual)
            actual_anchor_index = true_nfe - 1
        else:
            model_started = time.perf_counter()
            denoised = model(x, sigma * s_in, **extra_args)
            model_call_seconds += time.perf_counter() - model_started
            derivative = _to_d(x, sigma, denoised)
            if not _finite(derivative):
                raise RuntimeError(f"model returned a non-finite derivative at step {i}")
            true_nfe += 1
            if predictor is not None:
                previous_actual_sigma = predictor.last_actual_sigma
                anchor_row = diagnostics.observe_actual_anchor(
                    i, _scalar_sigma(sigma), x=x,
                    actual_derivative=derivative,
                    counterfactual=counterfactual,
                    previous_actual_sigma=previous_actual_sigma,
                    fallback_reason=fallback_reason,
                    policy_reason=decision.reason,
                    policy_risk=decision.risk,
                )
                prediction_metrics = anchor_row.get("prediction_metrics")
                policy.observe_actual(i, prediction_metrics=prediction_metrics)
                predictor.observe_actual(x, sigma, derivative)
                actual_anchor_index = true_nfe - 1
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
            "h3_vector_policy": config.policy,
            "h3_vector_policy_reason": decision.reason,
            "h3_vector_policy_risk": decision.risk,
            "h3_vector_fallback_reason": fallback_reason,
            "h3_vector_guard_predicted_derivative_ratio": guard_metrics.get("predicted_derivative_ratio"),
            "h3_vector_guard_curvature_correction_ratio": guard_metrics.get("curvature_correction_ratio"),
            "h3_vector_guard_anchor_direction_cosine": guard_metrics.get("anchor_direction_cosine"),
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
                                 method=config.method, profile=config.evaluation_profile,
                                 policy=config.policy, policy_reason=decision.reason,
                                 policy_risk=decision.risk,
                                 video_risk=decision.video_risk,
                                 audio_risk=decision.audio_risk,
                                 guard_predicted_derivative_ratio=guard_metrics.get("predicted_derivative_ratio"),
                                 guard_curvature_correction_ratio=guard_metrics.get("curvature_correction_ratio"),
                                 guard_anchor_direction_cosine=guard_metrics.get("anchor_direction_cosine"))
        policy.observe_step(use_forecast)
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
