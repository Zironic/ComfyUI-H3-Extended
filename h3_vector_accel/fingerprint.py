"""Stable identities for vector acceleration configurations and schedules."""

import hashlib
import json

import torch

from .config import ADAPTIVE_PROFILES, CONTINUOUS_PROFILES, SamplerConfig
from .adaptive_res import controller_identity
from .schedules import continuous_schedule_identity


def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sigma_hash(sigmas) -> str:
    tensor = torch.as_tensor(sigmas).detach().to(device="cpu", dtype=torch.float64).contiguous()
    payload = tensor.numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def model_fingerprint(model) -> str:
    """Best-effort stable model identity without reading model weights."""
    if model is None:
        return "unknown"
    for attr in ("h3_vector_model_fingerprint", "model_fingerprint", "checkpoint_fingerprint"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    cls = type(model)
    sampling = getattr(model, "model_sampling", None)
    if sampling is None:
        inner = getattr(model, "inner_model", None)
        sampling = getattr(inner, "model_sampling", None)
    bits = [f"{cls.__module__}.{cls.__qualname__}"]
    if sampling is not None:
        bits.append(type(sampling).__name__)
        for name in ("shift", "audio_shift", "multiplier"):
            if hasattr(sampling, name):
                bits.append(f"{name}={getattr(sampling, name)!r}")
    return hashlib.sha256("|".join(bits).encode("utf-8")).hexdigest()[:24]


def configuration_payload(config: SamplerConfig, sigmas=None, model_identity=None,
                          effective_sigmas=None, actual_indices=None) -> dict:
    payload = {
        "method": config.method,
        "evaluation_profile": config.evaluation_profile,
        "actual_mask": (
            None if (config.evaluation_profile in ADAPTIVE_PROFILES or
                     config.evaluation_profile in CONTINUOUS_PROFILES)
            else list(config.actual_indices)
        ),
        "mask_version": config.mask_version,
        "predictor_version": config.predictor_version,
        "guards": {
            "max_extrapolation_ratio": config.max_extrapolation_ratio,
            "curvature_ratio": config.curvature_ratio,
            "min_direction_cosine": config.min_direction_cosine,
        },
        "max_adaptive_step_scale": config.max_adaptive_step_scale,
        "embedded_video_tolerance": config.embedded_video_tolerance,
        "adaptive_safety_factor": config.adaptive_safety_factor,
        "max_adaptive_growth_ratio": config.max_adaptive_growth_ratio,
        "fallback_on_guard": config.fallback_on_guard,
        "policy": config.policy,
        "quality_preset": config.quality_preset,
        "repairability_profile": config.repairability_profile,
        "conditioning_mode": config.conditioning_mode,
        "safety_factor": config.safety_factor,
        "recovery_actual_steps": config.recovery_actual_steps,
        "max_consecutive_forecasts": config.max_consecutive_forecasts,
        "protected_prefix_steps": config.protected_prefix_steps,
        "audio_emergency_multiplier": config.audio_emergency_multiplier,
        "adaptive_profile_hash": config.adaptive_profile_hash,
    }
    if config.evaluation_profile in ADAPTIVE_PROFILES:
        payload["adaptive_controller"] = controller_identity(
            config.evaluation_profile, config.max_adaptive_step_scale
        )
    elif config.evaluation_profile in CONTINUOUS_PROFILES:
        payload["continuous_schedule"] = continuous_schedule_identity(
            config.evaluation_profile
        )
    if sigmas is not None:
        payload["sigma_hash"] = sigma_hash(sigmas)
        if effective_sigmas is not None or actual_indices is not None:
            payload["source_sigma_sequence"] = torch.as_tensor(sigmas).detach().cpu().tolist()
    if effective_sigmas is not None:
        payload["effective_sigma_hash"] = sigma_hash(effective_sigmas)
        payload["effective_sigma_sequence"] = torch.as_tensor(effective_sigmas).detach().cpu().tolist()
    if actual_indices is not None:
        payload["actual_indices"] = [int(index) for index in actual_indices]
    if model_identity is not None:
        payload["model_fingerprint"] = str(model_identity)
    return payload


def configuration_fingerprint(config: SamplerConfig, sigmas=None, model_identity=None,
                              effective_sigmas=None, actual_indices=None) -> str:
    encoded = canonical_json(configuration_payload(
        config, sigmas, model_identity, effective_sigmas, actual_indices
    )).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


sampler_fingerprint = configuration_fingerprint
fingerprint = configuration_fingerprint
