"""Deterministic Euler-compatible H3 vector acceleration loop."""

from dataclasses import dataclass, replace
import logging
import math
import time

import torch
from comfy.utils import model_trange as trange
from comfy.k_diffusion.sampling import sample_euler, sample_res_multistep

from .config import ADAPTIVE_PROFILES, CORE_SOLVER_METHODS, PREDICTOR_METHODS, SamplerConfig
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
from .adaptive_res import (
    AdaptiveHistoryController,
    AdaptiveHistoryControllerV2,
    AdaptiveHistoryControllerV3,
    AdaptiveEmbeddedRESController,
    IncrementalRES,
)


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


class _TimedModel:
    """Proxy that preserves the model interface while measuring model calls."""

    def __init__(self, model):
        self._model = model
        self.calls = 0
        self.elapsed = 0.0

    def __call__(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return self._model(*args, **kwargs)
        finally:
            self.calls += 1
            self.elapsed += time.perf_counter() - started

    def __getattr__(self, name):
        return getattr(self._model, name)


def _adaptive_source_coordinate(controller, sigma):
    """Map an adaptive sigma to a fractional coordinate on the source grid."""
    sigma = _scalar_sigma(sigma)
    if sigma <= 0.0:
        return float(len(controller.source) - 1)
    source_index = controller._source_index(sigma)
    if source_index is not None:
        return float(source_index)
    left = controller._containing_index(sigma)
    right_sigma = controller.source[left + 1]
    if right_sigma <= 0.0:
        return float(left)
    left_t = -math.log(controller.source[left])
    right_t = -math.log(right_sigma)
    fraction = (-math.log(sigma) - left_t) / (right_t - left_t)
    return left + min(1.0, max(0.0, fraction))


def _adaptive_estimated_total_nfe(controller, current_nfe, next_sigma):
    """Estimate total genuine calls from the current scale and terminal policy."""
    if next_sigma <= 0.0:
        return current_nfe
    next_coordinate = _adaptive_source_coordinate(controller, next_sigma)
    tail = controller.constants["protected_tail"]
    if not tail:
        scale = max(controller.constants["step_scale_min"], controller.step_scale)
        remaining = max(1, math.ceil((len(controller.source) - 1 - next_coordinate) / scale))
    else:
        tail_start = tail[0]
        if next_coordinate < tail_start:
            scale = max(controller.constants["step_scale_min"], controller.step_scale)
            intermediate = max(1, math.ceil((tail_start - next_coordinate) / scale))
            remaining = intermediate + len(tail)
        else:
            remaining = sum(index + 1e-6 >= next_coordinate for index in tail)
    return min(controller.max_nfe, current_nfe + remaining)


def _adaptive_metric(value):
    return "n/a" if value is None or not math.isfinite(value) else f"{value:.4f}"


def _log_adaptive_v2_progress(controller, current_nfe, sigma, next_sigma, decision, observation):
    current_coordinate = _adaptive_source_coordinate(controller, sigma)
    next_coordinate = _adaptive_source_coordinate(controller, next_sigma)
    estimated_total = _adaptive_estimated_total_nfe(controller, current_nfe, next_sigma)
    schedule_steps = len(controller.source) - 1
    reference = controller.reference_video_rate
    ratio = (
        None if observation.video_rate is None or reference is None else
        observation.video_rate / max(reference, 1e-8)
    )
    logging.info(
        "[H3 Adaptive RES v2] NFE %d/~%d (est. compute %.0f%%) | "
        "schedule %.2f/%d (%.0f%%) -> %.2f | scale %.2fx | %s | "
        "delta_t=%s | video raw v=%s x0=%s | "
        "video rate v=%s x0=%s combined=%s ref=%s ratio=%s",
        current_nfe, estimated_total, current_nfe / max(1, estimated_total) * 100.0,
        current_coordinate, schedule_steps, current_coordinate / schedule_steps * 100.0,
        next_coordinate, decision["step_scale"], decision["reason"],
        _adaptive_metric(observation.actual_delta_t),
        _adaptive_metric(observation.video_change),
        _adaptive_metric(observation.video_x0_change),
        _adaptive_metric(observation.video_velocity_rate),
        _adaptive_metric(observation.video_x0_rate),
        _adaptive_metric(observation.video_rate),
        _adaptive_metric(reference), _adaptive_metric(ratio),
    )


def _log_adaptive_v3_progress(controller, current_nfe, sigma, next_sigma, decision):
    current_coordinate = _adaptive_source_coordinate(controller, sigma)
    next_coordinate = _adaptive_source_coordinate(controller, next_sigma)
    estimated_total = _adaptive_estimated_total_nfe(controller, current_nfe, next_sigma)
    schedule_steps = len(controller.source) - 1
    residuals = decision.get("residuals") or {}
    logging.info(
        "[H3 Adaptive RES v3] NFE %d/~%d (est. compute %.0f%%) | "
        "schedule %.2f/%d (%.0f%%) -> %.2f | previous scale=%s delta_t=%s | "
        "error video v=%s x0=%s max=%s ref=%s ratio=%s | "
        "audio v=%s x0=%s max=%s ref=%s ratio=%s | "
        "action=%s next scale=%.2fx",
        current_nfe, estimated_total, current_nfe / max(1, estimated_total) * 100.0,
        current_coordinate, schedule_steps, current_coordinate / schedule_steps * 100.0,
        next_coordinate,
        _adaptive_metric(decision.get("previous_step_scale")),
        _adaptive_metric(decision.get("actual_delta_t")),
        _adaptive_metric(residuals.get("video_v_error")),
        _adaptive_metric(residuals.get("video_x0_error")),
        _adaptive_metric(residuals.get("video_error")),
        _adaptive_metric(decision.get("reference_video_error")),
        _adaptive_metric(decision.get("video_error_ratio")),
        _adaptive_metric(residuals.get("audio_v_error")),
        _adaptive_metric(residuals.get("audio_x0_error")),
        _adaptive_metric(residuals.get("audio_error")),
        _adaptive_metric(decision.get("reference_audio_error")),
        _adaptive_metric(decision.get("audio_error_ratio")),
        decision.get("action"), decision["step_scale"],
    )


def _log_embedded_progress(controller, current_nfe, sigma, next_sigma, decision):
    current_coordinate = _adaptive_source_coordinate(controller, sigma)
    next_coordinate = _adaptive_source_coordinate(controller, next_sigma)
    estimated_total = _adaptive_estimated_total_nfe(controller, current_nfe, next_sigma)
    schedule_steps = len(controller.source) - 1
    logging.info("[H3 Adaptive RES embedded v1] NFE %d/~%d (est. compute %.0f%%) | "
                 "schedule %.2f/%d (%.0f%%) -> %.2f | h=%s defect=%s tol=%s clamp=%s",
                 current_nfe, estimated_total, current_nfe / max(1, estimated_total) * 100.0,
                 current_coordinate, schedule_steps, current_coordinate / schedule_steps * 100.0,
                 next_coordinate,
                 _adaptive_metric(decision.get("accepted_h")),
                 _adaptive_metric(decision.get("defect_at_accepted_h")),
                 _adaptive_metric(decision.get("video_tolerance")), decision.get("clamp_selected"))


def _sample_adaptive_res(model, x, source_sigmas, extra_args, callback, disable, config,
                         diagnostics, context):
    """Run causal adaptive RES with one genuine model call per accepted anchor."""
    source = source_sigmas.detach().cpu().tolist()
    source_hash = sigma_hash(source_sigmas)
    controller_class = (AdaptiveEmbeddedRESController if config.evaluation_profile == "adaptive_embedded_res_v1"
                        else AdaptiveHistoryControllerV3 if config.evaluation_profile == "adaptive_history_v3"
                        else AdaptiveHistoryControllerV2 if config.evaluation_profile == "adaptive_history_v2"
                        else AdaptiveHistoryController)
    controller_kwargs = {"latent_shapes": context.latent_shapes,
                         "max_step_scale": config.max_adaptive_step_scale}
    if config.evaluation_profile == "adaptive_embedded_res_v1":
        controller_kwargs.update(video_tolerance=config.embedded_video_tolerance,
                                safety_factor=config.adaptive_safety_factor,
                                max_growth_ratio=config.max_adaptive_growth_ratio)
    controller = controller_class(source, **controller_kwargs)
    stepper = IncrementalRES()
    timed_model = _TimedModel(model)
    diagnostics = diagnostics or RunDiagnostics(config=config, latent_shapes=context.latent_shapes,
                                                model_fingerprint=context.model_fingerprint)
    diagnostics.start_run(
        sigmas=source, method=config.method, evaluation_profile=config.evaluation_profile,
        configuration_fingerprint=configuration_fingerprint(config, source_sigmas, context.model_fingerprint),
        configuration=configuration_payload(config, source_sigmas, context.model_fingerprint),
        sigma_hash=source_hash, source_sigma_hash=source_hash,
        source_sigma_sequence=source, controller_version=controller.version,
        controller_constants=dict(controller.constants), effective_sigma_sequence=[],
    )
    s_in = x.new_ones([x.shape[0]])
    sigma = source_sigmas[0]
    prev_derivative = prev_denoised = prev_sigma = None
    effective = []
    decisions = []
    started = time.perf_counter()
    while True:
        denoised = timed_model(x, sigma * s_in, **extra_args)
        derivative = _to_d(x, sigma, denoised)
        if not _finite(derivative):
            raise RuntimeError("model returned a non-finite derivative")
        effective.append(_scalar_sigma(sigma))
        observation = controller.observe(
            sigma, derivative, denoised,
            prev_derivative, prev_denoised, prev_sigma,
        )
        if config.evaluation_profile == "adaptive_embedded_res_v1":
            sigma_next_value, decision = controller.propose(
                sigma, observation, current_x=x, current_x0=denoised
            )
        else:
            sigma_next_value, decision = controller.propose(sigma, observation)
        decisions.append(decision)
        current_source = controller._source_index(_scalar_sigma(sigma))
        tail = controller.constants["protected_tail"]
        if tail:
            tail_start = tail[0]
            tail_count = len(tail)
            if (len(effective) >= controller.max_nfe - tail_count and
                    (current_source is None or current_source < tail_start)):
                sigma_next_value = source[tail_start]
                decision = dict(
                    decision,
                    reason="reserve_protected_tail",
                    next_sigma=sigma_next_value,
                    proposed_interval_t=-math.log(sigma_next_value) + math.log(_scalar_sigma(sigma)),
                    protected_region="tail",
                )
                decisions[-1] = decision
        elif (len(effective) >= (controller.max_nfe - 1 if config.evaluation_profile == "adaptive_embedded_res_v1" else controller.max_nfe)
              and sigma_next_value > 0.0):
            if config.evaluation_profile == "adaptive_embedded_res_v1":
                sigma_next_value = source[controller.constants["terminal_positive_index"]]
                accepted_h = -math.log(sigma_next_value) + math.log(_scalar_sigma(sigma))
                previous_h = decision.get("previous_accepted_h")
                video_defect = (
                    controller.embedded_defect(
                        accepted_h, previous_h,
                        decision.get("video_x0_difference_rms"),
                        decision.get("video_normalization_scale"),
                    ) if previous_h and decision.get("video_x0_difference_rms") is not None else None
                )
                audio_defect = (
                    controller.embedded_defect(
                        accepted_h, previous_h,
                        decision.get("audio_x0_difference_rms"),
                        decision.get("audio_normalization_scale"),
                    ) if previous_h and decision.get("audio_x0_difference_rms") is not None and
                    decision.get("audio_normalization_scale") else None
                )
                decision = dict(decision, action="terminal_floor", reason="max_nfe_terminal_floor",
                                next_sigma=sigma_next_value, clamp_selected="terminal_floor",
                                proposed_interval_t=accepted_h, accepted_h=accepted_h,
                                step_scale=accepted_h / max(decision.get("local_base_interval") or 0.0, 1e-8),
                                growth_ratio=accepted_h / previous_h if previous_h else None,
                                defect_at_accepted_h=video_defect,
                                audio_defect_at_accepted_h=audio_defect,
                                protected_region="terminal_floor")
                controller.previous_accepted_h = accepted_h
                controller.step_scale = decision["step_scale"]
                decisions[-1] = decision
            else:
                sigma_next_value = 0.0
                decision = dict(
                    decision,
                    action="terminal", reason="max_nfe_terminal", next_sigma=0.0,
                    proposed_interval_t=None, protected_region=None,
                )
                decisions[-1] = decision
        if not math.isfinite(sigma_next_value) or not 0.0 <= sigma_next_value < _scalar_sigma(sigma):
            raise RuntimeError("adaptive RES proposed an invalid sigma")
        if config.evaluation_profile == "adaptive_history_v2":
            _log_adaptive_v2_progress(
                controller, len(effective), sigma, sigma_next_value, decision, observation
            )
        elif config.evaluation_profile == "adaptive_history_v3":
            _log_adaptive_v3_progress(
                controller, len(effective), sigma, sigma_next_value, decision
            )
        elif config.evaluation_profile == "adaptive_embedded_res_v1":
            _log_embedded_progress(controller, len(effective), sigma, sigma_next_value, decision)
        sigma_next = sigma.new_tensor(sigma_next_value)
        source_index = controller._source_index(_scalar_sigma(sigma))
        logical_step = controller._containing_index(_scalar_sigma(sigma)) if source_index is None else min(source_index, 19)
        anchor_index = len(effective) - 1
        metadata = {
            "x": x, "i": logical_step, "sigma": sigma, "sigma_hat": sigma,
            "denoised": denoised, "h3_vector_forecast": False,
            "h3_vector_true_nfe": len(effective),
            "h3_vector_actual_anchor_index": anchor_index,
            "h3_vector_logical_step": logical_step,
            "h3_vector_method": config.method, "h3_vector_profile": config.evaluation_profile,
            "h3_vector_policy": config.policy, "h3_vector_policy_reason": decision["reason"],
            "h3_vector_step_scale": decision["step_scale"],
            "h3_vector_local_base_interval": decision.get("local_base_interval"),
            "h3_vector_proposed_interval_t": decision.get("proposed_interval_t"),
            "h3_vector_next_sigma": sigma_next_value,
            "h3_vector_protected_region": decision.get("protected_region"),
            "h3_vector_video_change": observation.video_change,
            "h3_vector_audio_change": observation.audio_change,
            "h3_vector_video_direction_cosine": observation.video_cosine,
            "h3_vector_audio_direction_cosine": observation.audio_cosine,
            "h3_vector_video_rate": observation.video_rate,
            "h3_vector_video_velocity_rate": observation.video_velocity_rate,
            "h3_vector_video_x0_rate": observation.video_x0_rate,
            "h3_vector_reference_video_rate": controller.reference_video_rate,
            "h3_vector_video_rate_ratio": (
                None if observation.video_rate is None or controller.reference_video_rate is None
                else observation.video_rate / max(controller.reference_video_rate, 1e-8)
            ),
            "h3_vector_audio_rate": observation.audio_rate,
            "h3_vector_video_x0_change": observation.video_x0_change,
            "h3_vector_audio_x0_change": observation.audio_x0_change,
            "h3_vector_residuals": observation.residuals,
            "h3_vector_previous_step_scale": decision.get("previous_step_scale"),
            "h3_vector_actual_delta_t": decision.get("actual_delta_t", observation.actual_delta_t),
            "h3_vector_action": decision.get("action"),
            "h3_vector_video_v_error": (observation.residuals or {}).get("video_v_error"),
            "h3_vector_video_x0_error": (observation.residuals or {}).get("video_x0_error"),
            "h3_vector_video_error": (observation.residuals or {}).get("video_error"),
            "h3_vector_reference_video_error": decision.get("reference_video_error"),
            "h3_vector_video_error_ratio": decision.get("video_error_ratio"),
            "h3_vector_audio_v_error": (observation.residuals or {}).get("audio_v_error"),
            "h3_vector_audio_x0_error": (observation.residuals or {}).get("audio_x0_error"),
            "h3_vector_audio_error": (observation.residuals or {}).get("audio_error"),
            "h3_vector_reference_audio_error": decision.get("reference_audio_error"),
            "h3_vector_audio_error_ratio": decision.get("audio_error_ratio"),
            "h3_vector_source_sigma_hash": source_hash,
            "h3_vector_callback_context": "h3_vector_adaptive_actual_only",
            "h3_vector_controller_version": controller.version,
        }
        for key in ("tolerance_solution_h", "safety_adjusted_h", "accepted_h",
                    "previous_accepted_h", "growth_ratio", "defect_at_accepted_h",
                    "audio_defect_at_accepted_h",
                    "video_x0_difference_rms", "audio_x0_difference_rms",
                    "video_normalization_scale", "audio_normalization_scale",
                    "clamp_selected", "video_tolerance"):
            if key in decision:
                metadata[f"h3_vector_{key}"] = decision[key]
        if callback is not None:
            with callback_metadata_scope({k: v for k, v in metadata.items() if k.startswith("h3_vector_")}):
                callback(metadata)
        diagnostics.observe_actual_anchor(
            logical_step, _scalar_sigma(sigma), x=x, actual_derivative=derivative,
            policy_reason=decision["reason"], step_scale=decision["step_scale"],
            previous_step_scale=decision.get("previous_step_scale"),
            actual_delta_t=decision.get("actual_delta_t", observation.actual_delta_t),
            action=decision.get("action"), residuals=observation.residuals,
            source_index=source_index, next_sigma=sigma_next_value,
            local_base_interval=decision.get("local_base_interval"),
            proposed_interval_t=decision.get("proposed_interval_t"),
            protected_region=decision.get("protected_region"), trajectory_metrics={
                "video_change": observation.video_change, "audio_change": observation.audio_change,
                "video_x0_change": observation.video_x0_change,
                "audio_x0_change": observation.audio_x0_change,
                "video_direction_cosine": observation.video_cosine,
                "audio_direction_cosine": observation.audio_cosine,
                "video_rate": observation.video_rate, "audio_rate": observation.audio_rate,
                "video_velocity_rate": observation.video_velocity_rate,
                "video_x0_rate": observation.video_x0_rate,
                "reference_video_rate": controller.reference_video_rate,
                "reference_video_error": decision.get("reference_video_error"),
                "video_error_ratio": decision.get("video_error_ratio"),
                "reference_audio_error": decision.get("reference_audio_error"),
                "audio_error_ratio": decision.get("audio_error_ratio"),
                "video_rate_ratio": (
                    None if observation.video_rate is None or controller.reference_video_rate is None
                    else observation.video_rate / max(controller.reference_video_rate, 1e-8)
                ),
                "video_score": observation.video_score, "audio_score": observation.audio_score,
                "residuals": observation.residuals,
                "tolerance_solution_h": decision.get("tolerance_solution_h"),
                "safety_adjusted_h": decision.get("safety_adjusted_h"),
                "accepted_h": decision.get("accepted_h"),
                "previous_accepted_h": decision.get("previous_accepted_h"),
                "growth_ratio": decision.get("growth_ratio"),
                "defect_at_accepted_h": decision.get("defect_at_accepted_h"),
                "audio_defect_at_accepted_h": decision.get("audio_defect_at_accepted_h"),
                "video_x0_difference_rms": decision.get("video_x0_difference_rms"),
                "audio_x0_difference_rms": decision.get("audio_x0_difference_rms"),
                "video_normalization_scale": decision.get("video_normalization_scale"),
                "audio_normalization_scale": decision.get("audio_normalization_scale"),
                "clamp_selected": decision.get("clamp_selected"),
                **(observation.residuals or {}),
            },
        )
        diagnostics.observe_step(logical_step, _scalar_sigma(sigma), False, len(effective),
                                 actual_anchor_index=anchor_index, method=config.method,
                                 profile=config.evaluation_profile, policy=config.policy,
                                 policy_reason=decision["reason"], step_scale=decision["step_scale"],
                                 previous_step_scale=decision.get("previous_step_scale"),
                                 actual_delta_t=decision.get("actual_delta_t", observation.actual_delta_t),
                                 action=decision.get("action"),
                                 next_sigma=sigma_next_value,
                                 proposed_interval_t=decision.get("proposed_interval_t"),
                                 protected_region=decision.get("protected_region"))
        terminal_ready = (
            config.evaluation_profile == "adaptive_embedded_res_v1" and
            controller._source_index(_scalar_sigma(sigma)) == controller.constants.get("terminal_positive_index")
        )
        if sigma_next_value <= 0.0 or (len(effective) >= controller.max_nfe and not (
                config.evaluation_profile == "adaptive_embedded_res_v1" and not terminal_ready
        )):
            if sigma_next_value > 0.0:
                sigma_next = sigma.new_zeros(())
            x = stepper.step(x, sigma, denoised, sigma_next)
            break
        x = stepper.step(x, sigma, denoised, sigma_next)
        prev_derivative, prev_denoised, prev_sigma = derivative, denoised, sigma
        sigma = sigma_next
    effective.append(0.0)
    effective_tensor = source_sigmas.new_tensor(effective)
    run_fingerprint = configuration_fingerprint(config, source_sigmas, context.model_fingerprint,
                                                effective_sigmas=effective_tensor)
    diagnostics.update_run_metadata(
        effective_sigma_hash=sigma_hash(effective_tensor),
        effective_sigma_sequence=effective,
        configuration_fingerprint=run_fingerprint,
        effective_schedule_fingerprint=run_fingerprint,
        adaptive_decisions=decisions,
    )
    wall_seconds = time.perf_counter() - started
    diagnostics.finish_run(model_call_seconds=timed_model.elapsed,
                           sampler_overhead_seconds=max(0.0, wall_seconds - timed_model.elapsed),
                           wall_seconds=wall_seconds)
    return x


def _sample_core_solver(model, x, source_sigmas, extra_args, callback, disable, config,
                        diagnostics, context):
    logical_steps = int(source_sigmas.numel() - 1)
    if logical_steps != 20:
        raise ValueError("core solver methods require the existing 20-interval sigma schedule")
    if not _finite(source_sigmas):
        raise ValueError("core solver methods require finite sigma values")
    differences = source_sigmas[1:] - source_sigmas[:-1]
    if bool((differences >= 0).any().item()):
        raise ValueError("core solver methods require a strictly descending sigma schedule")
    if config.policy != "fixed":
        raise ValueError("core solver methods do not support adaptive policy")
    if not context.is_h3_flow_av:
        raise RuntimeError("core solver methods require ModelSamplingAV combined with CONST flow sampling")

    if config.evaluation_profile in ADAPTIVE_PROFILES:
        return _sample_adaptive_res(model, x, source_sigmas, extra_args, callback, disable,
                                    config, diagnostics, context)
    actual_indices = config.actual_indices
    index_tensor = torch.as_tensor(actual_indices, device=source_sigmas.device)
    effective_sigmas = torch.cat((source_sigmas.index_select(0, index_tensor), source_sigmas[-1:]))
    source_hash = sigma_hash(source_sigmas)
    effective_hash = sigma_hash(effective_sigmas)
    fingerprint = configuration_fingerprint(
        config, source_sigmas, context.model_fingerprint,
        effective_sigmas=effective_sigmas, actual_indices=actual_indices,
    )
    diagnostics.start_run(
        sigmas=source_sigmas.tolist(),
        method=config.method,
        evaluation_profile=config.evaluation_profile,
        configuration_fingerprint=fingerprint,
        configuration=configuration_payload(
            config, source_sigmas, context.model_fingerprint,
            effective_sigmas=effective_sigmas, actual_indices=actual_indices,
        ),
        sigma_hash=source_hash,
        source_sigma_hash=source_hash,
        source_sigma_sequence=source_sigmas.tolist(),
        effective_sigma_hash=effective_hash,
        effective_sigma_sequence=effective_sigmas.tolist(),
        actual_indices=list(actual_indices),
    )

    timed_model = _TimedModel(model)
    core_sampler = sample_euler if config.method == "euler" else sample_res_multistep
    callback_count = 0

    def core_callback(data):
        nonlocal callback_count
        callback_count += 1
        reduced_index = int(data["i"])
        logical_index = int(actual_indices[reduced_index])
        true_nfe = timed_model.calls
        payload = dict(data)
        payload["i"] = logical_index
        payload["sigma"] = source_sigmas[logical_index]
        payload["sigma_hat"] = source_sigmas[logical_index]
        payload.update({
            "h3_vector_forecast": False,
            "h3_vector_true_nfe": true_nfe,
            "h3_vector_actual_anchor_index": true_nfe - 1,
            "h3_vector_logical_step": logical_index,
            "h3_vector_method": config.method,
            "h3_vector_profile": config.evaluation_profile,
            "h3_vector_policy": config.policy,
            "h3_vector_actual_only": True,
            "h3_vector_core_solver": config.method,
            "h3_vector_source_sigma_hash": source_hash,
            "h3_vector_effective_sigma_hash": effective_hash,
            "h3_vector_actual_indices": list(actual_indices),
            "h3_vector_callback_context": "h3_vector_actual_only",
        })
        derivative = _to_d(payload["x"], payload["sigma"], payload["denoised"])
        diagnostics.observe_actual_anchor(
            logical_index, _scalar_sigma(payload["sigma"]), x=payload["x"],
            actual_derivative=derivative,
        )
        diagnostics.observe_step(
            logical_index, _scalar_sigma(payload["sigma"]), False, true_nfe,
            actual_anchor_index=true_nfe - 1, method=config.method,
            profile=config.evaluation_profile, policy=config.policy,
        )
        if callback is not None:
            context_metadata = {
                key: value for key, value in payload.items()
                if key.startswith("h3_vector_")
            }
            with callback_metadata_scope(context_metadata):
                callback(payload)

    started = time.perf_counter()
    result = core_sampler(
        timed_model, x, effective_sigmas, extra_args=extra_args,
        callback=core_callback, disable=disable,
    )
    wall_seconds = time.perf_counter() - started
    if callback_count != timed_model.calls:
        raise RuntimeError(
            f"core sampler callback count {callback_count} did not match true NFE {timed_model.calls}"
        )
    diagnostics.finish_run(
        model_call_seconds=timed_model.elapsed,
        sampler_overhead_seconds=max(0.0, wall_seconds - timed_model.elapsed),
        wall_seconds=wall_seconds,
    )
    return result


def sample_vector_accel(model, x, sigmas, extra_args=None, callback=None, disable=None,
                        config=None, diagnostics=None, latent_shapes=None):
    """Run an actual-only core solver or a forecast method over a source sigma grid."""
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
    if config.method in PREDICTOR_METHODS:
        if not context.is_h3_flow_av:
            raise RuntimeError("forecast methods require ModelSamplingAV combined with CONST flow sampling")
        if not context.latent_shapes or len(context.latent_shapes) < 2:
            raise RuntimeError("forecast methods require video and audio latent_shapes")
    if config.method in CORE_SOLVER_METHODS:
        return _sample_core_solver(
            model, x, sigma_values, extra_args, callback, disable, config,
            diagnostics or RunDiagnostics(config=config, latent_shapes=context.latent_shapes,
                                          model_fingerprint=context.model_fingerprint), context,
        )

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

    predictor = make_predictor(config.method, latent_shapes=context.latent_shapes)
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
