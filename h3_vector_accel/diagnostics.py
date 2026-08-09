"""Run-scoped diagnostics and callback metadata bridge."""

from contextlib import contextmanager
from contextvars import ContextVar
import json
import logging
import math
import os
import time
import uuid

import torch


_callback_metadata: ContextVar[dict | None] = ContextVar("h3_vector_callback_metadata", default=None)


def _json_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def current_callback_metadata():
    value = _callback_metadata.get()
    return None if value is None else dict(value)


get_callback_metadata = current_callback_metadata


@contextmanager
def callback_metadata_scope(metadata):
    token = _callback_metadata.set(dict(metadata))
    try:
        yield
    finally:
        _callback_metadata.reset(token)


def _rms(value):
    if value.numel() == 0:
        return 0.0
    return float(torch.sqrt(torch.mean(value.float() ** 2)).item())


def _relative_l2(actual, predicted, eps=1e-8):
    return float(torch.linalg.vector_norm((actual - predicted).float()).item() /
                 (torch.linalg.vector_norm(actual.float()).item() + eps))


def _cosine(actual, predicted, eps=1e-8):
    a, b = actual.float().reshape(-1), predicted.float().reshape(-1)
    return float(torch.dot(a, b).item() /
                 (torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item() + eps))


def _split_modalities(value, latent_shapes):
    if latent_shapes and len(latent_shapes) >= 2:
        try:
            from comfy.utils import unpack_latents
            return tuple(unpack_latents(value, latent_shapes)[:2])
        except (ImportError, RuntimeError, ValueError, TypeError):
            pass
    return (value,)


def modality_metrics(actual, predicted, latent_shapes=None, integration_span=None,
                     eps=1e-8):
    """Return video/audio/packed scalar errors, preserving audio visibility."""
    actual_parts = _split_modalities(actual, latent_shapes)
    predicted_parts = _split_modalities(predicted, latent_shapes)
    names = ("video", "audio") if len(actual_parts) >= 2 else ("packed",)
    result = {}
    for index, name in enumerate(names):
        a, p = actual_parts[index], predicted_parts[index]
        delta = (a - p).float()
        result[name] = {
            "relative_l2": _relative_l2(a, p, eps),
            "rms": _rms(delta),
            "direction_cosine": _cosine(a, p, eps),
        }
        if integration_span is not None:
            result[name]["integration_error_proxy"] = (
                abs(float(integration_span)) * result[name]["rms"]
            )
    if len(actual_parts) >= 2:
        result["packed"] = {
            "relative_l2": _relative_l2(actual, predicted, eps),
            "rms": _rms(actual - predicted),
            "direction_cosine": _cosine(actual, predicted, eps),
        }
        result["modal_max"] = max(result["video"]["relative_l2"], result["audio"]["relative_l2"])
    return result


class RunDiagnostics:
    def __init__(self, config=None, output_root=None, run_id=None, latent_shapes=None, model_fingerprint=None):
        self.config = config
        self.output_root = output_root
        self.run_id = run_id or uuid.uuid4().hex
        self.latent_shapes = latent_shapes
        self.model_fingerprint = model_fingerprint
        self._started = None
        self._steps = []
        self._anchors = []
        self._true_nfe = 0
        self._forecast_count = 0
        self._fallback_count = 0
        self._last_actual_step = None

    @property
    def true_nfe(self):
        return self._true_nfe

    def start_run(self, sigmas=None, **metadata):
        self._started = time.perf_counter()
        self._steps.clear()
        self._anchors.clear()
        self._true_nfe = 0
        self._forecast_count = 0
        self._fallback_count = 0
        self._last_actual_step = None
        self._sigmas = [_json_float(v) for v in sigmas] if sigmas is not None else None
        self._run_metadata = dict(metadata)

    def policy_state(self):
        return {"true_nfe": self._true_nfe, "forecast_count": self._forecast_count}

    def observe_actual_anchor(self, step, sigma, x=None, actual_derivative=None,
                              counterfactual=None, previous_actual_sigma=None,
                              fallback_reason=None, **metadata):
        row = {"step": int(step), "sigma": _json_float(sigma), "actual": True,
               "previous_actual_sigma": (
                   None if previous_actual_sigma is None
                   else _json_float(previous_actual_sigma)
               ),
               "fallback_reason": fallback_reason}
        row["logical_span"] = (
            None if self._last_actual_step is None else int(step) - self._last_actual_step
        )
        if actual_derivative is not None and counterfactual is not None and counterfactual.valid:
            span = (
                None
                if previous_actual_sigma is None
                else float(sigma) - float(previous_actual_sigma)
            )
            row["prediction_metrics"] = modality_metrics(
                actual_derivative,
                counterfactual.derivative,
                self.latent_shapes,
                integration_span=span,
            )
        row.update(metadata)
        self._anchors.append(row)
        self._last_actual_step = int(step)
        if fallback_reason:
            self._fallback_count += 1

    def observe_step(self, step, sigma, forecast, true_nfe, fallback_reason=None, **metadata):
        self._true_nfe = int(true_nfe)
        self._forecast_count += int(bool(forecast))
        self._steps.append({"step": int(step), "sigma": _json_float(sigma), "forecast": bool(forecast),
                            "true_nfe": int(true_nfe), "fallback_reason": fallback_reason, **metadata})

    def finish_run(self, **metadata):
        self._run_metadata.update(metadata)
        if self.config is None or getattr(self.config, "diagnostics", "off") == "off":
            return None
        elapsed = time.perf_counter() - (self._started or time.perf_counter())
        summary = {
            **self._run_metadata,
            "run_id": self.run_id,
            "nominal_steps": len(self._steps),
            "true_nfe": self._true_nfe,
            "forecast_count": self._forecast_count,
            "fallback_count": self._fallback_count,
            "elapsed_seconds": elapsed,
            "model_fingerprint": self.model_fingerprint,
            "sigma_sequence": self._sigmas,
            "steps": self._steps,
            "anchors": self._anchors,
        }
        video_errors = [
            row["prediction_metrics"]["video"]["relative_l2"]
            for row in self._anchors
            if "video" in row.get("prediction_metrics", {})
        ]
        audio_errors = [
            row["prediction_metrics"]["audio"]["relative_l2"]
            for row in self._anchors
            if "audio" in row.get("prediction_metrics", {})
        ]
        summary["maximum_video_prediction_error"] = max(video_errors, default=None)
        summary["maximum_audio_prediction_error"] = max(audio_errors, default=None)
        summary["actual_forecast_mask"] = [not row["forecast"] for row in self._steps]
        if getattr(self.config, "diagnostics", "off") == "summary":
            logging.info(
                "[H3 Vector Accel] method=%s profile=%s steps=%d true_nfe=%d forecasts=%d fallbacks=%d fingerprint=%s",
                summary.get("method"), summary.get("evaluation_profile"),
                len(self._steps), self._true_nfe, self._forecast_count,
                self._fallback_count,
                str(summary.get("configuration_fingerprint", "unknown"))[:12],
            )
            return summary
        root = self.output_root
        if root is None:
            try:
                import folder_paths
                root = folder_paths.get_output_directory()
            except ImportError:
                root = os.path.join(os.getcwd(), "output")
        run_dir = os.path.join(root, "h3_vector_accel", self.run_id)
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, "diagnostics.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, sort_keys=True, indent=2, allow_nan=False)
        return summary
