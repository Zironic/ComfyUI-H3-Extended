"""Trajectory replay, survival profiles, and adaptive-profile validation."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch

from .fingerprint import canonical_json, sigma_hash
from .predictor import make_predictor


PROFILE_SCHEMA_VERSION = 1
PROFILE_ROOT = Path(__file__).with_name("profiles")
DEFAULT_PERTURBATION_STEPS = (2, 5, 8, 11, 14, 16, 18)


def _parts(value, latent_shapes):
    if latent_shapes and len(latent_shapes) >= 2:
        offset, out = 0, []
        for shape in latent_shapes[:2]:
            shape = tuple(shape)
            count = math.prod(shape[1:])
            out.append(value[:, :, offset:offset + count].reshape((value.shape[0],) + shape[1:]))
            offset += count
        return out
    return [value]


def _sigma_broadcast(sigma, value):
    sigma = torch.as_tensor(sigma, device=value.device, dtype=value.dtype)
    if sigma.numel() == 1:
        return sigma.reshape((1,) + (1,) * (value.ndim - 1))
    return sigma.reshape((sigma.shape[0],) + (1,) * (value.ndim - 1))


def _to_d(x, sigma, denoised):
    return (x - denoised) / _sigma_broadcast(sigma, x)


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).clone()
    if isinstance(value, dict):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    return value


def _device_snapshot(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _device_snapshot(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_device_snapshot(item, device) for item in value)
    if isinstance(value, list):
        return [_device_snapshot(item, device) for item in value]
    return value


def per_modality_divergence(branch, native, latent_shapes=None, eps=1e-8):
    result = {}
    native = native.to(device=branch.device)
    left_parts, right_parts = _parts(branch, latent_shapes), _parts(native, latent_shapes)
    names = ("video", "audio") if len(left_parts) >= 2 else ("packed",)
    for name, left, right in zip(names, left_parts, right_parts):
        numerator = torch.sqrt(torch.mean((left.float() - right.float()) ** 2)).item()
        denominator = torch.sqrt(torch.mean(right.float() ** 2)).item() + eps
        result[name] = float(numerator / denominator)
    if len(left_parts) >= 2:
        result["modal_max"] = max(result["video"], result["audio"])
    return result


def normalized_perturbation(delta, latent_shapes=None, target_rms=1.0, modality="joint", eps=1e-8):
    if modality not in ("joint", "video", "audio"):
        raise ValueError("modality must be joint, video, or audio")
    if not math.isfinite(float(target_rms)) or target_rms <= 0:
        raise ValueError("target_rms must be finite and positive")
    parts = _parts(delta, latent_shapes)
    if len(parts) == 1:
        rms = torch.sqrt(torch.mean(delta.float() ** 2)).item()
        if rms <= eps:
            raise ValueError("cannot normalize a zero perturbation")
        return delta * (float(target_rms) / (rms + eps))

    result = torch.zeros_like(delta)
    selected = (0, 1) if modality == "joint" else ((0,) if modality == "video" else (1,))
    offset = 0
    for index, part in enumerate(parts):
        count = part.numel() // delta.shape[0]
        if index in selected:
            rms = torch.sqrt(torch.mean(part.float() ** 2)).item()
            if rms <= eps:
                raise ValueError(f"cannot normalize a zero {('video', 'audio')[index]} perturbation")
            scaled = part * (float(target_rms) / (rms + eps))
            result[:, :, offset:offset + count] = scaled.reshape(delta.shape[0], 1, count)
        offset += count
    return result


def survival_factor(divergence_at_introduction, final_divergence, eps=1e-8):
    return float(final_divergence) / (float(divergence_at_introduction) + eps)


@dataclass(frozen=True)
class TrajectorySnapshot:
    step: int
    sigma: float
    x: torch.Tensor
    derivative: torch.Tensor
    predictor_states: dict


@dataclass(frozen=True)
class NativeTrajectory:
    sigmas: torch.Tensor
    snapshots: tuple[TrajectorySnapshot, ...]
    final_state: torch.Tensor
    latent_shapes: object
    metadata: dict
    runtime_device: str
    runtime_dtype: torch.dtype

    @property
    def logical_steps(self):
        return len(self.snapshots)

    def state(self, index):
        if index == self.logical_steps:
            return self.final_state
        return self.snapshots[index].x


@dataclass(frozen=True)
class BranchResult:
    step: int
    method: str
    branch_type: str
    modality: str
    progress: float
    divergence_curve: tuple[dict, ...]
    survival: dict
    natural_delta: torch.Tensor | None = None

    def as_dict(self):
        return {
            "step": self.step,
            "method": self.method,
            "branch_type": self.branch_type,
            "modality": self.modality,
            "progress": self.progress,
            "divergence_curve": list(self.divergence_curve),
            "survival": dict(self.survival),
        }


def capture_native_trajectory(model, x, sigmas, extra_args=None, latent_shapes=None,
                              predictor_methods=("hold", "linear_velocity", "vde"), metadata=None):
    """Capture a deterministic native Euler trajectory with FP32 CPU snapshots."""
    extra_args = {} if extra_args is None else extra_args
    sigma_values = torch.as_tensor(sigmas)
    if sigma_values.ndim != 1 or sigma_values.numel() < 2:
        raise ValueError("sigma schedule must contain at least two values")
    if not bool(torch.isfinite(sigma_values).all().item()) or bool(((sigma_values[1:] - sigma_values[:-1]) >= 0).any().item()):
        raise ValueError("repairability capture requires a finite decreasing sigma schedule")
    predictors = {
        method: make_predictor(method, latent_shapes=latent_shapes)
        for method in predictor_methods
    }
    s_in = x.new_ones([x.shape[0]])
    snapshots = []
    for step in range(sigma_values.numel() - 1):
        sigma = sigma_values[step].to(device=x.device)
        sigma_next = sigma_values[step + 1].to(device=x.device)
        states = {method: _cpu_snapshot(predictor.snapshot()) for method, predictor in predictors.items()}
        denoised = model(x, sigma * s_in, **extra_args)
        derivative = _to_d(x, sigma, denoised)
        if not bool(torch.isfinite(derivative).all().item()):
            raise RuntimeError(f"model returned a non-finite derivative at step {step}")
        snapshots.append(TrajectorySnapshot(
            step=step,
            sigma=float(sigma.detach().float().item()),
            x=_cpu_snapshot(x),
            derivative=_cpu_snapshot(derivative),
            predictor_states=states,
        ))
        for predictor in predictors.values():
            predictor.observe_actual(x, sigma, derivative)
        x = x + derivative * (sigma_next - sigma)
    return NativeTrajectory(
        sigmas=sigma_values.detach().to(device="cpu", dtype=torch.float64),
        snapshots=tuple(snapshots),
        final_state=_cpu_snapshot(x),
        latent_shapes=latent_shapes,
        metadata=dict(metadata or {}),
        runtime_device=str(x.device),
        runtime_dtype=x.dtype,
    )


def _branch_continuation(model, trajectory, start_state, first_state_index, extra_args=None):
    extra_args = {} if extra_args is None else extra_args
    device = start_state.device
    x = start_state
    curve = [{"state_index": first_state_index, **per_modality_divergence(
        x, trajectory.state(first_state_index), trajectory.latent_shapes
    )}]
    s_in = x.new_ones([x.shape[0]])
    for step in range(first_state_index, trajectory.logical_steps):
        sigma = trajectory.sigmas[step].to(device=device, dtype=x.dtype)
        sigma_next = trajectory.sigmas[step + 1].to(device=device, dtype=x.dtype)
        denoised = model(x, sigma * s_in, **extra_args)
        derivative = _to_d(x, sigma, denoised)
        x = x + derivative * (sigma_next - sigma)
        curve.append({"state_index": step + 1, **per_modality_divergence(
            x, trajectory.state(step + 1), trajectory.latent_shapes
        )})
    return x, tuple(curve)


def _survival_from_curve(curve):
    introduced, final = curve[0], curve[-1]
    names = ("video", "audio") if "video" in introduced else ("packed",)
    result = {name: survival_factor(introduced[name], final[name]) for name in names}
    if "video" in result:
        result["joint_conservative_max"] = max(result["video"], result["audio"])
    return result


def run_natural_omission(model, trajectory, step, method, extra_args=None, device=None):
    """Omit one native evaluation, then continue with genuine evaluations."""
    if step < 0 or step >= trajectory.logical_steps:
        raise ValueError("omission step is outside the trajectory")
    snapshot = trajectory.snapshots[step]
    if method not in snapshot.predictor_states:
        raise ValueError(f"trajectory does not contain {method!r} predictor state")
    target_device = torch.device(device) if device is not None else torch.device(trajectory.runtime_device)
    x = snapshot.x.to(device=target_device, dtype=trajectory.runtime_dtype)
    predictor = make_predictor(method, latent_shapes=trajectory.latent_shapes)
    predictor.restore(_device_snapshot(snapshot.predictor_states[method], target_device))
    sigma = trajectory.sigmas[step].to(device=target_device, dtype=x.dtype)
    sigma_next = trajectory.sigmas[step + 1].to(device=target_device, dtype=x.dtype)
    prediction = predictor.predict(x, sigma)
    if not prediction.valid:
        raise ValueError(f"predictor cannot omit step {step}: {prediction.failure_reason}")
    branch_next = predictor.integrate(x, sigma, sigma_next, prediction)
    native_next = trajectory.state(step + 1).to(device=target_device, dtype=trajectory.runtime_dtype)
    natural_delta = branch_next.float() - native_next.float()
    _, curve = _branch_continuation(model, trajectory, branch_next, step + 1, extra_args=extra_args)
    progress = step / max(1, trajectory.logical_steps - 1)
    return BranchResult(step, method, "natural_omission", "joint", progress, curve,
                        _survival_from_curve(curve), _cpu_snapshot(natural_delta))


def run_normalized_perturbation(model, trajectory, natural_result, modality,
                                target_rms, extra_args=None, device=None):
    if natural_result.natural_delta is None:
        raise ValueError("normalized perturbation requires a natural omission delta")
    step = natural_result.step
    target_device = torch.device(device) if device is not None else torch.device(trajectory.runtime_device)
    native_next = trajectory.state(step + 1).to(device=target_device, dtype=trajectory.runtime_dtype)
    delta = natural_result.natural_delta.to(device=target_device)
    injected = normalized_perturbation(
        delta, trajectory.latent_shapes, target_rms, modality
    ).to(dtype=native_next.dtype)
    branch_next = native_next + injected
    _, curve = _branch_continuation(model, trajectory, branch_next, step + 1, extra_args=extra_args)
    return BranchResult(step, natural_result.method, "normalized_perturbation", modality,
                        natural_result.progress, curve, _survival_from_curve(curve))


def run_repairability_sweep(model, trajectory, method="linear_velocity", steps=None,
                            target_rms=0.01, extra_args=None, device=None):
    results = []
    for step in DEFAULT_PERTURBATION_STEPS if steps is None else tuple(steps):
        natural = run_natural_omission(model, trajectory, step, method, extra_args, device)
        results.append(natural)
        for modality in ("joint", "video", "audio"):
            results.append(run_normalized_perturbation(
                model, trajectory, natural, modality, target_rms, extra_args, device
            ))
    return tuple(results)


def _quantile(values, probability):
    values = sorted(float(value) for value in values)
    if not values:
        raise ValueError("cannot calculate a quantile without values")
    position = (len(values) - 1) * float(probability)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def build_repairability_profile(entries, sigmas=None, model_fingerprint=None,
                                video_shift=None, audio_shift=None,
                                nominal_steps=None, predictor_method=None,
                                conditioning_mode=None, quality_presets=None,
                                quantiles=(0.5, 0.9, 0.95), adaptive_methods=None,
                                **metadata):
    quantiles = tuple(float(value) for value in quantiles)
    if not quantiles or any(not 0 <= value <= 1 for value in quantiles):
        raise ValueError("profile quantiles must be probabilities between zero and one")
    rows = [entry.as_dict() if isinstance(entry, BranchResult) else dict(entry) for entry in entries]
    if not rows:
        raise ValueError("repairability profile requires at least one branch result")
    grouped = {}
    for row in rows:
        if "survival" not in row:
            survival = {
                "video": float(row["video"]),
                "audio": float(row.get("audio", row["video"])),
            }
        else:
            survival = row["survival"]
        progress = float(row.get("progress", row.get("progress_bin", 0.0)))
        grouped.setdefault(progress, []).append(survival)
    bins = []
    for progress, samples in sorted(grouped.items()):
        video = [float(sample.get("video", sample.get("packed", 0.0))) for sample in samples]
        audio = [float(sample.get("audio", sample.get("packed", 0.0))) for sample in samples]
        joint = [float(sample.get("joint_conservative_max", max(
            sample.get("video", sample.get("packed", 0.0)),
            sample.get("audio", sample.get("packed", 0.0)),
        ))) for sample in samples]
        video_q = {str(value): _quantile(video, value) for value in quantiles}
        audio_q = {str(value): _quantile(audio, value) for value in quantiles}
        conservative = max(
            video_q[str(max(quantiles))],
            audio_q[str(max(quantiles))],
            _quantile(joint, max(quantiles)),
        )
        bins.append({
            "progress": progress,
            "sample_count": len(samples),
            "video_survival_quantiles": video_q,
            "audio_survival_quantiles": audio_q,
            "joint_conservative_max": conservative,
        })
    compatibility = {
        "model_fingerprint": model_fingerprint,
        "sigma_hash": sigma_hash(sigmas) if sigmas is not None else None,
        "video_shift": None if video_shift is None else float(video_shift),
        "audio_shift": None if audio_shift is None else float(audio_shift),
        "nominal_steps": (
            int(nominal_steps)
            if nominal_steps is not None
            else (len(sigmas) - 1 if sigmas is not None else None)
        ),
        "predictor_method": predictor_method,
        "conditioning_mode": conditioning_mode,
    }
    profile = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "compatibility": compatibility,
        "quantile": str(max(quantiles)),
        "quality_presets": dict(quality_presets or {}),
        "adaptive_methods": list((predictor_method,) if adaptive_methods is None else adaptive_methods),
        "sample_count": len(rows),
        "bins": bins,
        "metadata": dict(metadata),
    }
    canonical_json(profile)
    return profile


def profile_json(profile):
    return json.dumps(profile, sort_keys=True, indent=2, allow_nan=False)


def resolve_profile_path(filename):
    name = Path(str(filename))
    if name.name != str(filename) or name.suffix.lower() != ".json":
        raise ValueError("repairability profile must be a JSON filename in h3_vector_accel/profiles")
    resolved = (PROFILE_ROOT / name.name).resolve()
    if resolved.parent != PROFILE_ROOT.resolve():
        raise ValueError("repairability profile escapes the profile directory")
    return resolved


@dataclass(frozen=True)
class ProfileCompatibility:
    model_fingerprint: str
    sigma_hash: str
    video_shift: float | None
    audio_shift: float | None
    nominal_steps: int
    predictor_method: str
    conditioning_mode: str


class RepairabilityProfile:
    def __init__(self, payload, source=None):
        if payload.get("schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported repairability profile schema")
        if not payload.get("bins"):
            raise ValueError("repairability profile has no survival bins")
        self.payload = payload
        self.source = source
        self.hash = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    @classmethod
    def load(cls, filename):
        path = resolve_profile_path(filename)
        with path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle), source=str(path))

    def validate_compatibility(self, context):
        expected = self.payload["compatibility"]
        actual = {
            "model_fingerprint": context.model_fingerprint,
            "sigma_hash": context.sigma_hash,
            "video_shift": context.video_shift,
            "audio_shift": context.audio_shift,
            "nominal_steps": context.nominal_steps,
            "predictor_method": context.predictor_method,
            "conditioning_mode": context.conditioning_mode,
        }
        mismatches = []
        for key, value in actual.items():
            wanted = expected.get(key)
            if key in ("video_shift", "audio_shift") and wanted is not None and value is not None:
                matches = math.isclose(float(wanted), float(value), rel_tol=0.0, abs_tol=1e-9)
            else:
                matches = wanted == value
            if not matches:
                mismatches.append(f"{key}: profile={wanted!r}, run={value!r}")
        if context.predictor_method not in self.payload.get("adaptive_methods", []):
            mismatches.append(f"predictor {context.predictor_method!r} is not approved for adaptive use")
        if mismatches:
            raise ValueError("repairability profile mismatch: " + "; ".join(mismatches))
        return True

    def tolerance(self, preset):
        value = self.payload.get("quality_presets", {}).get(preset)
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"repairability profile has no valid {preset!r} tolerance")
        return float(value)

    def survival(self, progress):
        bins = sorted(self.payload["bins"], key=lambda item: float(item["progress"]))
        progress = min(1.0, max(0.0, float(progress)))
        lower = [item for item in bins if float(item["progress"]) <= progress]
        upper = [item for item in bins if float(item["progress"]) >= progress]
        candidates = []
        if lower:
            candidates.append(lower[-1])
        if upper and (not candidates or upper[0] is not candidates[0]):
            candidates.append(upper[0])
        quantile = str(self.payload["quantile"])
        return {
            "video": max(float(item["video_survival_quantiles"][quantile]) for item in candidates),
            "audio": max(float(item["audio_survival_quantiles"][quantile]) for item in candidates),
        }
