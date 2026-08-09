"""CPU-only pure helpers for offline repairability analysis."""

import json
import math

import torch

from .fingerprint import canonical_json, sigma_hash


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


def per_modality_divergence(branch, native, latent_shapes=None, eps=1e-8):
    result = {}
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
    if modality not in ("video", "audio"):
        if modality != "joint":
            raise ValueError("modality must be joint, video, or audio")
    parts = _parts(delta, latent_shapes)
    if len(parts) == 1:
        rms = torch.sqrt(torch.mean(delta.float() ** 2)).item()
        return delta * (float(target_rms) / (rms + eps))

    result = torch.zeros_like(delta)
    selected = (0, 1) if modality == "joint" else ((0,) if modality == "video" else (1,))
    offset = 0
    for index, part in enumerate(parts):
        count = part.numel() // delta.shape[0]
        if index in selected:
            rms = torch.sqrt(torch.mean(part.float() ** 2)).item()
            scaled = part * (float(target_rms) / (rms + eps))
            result[:, :, offset:offset + count] = scaled.reshape(delta.shape[0], 1, count)
        offset += count
    return result


def survival_factor(divergence_at_introduction, final_divergence, eps=1e-8):
    return float(final_divergence) / (float(divergence_at_introduction) + eps)


def build_repairability_profile(entries, sigmas=None, model_fingerprint=None, **metadata):
    entries = list(entries)
    video = [float(item["video"] if isinstance(item, dict) else item) for item in entries]
    audio = [float(item.get("audio", item.get("video", 0.0))) for item in entries if isinstance(item, dict)]
    profile = {"metadata": dict(metadata), "model_fingerprint": model_fingerprint,
               "sigma_hash": sigma_hash(sigmas) if sigmas is not None else None,
               "sample_count": len(entries), "video_survival": video, "audio_survival": audio,
               "joint_conservative_max": max(video + audio, default=0.0)}
    canonical_json(profile)
    return profile


def profile_json(profile):
    return json.dumps(profile, sort_keys=True, separators=(",", ":"), allow_nan=False)
