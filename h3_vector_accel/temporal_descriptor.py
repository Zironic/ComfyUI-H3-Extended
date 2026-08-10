"""Compact temporal measurements for batch-one H3 video latents."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import torch
import torch.nn.functional as F


SUMMARY_KEYS = ("p10", "p50", "p90", "p95", "min", "max", "mean")


@dataclass
class TemporalDescriptor:
    pool_size: int
    channels: int
    temporal_length: int
    pooled_frames: torch.Tensor
    uncentered_features: torch.Tensor
    centered_features: torch.Tensor
    first_differences: torch.Tensor
    second_differences: torch.Tensor
    frame_rms: torch.Tensor
    spatial_contrast: torch.Tensor
    spatial_gradient_energy: torch.Tensor
    spatial_structure_score: torch.Tensor
    first_difference_relative_energy: torch.Tensor
    second_difference_relative_energy: torch.Tensor
    frame_ssm: torch.Tensor
    motion_ssm: torch.Tensor
    source_scale: float


@dataclass
class TemporalTransition:
    frame_ssm_change: float
    motion_ssm_change: float | None
    frame_correspondence: torch.Tensor
    motion_correspondence: torch.Tensor | None
    frame_same_position_score: float | None
    frame_distant_competitor_score: float | None
    frame_alignment_margin: float | None
    frame_normalized_argmax_warp: float | None
    motion_same_position_score: float | None
    motion_distant_competitor_score: float | None
    motion_alignment_margin: float | None
    motion_normalized_argmax_warp: float | None
    localized_structure_change: torch.Tensor
    localized_first_difference_change: torch.Tensor
    localized_second_difference_change: torch.Tensor
    motion_observable_mask: torch.Tensor
    motion_observable_fraction: float


def _validate_video(video_x0: torch.Tensor) -> tuple[int, int, int, int]:
    if not isinstance(video_x0, torch.Tensor):
        raise TypeError("video_x0 must be a torch.Tensor")
    if video_x0.ndim != 5 or video_x0.shape[0] != 1:
        raise ValueError("video_x0 must have batch-one shape [1,C,T,H,W]")
    if not video_x0.is_floating_point() or not bool(torch.isfinite(video_x0).all().item()):
        raise ValueError("video_x0 must contain finite floating-point values")
    _, channels, temporal, height, width = map(int, video_x0.shape)
    if min(channels, temporal, height, width) <= 0:
        raise ValueError("video_x0 dimensions must be positive")
    return channels, temporal, height, width


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _row_rms(value: torch.Tensor) -> torch.Tensor:
    if value.shape[0] == 0:
        return value.new_empty((0,))
    return torch.sqrt(value.float().reshape(value.shape[0], -1).square().mean(dim=1).clamp_min(0.0))


def _safe_denominator(value: torch.Tensor, epsilon: float) -> torch.Tensor:
    return value.clamp_min(epsilon)


def _cosine_matrix(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> torch.Tensor:
    left = left.float().reshape(left.shape[0], -1)
    right = right.float().reshape(right.shape[0], -1)
    left = left / _safe_denominator(torch.linalg.vector_norm(left, dim=1, keepdim=True), epsilon)
    right = right / _safe_denominator(torch.linalg.vector_norm(right, dim=1, keepdim=True), epsilon)
    return (left @ right.transpose(0, 1)).clamp(-1.0, 1.0)


def _ssm(features: torch.Tensor, epsilon: float) -> torch.Tensor:
    return _cosine_matrix(features, features, epsilon)


def _summary_values(values: torch.Tensor | Iterable[float]) -> dict[str, float]:
    if isinstance(values, torch.Tensor):
        flat = values.detach().float().reshape(-1)
        if flat.numel() == 0:
            return {key: 0.0 for key in SUMMARY_KEYS}
        flat = flat[torch.isfinite(flat)]
        numbers = flat.cpu().tolist()
    else:
        numbers = [float(value) for value in values if math.isfinite(float(value))]
    if not numbers:
        return {key: 0.0 for key in SUMMARY_KEYS}
    numbers.sort()

    def quantile(fraction: float) -> float:
        if len(numbers) == 1:
            return numbers[0]
        position = fraction * (len(numbers) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        weight = position - lower
        return numbers[lower] * (1.0 - weight) + numbers[upper] * weight

    return {
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "min": numbers[0],
        "max": numbers[-1],
        "mean": sum(numbers) / len(numbers),
    }


def extract_temporal_descriptor(video_x0: torch.Tensor, pool_size: int = 8,
                                epsilon: float = 1e-8) -> TemporalDescriptor:
    """Pool an H3 video x0 without pooling time and calculate latent proxies."""
    channels, temporal, height, width = _validate_video(video_x0)
    if isinstance(pool_size, bool) or int(pool_size) != pool_size or int(pool_size) <= 0:
        raise ValueError("pool_size must be a positive integer")
    pool_size = int(pool_size)
    epsilon = _positive_finite(epsilon, "epsilon")

    # Keep the large source in its native dtype. Only the compact pooled tensor is
    # promoted to float32 for stable descriptor reductions.
    pooled = F.adaptive_avg_pool2d(
        video_x0.permute(0, 2, 1, 3, 4).reshape(temporal, channels, height, width),
        (pool_size, pool_size),
    ).reshape(temporal, channels, pool_size, pool_size)
    pooled_float = pooled.float()
    source_scale_tensor = torch.sqrt(pooled_float.square().mean().clamp_min(0.0)).clamp_min(epsilon)
    normalized = pooled_float / source_scale_tensor
    features = normalized.reshape(temporal, -1)
    centered = features - features.mean(dim=0, keepdim=True)
    frame_rms = _row_rms(features)

    channel_rms = torch.sqrt(normalized.square().mean(dim=(2, 3)).clamp_min(0.0))
    spatial_centered = normalized - normalized.mean(dim=(2, 3), keepdim=True)
    contrast = torch.sqrt(spatial_centered.square().mean(dim=(2, 3)).clamp_min(0.0))
    contrast = contrast / _safe_denominator(channel_rms, epsilon)
    if pool_size == 1:
        gradient = torch.zeros_like(contrast)
    else:
        horizontal = normalized[..., 1:] - normalized[..., :-1]
        vertical = normalized[:, :, 1:, :] - normalized[:, :, :-1, :]
        gradient_rms = torch.sqrt(
            (horizontal.square().mean(dim=(2, 3)) + vertical.square().mean(dim=(2, 3))).mul(0.5).clamp_min(0.0)
        )
        gradient = gradient_rms / _safe_denominator(channel_rms, epsilon)
    contrast_per_frame = contrast.mean(dim=1)
    gradient_per_frame = gradient.mean(dim=1)
    structure = torch.sqrt((contrast_per_frame * gradient_per_frame).clamp_min(0.0))

    first = features[1:] - features[:-1] if temporal > 1 else features.new_empty((0, features.shape[1]))
    second = first[1:] - first[:-1] if temporal > 2 else features.new_empty((0, features.shape[1]))
    if first.shape[0]:
        first_energy = _row_rms(first) / _safe_denominator(
            (_row_rms(features[1:]) + _row_rms(features[:-1])) * 0.5, epsilon
        )
    else:
        first_energy = features.new_empty((0,))
    if second.shape[0]:
        second_energy = _row_rms(second) / _safe_denominator(
            (_row_rms(first[1:]) + _row_rms(first[:-1])) * 0.5, epsilon
        )
    else:
        second_energy = features.new_empty((0,))

    return TemporalDescriptor(
        pool_size=pool_size,
        channels=channels,
        temporal_length=temporal,
        pooled_frames=pooled,
        uncentered_features=features,
        centered_features=centered,
        first_differences=first,
        second_differences=second,
        frame_rms=frame_rms,
        spatial_contrast=contrast,
        spatial_gradient_energy=gradient,
        spatial_structure_score=structure,
        first_difference_relative_energy=first_energy,
        second_difference_relative_energy=second_energy,
        frame_ssm=_ssm(centered, epsilon),
        motion_ssm=_ssm(first, epsilon) if first.shape[0] else features.new_empty((0, 0)),
        source_scale=float(source_scale_tensor.item()),
    )


def _validate_pair(previous: TemporalDescriptor, current: TemporalDescriptor) -> None:
    if not isinstance(previous, TemporalDescriptor) or not isinstance(current, TemporalDescriptor):
        raise TypeError("previous and current must be TemporalDescriptor instances")
    identity = (previous.pool_size, previous.channels, previous.temporal_length)
    if identity != (current.pool_size, current.channels, current.temporal_length):
        raise ValueError("descriptor shapes are incompatible")
    if previous.uncentered_features.device != current.uncentered_features.device:
        raise ValueError("descriptors must be on the same device")
    temporal, channels, pool_size = previous.temporal_length, previous.channels, previous.pool_size
    expected_features = (temporal, channels * pool_size * pool_size)
    for descriptor in (previous, current):
        if tuple(descriptor.pooled_frames.shape) != (temporal, channels, pool_size, pool_size):
            raise ValueError("descriptor pooled-frame shape is incompatible")
        if tuple(descriptor.uncentered_features.shape) != expected_features:
            raise ValueError("descriptor feature shape is incompatible")
        if tuple(descriptor.centered_features.shape) != expected_features:
            raise ValueError("descriptor centered-feature shape is incompatible")
        if tuple(descriptor.first_differences.shape) != (max(temporal - 1, 0), expected_features[1]):
            raise ValueError("descriptor first-difference shape is incompatible")
        if tuple(descriptor.second_differences.shape) != (max(temporal - 2, 0), expected_features[1]):
            raise ValueError("descriptor second-difference shape is incompatible")
        tensors = (
            descriptor.uncentered_features, descriptor.centered_features,
            descriptor.first_differences, descriptor.second_differences,
            descriptor.spatial_structure_score, descriptor.frame_ssm, descriptor.motion_ssm,
        )
        if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
            raise ValueError("descriptor tensors must be finite")


def _symmetric_relative_rms(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> float:
    numerator = torch.sqrt((left.float() - right.float()).square().mean().clamp_min(0.0))
    left_rms = torch.sqrt(left.float().square().mean().clamp_min(0.0))
    right_rms = torch.sqrt(right.float().square().mean().clamp_min(0.0))
    denominator = (left_rms + right_rms) * 0.5
    return float((numerator / denominator.clamp_min(epsilon)).item())


def _per_position_symmetric_change(left: torch.Tensor, right: torch.Tensor, epsilon: float) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError("localized descriptor tensors are incompatible")
    if left.shape[0] == 0:
        return left.new_empty((0,), dtype=torch.float32)
    left_rms = _row_rms(left)
    right_rms = _row_rms(right)
    difference_rms = _row_rms(left.float() - right.float())
    return difference_rms / _safe_denominator((left_rms + right_rms) * 0.5, epsilon)


def _alignment_metrics(matrix: torch.Tensor, row_mask: torch.Tensor | None) -> tuple[float | None, float | None, float | None, float | None]:
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        return None, None, None, None
    usable = torch.ones(rows, dtype=torch.bool, device=matrix.device) if row_mask is None else row_mask
    if tuple(usable.shape) != (rows,) or not bool(usable.any().item()):
        return None, None, None, None
    same_values = []
    competitor_values = []
    margin_values = []
    warp_values = []
    for row in range(rows):
        if not bool(usable[row].item()):
            continue
        same = matrix[row, row] if row < columns else None
        distant = [column for column in range(columns) if abs(column - row) > 1]
        competitor = matrix[row, distant].max() if distant else None
        if same is not None:
            same_values.append(same)
        if same is not None and competitor is not None:
            competitor_values.append(competitor)
            margin_values.append(same - competitor)
        best = int(matrix[row].argmax().item())
        warp_values.append(abs(best - row) / max(max(rows, columns) - 1, 1))

    def mean_or_none(values: list[torch.Tensor]) -> float | None:
        return float(torch.stack(values).mean().item()) if values else None

    return (
        mean_or_none(same_values),
        mean_or_none(competitor_values),
        mean_or_none(margin_values),
        sum(warp_values) / len(warp_values) if warp_values else None,
    )


def compare_temporal_descriptors(previous: TemporalDescriptor, current: TemporalDescriptor,
                                 motion_energy_floor: float = 1e-6,
                                 epsilon: float = 1e-8) -> TemporalTransition:
    _validate_pair(previous, current)
    epsilon = _positive_finite(epsilon, "epsilon")
    motion_energy_floor = float(motion_energy_floor)
    if not math.isfinite(motion_energy_floor) or motion_energy_floor < 0.0:
        raise ValueError("motion_energy_floor must be finite and non-negative")

    frame_correspondence = _cosine_matrix(previous.uncentered_features, current.uncentered_features, epsilon)
    frame_alignment = _alignment_metrics(frame_correspondence, None)
    motion_mask = (
        (previous.first_difference_relative_energy > motion_energy_floor)
        & (current.first_difference_relative_energy > motion_energy_floor)
    )
    motion_observable_fraction = float(motion_mask.float().mean().item()) if motion_mask.numel() else 0.0
    motion_correspondence = None
    motion_alignment = (None, None, None, None)
    motion_ssm_change = None
    if motion_mask.numel() and bool(motion_mask.any().item()):
        motion_correspondence = _cosine_matrix(previous.first_differences, current.first_differences, epsilon)
        motion_alignment = _alignment_metrics(motion_correspondence, motion_mask)
        indices = motion_mask.nonzero(as_tuple=False).flatten()
        previous_ssm = previous.motion_ssm.index_select(0, indices).index_select(1, indices)
        current_ssm = current.motion_ssm.index_select(0, indices).index_select(1, indices)
        motion_ssm_change = _symmetric_relative_rms(previous_ssm, current_ssm, epsilon)

    return TemporalTransition(
        frame_ssm_change=_symmetric_relative_rms(previous.frame_ssm, current.frame_ssm, epsilon),
        motion_ssm_change=motion_ssm_change,
        frame_correspondence=frame_correspondence,
        motion_correspondence=motion_correspondence,
        frame_same_position_score=frame_alignment[0],
        frame_distant_competitor_score=frame_alignment[1],
        frame_alignment_margin=frame_alignment[2],
        frame_normalized_argmax_warp=frame_alignment[3],
        motion_same_position_score=motion_alignment[0],
        motion_distant_competitor_score=motion_alignment[1],
        motion_alignment_margin=motion_alignment[2],
        motion_normalized_argmax_warp=motion_alignment[3],
        localized_structure_change=_per_position_symmetric_change(
            previous.spatial_structure_score[:, None], current.spatial_structure_score[:, None], epsilon
        ),
        localized_first_difference_change=_per_position_symmetric_change(
            previous.first_differences, current.first_differences, epsilon
        ),
        localized_second_difference_change=_per_position_symmetric_change(
            previous.second_differences, current.second_differences, epsilon
        ),
        motion_observable_mask=motion_mask,
        motion_observable_fraction=motion_observable_fraction,
    )


def descriptor_summary(descriptor: TemporalDescriptor,
                       structure_threshold: float | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pool_size": descriptor.pool_size,
        "channels": descriptor.channels,
        "temporal_length": descriptor.temporal_length,
        "source_scale": descriptor.source_scale,
        "frame_rms": _summary_values(descriptor.frame_rms),
        "spatial_contrast": _summary_values(descriptor.spatial_contrast),
        "spatial_gradient_energy": _summary_values(descriptor.spatial_gradient_energy),
        "spatial_structure_score": _summary_values(descriptor.spatial_structure_score),
        "first_difference_relative_energy": _summary_values(descriptor.first_difference_relative_energy),
        "second_difference_relative_energy": _summary_values(descriptor.second_difference_relative_energy),
    }
    if structure_threshold is not None:
        threshold = float(structure_threshold)
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError("structure_threshold must be finite and non-negative")
        result["structure_coverage"] = float(
            (descriptor.spatial_structure_score >= threshold).float().mean().item()
        )
    return result


def transition_summary(transition: TemporalTransition) -> dict[str, Any]:
    def finite_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    return {
        "frame_ssm_change": finite_or_none(transition.frame_ssm_change),
        "motion_ssm_change": finite_or_none(transition.motion_ssm_change),
        "frame_same_position_score": finite_or_none(transition.frame_same_position_score),
        "frame_distant_competitor_score": finite_or_none(transition.frame_distant_competitor_score),
        "frame_alignment_margin": finite_or_none(transition.frame_alignment_margin),
        "frame_normalized_argmax_warp": finite_or_none(transition.frame_normalized_argmax_warp),
        "motion_same_position_score": finite_or_none(transition.motion_same_position_score),
        "motion_distant_competitor_score": finite_or_none(transition.motion_distant_competitor_score),
        "motion_alignment_margin": finite_or_none(transition.motion_alignment_margin),
        "motion_normalized_argmax_warp": finite_or_none(transition.motion_normalized_argmax_warp),
        "motion_observable_fraction": transition.motion_observable_fraction,
        "localized_spatial_structure_change": _summary_values(transition.localized_structure_change),
        "localized_first_difference_change": _summary_values(transition.localized_first_difference_change),
        "localized_second_difference_change": _summary_values(transition.localized_second_difference_change),
    }


__all__ = [
    "TemporalDescriptor",
    "TemporalTransition",
    "extract_temporal_descriptor",
    "compare_temporal_descriptors",
    "descriptor_summary",
    "transition_summary",
]
