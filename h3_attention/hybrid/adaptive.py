"""Budget-neutral adaptive density policy for H3 Sparse-Sage routing."""

from dataclasses import dataclass
import math

import torch

from .config import DENSITY_ADAPTIVE_BUDGET, DENSITY_FIXED


class AdaptiveDensityError(ValueError):
    pass


@dataclass(frozen=True)
class DensityPlan:
    mode: str
    target: int
    minimum: int
    maximum: int
    temperature: float
    target_mass: float


def resolve_density_plan(config, video_budget, pure_kv):
    """Quantize density controls to one static video-KV geometry."""
    mode = DENSITY_FIXED if config is None else str(config.density_mode)
    if mode not in (DENSITY_FIXED, DENSITY_ADAPTIVE_BUDGET):
        raise AdaptiveDensityError("unknown density_mode %r" % mode)
    target = min(pure_kv, max(1, math.ceil(float(video_budget) * pure_kv)))
    if mode == DENSITY_FIXED:
        return DensityPlan(mode, target, target, target, 1.0, 1.0)

    minimum_density = float(config.min_video_density)
    maximum_density = float(config.max_video_density)
    temperature = float(config.adaptive_temperature)
    target_mass = float(config.adaptive_target_mass)
    if not (0.0 < minimum_density <= maximum_density <= 1.0):
        raise AdaptiveDensityError(
            "adaptive density rails must satisfy 0 < min <= max <= 1"
        )
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise AdaptiveDensityError(
            "adaptive_temperature must be finite and positive"
        )
    if not math.isfinite(target_mass) or not 0.0 < target_mass <= 1.0:
        raise AdaptiveDensityError(
            "adaptive_target_mass must be finite and in (0, 1]"
        )
    minimum = min(pure_kv, max(1, math.ceil(minimum_density * pure_kv)))
    maximum = min(pure_kv, max(1, math.ceil(maximum_density * pure_kv)))
    if not minimum <= target <= maximum:
        raise AdaptiveDensityError(
            "quantized adaptive target %d lies outside quantized rails [%d, %d]"
            % (target, minimum, maximum)
        )
    return DensityPlan(
        mode, target, minimum, maximum, temperature, target_mass
    )


def allocate_adaptive_rows(scores, plan, head_dim):
    """Return exact-budget row counts and score-ordered block candidates.

    Coarse top-p gives each row an unconstrained demand. A fixed scalar bisection
    shifts those demands under the rails, then largest-remainder integerization
    preserves exactly the block count of the corresponding fixed-density route.
    """
    scores_fp32 = scores.float()
    finite_rows = torch.isfinite(scores_fp32).all(dim=-1)
    safe = torch.nan_to_num(
        scores_fp32, nan=0.0, posinf=1.0e4, neginf=-1.0e4
    )
    top_scores, top_indices = torch.topk(
        safe, k=plan.maximum, dim=-1, sorted=True
    )
    row_shape = scores.shape[:-1]
    row_count = math.prod(int(value) for value in row_shape)
    if plan.minimum == plan.maximum:
        return torch.full(
            row_shape, plan.minimum, dtype=torch.int32, device=scores.device
        ), top_indices

    scale = (float(head_dim) ** -0.5) / plan.temperature
    probability = torch.exp(
        top_scores * scale
        - torch.logsumexp(safe * scale, dim=-1, keepdim=True)
    )
    requested = (
        (torch.cumsum(probability, dim=-1) < plan.target_mass).sum(dim=-1)
        + 1
    ).clamp(min=plan.minimum, max=plan.maximum).float()
    requested = torch.where(
        finite_rows,
        requested,
        torch.full_like(requested, float(plan.maximum)),
    )
    if plan.target in (plan.minimum, plan.maximum):
        return torch.full(
            row_shape, plan.target, dtype=torch.int32, device=scores.device
        ), top_indices

    low = requested.new_tensor(float(plan.minimum - plan.maximum))
    high = requested.new_tensor(float(plan.maximum - plan.minimum))
    target = requested.new_tensor(float(plan.target))
    for _ in range(32):
        midpoint = (low + high) * 0.5
        candidate = torch.clamp(
            requested + midpoint,
            min=float(plan.minimum),
            max=float(plan.maximum),
        )
        feasible = candidate.mean() <= target
        low = torch.where(feasible, midpoint, low)
        high = torch.where(feasible, high, midpoint)

    allocated = torch.clamp(
        requested + low,
        min=float(plan.minimum),
        max=float(plan.maximum),
    )
    base = torch.floor(allocated).to(torch.int32)
    deficit = (
        base.new_tensor(plan.target * row_count)
        - base.sum(dtype=torch.int32)
    ).clamp(min=0, max=row_count)
    fraction = torch.where(
        base < plan.maximum,
        allocated - base.float(),
        torch.full_like(allocated, float("-inf")),
    ).reshape(-1)
    order = torch.argsort(fraction, descending=True, stable=True)
    rank = torch.arange(row_count, device=scores.device, dtype=torch.int32)
    ordered_increment = (rank < deficit).to(torch.int32)
    increment = torch.zeros(
        row_count, dtype=torch.int32, device=scores.device
    ).scatter(0, order, ordered_increment)
    return (base.reshape(-1) + increment).reshape(row_shape), top_indices
