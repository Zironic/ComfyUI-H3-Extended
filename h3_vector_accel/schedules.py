"""Fixed continuous sigma schedules for H3 characterization."""

import math

import torch


GEOMETRIC_SCHEDULE_VERSION = "v1"
GEOMETRIC_PROFILE_NFE = {
    "geometric_11": 11,
    "geometric_linear_ends_11": 11,
}


def _geometric_sum(ratio: float, count: int) -> float:
    if abs(ratio - 1.0) < 1e-12:
        return float(count)
    return (ratio ** count - 1.0) / (ratio - 1.0)


def _solve_ratio(first_interval: float, total: float, count: int) -> float:
    target = total / first_interval
    if target < count - 1e-10:
        raise ValueError("geometric schedule span is too short for its fixed first interval")
    if abs(target - count) <= 1e-10:
        return 1.0
    low, high = 1.0, 2.0
    while _geometric_sum(high, count) < target:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) * 0.5
        if _geometric_sum(middle, count) < target:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def _times_from_intervals(start: float, first: float, ratio: float, count: int,
                          end: float) -> list[float]:
    times = [start]
    current = start
    for index in range(count):
        current += first * ratio ** index
        times.append(current)
    times[-1] = end
    return times


def _source_times(source_sigmas: torch.Tensor) -> list[float]:
    if source_sigmas.ndim != 1 or source_sigmas.numel() < 4:
        raise ValueError("geometric schedules require a one-dimensional sigma schedule")
    if not bool(torch.isfinite(source_sigmas).all().item()):
        raise ValueError("geometric schedules require finite sigma values")
    if bool(((source_sigmas[1:] - source_sigmas[:-1]) >= 0).any().item()):
        raise ValueError("geometric schedules require strictly descending sigma values")
    values = [float(value) for value in source_sigmas.detach().cpu()]
    if values[-1] != 0.0 or values[-2] <= 0.0:
        raise ValueError("geometric schedules require one terminal zero after positive sigmas")
    return [-math.log(value) for value in values[:-1]]


def geometric_schedule(source_sigmas, profile: str) -> tuple[torch.Tensor, tuple[float, ...], float]:
    """Build a fixed geometric positive-sigma schedule plus terminal zero."""
    if profile not in GEOMETRIC_PROFILE_NFE:
        raise ValueError(f"unknown geometric schedule profile: {profile}")
    source = torch.as_tensor(source_sigmas)
    times = _source_times(source)
    nfe = GEOMETRIC_PROFILE_NFE[profile]

    if profile == "geometric_11":
        interval_count = nfe - 1
        first = times[1] - times[0]
        ratio = _solve_ratio(first, times[-1] - times[0], interval_count)
        effective_times = _times_from_intervals(
            times[0], first, ratio, interval_count, times[-1]
        )
    else:
        middle_count = nfe - 3
        first = times[2] - times[1]
        ratio = _solve_ratio(first, times[-2] - times[1], middle_count)
        middle_times = _times_from_intervals(
            times[1], first, ratio, middle_count, times[-2]
        )
        effective_times = [times[0], *middle_times, times[-1]]

    positive = torch.exp(-torch.tensor(
        effective_times, device=source.device, dtype=torch.float64,
    )).to(dtype=source.dtype)
    positive[0] = source[0]
    if profile == "geometric_11":
        positive[1] = source[1]
    else:
        positive[1] = source[1]
        positive[2] = source[2]
        positive[-2] = source[-3]
    positive[-1] = source[-2]
    effective = torch.cat((positive, source[-1:]))

    coordinates = []
    for sigma in effective[:-1]:
        value = float(sigma)
        sigma_t = -math.log(value)
        if sigma_t <= times[0]:
            coordinates.append(0.0)
            continue
        if sigma_t >= times[-1]:
            coordinates.append(float(len(times) - 1))
            continue
        left = next(index for index in range(len(times) - 1)
                    if times[index] <= sigma_t <= times[index + 1])
        fraction = (sigma_t - times[left]) / (times[left + 1] - times[left])
        coordinates.append(left + fraction)
    return effective, tuple(coordinates), ratio


def geometric_schedule_identity(profile: str) -> dict:
    if profile not in GEOMETRIC_PROFILE_NFE:
        raise ValueError(f"unknown geometric schedule profile: {profile}")
    return {
        "version": GEOMETRIC_SCHEDULE_VERSION,
        "profile": profile,
        "true_nfe": GEOMETRIC_PROFILE_NFE[profile],
        "time_coordinate": "negative_log_sigma",
        "interval_rule": (
            "native_first_then_geometric" if profile == "geometric_11"
            else "native_0_1_2_geometric_interior_native_18_19"
        ),
    }


__all__ = [
    "GEOMETRIC_PROFILE_NFE",
    "GEOMETRIC_SCHEDULE_VERSION",
    "geometric_schedule",
    "geometric_schedule_identity",
]
