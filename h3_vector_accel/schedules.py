"""Continuous scheduler families and fixed compatibility profiles for H3."""

import math

import torch


CONTINUOUS_SCHEDULE_VERSION = "v2"
CONTINUOUS_SCHEDULE_FAMILIES = (
    "geometric",
    "geometric_linear_ends",
    "multiplicative_stride",
    "multiplicative_stride_linear_ends",
)
CONTINUOUS_PROFILE_NFE = {
    "geometric_11": 11,
    "geometric_linear_ends_11": 11,
    "multiplicative_stride_11": 11,
    "multiplicative_stride_linear_ends_11": 11,
}


def _geometric_sum(ratio: float, count: int) -> float:
    if abs(ratio - 1.0) < 1e-12:
        return float(count)
    return (ratio ** count - 1.0) / (ratio - 1.0)


def _solve_ratio(first_interval: float, total: float, count: int) -> float:
    if count <= 0:
        raise ValueError("geometric schedule requires at least one interval")
    if not math.isfinite(first_interval) or first_interval <= 0.0:
        raise ValueError("geometric schedule requires a positive first interval")
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("geometric schedule requires a positive span")
    target = total / first_interval
    if target < 1.0 - 1e-10:
        raise ValueError("geometric schedule span is too short for its fixed first interval")
    if count == 1:
        return target
    if abs(target - count) <= 1e-10:
        return 1.0
    if target < count:
        low, high = 0.0, 1.0
    else:
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


def _coordinates_from_effective(source: torch.Tensor, times: list[float],
                                effective: torch.Tensor) -> tuple[float, ...]:
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
    return tuple(coordinates)


def continuous_schedule_family(
    source_sigmas, family: str, steps: int,
) -> tuple[torch.Tensor, tuple[float, ...], float]:
    """Build a named continuous schedule for an arbitrary true NFE count."""
    if family not in CONTINUOUS_SCHEDULE_FAMILIES:
        raise ValueError(f"unknown continuous schedule family: {family}")
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise ValueError("continuous schedule steps must be an integer")
    if steps <= 0:
        raise ValueError("continuous schedule steps must be positive")
    if family.endswith("_linear_ends") and steps < 5:
        raise ValueError(
            f"{family} requires at least 5 steps to preserve native head and tail endpoints"
        )

    source = torch.as_tensor(source_sigmas)
    times = _source_times(source)
    nfe = steps

    if family == "geometric":
        base = _solve_unit_geometric_base(nfe)
        ratio = 1.0 / base
        span = float(source[0] - source[-1])
        current = float(source[0])
        effective_values = [current]
        for power in range(nfe, 1, -1):
            current -= span * base ** power
            effective_values.append(current)
        effective_values.append(float(source[-1]))
        effective = torch.tensor(
            effective_values, device=source.device, dtype=torch.float64,
        ).to(dtype=source.dtype)
        effective[0] = source[0]
        effective[-1] = source[-1]
        return effective, _coordinates_from_effective(source, times, effective), ratio

    if family == "geometric_linear_ends":
        middle_count = nfe - 3
        first = times[2] - times[1]
        ratio = _solve_ratio(first, times[-2] - times[1], middle_count)
        middle_times = _values_from_intervals(
            times[1], first, ratio, middle_count, times[-2]
        )
        effective_times = [times[0], *middle_times, times[-1]]
        positive = torch.exp(-torch.tensor(
            effective_times, device=source.device, dtype=torch.float64,
        )).to(dtype=source.dtype)
        positive[0] = source[0]
        positive[1] = source[1]
        positive[2] = source[2]
        positive[-2] = source[-3]
        positive[-1] = source[-2]
        effective = torch.cat((positive, source[-1:]))
        return effective, _coordinates_from_effective(source, times, effective), ratio

    if source.numel() != 21:
        raise ValueError(f"{family} requires exactly 20 source intervals")
    if family == "multiplicative_stride":
        if nfe < 2:
            raise ValueError("multiplicative_stride requires at least 2 steps for a unit first interval")
        ratio = _solve_ratio(1.0, 20.0, nfe)
        schedule_coordinates = _values_from_intervals(
            0.0, 1.0, ratio, nfe, 20.0,
        )
    else:
        middle_count = nfe - 4
        ratio = _solve_ratio(1.0, 16.0, middle_count)
        middle_coordinates = _values_from_intervals(
            2.0, 1.0, ratio, middle_count, 18.0,
        )
        schedule_coordinates = [0.0, 1.0, *middle_coordinates, 19.0, 20.0]
    effective = _sigmas_from_coordinates(source, times, schedule_coordinates)
    return effective, tuple(schedule_coordinates[:-1]), ratio


def _solve_unit_geometric_base(count: int) -> float:
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) * 0.5
        total = sum(middle ** power for power in range(1, count + 1))
        if total < 1.0:
            low = middle
        else:
            high = middle
    return (low + high) * 0.5


def _values_from_intervals(start: float, first: float, ratio: float, count: int,
                           end: float) -> list[float]:
    values = [start]
    current = start
    for index in range(count):
        current += first * ratio ** index
        values.append(current)
    values[-1] = end
    return values


def _source_times(source_sigmas: torch.Tensor) -> list[float]:
    if source_sigmas.ndim != 1 or source_sigmas.numel() < 4:
        raise ValueError("continuous schedules require a one-dimensional sigma schedule")
    if not bool(torch.isfinite(source_sigmas).all().item()):
        raise ValueError("continuous schedules require finite sigma values")
    if bool(((source_sigmas[1:] - source_sigmas[:-1]) >= 0).any().item()):
        raise ValueError("continuous schedules require strictly descending sigma values")
    values = [float(value) for value in source_sigmas.detach().cpu()]
    if values[-1] != 0.0 or values[-2] <= 0.0:
        raise ValueError("continuous schedules require one terminal zero after positive sigmas")
    return [-math.log(value) for value in values[:-1]]


def _sigmas_from_coordinates(source: torch.Tensor, times: list[float],
                             coordinates: list[float]) -> torch.Tensor:
    values = []
    terminal = len(times)
    for coordinate in coordinates:
        if abs(coordinate - terminal) <= 1e-10:
            values.append(float(source[-1]))
            continue
        if coordinate >= terminal - 1.0:
            fraction = coordinate - (terminal - 1.0)
            if fraction < -1e-10 or fraction > 1.0 + 1e-10:
                raise ValueError("continuous schedule coordinate leaves the source range")
            values.append(float(source[-2]) * max(0.0, 1.0 - fraction))
            continue
        left = int(math.floor(coordinate))
        fraction = coordinate - left
        if not 0 <= left < len(times):
            raise ValueError("continuous schedule coordinate enters the terminal-zero interval")
        if fraction <= 1e-10:
            values.append(float(source[left]))
        else:
            sigma_t = times[left] + fraction * (times[left + 1] - times[left])
            values.append(math.exp(-sigma_t))
    return torch.tensor(values, device=source.device, dtype=torch.float64).to(dtype=source.dtype)


def continuous_schedule(source_sigmas, profile: str) -> tuple[torch.Tensor, tuple[float, ...], float]:
    """Build a fixed continuous sigma schedule for characterization."""
    if profile not in CONTINUOUS_PROFILE_NFE:
        raise ValueError(f"unknown continuous schedule profile: {profile}")
    family = profile[:-3]
    return continuous_schedule_family(source_sigmas, family, CONTINUOUS_PROFILE_NFE[profile])


def continuous_schedule_identity(profile: str) -> dict:
    if profile not in CONTINUOUS_PROFILE_NFE:
        raise ValueError(f"unknown continuous schedule profile: {profile}")
    coordinate = {
        "geometric_11": "sigma",
        "geometric_linear_ends_11": "negative_log_sigma",
        "multiplicative_stride_11": "source_log_sigma_logical_coordinate",
        "multiplicative_stride_linear_ends_11": "source_log_sigma_logical_coordinate",
    }[profile]
    interval_rule = {
        "geometric_11": "normalized_reverse_powers_sum_to_sigma_span",
        "geometric_linear_ends_11": "native_0_1_2_geometric_interior_native_18_19",
        "multiplicative_stride_11": "unit_first_multiplicative_stride_sum_to_20",
        "multiplicative_stride_linear_ends_11": "native_0_1_2_multiplicative_interior_native_18_19_20",
    }[profile]
    return {
        "version": CONTINUOUS_SCHEDULE_VERSION,
        "profile": profile,
        "true_nfe": CONTINUOUS_PROFILE_NFE[profile],
        "time_coordinate": coordinate,
        "interval_rule": interval_rule,
    }


__all__ = [
    "CONTINUOUS_PROFILE_NFE",
    "CONTINUOUS_SCHEDULE_FAMILIES",
    "CONTINUOUS_SCHEDULE_VERSION",
    "continuous_schedule_family",
    "continuous_schedule",
    "continuous_schedule_identity",
]
