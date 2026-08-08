"""Pure causal previous-activity to current-update evaluation."""

import torch
import torch.nn.functional as F

from .ops import active_fraction, captured_energy_fraction

RUNTIME_THRESHOLDS = (0.01, 0.02, 0.05)
PROFILES = {
    "exact": (1, 1, 1),
    "spatial_1": (1, 3, 3),
    "spatial_2": (1, 5, 5),
    "temporal_1": (3, 1, 1),
    "spatiotemporal_1": (3, 3, 3),
}


def _key(value):
    return "%g%%" % (100.0 * float(value))


def _profile_mask(active, kernel):
    if kernel == (1, 1, 1):
        return active.clone()
    t, h, w = active.shape
    kt, kh, kw = kernel
    x = active.float()[None, None]
    x = F.max_pool3d(x, kernel_size=kernel, stride=1,
                     padding=(kt // 2, kh // 2, kw // 2))
    return x.reshape(t, h, w) > 0.5


def _energy_metrics(predicted, current, energy, patch_base_ss, total_values=None):
    energy = energy.float() if energy is not None else torch.zeros_like(current, dtype=torch.float32)
    current_energy = float(energy.sum().item())
    captured = float(energy[predicted].sum().item())
    missed = current_energy - captured
    captured_fraction = captured_energy_fraction(energy, predicted)
    delta_values = int(total_values if total_values is not None else energy.numel())
    base_total = float(patch_base_ss.float().sum().item()) if patch_base_ss is not None else 0.0
    return {
        "current_update_energy": current_energy,
        "captured_energy_fraction": captured_fraction,
        "missed_energy_fraction": 1.0 - captured_fraction,
        "missed_energy_sum": missed,
        "freeze_surrogate_rms": (max(missed, 0.0) / max(delta_values, 1)) ** 0.5,
        "freeze_surrogate_relative_l2": (max(missed, 0.0) / max(base_total, 1e-12)) ** 0.5,
    }


def evaluate_predictability(previous_activity, current_activity, patch_delta_ss=None,
                            patch_base_ss=None, thresholds=RUNTIME_THRESHOLDS,
                            total_values=None):
    """Evaluate all runtime threshold/profile masks for one causal transition.

    Maps are frame-major ``[T,H,W]`` and are compared with strict ``>``
    threshold semantics. The function is pure and does not retain state.
    """
    previous_activity = previous_activity.float()
    current_activity = current_activity.float()
    if previous_activity.shape != current_activity.shape or previous_activity.ndim != 3:
        raise ValueError("activity maps must both be [T,H,W] with equal shape")
    if patch_delta_ss is None:
        patch_delta_ss = current_activity
    if patch_base_ss is None:
        patch_base_ss = torch.zeros_like(current_activity)
    if patch_delta_ss.shape != current_activity.shape or patch_base_ss.shape != current_activity.shape:
        raise ValueError("energy maps must match activity map shape")

    rows = []
    for threshold in thresholds:
        current = current_activity > float(threshold)
        previous = previous_activity > float(threshold)
        for profile, kernel in PROFILES.items():
            predicted = _profile_mask(previous, kernel)
            tp = int((predicted & current).sum().item())
            current_count = int(current.sum().item())
            fn = int((~predicted & current).sum().item())
            row = {
                "threshold": float(threshold),
                "threshold_label": _key(threshold),
                "profile": profile,
                "kernel": list(kernel),
                "predicted_active_fraction": active_fraction(predicted),
                "previous_active_fraction": active_fraction(previous),
                "current_active_fraction": active_fraction(current),
                "next_active_recall": 1.0 if current_count == 0 else tp / current_count,
                "false_freeze_rate": fn / max(current.numel(), 1),
            }
            row.update(_energy_metrics(predicted, current, patch_delta_ss, patch_base_ss,
                                       total_values))
            rows.append(row)

    transitions = {}
    all_tokens = previous_activity.numel()
    for x in thresholds:
        xmask = previous_activity <= float(x)
        denom = int(xmask.sum().item())
        values = {}
        for y in thresholds:
            count = int((xmask & (current_activity > float(y))).sum().item())
            values[_key(y)] = {
                "count": count,
                "denominator": denom,
                "probability": None if denom == 0 else count / denom,
            }
        transitions[_key(x)] = values
    return {"thresholds": list(thresholds), "profiles": list(PROFILES),
            "rows": rows, "transition_probabilities": transitions,
            "token_count": int(all_tokens)}
