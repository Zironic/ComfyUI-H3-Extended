"""Reproducible arm definitions and result collection for milestones 4 and 7."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .config import SamplerConfig, actual_mask


@dataclass(frozen=True)
class StudyArm:
    label: str
    phase: str
    method: str
    evaluation_profile: str
    expected_true_nfe: int | None
    policy: str = "fixed"
    quality_preset: str = "balanced"
    repairability_profile: str | None = None
    conditioning_mode: str = "default"

    def config(self):
        return SamplerConfig(
            method=self.method,
            evaluation_profile=self.evaluation_profile,
            policy=self.policy,
            diagnostics="full",
            quality_preset=self.quality_preset,
            repairability_profile=self.repairability_profile,
            conditioning_mode=self.conditioning_mode,
        )

    def as_dict(self):
        value = asdict(self)
        value["actual_indices"] = (
            list(actual_mask(self.evaluation_profile)) if self.policy == "fixed" else None
        )
        return value


def fixed_policy_arms(include_vde=False):
    """Return ordered predictor and equal-NFE placement study arms."""
    arms = [
        StudyArm("native_20", "predictor", "native", "native_20", 20),
        StudyArm("hold_conservative_12", "predictor", "hold", "conservative_12", 12),
        StudyArm("linear_conservative_12", "predictor", "linear_velocity", "conservative_12", 12),
        StudyArm("linear_early_aggressive_13", "placement", "linear_velocity", "early_aggressive_13", 13),
        StudyArm("linear_uniform_13", "placement", "linear_velocity", "uniform_13", 13),
        StudyArm("linear_late_aggressive_13", "placement", "linear_velocity", "late_aggressive_13", 13),
        StudyArm("linear_late_cautious_14", "late_tail_pace", "linear_velocity", "late_cautious_14", 14),
        StudyArm("linear_late_aggressive_12", "late_tail_pace", "linear_velocity", "late_aggressive_12", 12),
        StudyArm("linear_late_max_11", "late_tail_pace", "linear_velocity", "late_max_11", 11),
    ]
    if include_vde:
        arms.append(StudyArm("vde_conservative_12", "vde_fixed", "vde", "conservative_12", 12))
    return tuple(arms)


def adaptive_comparison_arms(best_fixed_method, best_fixed_profile,
                             repairability_profile, predictor_method,
                             conditioning_mode, quality_presets=("conservative", "balanced", "aggressive")):
    """Compare a selected fixed arm with profile-gated adaptive presets."""
    arms = [StudyArm(
        f"best_fixed_{best_fixed_method}_{best_fixed_profile}",
        "adaptive_compare", best_fixed_method, best_fixed_profile,
        len(actual_mask(best_fixed_profile)), conditioning_mode=conditioning_mode,
    )]
    arms.extend(StudyArm(
        f"adaptive_{predictor_method}_{preset}",
        "adaptive_compare", predictor_method, "native_20", None,
        policy="adaptive_repair", quality_preset=preset,
        repairability_profile=repairability_profile,
        conditioning_mode=conditioning_mode,
    ) for preset in quality_presets)
    return tuple(arms)


def _validate_result(arm, result):
    if not isinstance(result, dict):
        raise TypeError(f"runner result for {arm.label} must be a dictionary")
    true_nfe = result.get("true_nfe")
    if arm.expected_true_nfe is None:
        if not isinstance(true_nfe, int) or not 1 <= true_nfe <= 20:
            raise ValueError(f"{arm.label} reported invalid adaptive true NFE {true_nfe!r}")
    elif true_nfe != arm.expected_true_nfe:
        raise ValueError(
            f"{arm.label} reported {true_nfe!r} true NFE; expected {arm.expected_true_nfe}"
        )
    for modality in ("video", "audio"):
        if modality not in result or not isinstance(result[modality], dict):
            raise ValueError(f"{arm.label} must report separate {modality} metrics")


def candidate_profiles(records):
    """Return only explicitly quality-approved candidates, ordered by cost."""
    candidates = [record for record in records if record["result"].get("quality_pass") is True]
    return sorted(candidates, key=lambda row: (
        row["result"]["true_nfe"],
        float(row["result"].get("wall_seconds", float("inf"))),
        row["arm"]["label"],
    ))


def run_study_arms(runner, arms, metadata=None):
    records = []
    for arm in arms:
        result = runner(arm)
        _validate_result(arm, result)
        records.append({"arm": arm.as_dict(), "result": result})
    return {
        "schema_version": 1,
        "metadata": dict(metadata or {}),
        "records": records,
        "candidates": candidate_profiles(records),
    }


def run_fixed_policy_study(runner, include_vde=False, metadata=None):
    return run_study_arms(
        runner, fixed_policy_arms(include_vde=include_vde), metadata=metadata
    )


def write_study_result(result, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
