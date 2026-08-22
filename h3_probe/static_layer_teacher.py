"""Exact same-budget static layer allocation over dense-teacher captures."""

from __future__ import annotations

import math


BASELINE_BUDGET = 0.20


def _budget_key(value):
    return "%.6f" % float(value)


def _fixed_metrics(row):
    teacher = row.get("adaptive_teacher")
    if teacher is not None:
        return teacher["sampled_teacher_rows"]["fixed"]
    precision = row.get("router_precision_teacher")
    if precision is not None:
        return precision["arms"]["bf16"]
    return None


def _record_signature(record):
    return (
        int(record.get("step", -1)),
        int(record.get("cond_or_uncond", 0)),
        str(record.get("kind", "unknown")),
        int(record.get("start", -1)),
        int(record.get("stop", -1)),
    )


def _curves(records):
    layers = {}
    for record in records:
        if record.get("kind") != "video":
            continue
        layer = int(record["layer"])
        layer_rows = layers.setdefault(layer, {})
        for row in record["moba3d"].get("budgets", ()):
            metrics = _fixed_metrics(row)
            if metrics is None or "squared_error" not in metrics:
                continue
            key = _budget_key(row["budget"])
            option = layer_rows.setdefault(key, {
                "budget": float(row["budget"]),
                "retained_video_kv_tiles": int(row["direct_tile_keep_video_kv_tiles"]),
                "pure_video_kv_tiles": int(row["direct_tile_video_kv_tiles"]),
                "squared_error": 0.0,
                "teacher_energy": 0.0,
                "captures": 0,
            })
            if option["retained_video_kv_tiles"] != int(row["direct_tile_keep_video_kv_tiles"]):
                raise ValueError("one layer/budget resolved multiple retained K values")
            if option["pure_video_kv_tiles"] != int(row["direct_tile_video_kv_tiles"]):
                raise ValueError("one layer/budget mixed Sparse-Sage geometries")
            option["squared_error"] += float(metrics["squared_error"])
            option["teacher_energy"] += float(metrics["teacher_energy"])
            option["captures"] += 1
    return layers


def _deduplicate_options(options):
    by_k = {}
    for option in options.values():
        retained = int(option["retained_video_kv_tiles"])
        current = by_k.get(retained)
        if current is None or option["squared_error"] < current["squared_error"]:
            by_k[retained] = option
    return [by_k[key] for key in sorted(by_k)]


def _objective(option, name):
    if name == "micro_squared_error":
        return float(option["squared_error"])
    if name == "layer_relative_squared_error":
        return float(option["squared_error"]) / max(float(option["teacher_energy"]), 1.0e-24)
    raise ValueError("unknown static layer teacher objective %r" % name)


def _optimize(curves, target, objective):
    layers = sorted(curves)
    states = {0: (0.0, ())}
    for layer in layers:
        next_states = {}
        for spent, (score, choices) in states.items():
            for option in _deduplicate_options(curves[layer]):
                total = spent + int(option["retained_video_kv_tiles"])
                if total > target:
                    continue
                candidate = (score + _objective(option, objective), choices + (option,))
                current = next_states.get(total)
                if current is None or candidate[0] < current[0]:
                    next_states[total] = candidate
        states = next_states
    if target not in states:
        return None
    return {
        layer: option
        for layer, option in zip(layers, states[target][1])
    }


def _density_groups(selected):
    groups = []
    for layer in sorted(selected):
        budget = float(selected[layer]["budget"])
        if groups and groups[-1]["last_layer"] + 1 == layer and groups[-1]["budget"] == budget:
            groups[-1]["last_layer"] = layer
        else:
            groups.append({
                "first_layer": layer,
                "last_layer": layer,
                "budget": budget,
            })
    return groups


def _evaluate(selected):
    squared_error = sum(float(option["squared_error"]) for option in selected.values())
    teacher_energy = sum(float(option["teacher_energy"]) for option in selected.values())
    relative_by_layer = [
        float(option["squared_error"]) / max(float(option["teacher_energy"]), 1.0e-24)
        for option in selected.values()
    ]
    return {
        "selected_video_kv_tiles": sum(
            int(option["retained_video_kv_tiles"])
            for option in selected.values()
        ),
        "squared_error": squared_error,
        "teacher_energy": teacher_energy,
        "micro_relative_l2": math.sqrt(squared_error / max(teacher_energy, 1.0e-24)),
        "mean_layer_relative_squared_error": sum(relative_by_layer) / len(relative_by_layer),
        "density_by_layer": {
            str(layer): float(option["budget"])
            for layer, option in sorted(selected.items())
        },
        "retained_k_by_layer": {
            str(layer): int(option["retained_video_kv_tiles"])
            for layer, option in sorted(selected.items())
        },
        "density_groups": _density_groups(selected),
    }


def _uniform_selection(curves, baseline_key):
    return {
        layer: options[baseline_key]
        for layer, options in curves.items()
    }


def _schedule_report(curves, uniform, objective):
    target = sum(
        int(option["retained_video_kv_tiles"])
        for option in uniform.values()
    )
    selected = _optimize(curves, target, objective)
    if selected is None:
        return None
    baseline = _evaluate(uniform)
    result = _evaluate(selected)
    result.update({
        "objective": objective,
        "solver": "exact discrete dynamic programming",
        "proven_optimal_for_local_teacher_surrogate": True,
        "same_budget_as_uniform": result["selected_video_kv_tiles"] == target,
        "squared_error_reduction_vs_uniform": (
            (baseline["squared_error"] - result["squared_error"])
            / max(baseline["squared_error"], 1.0e-24)
        ),
        "mean_layer_relative_squared_error_reduction_vs_uniform": (
            (
                baseline["mean_layer_relative_squared_error"]
                - result["mean_layer_relative_squared_error"]
            )
            / max(baseline["mean_layer_relative_squared_error"], 1.0e-24)
        ),
    })
    return result


def _evaluate_schedule(curves, density_by_layer):
    selected = {}
    for layer, options in curves.items():
        key = _budget_key(density_by_layer[str(layer)])
        if key not in options:
            return None
        selected[layer] = options[key]
    return _evaluate(selected)


def _region_holdouts(records, expected_layers, baseline_key, objective):
    signatures = sorted({_record_signature(record) for record in records if record.get("kind") == "video"})
    if len(signatures) != 2:
        return []
    result = []
    for training_signature, evaluation_signature in (
        (signatures[0], signatures[1]),
        (signatures[1], signatures[0]),
    ):
        training = _curves([
            record for record in records
            if _record_signature(record) == training_signature
        ])
        evaluation = _curves([
            record for record in records
            if _record_signature(record) == evaluation_signature
        ])
        if sorted(training) != expected_layers or sorted(evaluation) != expected_layers:
            continue
        if any(baseline_key not in training[layer] or baseline_key not in evaluation[layer] for layer in expected_layers):
            continue
        uniform = _uniform_selection(training, baseline_key)
        schedule = _schedule_report(training, uniform, objective)
        if schedule is None:
            continue
        evaluated = _evaluate_schedule(evaluation, schedule["density_by_layer"])
        baseline_evaluated = _evaluate(_uniform_selection(evaluation, baseline_key))
        if evaluated is None:
            continue
        result.append({
            "training_region": list(training_signature),
            "evaluation_region": list(evaluation_signature),
            "training_squared_error_reduction_vs_uniform": schedule["squared_error_reduction_vs_uniform"],
            "evaluation_squared_error_reduction_vs_uniform": (
                (baseline_evaluated["squared_error"] - evaluated["squared_error"])
                / max(baseline_evaluated["squared_error"], 1.0e-24)
            ),
            "evaluation_micro_relative_l2": evaluated["micro_relative_l2"],
            "evaluation_uniform_micro_relative_l2": baseline_evaluated["micro_relative_l2"],
            "density_by_layer": schedule["density_by_layer"],
        })
    return result


def summarize(records, expected_layers=(), baseline_budget=BASELINE_BUDGET):
    """Build exact static per-layer schedules from existing fixed-route metrics."""
    curves = _curves(records)
    if not curves:
        return {"status": "unavailable", "reason": "no fixed dense-teacher records"}
    layers = sorted(curves)
    expected = sorted(int(layer) for layer in expected_layers) or layers
    baseline_key = _budget_key(baseline_budget)
    missing_layers = sorted(set(expected) - set(layers))
    missing_baseline = [layer for layer in layers if baseline_key not in curves[layer]]
    if missing_baseline:
        return {
            "status": "unavailable",
            "reason": "baseline budget is absent from one or more layers",
            "missing_baseline_layers": missing_baseline,
        }
    uniform = _uniform_selection(curves, baseline_key)
    micro = _schedule_report(curves, uniform, "micro_squared_error")
    balanced = _schedule_report(curves, uniform, "layer_relative_squared_error")
    if micro is None or balanced is None:
        return {
            "status": "unavailable",
            "reason": "candidate layer densities cannot reach the exact uniform budget",
        }
    candidates = sorted({
        (float(option["budget"]), int(option["retained_video_kv_tiles"]))
        for options in curves.values()
        for option in options.values()
    })
    return {
        "status": "partial" if missing_layers else "complete",
        "scope": "local dense-attention output error; downstream layer interactions and runtime are not measured",
        "baseline_budget": float(baseline_budget),
        "layers": layers,
        "expected_layers": expected,
        "missing_layers": missing_layers,
        "capture_count": sum(
            option["captures"]
            for options in curves.values()
            for key, option in options.items()
            if key == baseline_key
        ),
        "candidate_budget_and_k": [
            {"budget": budget, "retained_video_kv_tiles": retained}
            for budget, retained in candidates
        ],
        "uniform": _evaluate(uniform),
        "optimized": {
            "micro_squared_error": micro,
            "layer_relative_squared_error": balanced,
        },
        "two_region_holdout": {
            "description": "internal first/last sampled query-region transfer; not a held-out prompt or trajectory",
            "micro_squared_error": _region_holdouts(
                records, layers, baseline_key, "micro_squared_error"
            ),
            "layer_relative_squared_error": _region_holdouts(
                records, layers, baseline_key, "layer_relative_squared_error"
            ),
        },
    }
