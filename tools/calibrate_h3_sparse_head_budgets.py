"""Calibrate H3 direct-tile sparsity budgets per layer/head from probe snapshots.

The MoBA3D probe on the q-mask-sharing experiment branch records exact direct
128x64 sparse-output error for every sampled budget and head.  This tool pools
those curves and redistributes one global average density budget across the
observed (layer, head) pairs using greedy marginal error reduction.

Usage:

    python custom_nodes/ComfyUI-H3-Extended/tools/calibrate_h3_sparse_head_budgets.py \
        output/h3_probe/run/moba3d_summary.json --target 0.50

The result is calibration evidence, not a production policy: quality still needs
video-level validation on held-out generations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np


def _resolve(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        path = os.path.join(path, "moba3d_summary.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def _load(paths):
    payloads = []
    for path in paths:
        resolved = _resolve(path)
        with open(resolved, encoding="utf-8") as handle:
            payloads.append((resolved, json.load(handle)))
    return payloads


def collect_curves(payloads, aggregate="mean"):
    """Return observed direct-tile error curves keyed by (layer, head)."""
    raw = defaultdict(lambda: defaultdict(list))
    geometry = set()
    for _path, payload in payloads:
        for rec in payload.get("records", ()):
            layer = int(rec["layer"])
            for row in rec.get("moba3d", {}).get("budgets", ()):
                errors = row.get("direct_tile_head_rel_l2")
                if errors is None:
                    continue
                budget = float(row["direct_tile_video_density"])
                geometry.add(
                    (
                        int(row.get("direct_tile_q_tile", 128)),
                        int(row.get("direct_tile_kv_tile", 64)),
                    )
                )
                for head, error in enumerate(errors):
                    raw[(layer, int(head))][budget].append(float(error))
    if not raw:
        raise ValueError(
            "no direct_tile_head_rel_l2 data found; run the improved sage_sparse MoBA probe first"
        )
    if len(geometry) != 1:
        raise ValueError("probe inputs use different direct-tile geometries: %s" % sorted(geometry))

    curves = {}
    for key, by_budget in raw.items():
        curve = []
        for budget, values in sorted(by_budget.items()):
            if aggregate == "max":
                error = max(values)
            else:
                error = float(np.mean(values))
            curve.append(
                {
                    "budget": float(budget),
                    "error": float(error),
                    "samples": len(values),
                    "error_max": float(max(values)),
                    "error_mean": float(np.mean(values)),
                }
            )
        curves[key] = curve
    return curves, next(iter(geometry))


def _common_levels(curves):
    levels = None
    for curve in curves.values():
        current = {round(row["budget"], 9) for row in curve}
        levels = current if levels is None else levels & current
    levels = sorted(levels or ())
    if len(levels) < 2:
        raise ValueError("need at least two common calibrated budgets per layer/head")
    return levels


def _curve_map(curve):
    return {round(row["budget"], 9): row for row in curve}


def calibrate(curves, target):
    """Greedily spend density where the next increment reduces error most."""
    levels = _common_levels(curves)
    target = float(target)
    if target > 1.0:
        target /= 100.0
    if not levels[0] <= target <= levels[-1]:
        raise ValueError(
            "target %.4f lies outside calibrated range [%.4f, %.4f]"
            % (target, levels[0], levels[-1])
        )

    keys = sorted(curves)
    maps = {key: _curve_map(curves[key]) for key in keys}
    state = {key: 0 for key in keys}
    total_target = target * len(keys)
    total_density = levels[0] * len(keys)

    def next_candidate(key):
        index = state[key]
        if index + 1 >= len(levels):
            return None
        left = maps[key][levels[index]]
        right = maps[key][levels[index + 1]]
        delta_density = levels[index + 1] - levels[index]
        delta_error = left["error"] - right["error"]
        # Highest positive error reduction per density wins.  Negative values
        # remain valid when noisy measurements force us to spend more budget.
        score = delta_error / max(delta_density, 1e-12)
        return score, delta_density, delta_error

    while total_density + 1e-12 < total_target:
        candidates = []
        for key in keys:
            candidate = next_candidate(key)
            if candidate is not None:
                candidates.append((candidate[0], key, candidate[1], candidate[2]))
        if not candidates:
            break
        candidates.sort(key=lambda item: (item[0], -item[2]), reverse=True)
        _score, key, delta_density, _delta_error = candidates[0]
        state[key] += 1
        total_density += delta_density

    allocation = {}
    errors = []
    for key in keys:
        level = levels[state[key]]
        row = maps[key][level]
        layer, head = key
        allocation.setdefault(str(layer), {})[str(head)] = {
            "budget": float(level),
            "estimated_rel_l2": float(row["error"]),
            "samples": int(row["samples"]),
        }
        errors.append(float(row["error"]))

    achieved = sum(
        allocation[str(layer)][str(head)]["budget"] for layer, head in keys
    ) / len(keys)
    return {
        "target_average_budget": float(target),
        "achieved_average_budget": float(achieved),
        "calibrated_budget_levels": [float(x) for x in levels],
        "layer_heads": len(keys),
        "estimated_rel_l2_mean": float(np.mean(errors)),
        "estimated_rel_l2_max": float(np.max(errors)),
        "allocation": allocation,
    }


def uniform_baseline(curves, target):
    levels = _common_levels(curves)
    chosen = min(levels, key=lambda value: abs(value - target))
    errors = []
    for curve in curves.values():
        row = _curve_map(curve)[chosen]
        errors.append(float(row["error"]))
    return {
        "budget": float(chosen),
        "estimated_rel_l2_mean": float(np.mean(errors)),
        "estimated_rel_l2_max": float(np.max(errors)),
    }


def analyze(paths, target=0.5, aggregate="mean"):
    payloads = _load(paths)
    curves, geometry = collect_curves(payloads, aggregate=aggregate)
    calibrated = calibrate(curves, target)
    baseline = uniform_baseline(curves, float(target) if float(target) <= 1 else float(target) / 100)
    calibrated.update(
        {
            "inputs": [path for path, _payload in payloads],
            "aggregation": aggregate,
            "q_tile": int(geometry[0]),
            "kv_tile": int(geometry[1]),
            "uniform_baseline": baseline,
            "estimated_mean_error_reduction_vs_uniform": float(
                baseline["estimated_rel_l2_mean"] - calibrated["estimated_rel_l2_mean"]
            ),
            "estimated_max_error_reduction_vs_uniform": float(
                baseline["estimated_rel_l2_max"] - calibrated["estimated_rel_l2_max"]
            ),
        }
    )
    return calibrated


def _pct(value):
    return "%6.2f%%" % (100.0 * float(value))


def render(result):
    baseline = result["uniform_baseline"]
    lines = [
        "H3 layer/head sparse-budget calibration",
        "=" * 88,
        "geometry: %dQ x %dKV" % (result["q_tile"], result["kv_tile"]),
        "aggregate: %s" % result["aggregation"],
        "layer/head pairs: %d" % result["layer_heads"],
        "target average density: %s" % _pct(result["target_average_budget"]),
        "achieved average density: %s" % _pct(result["achieved_average_budget"]),
        "uniform comparison density: %s" % _pct(baseline["budget"]),
        "",
        "estimated mean rel-L2: calibrated %s | uniform %s | delta %s"
        % (
            _pct(result["estimated_rel_l2_mean"]),
            _pct(baseline["estimated_rel_l2_mean"]),
            _pct(result["estimated_mean_error_reduction_vs_uniform"]),
        ),
        "estimated max  rel-L2: calibrated %s | uniform %s | delta %s"
        % (
            _pct(result["estimated_rel_l2_max"]),
            _pct(baseline["estimated_rel_l2_max"]),
            _pct(result["estimated_max_error_reduction_vs_uniform"]),
        ),
        "",
        "Allocation",
        "-" * 88,
    ]
    for layer in sorted(result["allocation"], key=int):
        entries = result["allocation"][layer]
        budgets = [entries[head]["budget"] for head in sorted(entries, key=int)]
        lines.append(
            "  L%-2s mean %s min %s max %s"
            % (
                layer,
                _pct(np.mean(budgets)),
                _pct(min(budgets)),
                _pct(max(budgets)),
            )
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="moba3d_summary.json files or run directories")
    parser.add_argument("--target", type=float, default=0.50, help="target mean density, fraction or percent")
    parser.add_argument("--aggregate", choices=("mean", "max"), default="mean")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()
    result = analyze(args.paths, target=args.target, aggregate=args.aggregate)
    print(render(result))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
