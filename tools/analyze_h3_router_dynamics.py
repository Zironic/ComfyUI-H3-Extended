"""Analyze H3 router temporal reuse and sampled static topology.

Usage from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tools/analyze_h3_router_dynamics.py \
        output/h3_probe/run_a output/h3_probe/run_b

Each input may be a run directory, router_dynamics.json, or router_topology.npz.
The temporal analysis estimates HASTE-style reuse quality at several Q/K drift
thresholds.  With two or more compatible topology archives it also reports how
stable the sampled direct-tile topology is across runs.
"""

from __future__ import annotations

import argparse
import json
import os
from itertools import combinations

import numpy as np


DEFAULT_DRIFT_THRESHOLDS = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05)


def _resolve(path):
    path = os.path.abspath(path)
    if os.path.isdir(path):
        return (
            os.path.join(path, "router_dynamics.json"),
            os.path.join(path, "router_topology.npz"),
        )
    name = os.path.basename(path)
    if name == "router_dynamics.json":
        return path, os.path.join(os.path.dirname(path), "router_topology.npz")
    if name == "router_topology.npz":
        return os.path.join(os.path.dirname(path), "router_dynamics.json"), path
    raise ValueError("expected a run directory, router_dynamics.json, or router_topology.npz")


def _load_runs(paths):
    runs = []
    for path in paths:
        dynamics_path, topology_path = _resolve(path)
        payload = None
        if os.path.exists(dynamics_path):
            with open(dynamics_path, encoding="utf-8") as handle:
                payload = json.load(handle)
        runs.append(
            {
                "root": os.path.dirname(dynamics_path),
                "dynamics_path": dynamics_path,
                "topology_path": topology_path,
                "payload": payload,
            }
        )
    return runs


def _flatten_transitions(runs):
    rows = []
    for run_index, run in enumerate(runs):
        payload = run["payload"] or {}
        for rec in payload.get("records", ()):
            q = rec.get("q_cosine_by_head")
            k = rec.get("k_cosine_by_head")
            reuse = rec.get("exact_route_reuse_fraction_by_head")
            jaccard = rec.get("sampled_route_jaccard_by_head")
            if q is None or k is None or reuse is None or jaccard is None:
                continue
            for head, (qv, kv, rv, jv) in enumerate(zip(q, k, reuse, jaccard)):
                rows.append(
                    {
                        "run": run_index,
                        "step": int(rec["step"]),
                        "layer": int(rec["layer"]),
                        "head": int(head),
                        "q_cosine": float(qv),
                        "k_cosine": float(kv),
                        "exact_reuse": float(rv),
                        "jaccard": float(jv),
                    }
                )
    return rows


def _pearson(x, y):
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def temporal_analysis(rows, thresholds=DEFAULT_DRIFT_THRESHOLDS):
    if not rows:
        return {}
    drift = np.asarray(
        [max(0.0, 1.0 - row["q_cosine"], 1.0 - row["k_cosine"]) for row in rows],
        dtype=np.float64,
    )
    jaccard = np.asarray([row["jaccard"] for row in rows], dtype=np.float64)
    reuse = np.asarray([row["exact_reuse"] for row in rows], dtype=np.float64)
    route_change = 1.0 - jaccard

    threshold_rows = []
    for threshold in thresholds:
        eligible = drift <= float(threshold)
        count = int(eligible.sum())
        threshold_rows.append(
            {
                "drift_threshold": float(threshold),
                "eligible_fraction": float(eligible.mean()),
                "eligible_samples": count,
                "sampled_route_jaccard_mean": (
                    float(jaccard[eligible].mean()) if count else None
                ),
                "exact_route_reuse_fraction_mean": (
                    float(reuse[eligible].mean()) if count else None
                ),
                "bad_reuse_fraction_jaccard_lt_0_95": (
                    float((jaccard[eligible] < 0.95).mean()) if count else None
                ),
                "bad_reuse_fraction_jaccard_lt_0_90": (
                    float((jaccard[eligible] < 0.90).mean()) if count else None
                ),
            }
        )

    by_layer = {}
    for layer in sorted({row["layer"] for row in rows}):
        subset = [row for row in rows if row["layer"] == layer]
        by_layer[str(layer)] = {
            "samples": len(subset),
            "q_cosine_mean": float(np.mean([row["q_cosine"] for row in subset])),
            "k_cosine_mean": float(np.mean([row["k_cosine"] for row in subset])),
            "exact_route_reuse_fraction_mean": float(
                np.mean([row["exact_reuse"] for row in subset])
            ),
            "sampled_route_jaccard_mean": float(
                np.mean([row["jaccard"] for row in subset])
            ),
        }

    return {
        "samples": len(rows),
        "drift_vs_route_change_pearson": _pearson(drift, route_change),
        "q_drift_vs_route_change_pearson": _pearson(
            [1.0 - row["q_cosine"] for row in rows], route_change
        ),
        "k_drift_vs_route_change_pearson": _pearson(
            [1.0 - row["k_cosine"] for row in rows], route_change
        ),
        "exact_route_reuse_fraction_mean": float(reuse.mean()),
        "sampled_route_jaccard_mean": float(jaccard.mean()),
        "thresholds": threshold_rows,
        "by_layer": by_layer,
    }


def _topology_layers(npz):
    layers = []
    for key in npz.files:
        if key.startswith("layer_") and key.endswith("_counts"):
            layers.append(int(key.split("_")[1]))
    return sorted(set(layers))


def _topk_mask(frequency, target):
    target = min(frequency.shape[-1], max(1, int(target)))
    order = np.argpartition(frequency, -target, axis=-1)[..., -target:]
    mask = np.zeros(frequency.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


def _jaccard(a, b):
    intersection = np.logical_and(a, b).sum(axis=-1)
    union = np.logical_or(a, b).sum(axis=-1)
    return intersection / np.maximum(union, 1)


def static_topology_analysis(runs):
    archives = []
    for run in runs:
        path = run["topology_path"]
        if not os.path.exists(path):
            continue
        archive = np.load(path)
        archives.append((path, archive))
    if len(archives) < 2:
        for _path, archive in archives:
            archive.close()
        return {"compatible_runs": len(archives), "status": "need at least two runs"}

    try:
        q_tiles = {int(archive["q_tile"][0]) for _, archive in archives}
        kv_tiles = {int(archive["kv_tile"][0]) for _, archive in archives}
        budgets = {round(float(archive["budget"][0]), 6) for _, archive in archives}
        if len(q_tiles) != 1 or len(kv_tiles) != 1 or len(budgets) != 1:
            return {
                "compatible_runs": 0,
                "status": "router geometry/budget differs across archives",
            }

        common_layers = set(_topology_layers(archives[0][1]))
        for _, archive in archives[1:]:
            common_layers &= set(_topology_layers(archive))

        by_layer = {}
        for layer in sorted(common_layers):
            frequencies = []
            q_samples = None
            pure_kv = None
            valid = True
            for _path, archive in archives:
                prefix = "layer_%02d_" % layer
                q = archive[prefix + "q_tiles"]
                if q_samples is None:
                    q_samples = q
                elif not np.array_equal(q_samples, q):
                    valid = False
                    break
                observations = max(1, int(archive[prefix + "observations"][0]))
                this_pure_kv = int(archive[prefix + "pure_kv"][0])
                if pure_kv is None:
                    pure_kv = this_pure_kv
                elif pure_kv != this_pure_kv:
                    valid = False
                    break
                frequencies.append(archive[prefix + "counts"].astype(np.float32) / observations)
            if not valid:
                continue

            budget = next(iter(budgets))
            target = max(1, int(np.ceil(budget * pure_kv)))
            masks = [_topk_mask(freq, target) for freq in frequencies]
            pair_values = []
            for left, right in combinations(masks, 2):
                pair_values.append(_jaccard(left, right))
            joined = np.concatenate([value.reshape(-1) for value in pair_values])
            by_layer[str(layer)] = {
                "pairs": len(pair_values),
                "heads": int(masks[0].shape[0]),
                "sampled_q_tiles": int(masks[0].shape[1]),
                "static_topk_jaccard_mean": float(joined.mean()),
                "static_topk_jaccard_min": float(joined.min()),
                "static_topk_jaccard_p10": float(np.quantile(joined, 0.10)),
            }

        return {
            "compatible_runs": len(archives),
            "q_tile": next(iter(q_tiles)),
            "kv_tile": next(iter(kv_tiles)),
            "budget": next(iter(budgets)),
            "by_layer": by_layer,
        }
    finally:
        for _path, archive in archives:
            archive.close()


def analyze(paths):
    runs = _load_runs(paths)
    rows = _flatten_transitions(runs)
    return {
        "inputs": [run["root"] for run in runs],
        "temporal": temporal_analysis(rows),
        "static_topology": static_topology_analysis(runs),
    }


def _pct(value):
    return "n/a" if value is None else "%6.2f%%" % (100.0 * float(value))


def render(result):
    lines = ["H3 sparse-router dynamics analysis", "=" * 88]
    temporal = result.get("temporal") or {}
    if temporal:
        lines.extend(
            [
                "Temporal mask reuse",
                "-" * 88,
                "samples: %d" % temporal["samples"],
                "mean exact row reuse: %s" % _pct(temporal["exact_route_reuse_fraction_mean"]),
                "mean sampled Jaccard: %s" % _pct(temporal["sampled_route_jaccard_mean"]),
                "drift vs route-change Pearson: %s" % (
                    "n/a" if temporal["drift_vs_route_change_pearson"] is None
                    else "%+.4f" % temporal["drift_vs_route_change_pearson"]
                ),
                "",
                "  drift <=   eligible    Jaccard   exact reuse   bad<.95   bad<.90",
            ]
        )
        for row in temporal["thresholds"]:
            lines.append(
                "  %-9.4g %-10s %-9s %-12s %-9s %-9s" % (
                    row["drift_threshold"],
                    _pct(row["eligible_fraction"]).strip(),
                    _pct(row["sampled_route_jaccard_mean"]).strip(),
                    _pct(row["exact_route_reuse_fraction_mean"]).strip(),
                    _pct(row["bad_reuse_fraction_jaccard_lt_0_95"]).strip(),
                    _pct(row["bad_reuse_fraction_jaccard_lt_0_90"]).strip(),
                )
            )
        lines.append("")

    static = result.get("static_topology") or {}
    lines.extend(["Static topology across runs", "-" * 88])
    if static.get("by_layer"):
        lines.append(
            "compatible runs=%d q=%d kv=%d budget=%s"
            % (
                static["compatible_runs"],
                static["q_tile"],
                static["kv_tile"],
                _pct(static["budget"]).strip(),
            )
        )
        for layer, row in sorted(static["by_layer"].items(), key=lambda item: int(item[0])):
            lines.append(
                "  L%-2s static top-k Jaccard mean %s p10 %s min %s"
                % (
                    layer,
                    _pct(row["static_topk_jaccard_mean"]).strip(),
                    _pct(row["static_topk_jaccard_p10"]).strip(),
                    _pct(row["static_topk_jaccard_min"]).strip(),
                )
            )
    else:
        lines.append(static.get("status", "no compatible topology data"))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="run directories or router artifact paths")
    parser.add_argument("--json", dest="json_path", default=None)
    args = parser.parse_args()
    result = analyze(args.paths)
    text = render(result)
    print(text)
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
