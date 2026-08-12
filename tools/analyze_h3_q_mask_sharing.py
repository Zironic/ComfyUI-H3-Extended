#!/usr/bin/env python3
"""Aggregate Q-mask sharing sweeps from H3 MoBA3D probe reports.

Examples:

    python custom_nodes/ComfyUI-H3-Extended/tools/analyze_h3_q_mask_sharing.py \
        output/h3_probe/my_run_*/moba3d_summary.json

    python custom_nodes/ComfyUI-H3-Extended/tools/analyze_h3_q_mask_sharing.py \
        output/h3_probe/my_run_20260812-120000 \
        --json output/h3_probe/q_mask_sharing_analysis.json

The reported mean is an equal-weight mean across captured probe records, matching
the way the original execution-mask analysis summarized its 54 records. Each
record's value is already averaged across its heads and evaluated Q tokens.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def _expand_inputs(specs):
    paths = []
    for spec in specs:
        matches = glob.glob(spec)
        if not matches:
            matches = [spec]
        for path in matches:
            if os.path.isdir(path):
                path = os.path.join(path, "moba3d_summary.json")
            if path not in paths:
                paths.append(path)
    return paths


def _budget_key(value):
    return "%.6f" % float(value)


def collect(paths):
    buckets = {}
    sources = []
    kv_tiles = set()
    total_records = 0
    sweep_records = 0

    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        sources.append(os.path.abspath(path))
        top_kv = payload.get("sage_kv_tile")
        if top_kv is not None:
            kv_tiles.add(int(top_kv))

        for record in payload.get("records", ()):
            total_records += 1
            moba = record.get("moba3d") or {}
            record_kv = moba.get("execution_q_tile_density_sweep_kv_tile")
            if record_kv is not None:
                kv_tiles.add(int(record_kv))
            record_had_sweep = False

            for row in moba.get("budgets", ()):
                sweep = row.get("executable_q_tile_density_sweep")
                if not sweep:
                    continue
                record_had_sweep = True
                budget = float(row["budget"])
                bkey = _budget_key(budget)
                budget_bucket = buckets.setdefault(
                    bkey,
                    {"budget": budget, "q_tiles": {}},
                )
                for q_tile_text, stats in sweep.items():
                    q_tile = int(q_tile_text)
                    target = budget_bucket["q_tiles"].setdefault(
                        q_tile,
                        {"means": [], "maxima": []},
                    )
                    target["means"].append(float(stats["mean"]))
                    target["maxima"].append(float(stats["max"]))

            if record_had_sweep:
                sweep_records += 1

    summary = {
        "sources": sources,
        "total_records": total_records,
        "sweep_records": sweep_records,
        "kv_tiles": sorted(kv_tiles),
        "budgets": [],
    }

    for bkey in sorted(buckets, key=lambda key: buckets[key]["budget"]):
        source = buckets[bkey]
        q_rows = []
        q1_mean = None
        if 1 in source["q_tiles"]:
            q1_values = source["q_tiles"][1]["means"]
            if q1_values:
                q1_mean = statistics.fmean(q1_values)

        for q_tile in sorted(source["q_tiles"], reverse=True):
            values = source["q_tiles"][q_tile]
            means = values["means"]
            maxima = values["maxima"]
            mean_density = statistics.fmean(means)
            q_rows.append(
                {
                    "q_tile": q_tile,
                    "records": len(means),
                    "mean_executable_density": mean_density,
                    "min_record_mean_density": min(means),
                    "max_record_mean_density": max(means),
                    "worst_record_max_density": max(maxima),
                    "density_overhead_vs_q1": (
                        mean_density - q1_mean if q1_mean is not None else None
                    ),
                }
            )

        summary["budgets"].append(
            {
                "budget": source["budget"],
                "q_tiles": q_rows,
            }
        )

    return summary


def _pct(value):
    if value is None:
        return "n/a"
    return "%.2f%%" % (100.0 * float(value))


def render(summary):
    lines = []
    kv_tiles = summary.get("kv_tiles") or []
    lines.append("H3 Q-mask sharing sweep")
    lines.append("=======================")
    lines.append("sources: %d" % len(summary.get("sources", ())))
    lines.append(
        "records: %d with sweep / %d total"
        % (summary.get("sweep_records", 0), summary.get("total_records", 0))
    )
    lines.append(
        "fixed KV tile: %s"
        % (", ".join(str(x) for x in kv_tiles) if kv_tiles else "unknown")
    )
    lines.append("")

    for budget in summary.get("budgets", ()):
        lines.append("Logical video budget %s" % _pct(budget["budget"]))
        lines.append("")
        lines.append(
            "| Q sharing | Records | Mean executable density | Worst record mean | Worst record max | Overhead vs Q=1 |"
        )
        lines.append(
            "| ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in budget["q_tiles"]:
            lines.append(
                "| %d | %d | %s | %s | %s | %s |"
                % (
                    row["q_tile"],
                    row["records"],
                    _pct(row["mean_executable_density"]),
                    _pct(row["max_record_mean_density"]),
                    _pct(row["worst_record_max_density"]),
                    _pct(row["density_overhead_vs_q1"]),
                )
            )
        lines.append("")

    if not summary.get("budgets"):
        lines.append(
            "No executable_q_tile_density_sweep records found. Run this branch "
            "with the MoBA 3D probe set to execution_geometry=sage_sparse."
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Aggregate H3 Sparse-Sage Q-mask sharing density sweeps."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="moba3d_summary.json files, directories containing them, or globs",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="optionally write the aggregate data as JSON",
    )
    args = parser.parse_args(argv)

    paths = _expand_inputs(args.inputs)
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        parser.error("missing input: %s" % missing[0])

    summary = collect(paths)
    print(render(summary))

    if args.json_path:
        directory = os.path.dirname(os.path.abspath(args.json_path))
        os.makedirs(directory, exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    return 0 if summary.get("budgets") else 2


if __name__ == "__main__":
    sys.exit(main())
