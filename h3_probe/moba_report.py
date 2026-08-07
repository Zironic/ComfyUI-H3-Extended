"""Human-readable and JSON reports for the probe-only 3D MoBA simulator."""

from __future__ import annotations

import json
import os


def _pct(x):
    return "%5.1f%%" % (100.0 * float(x))


def _budget_key(row):
    return "%.6f" % float(row["budget"])


def _head_maxima(rows, key="head_rel_l2"):
    width = max((len(r.get(key, ())) for r in rows), default=0)
    maxima = [0.0] * width
    for row in rows:
        for i, value in enumerate(row.get(key, ())):
            maxima[i] = max(maxima[i], float(value))
    return maxima


def _worst_head_list(values, limit=5):
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    return [
        {"head": int(i), "rel_l2": float(values[i])}
        for i in order[:limit]
    ]


def _threshold_list(values, threshold):
    return [i for i, value in enumerate(values) if value > threshold]


def summarize(records):
    """Worst-case routing and sparse-output quality, overall and by layer."""
    if not records:
        return {}

    by_budget = {}
    by_layer_budget = {}
    for rec in records:
        layer = int(rec["layer"])
        for row in rec["moba3d"]["budgets"]:
            key = _budget_key(row)
            by_budget.setdefault(key, []).append(row)
            by_layer_budget.setdefault((layer, key), []).append(row)

    overall = {}
    for key, rows in by_budget.items():
        overall[key] = {
            "budget": rows[0]["budget"],
            "video_block_density": rows[0]["video_block_density"],
            "routed_mass_min": min(r["routed_mass_min"] for r in rows),
            "routed_mass_mean_min": min(r["routed_mass_mean"] for r in rows),
            "oracle_mass_min": min(r["oracle_mass_min"] for r in rows),
            "regret_max": max(r["routing_regret_max"] for r in rows),
            "regret_mean_max": max(r["routing_regret_mean"] for r in rows),
            "oracle_overlap_min": min(r["oracle_block_overlap_min"] for r in rows),
            "effective_token_density_max": max(
                r["effective_token_density_max"] for r in rows
            ),
            "sparse_output_rel_l2_mean_head_max": max(
                r["sparse_output_rel_l2_mean_head"] for r in rows
            ),
            "sparse_output_rel_l2_max_head": max(
                r["sparse_output_rel_l2_max_head"] for r in rows
            ),
            "oracle_output_rel_l2_max_head": max(
                r["oracle_output_rel_l2_max_head"] for r in rows
            ),
        }

    by_layer = {}
    for (layer, key), rows in by_layer_budget.items():
        maxima = _head_maxima(rows, "head_rel_l2")
        oracle_maxima = _head_maxima(rows, "oracle_head_rel_l2")
        by_layer.setdefault(str(layer), {})[key] = {
            "budget": rows[0]["budget"],
            "sparse_output_rel_l2_mean_head_max": max(
                r["sparse_output_rel_l2_mean_head"] for r in rows
            ),
            "sparse_output_rel_l2_max_head": max(maxima, default=0.0),
            "oracle_output_rel_l2_max_head": max(oracle_maxima, default=0.0),
            "heads_rel_l2_gt_1pct": _threshold_list(maxima, 0.01),
            "heads_rel_l2_gt_2pct": _threshold_list(maxima, 0.02),
            "heads_rel_l2_gt_5pct": _threshold_list(maxima, 0.05),
            "worst_heads": _worst_head_list(maxima),
            "oracle_worst_heads": _worst_head_list(oracle_maxima),
        }

    return {"overall": overall, "by_layer": by_layer}


def render(run, summary):
    lines = [
        "MiniMax H3 3D MoBA-style routing probe - %s" % run.tag,
        "=" * 92,
        "layout:       %s" % (run.layout.describe() if run.layout else "n/a"),
        "layers:       %s of %s" % (sorted(run.layers), run.notes.get("num_layers")),
        "steps:        %s of %s" % (sorted(run.steps), run.notes.get("total_steps")),
        "3D block:     %dx%dx%d (latent-time x patch-height x patch-width)" % (
            run.block_t,
            run.block_h,
            run.block_w,
        ),
        "budgets:      %s" % ", ".join(_pct(x).strip() for x in run.budgets),
        "query blocks: %d" % len(run.records),
        "routing:      per query token and per head",
        "",
        "Interpretation: non-video KV tokens are always retained. Each query token",
        "independently selects target-video 3D blocks using dot products against",
        "mean-pooled block keys. 'Routed mass' is the original dense-softmax mass",
        "inside that mask. 'Sparse output error' is more important: it compares the",
        "dense attention output with the exact masked-and-renormalized output using V.",
        "The oracle uses the same per-query block budget but selects blocks by their",
        "true dense attention mass. This remains a probe of MiniMax's public description,",
        "not a reproduction of their unreleased train-aware implementation.",
        "",
    ]

    overall = summary.get("overall", {}) if summary else {}
    if overall:
        lines.extend([
            "SUMMARY (worst case across captured query blocks)",
            "-" * 92,
        ])
        for key in sorted(overall, key=lambda x: overall[x]["budget"]):
            s = overall[key]
            lines.append(
                "  video blocks %-5s | effective KV <= %-5s | "
                "sparse output rel-L2: record mean-head <= %-6s worst head %-6s | "
                "oracle worst %-6s" % (
                    _pct(s["video_block_density"]).strip(),
                    _pct(s["effective_token_density_max"]).strip(),
                    _pct(s["sparse_output_rel_l2_mean_head_max"]).strip(),
                    _pct(s["sparse_output_rel_l2_max_head"]).strip(),
                    _pct(s["oracle_output_rel_l2_max_head"]).strip(),
                )
            )
            lines.append(
                "                    routed mass min %-6s | oracle mass min %-6s | "
                "max routing regret %-6s | oracle overlap min %-6s" % (
                    _pct(s["routed_mass_min"]).strip(),
                    _pct(s["oracle_mass_min"]).strip(),
                    _pct(s["regret_max"]).strip(),
                    _pct(s["oracle_overlap_min"]).strip(),
                )
            )
        lines.append("")

    by_layer = summary.get("by_layer", {}) if summary else {}
    if by_layer:
        lines.extend([
            "LAYER / HEAD DIAGNOSTICS",
            "-" * 92,
        ])
        for layer in sorted(by_layer, key=int):
            lines.append("  Layer %s" % layer)
            layer_rows = by_layer[layer]
            for key in sorted(layer_rows, key=lambda x: layer_rows[x]["budget"]):
                s = layer_rows[key]
                worst = ", ".join(
                    "h%d=%s" % (item["head"], _pct(item["rel_l2"]).strip())
                    for item in s["worst_heads"][:3]
                )
                lines.append(
                    "    %-5s blocks: worst rel-L2 %-6s | oracle %-6s | "
                    "heads >1%%/%s >2%%/%s >5%%/%s | %s" % (
                        _pct(s["budget"]).strip(),
                        _pct(s["sparse_output_rel_l2_max_head"]).strip(),
                        _pct(s["oracle_output_rel_l2_max_head"]).strip(),
                        len(s["heads_rel_l2_gt_1pct"]),
                        len(s["heads_rel_l2_gt_2pct"]),
                        len(s["heads_rel_l2_gt_5pct"]),
                        worst or "n/a",
                    )
                )
        lines.append("")

    lines.extend(["PER QUERY BLOCK", "-" * 92])
    for rec in run.records:
        frame = rec.get("frame")
        label = "%s query" % rec["kind"] if frame is None else "video query t=%d" % frame
        m = rec["moba3d"]
        lines.append(
            "Layer %d, step %d, %s rows %d-%d | grid=%s blocks=%d | "
            "dense non-video=%s" % (
                rec["layer"],
                rec["step"],
                label,
                rec["start"],
                rec["stop"],
                "x".join(str(x) for x in m["block_grid"]),
                m["video_blocks"],
                _pct(m["dense_nonvideo_mass_mean"]).strip(),
            )
        )
        for row in m["budgets"]:
            worst = ", ".join(
                "h%d=%s" % (item["head"], _pct(item["rel_l2"]).strip())
                for item in row["worst_heads"][:3]
            )
            lines.append(
                "  %-5s video blocks (%d/%d): mass mean %s min %s | "
                "oracle %s | regret mean %s max %s | overlap %s | effective KV %s" % (
                    _pct(row["video_block_density"]).strip(),
                    row["keep_blocks"],
                    row["video_blocks"],
                    _pct(row["routed_mass_mean"]).strip(),
                    _pct(row["routed_mass_min"]).strip(),
                    _pct(row["oracle_mass_mean"]).strip(),
                    _pct(row["routing_regret_mean"]).strip(),
                    _pct(row["routing_regret_max"]).strip(),
                    _pct(row["oracle_block_overlap_mean"]).strip(),
                    _pct(row["effective_token_density_mean"]).strip(),
                )
            )
            lines.append(
                "        sparse output rel-L2 mean-head %s max-head %s | "
                "oracle max-head %s | heads >1%%/%d >2%%/%d >5%%/%d | %s" % (
                    _pct(row["sparse_output_rel_l2_mean_head"]).strip(),
                    _pct(row["sparse_output_rel_l2_max_head"]).strip(),
                    _pct(row["oracle_output_rel_l2_max_head"]).strip(),
                    len(row["heads_rel_l2_gt_1pct"]),
                    len(row["heads_rel_l2_gt_2pct"]),
                    len(row["heads_rel_l2_gt_5pct"]),
                    worst or "n/a",
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_run(run):
    if not run.records:
        return None
    os.makedirs(run.out_dir, exist_ok=True)
    summary = summarize(run.records)
    report_path = os.path.join(run.out_dir, "moba3d_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(run, summary))
    with open(
        os.path.join(run.out_dir, "moba3d_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "tag": run.tag,
                "layout": run.layout.as_dict() if run.layout else None,
                "layers": sorted(run.layers),
                "steps": sorted(run.steps),
                "block_shape": [run.block_t, run.block_h, run.block_w],
                "budgets": list(run.budgets),
                "notes": run.notes,
                "summary": summary,
                "records": run.records,
            },
            f,
            indent=2,
        )
    return report_path
