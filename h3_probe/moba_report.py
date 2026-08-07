"""Human-readable and JSON reports for the probe-only 3D MoBA simulator."""

from __future__ import annotations

import json
import os


def _pct(x):
    return "%5.1f%%" % (100.0 * float(x))


def _budget_key(row):
    return "%.6f" % float(row["budget"])


def summarize(records):
    """Worst-case routing quality for each configured sparsity budget."""
    if not records:
        return {}
    by_budget = {}
    for rec in records:
        for row in rec["moba3d"]["budgets"]:
            by_budget.setdefault(_budget_key(row), []).append(row)

    out = {}
    for key, rows in by_budget.items():
        out[key] = {
            "budget": rows[0]["budget"],
            "video_block_density": rows[0]["video_block_density"],
            "routed_mass_min": min(r["routed_mass_min_head"] for r in rows),
            "routed_mass_mean_min": min(r["routed_mass_mean"] for r in rows),
            "oracle_mass_min": min(r["oracle_mass_min_head"] for r in rows),
            "regret_max": max(r["routing_regret_max_head"] for r in rows),
            "regret_mean_max": max(r["routing_regret_mean"] for r in rows),
            "oracle_overlap_min": min(r["oracle_block_overlap_mean"] for r in rows),
            "effective_token_density_max": max(r["effective_token_density_mean"] for r in rows),
        }
    return out


def render(run, summary):
    lines = [
        "MiniMax H3 3D MoBA-style routing probe - %s" % run.tag,
        "=" * 78,
        "layout:       %s" % (run.layout.describe() if run.layout else "n/a"),
        "layers:       %s of %s" % (sorted(run.layers), run.notes.get("num_layers")),
        "steps:        %s of %s" % (sorted(run.steps), run.notes.get("total_steps")),
        "3D block:     %dx%dx%d (latent-time x patch-height x patch-width)" % (
            run.block_t, run.block_h, run.block_w),
        "budgets:      %s" % ", ".join(_pct(x).strip() for x in run.budgets),
        "query blocks: %d" % len(run.records),
        "",
        "Interpretation: non-video KV tokens are always retained. Only target-video",
        "blocks are routed. 'Routed mass' is exact dense-softmax mass retained by",
        "mean-pooled routing; 'oracle' selects the same number of video blocks using",
        "their true dense mass. This simulates the public H3 description; it does not",
        "claim to reproduce MiniMax's unreleased training-aware implementation.",
        "",
    ]

    if summary:
        lines.extend(["SUMMARY (worst case across captured query blocks)", "-" * 78])
        for key in sorted(summary, key=lambda x: summary[x]["budget"]):
            s = summary[key]
            lines.append(
                "  video blocks %-5s | routed min %s | oracle min %s | max regret %s | "
                "oracle overlap min %s | effective token density <= %s" % (
                    _pct(s["video_block_density"]).strip(),
                    _pct(s["routed_mass_min"]).strip(),
                    _pct(s["oracle_mass_min"]).strip(),
                    _pct(s["regret_max"]).strip(),
                    _pct(s["oracle_overlap_min"]).strip(),
                    _pct(s["effective_token_density_max"]).strip(),
                )
            )
        lines.append("")

    lines.extend(["PER QUERY BLOCK", "-" * 78])
    for rec in run.records:
        frame = rec.get("frame")
        label = "%s query" % rec["kind"] if frame is None else "video query t=%d" % frame
        m = rec["moba3d"]
        lines.append(
            "Layer %d, step %d, %s rows %d-%d | grid=%s blocks=%d | dense non-video=%s" % (
                rec["layer"], rec["step"], label, rec["start"], rec["stop"],
                "x".join(str(x) for x in m["block_grid"]), m["video_blocks"],
                _pct(m["dense_nonvideo_mass_mean"]).strip(),
            )
        )
        for row in m["budgets"]:
            lines.append(
                "  %-5s video blocks (%d/%d): routed mean %s min-head %s | oracle mean %s | "
                "regret mean %s max-head %s | overlap %s | effective tokens %s" % (
                    _pct(row["video_block_density"]).strip(), row["keep_blocks"], row["video_blocks"],
                    _pct(row["routed_mass_mean"]).strip(), _pct(row["routed_mass_min_head"]).strip(),
                    _pct(row["oracle_mass_mean"]).strip(), _pct(row["routing_regret_mean"]).strip(),
                    _pct(row["routing_regret_max_head"]).strip(),
                    _pct(row["oracle_block_overlap_mean"]).strip(),
                    _pct(row["effective_token_density_mean"]).strip(),
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
    with open(os.path.join(run.out_dir, "moba3d_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tag": run.tag,
            "layout": run.layout.as_dict() if run.layout else None,
            "layers": sorted(run.layers),
            "steps": sorted(run.steps),
            "block_shape": [run.block_t, run.block_h, run.block_w],
            "budgets": list(run.budgets),
            "notes": run.notes,
            "summary": summary,
            "records": run.records,
        }, f, indent=2)
    return report_path
