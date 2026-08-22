"""Human-readable and JSON reports for the probe-only 3D MoBA simulator."""

from __future__ import annotations

import json
import os

import numpy as np

from . import latent_dynamics, static_layer_teacher


def _pct(x):
    if x is None:
        return "n/a"
    return "%5.1f%%" % (100.0 * float(x))


def _corr(x):
    if x is None:
        return "n/a"
    return "%+.3f" % float(x)


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
            "execution_geometry": rows[0].get("execution_geometry", "logical"),
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
        if overall[key]["execution_geometry"] == "sage_sparse":
            overall[key].update(
                {
                    "executable_effective_token_density_max": max(
                        r["executable_effective_token_density_max"] for r in rows
                    ),
                    "executable_sparse_output_rel_l2_mean_head_max": max(
                        r["executable_sparse_output_rel_l2_mean_head"] for r in rows
                    ),
                    "executable_sparse_output_rel_l2_max_head": max(
                        r["executable_sparse_output_rel_l2_max_head"] for r in rows
                    ),
                }
            )

    by_layer = {}
    for (layer, key), rows in by_layer_budget.items():
        maxima = _head_maxima(rows, "head_rel_l2")
        oracle_maxima = _head_maxima(rows, "oracle_head_rel_l2")
        by_layer.setdefault(str(layer), {})[key] = {
            "budget": rows[0]["budget"],
            "execution_geometry": rows[0].get("execution_geometry", "logical"),
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
        if by_layer[str(layer)][key]["execution_geometry"] == "sage_sparse":
            executable_maxima = _head_maxima(rows, "executable_head_rel_l2")
            by_layer[str(layer)][key].update(
                {
                    "executable_effective_token_density_max": max(
                        r["executable_effective_token_density_max"] for r in rows
                    ),
                    "executable_sparse_output_rel_l2_max_head": max(
                        executable_maxima, default=0.0
                    ),
                    "executable_heads_rel_l2_gt_1pct": _threshold_list(executable_maxima, 0.01),
                    "executable_heads_rel_l2_gt_2pct": _threshold_list(executable_maxima, 0.02),
                    "executable_heads_rel_l2_gt_5pct": _threshold_list(executable_maxima, 0.05),
                    "executable_worst_heads": _worst_head_list(executable_maxima),
                }
            )

    return {"overall": overall, "by_layer": by_layer}


def _dynamics_region_map(run):
    out = {}
    for dyn in getattr(run, "latent_dynamics", ()):
        step = int(dyn["step"])
        for region in dyn.get("query_regions", ()):
            out[(step, int(region["start"]), int(region["stop"]))] = region
    return out


def _render_dynamics(run, dynamics_summary):
    dynamics = getattr(run, "latent_dynamics", ())
    if not dynamics:
        return []

    lines = [
        "LATENT DYNAMICS (sampler callback, target video)",
        "-" * 92,
        "  Metrics compare each callback with the immediately preceding callback.",
        "  sample = sampler x trajectory; prediction = denoised x0 trajectory.",
        "  Stable-patch fractions use H3's 1x2x2 DiT patch granularity.",
        "  explicit anchor latent frames: %s" % (
            getattr(run, "anchor_frames", []) or "none"
        ),
    ]
    if not getattr(run, "capture_attention", True):
        lines.append("  attention: OFF (routing capture and interpretation disabled)")
    by_step = (dynamics_summary or {}).get("by_step", {})
    for dyn in dynamics:
        step = int(dyn["step"])
        sample = (dyn.get("global", {}).get("sample") or {})
        prediction = (dyn.get("global", {}).get("prediction") or {})
        sample_stable = sample.get("stable_patch_fraction", {})
        pred_stable = prediction.get("stable_patch_fraction", {})
        summary = by_step.get(str(step), {})
        lines.append(
            "  step %-3d | sample Δ %-6s stable<=1%% %-6s <=2%% %-6s <=5%% %-6s | "
            "x0 Δ %-6s stable<=1%% %-6s <=2%% %-6s <=5%% %-6s" % (
                step,
                _pct(sample.get("update_rel_l2")).strip(),
                _pct(sample_stable.get("1%")).strip(),
                _pct(sample_stable.get("2%")).strip(),
                _pct(sample_stable.get("5%")).strip(),
                _pct(prediction.get("update_rel_l2")).strip(),
                _pct(pred_stable.get("1%")).strip(),
                _pct(pred_stable.get("2%")).strip(),
                _pct(pred_stable.get("5%")).strip(),
            )
        )
        if getattr(run, "anchor_frames", []):
            lines.append(
                "           anchor distance correlation: sample Δ %s | x0 Δ %s" % (
                    _corr(summary.get("anchor_distance_vs_sample_update_pearson")),
                    _corr(summary.get("anchor_distance_vs_prediction_update_pearson")),
                )
            )
        pred = dyn.get("predictability") or {}
        rows = pred.get("rows", [])
        if rows:
            lines.append("           ACTIVE-SET PREDICTABILITY (x0) from step %s to %s" %
                         (pred.get("from_step"), pred.get("to_step")))
            for threshold in (0.02, 0.05):
                selected = [r for r in rows if r["threshold"] == threshold and
                            r["profile"] in (("exact", "spatial_1") if threshold == 0.02 else
                                              ("exact", "spatial_1", "spatiotemporal_1"))]
                for row in selected:
                    lines.append("             %s %-16s coverage %5s | energy %5s | miss %5s | freeze rel %5s" %
                                 (row["threshold_label"], row["profile"],
                                  _pct(row["predicted_active_fraction"]).strip(),
                                  _pct(row["captured_energy_fraction"]).strip(),
                                  _pct(row["missed_energy_fraction"]).strip(),
                                  _pct(row["freeze_surrogate_relative_l2"]).strip()))

    correlations = (dynamics_summary or {}).get("attention_correlations", {})
    if correlations:
        lines.extend([
            "",
            "  MATCHED UPDATE / SPARSE-ERROR CORRELATIONS (Pearson)",
            "  Positive means less-converged regions tended to be more sparse-sensitive.",
        ])
        for layer in sorted(correlations, key=int):
            for key in sorted(
                correlations[layer],
                key=lambda k: correlations[layer][k]["budget"],
            ):
                row = correlations[layer][key]
                lines.append(
                    "    L%-2s %-5s blocks n=%-2d | x Δ/error %s | x0 Δ/error %s | "
                    "x0 Δ/oracle %s | anchor dist/error %s" % (
                        layer,
                        _pct(row["budget"]).strip(),
                        row["samples"],
                        _corr(row.get("sample_update_vs_sparse_error_pearson")),
                        _corr(row.get("prediction_update_vs_sparse_error_pearson")),
                        _corr(row.get("prediction_update_vs_oracle_error_pearson")),
                        _corr(row.get("anchor_distance_vs_sparse_error_pearson")),
                    )
                )
    lines.append("")
    return lines


def render(run, summary, dynamics_summary=None):
    attention_enabled = bool(getattr(run, "capture_attention", True))
    lines = [
        ("MiniMax H3 3D MoBA-style routing probe - %s" if attention_enabled
         else "MiniMax H3 latent-dynamics probe - %s") % run.tag,
        "=" * 92,
        "layout:       %s" % (run.layout.describe() if run.layout else "n/a"),
        "attention:    %s" % (
            "OFF" if not attention_enabled
            else ("OFF (no records)" if not run.records else "on")
        ),
        "latent dyn:   %s" % (
            "sampler x/x0 per-step" if getattr(run, "capture_latent_dynamics", False) else "off"
        ),
        "",
    ]
    if attention_enabled:
        lines[3:3] = [
            "layers:       %s of %s" % (sorted(run.layers), run.notes.get("num_layers")),
            "steps:        %s of %s" % (sorted(run.steps), run.notes.get("total_steps")),
            "3D block:     %dx%dx%d (latent-time x patch-height x patch-width)" % (
                run.block_t, run.block_h, run.block_w
            ),
            "budgets:      %s" % ", ".join(_pct(x).strip() for x in run.budgets),
            "query blocks: %d" % len(run.records),
            "routing:      per query token and per head",
            "execution:    %s" % (
                getattr(run, "execution_geometry", "logical")
                + (
                    " (global Q/KV tiles q=%d kv=%d)" % (
                        int(getattr(run, "sage_q_tile", 128)),
                        int(getattr(run, "sage_kv_tile", 64)),
                    )
                    if getattr(run, "execution_geometry", "logical") == "sage_sparse"
                    else " (per-token logical masks)"
                )
            ),
        ]
        lines.extend([
            "Interpretation: non-video KV tokens are always retained. Each query token",
            "independently selects target-video 3D blocks using dot products against",
            "mean-pooled block keys. 'Routed mass' is the original dense-softmax mass",
            "inside that logical mask. 'Logical sparse output error' compares the",
            "dense attention output with the exact masked-and-renormalized output using V.",
            "When sage_sparse is selected, executable metrics are tile-coarsened and",
            "reported separately; logical metrics are never hardware estimates.",
            "",
        ])

    overall = summary.get("overall", {}) if summary else {}
    if overall:
        lines.extend([
            "SUMMARY (worst case across captured query blocks)",
            "-" * 92,
        ])
        for key in sorted(overall, key=lambda x: overall[x]["budget"]):
            s = overall[key]
            lines.append(
                "  %s video blocks %-5s | logical effective KV <= %-5s | "
                "sparse output rel-L2: record mean-head <= %-6s worst head %-6s | "
                "oracle worst %-6s" % (
                    s.get("execution_geometry", "logical"),
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
            if s.get("execution_geometry") == "sage_sparse":
                lines.append(
                    "                    executable effective KV <= %-5s | "
                    "executable sparse output mean-head <= %-6s worst head %-6s" % (
                        _pct(s["executable_effective_token_density_max"]).strip(),
                        _pct(s["executable_sparse_output_rel_l2_mean_head_max"]).strip(),
                        _pct(s["executable_sparse_output_rel_l2_max_head"]).strip(),
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
                    "    %-5s blocks (%s): logical worst rel-L2 %-6s | oracle %-6s | "
                    "heads >1%%/%s >2%%/%s >5%%/%s | %s" % (
                        _pct(s["budget"]).strip(),
                        s.get("execution_geometry", "logical"),
                        _pct(s["sparse_output_rel_l2_max_head"]).strip(),
                        _pct(s["oracle_output_rel_l2_max_head"]).strip(),
                        len(s["heads_rel_l2_gt_1pct"]),
                        len(s["heads_rel_l2_gt_2pct"]),
                        len(s["heads_rel_l2_gt_5pct"]),
                        worst or "n/a",
                    )
                )
        lines.append("")

    lines.extend(_render_dynamics(run, dynamics_summary or {}))

    if run.records:
        lines.extend(["PER QUERY BLOCK", "-" * 92])
    dyn_regions = _dynamics_region_map(run)
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
        if m.get("execution_geometry") == "sage_sparse":
            requested = m.get("requested_q_range")
            evaluated = m.get("evaluated_q_range") or m.get("execution_q_range")
            if requested and evaluated:
                lines.append(
                    "  sage Q range: requested %d-%d | evaluated %d-%d | Q tiles %d" % (
                        requested[0], requested[1], evaluated[0], evaluated[1],
                        m.get("execution_q_tiles", 0),
                    )
                )
        dyn = dyn_regions.get((int(rec["step"]), int(rec["start"]), int(rec["stop"])))
        if dyn is not None:
            lines.append(
                "  latent dynamics: sample Δ %s | x0 Δ %s | anchor distance %s" % (
                    _pct((dyn.get("sample") or {}).get("update_rel_l2")).strip(),
                    _pct((dyn.get("prediction") or {}).get("update_rel_l2")).strip(),
                    dyn.get("anchor_distance") if dyn.get("anchor_distance") is not None else "n/a",
                )
            )
        for row in m["budgets"]:
            worst = ", ".join(
                "h%d=%s" % (item["head"], _pct(item["rel_l2"]).strip())
                for item in row["worst_heads"][:3]
            )
            lines.append(
                "  %-5s video blocks (%d/%d): mass mean %s min %s | "
                "oracle %s | regret mean %s max %s | overlap %s | logical effective KV %s" % (
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
            if row.get("execution_geometry") == "sage_sparse":
                exec_worst = ", ".join(
                    "h%d=%s" % (item["head"], _pct(item["rel_l2"]).strip())
                    for item in row.get("executable_worst_heads", [])[:3]
                )
                lines.append(
                    "        executable effective KV %s | sparse output rel-L2 "
                    "mean-head %s max-head %s | %s" % (
                        _pct(row["executable_effective_token_density_mean"]).strip(),
                        _pct(row["executable_sparse_output_rel_l2_mean_head"]).strip(),
                        _pct(row["executable_sparse_output_rel_l2_max_head"]).strip(),
                        exec_worst or "n/a",
                    )
                )
            lines.append(
                "        logical sparse output rel-L2 mean-head %s max-head %s | "
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


def write_run(run, arrays=True):
    if not run.records and not getattr(run, "latent_dynamics", None):
        return None
    os.makedirs(run.out_dir, exist_ok=True)
    summary = summarize(run.records)
    layer_teacher = static_layer_teacher.summarize(
        run.records, expected_layers=sorted(run.layers)
    )
    dynamics_summary = latent_dynamics.summarize_dynamics(
        getattr(run, "latent_dynamics", ()), run.records
    )
    report_path = os.path.join(run.out_dir, "moba3d_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(run, summary, dynamics_summary))
    if arrays:
        raw = {}
        indices = []
        steps = []
        for item in getattr(run, "latent_activity_maps", ()):
            label = "%04d_step%d" % (int(item["index"]), int(item["step"]))
            value = item["activity"]
            raw["activity_" + label] = (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)).astype(np.float32)
            indices.append(int(item["index"]))
            steps.append(int(item["step"]))
        for item in getattr(run, "latent_energy_maps", ()):
            label = "%04d_step%d" % (int(item["index"]), int(item["step"]))
            value = item["energy"]
            raw["energy_" + label] = (value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)).astype(np.float32)
        if raw:
            raw["index"] = np.asarray(indices, dtype=np.int64)
            raw["step"] = np.asarray(steps, dtype=np.int64)
            np.savez_compressed(os.path.join(run.out_dir, "latent_dynamics.npz"), **raw)
    payload = {
        "tag": run.tag,
        "layout": run.layout.as_dict() if run.layout else None,
        "layers": sorted(run.layers),
        "steps": sorted(run.steps),
        "block_shape": [run.block_t, run.block_h, run.block_w],
        "budgets": list(run.budgets),
        "execution_geometry": getattr(run, "execution_geometry", "logical"),
        "sage_q_tile": int(getattr(run, "sage_q_tile", 128)),
        "sage_kv_tile": int(getattr(run, "sage_kv_tile", 64)),
        "notes": run.notes,
        "capture_attention": bool(getattr(run, "capture_attention", True)),
        "summary": summary,
        "static_layer_teacher": layer_teacher,
        "dynamics_summary": dynamics_summary,
        "latent_dynamics": list(getattr(run, "latent_dynamics", ())),
        "records": run.records,
    }
    with open(
        os.path.join(run.out_dir, "moba3d_summary.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(payload, f, indent=2)
    with open(
        os.path.join(run.out_dir, "static_layer_teacher.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(layer_teacher, f, indent=2)
    return report_path
