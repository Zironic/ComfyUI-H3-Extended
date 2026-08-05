"""Persistence and summaries for masked-cache measurement runs."""

import json
import os

import numpy as np

from . import mask as mask_ops
from .config import THRESHOLD_SWEEP


def _pct(x):
    return "  n/a " if x is None else "%5.1f%%" % (100.0 * x)


def _score_maps(run):
    guided = [x for label, x in run.score_maps if label != "final"]
    final = next((x for label, x in run.score_maps if label == "final"), None)
    return guided, final


def _policy_sweep(run):
    """Simulate the configured burn-in/freeze policy at every threshold."""
    cfg = run.config
    guided_scores, final_score = _score_maps(run)
    start, stop = cfg.freeze_start, cfg.freeze_stop
    out = []

    for threshold in THRESHOLD_SWEEP:
        masks = [mask_ops.build_mask(
            score, threshold, cfg.tile_h, cfg.tile_w,
            cfg.spatial_halo, cfg.temporal_halo)[1]
            for score in guided_scores]
        active = [mask_ops.active_fraction(m) for m in masks]
        row = {
            "threshold": float(threshold),
            "guided_active_mean": sum(active) / len(active) if active else None,
            "guided_active_max": max(active) if active else None,
            "frozen_active": None,
            "later_escaped_max": None,
            "later_escaped_mean": None,
            "final_active": None,
            "final_coverage": None,
            "final_escaped": None,
            "final_missed_excess_mass": None,
        }

        if len(masks) >= stop:
            frozen = masks[start].clone()
            for m in masks[start + 1:stop]:
                frozen |= m
            row["frozen_active"] = mask_ops.active_fraction(frozen)

            later_escape = [mask_ops.escaped_fraction(m, frozen) for m in masks[stop:]]
            later_escape = [x for x in later_escape if x is not None]
            if later_escape:
                row["later_escaped_max"] = max(later_escape)
                row["later_escaped_mean"] = sum(later_escape) / len(later_escape)

            if final_score is not None:
                final_mask = mask_ops.build_mask(
                    final_score, threshold, cfg.tile_h, cfg.tile_w,
                    cfg.spatial_halo, cfg.temporal_halo)[1]
                row["final_active"] = mask_ops.active_fraction(final_mask)
                row["final_coverage"] = mask_ops.coverage_fraction(final_mask, frozen)
                row["final_escaped"] = mask_ops.escaped_fraction(final_mask, frozen)
                row["final_missed_excess_mass"] = mask_ops.missed_score_mass(
                    final_score, frozen, threshold)
        out.append(row)
    return out


def aggregate(run):
    steps = run.steps
    if not steps:
        return None

    def series(key):
        return [s[key] for s in steps if s.get(key) is not None]

    jac = series("jaccard_prev")
    esc_union = series("escaped_union")
    esc_frozen = series("escaped_frozen")
    missed_mass = series("missed_score_mass_frozen")
    core = series("active_core")
    expanded = series("active_expanded")

    sweep = []
    for i, thr in enumerate(THRESHOLD_SWEEP):
        vals_core = [s["threshold_sweep"][i]["active_core"] for s in steps]
        vals_exp = [s["threshold_sweep"][i]["active_expanded"] for s in steps]
        sweep.append({
            "threshold": float(thr),
            "active_core_mean": sum(vals_core) / len(vals_core),
            "active_core_max": max(vals_core),
            "active_expanded_mean": sum(vals_exp) / len(vals_exp),
            "active_expanded_max": max(vals_exp),
        })

    return {
        "observed_forwards": len(steps),
        "distinct_sigmas": run.sigma_count,
        "burn_in_steps": run.config.burn_in_steps,
        "warmup_steps": run.config.warmup_steps,
        "frozen_range": list(run.frozen_range) if run.frozen_range is not None else None,
        "frozen_active": (float(run.frozen_mask.float().mean().item())
                          if run.frozen_mask is not None else None),
        "active_core": {"first": core[0], "last": core[-1], "min": min(core), "max": max(core)},
        "active_expanded": {"first": expanded[0], "last": expanded[-1],
                            "min": min(expanded), "max": max(expanded)},
        "jaccard_consecutive": {"min": min(jac), "mean": sum(jac) / len(jac)} if jac else None,
        "escaped_running_union": ({"max": max(esc_union), "mean": sum(esc_union) / len(esc_union)}
                                  if esc_union else None),
        "escaped_frozen": ({"max": max(esc_frozen), "mean": sum(esc_frozen) / len(esc_frozen)}
                           if esc_frozen else None),
        "missed_score_mass_frozen": ({"max": max(missed_mass), "mean": sum(missed_mass) / len(missed_mass)}
                                     if missed_mass else None),
        "union_active": (float(run.union_mask.float().mean().item())
                         if run.union_mask is not None else None),
        "final": run.final,
        "threshold_sweep": sweep,
        "policy_sweep": _policy_sweep(run),
    }


def render(run, summary):
    cfg = run.config
    layout = run.layout
    lines = [
        "MiniMax H3 masked Ref2V - measurement - %s" % run.tag,
        "=" * 78,
        "mode:        %s (strict=%s)" % (cfg.mode, cfg.strict),
        "source:      %s" % (run.source.describe() if run.source else "n/a"),
        "layout:      %s" % (layout.describe() if layout else "n/a"),
    ]
    if layout:
        t, ph, pw = layout.video_shape
        lines.append("token grid:  t=%d %dx%d = %d target-video rows of %d packed"
                     % (t, ph, pw, t * ph * pw, layout.seq_len))
    lines.extend([
        "threshold:   %.4g (absolute floor %.3g)" % (cfg.score_threshold, cfg.score_absolute_floor),
        "tiles:       %dx%d tokens, spatial halo %d tiles, temporal halo %d frames"
        % (cfg.tile_h, cfg.tile_w, cfg.spatial_halo, cfg.temporal_halo),
        "freeze:      discard %d burn-in prediction%s; union the next %d; never grow afterwards"
        % (cfg.burn_in_steps, "" if cfg.burn_in_steps == 1 else "s", cfg.warmup_steps),
    ])
    for k, v in sorted(run.notes.items()):
        lines.append("%-18s %s" % (k + ":", v))
    if run.disabled_reason:
        lines.extend(["", "!! MEASUREMENT DISABLED: %s" % run.disabled_reason])
    if run.fallbacks:
        lines.append("")
        lines.append("fallbacks:")
        for reason, n in run.fallbacks:
            lines.append("  %4dx %s" % (n, reason))
    lines.append("")

    if summary:
        lines.extend(["SUMMARY", "-" * 78])
        lines.append("  observed guided predictions: %d over %d distinct sigmas"
                     % (summary["observed_forwards"], summary["distinct_sigmas"]))
        lines.append("  frozen after burn-in+warmup:  %s" % _pct(summary["frozen_active"]))
        a = summary["active_expanded"]
        lines.append("  active (tiles + halo):        first %s  last %s  max %s"
                     % (_pct(a["first"]), _pct(a["last"]), _pct(a["max"])))
        if summary["escaped_frozen"]:
            e = summary["escaped_frozen"]
            lines.append("  escaped immutable frozen mask: max %s  mean %s"
                         % (_pct(e["max"]), _pct(e["mean"])))
        if summary["missed_score_mass_frozen"]:
            e = summary["missed_score_mass_frozen"]
            lines.append("  excess score outside frozen:   max %s  mean %s"
                         % (_pct(e["max"]), _pct(e["mean"])))
        if summary["final"]:
            f = summary["final"]
            lines.append("  FINAL sampled latent:")
            lines.append("    active after tiles+halo:      %s" % _pct(f["active_expanded"]))
            lines.append("    covered by frozen mask:       %s" % _pct(f["coverage_by_frozen"]))
            lines.append("    escaped frozen mask:          %s" % _pct(f["escaped_frozen"]))
            lines.append("    final excess score missed:    %s" % _pct(f["missed_score_mass_frozen"]))
        else:
            lines.append("  FINAL sampled latent:           not captured")

        lines.extend(["", "  THRESHOLD SWEEP (mean over guided predictions)",
                      "    threshold   active core        active after tiles+halo"])
        for row in summary["threshold_sweep"]:
            lines.append("    %9.4g   %s (max %s)   %s (max %s)" % (
                row["threshold"], _pct(row["active_core_mean"]), _pct(row["active_core_max"]),
                _pct(row["active_expanded_mean"]), _pct(row["active_expanded_max"])))

        lines.extend(["", "  FROZEN POLICY SWEEP",
                      "    threshold  guided mean  frozen  later escape  final active  final escape  excess missed"])
        for row in summary["policy_sweep"]:
            lines.append("    %9.4g  %s     %s   %s       %s      %s       %s" % (
                row["threshold"], _pct(row["guided_active_mean"]),
                _pct(row["frozen_active"]), _pct(row["later_escaped_max"]),
                _pct(row["final_active"]), _pct(row["final_escaped"]),
                _pct(row["final_missed_excess_mass"])))
        lines.append("")

    lines.extend(["PER GUIDED PREDICTION", "-" * 78,
                  "  step   sigma    active core   +tiles/halo   J(prev)  escaped(frozen)"])
    for s in run.steps:
        lines.append("  %4d %8.4f   %s        %s     %s   %s" % (
            s["step"], s["sigma"], _pct(s["active_core"]), _pct(s["active_expanded"]),
            "  n/a" if s["jaccard_prev"] is None else "%5.3f" % s["jaccard_prev"],
            _pct(s["escaped_frozen"])))
    lines.extend(["", "SCORE QUANTILES (guided relative token score)", "-" * 78])
    if run.steps:
        keys = list(run.steps[0]["score_quantiles"].keys())
        lines.append("  step  " + "  ".join("%8s" % ("q" + k) for k in keys))
        for s in run.steps:
            lines.append("  %4d  " % s["step"] + "  ".join(
                "%8.4f" % s["score_quantiles"][k] for k in keys))
    return "\n".join(lines)


def write_run(run):
    if not run.steps and run.disabled_reason is None and run.final is None:
        return None
    os.makedirs(run.out_dir, exist_ok=True)
    summary = aggregate(run)

    report_path = os.path.join(run.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(run, summary))

    payload = {
        "tag": run.tag,
        "config": run.config.as_dict(),
        "layout": run.layout.as_dict() if run.layout else None,
        "source": ({"ref_ordinal": run.source.ref_ordinal,
                    "payload_index": run.source.payload_index,
                    "kind": run.source.kind,
                    "latent_shape": list(run.source.latent.shape)}
                   if run.source and run.source.valid else None),
        "notes": run.notes,
        "disabled_reason": run.disabled_reason,
        "fallbacks": [{"reason": r, "count": n} for r, n in run.fallbacks],
        "summary": summary,
        "final": run.final,
        "steps": run.steps,
    }
    with open(os.path.join(run.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(os.path.join(run.out_dir, "steps.jsonl"), "w", encoding="utf-8") as f:
        for s in run.steps:
            f.write(json.dumps(s) + "\n")

    arrays = {}
    for label, x in run.score_maps:
        arrays["score_" + label] = x.numpy().astype(np.float32)
    for label, x in run.error_maps:
        arrays["error_rms_" + label] = x.numpy().astype(np.float32)
    for label, x in run.source_maps:
        arrays["source_rms_" + label] = x.numpy().astype(np.float32)
    for label, x in run.saliency_maps:
        arrays["saliency_" + label] = x.numpy().astype(np.float32)
    for label, x in run.masks:
        arrays["mask_" + label] = x.numpy()
    if run.union_mask is not None:
        arrays["mask_union"] = run.union_mask.detach().cpu().numpy()
    if run.frozen_mask is not None:
        arrays["mask_frozen"] = run.frozen_mask.detach().cpu().numpy()
    arrays["index"] = np.array(json.dumps({
        "tag": run.tag,
        "config": run.config.as_dict(),
        "video_shape": list(run.layout.video_shape) if run.layout else None,
        "steps": [{"i": i, "step": s["step"], "sigma": s["sigma"],
                   "source_kind": s.get("source_kind")} for i, s in enumerate(run.steps)],
        "has_final": run.final is not None,
    }))
    np.savez_compressed(os.path.join(run.out_dir, "mask.npz"), **arrays)
    return report_path
