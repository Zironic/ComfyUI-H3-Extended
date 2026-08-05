"""Persistence for masked-cache measurement runs.

    output/h3_masked_cache/<run_tag>_<timestamp>/
    |-- summary.json    run metadata, config, per-step rows, aggregate sweep
    |-- steps.jsonl     one line per observed forward, append-friendly
    |-- mask.npz        token score maps and expanded masks, compressed
    `-- report.txt      the same thing, readable

No pickles: the score maps are the evidence a threshold gets chosen from, and
they have to be loadable a year from now by something that is not this code.

Rewritten in full after every observed forward, so a cancelled or OOM-killed run
still leaves everything it had measured up to that point.
"""

import json
import os

import numpy as np

from .config import THRESHOLD_SWEEP


def _pct(x):
    return "  n/a " if x is None else "%5.1f%%" % (100.0 * x)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

def aggregate(run):
    """Run-level answers to the Stage 0 gate questions.

    Three numbers decide whether compaction is worth building:

    * how small the mask gets (`active_expanded` at the end of the run);
    * how stable consecutive masks are (`jaccard`);
    * how much of a late mask escapes the *union of everything seen so far*
      (`escaped_union`) - the direct measure of what freezing an early mask
      would lose.
    """
    steps = run.steps
    if not steps:
        return None

    def series(key):
        return [s[key] for s in steps if s.get(key) is not None]

    jac = series("jaccard_prev")
    esc = series("escaped_union")
    core = series("active_core")
    expanded = series("active_expanded")

    # sweep averaged over steps, so one threshold can be read off the whole run
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
        "active_core": {"first": core[0], "last": core[-1], "min": min(core), "max": max(core)},
        "active_expanded": {"first": expanded[0], "last": expanded[-1],
                            "min": min(expanded), "max": max(expanded)},
        "jaccard_consecutive": {"min": min(jac), "mean": sum(jac) / len(jac)} if jac else None,
        "escaped_union": {"max": max(esc), "mean": sum(esc) / len(esc)} if esc else None,
        "union_active": (float(run.union_mask.float().mean().item())
                         if run.union_mask is not None else None),
        "threshold_sweep": sweep,
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def render(run, summary):
    cfg = run.config
    layout = run.layout
    lines = []
    lines.append("MiniMax H3 masked Ref2V - measurement - %s" % run.tag)
    lines.append("=" * 78)
    lines.append("mode:        %s (strict=%s)" % (cfg.mode, cfg.strict))
    lines.append("source:      %s" % (run.source.describe() if run.source else "n/a"))
    lines.append("layout:      %s" % (layout.describe() if layout else "n/a"))
    if layout:
        t, ph, pw = layout.video_shape
        lines.append("token grid:  t=%d %dx%d = %d target-video rows of %d packed"
                     % (t, ph, pw, t * ph * pw, layout.seq_len))
    lines.append("threshold:   %.4g (absolute floor %.3g)" % (cfg.score_threshold,
                                                              cfg.score_absolute_floor))
    lines.append("tiles:       %dx%d tokens, spatial halo %d tiles, temporal halo %d frames"
                 % (cfg.tile_h, cfg.tile_w, cfg.spatial_halo, cfg.temporal_halo))
    for k, v in sorted(run.notes.items()):
        lines.append("%-18s %s" % (k + ":", v))
    if run.disabled_reason:
        lines.append("")
        lines.append("!! MEASUREMENT DISABLED: %s" % run.disabled_reason)
        lines.append("!! the rows below, if any, stop at that point - this is not a "
                     "complete observation of the run")
    if run.fallbacks:
        lines.append("")
        lines.append("fallbacks:")
        for reason, n in run.fallbacks:
            lines.append("  %4dx %s" % (n, reason))
    lines.append("")

    if summary:
        lines.append("SUMMARY")
        lines.append("-" * 78)
        lines.append("  observed forwards:      %d over %d distinct sigmas"
                     % (summary["observed_forwards"], summary["distinct_sigmas"]))
        a = summary["active_core"]
        lines.append("  active (threshold only):  first %s  last %s  max %s"
                     % (_pct(a["first"]), _pct(a["last"]), _pct(a["max"])))
        a = summary["active_expanded"]
        lines.append("  active (tiles + halo):    first %s  last %s  max %s"
                     % (_pct(a["first"]), _pct(a["last"]), _pct(a["max"])))
        lines.append("  union over the run:       %s" % _pct(summary["union_active"]))
        if summary["jaccard_consecutive"]:
            j = summary["jaccard_consecutive"]
            lines.append("  mask stability (J):       min %.3f  mean %.3f" % (j["min"], j["mean"]))
        if summary["escaped_union"]:
            e = summary["escaped_union"]
            lines.append("  escaped the running union: max %s  mean %s"
                         % (_pct(e["max"]), _pct(e["mean"])))
            lines.append("    (share of a step's active tokens that no earlier step covered -")
            lines.append("     this is what freezing an early mask would miss)")
        lines.append("")
        lines.append("  THRESHOLD SWEEP (mean over observed forwards)")
        lines.append("    threshold   active core        active after tiles+halo")
        for row in summary["threshold_sweep"]:
            lines.append("    %9.4g   %s (max %s)   %s (max %s)" % (
                row["threshold"], _pct(row["active_core_mean"]), _pct(row["active_core_max"]),
                _pct(row["active_expanded_mean"]), _pct(row["active_expanded_max"])))
        lines.append("")

    lines.append("PER OBSERVED FORWARD")
    lines.append("-" * 78)
    lines.append("  step   sigma    active core   +tiles/halo   J(prev)  escaped(union)")
    for s in run.steps:
        lines.append("  %4d %8.4f   %s        %s     %s   %s" % (
            s["step"], s["sigma"], _pct(s["active_core"]), _pct(s["active_expanded"]),
            "  n/a" if s["jaccard_prev"] is None else "%5.3f" % s["jaccard_prev"],
            _pct(s["escaped_union"])))
    lines.append("")
    lines.append("SCORE QUANTILES (token resolution)")
    lines.append("-" * 78)
    if run.steps:
        keys = list(run.steps[0]["score_quantiles"].keys())
        lines.append("  step  " + "  ".join("%8s" % ("q" + k) for k in keys))
        for s in run.steps:
            lines.append("  %4d  " % s["step"] + "  ".join(
                "%8.4f" % s["score_quantiles"][k] for k in keys))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------

def write_run(run):
    """Write every artifact for `run`. Safe to call after each observed forward."""
    if not run.steps and run.disabled_reason is None:
        return None
    os.makedirs(run.out_dir, exist_ok=True)
    summary = aggregate(run)

    report_path = os.path.join(run.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(run, summary))

    with open(os.path.join(run.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tag": run.tag,
            "config": run.config.as_dict(),
            "layout": run.layout.as_dict() if run.layout else None,
            "source": {
                "ref_ordinal": run.source.ref_ordinal,
                "payload_index": run.source.payload_index,
                "kind": run.source.kind,
                "latent_shape": list(run.source.latent.shape),
            } if run.source and run.source.valid else None,
            "notes": run.notes,
            "disabled_reason": run.disabled_reason,
            "fallbacks": [{"reason": r, "count": n} for r, n in run.fallbacks],
            "summary": summary,
            "steps": run.steps,
        }, f, indent=2)

    with open(os.path.join(run.out_dir, "steps.jsonl"), "w", encoding="utf-8") as f:
        for s in run.steps:
            f.write(json.dumps(s) + "\n")

    arrays = {}
    for label, scores in run.score_maps:
        arrays["score_" + label] = scores.numpy().astype(np.float16)
    for label, m in run.masks:
        arrays["mask_" + label] = m.numpy()
    if run.union_mask is not None:
        arrays["mask_union"] = run.union_mask.detach().to("cpu").numpy()
    arrays["index"] = np.array(json.dumps({
        "tag": run.tag,
        "config": run.config.as_dict(),
        "video_shape": list(run.layout.video_shape) if run.layout else None,
        "steps": [{"i": i, "step": s["step"], "sigma": s["sigma"],
                   "cond_or_uncond": s["cond_or_uncond"]} for i, s in enumerate(run.steps)],
    }))
    np.savez_compressed(os.path.join(run.out_dir, "mask.npz"), **arrays)
    return report_path
