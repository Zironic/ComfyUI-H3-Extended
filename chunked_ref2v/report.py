"""Run report: one JSON document and one human-readable summary.

`null` rather than a plausible number wherever a value is unknown. An arm that
cancelled on the VRAM guard has no metrics, and inventing zeros for it would put
a resource result on the same axis as a model result.
"""

import json

from .metrics import format_metrics

SCHEMA_VERSION = 3


def build(*, run_id, geometry, seeds, canvas, experiment_ids, results,
          chunk_a_reused, dependencies, notes=None, memory_status=None,
          monolithic_reused=None, tail_frames=17):
    from .geometry import UnalignedProfileError

    try:
        start, count = geometry.overlap_slice()
    except UnalignedProfileError:
        start, count = None, None

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "profile": {
            "chunk_frames": geometry.chunk_frames,
            "overlap_frames": geometry.overlap_frames,
            "stride_frames": geometry.stride_frames,
            "target_latent_t": geometry.target_latent_t,
            "overlap_latent_start": start,
            "overlap_latent_t": count,
            "fps": geometry.fps,
            "canvas": list(canvas),
        },
        "seeds": seeds.as_dict(),
        "common_assets": {
            "chunk_a_reused": bool(chunk_a_reused),
            "base_qwen_b_reused": True,
            "dynamic_dependencies": dependencies.as_dict() if dependencies else None,
            "monolithic_reference": None if monolithic_reused is None else {
                "total_frames": geometry.total_frames,
                "tail_frames": tail_frames,
                "tail_range": list(geometry.tail_range(tail_frames)),
                "reused": bool(monolithic_reused),
            },
        },
        # Every arm shares one armed model, so the backend belongs to the run
        # rather than to any single experiment - but each arm's resources carry
        # a copy of the selection, because a resource number read without its
        # backend is the one that gets misquoted later.
        "runtime": dict(memory_status or {}) or None,
        "selected": list(experiment_ids),
        "notes": list(notes or []),
        "experiments": {},
    }

    selected_attention = (memory_status or {}).get("attention_selected")
    for record in results:
        resources = dict(record.get("resources") or {})
        if selected_attention is not None:
            resources["attention_selected"] = selected_attention
        document["experiments"][record["experiment_id"]] = {
            "status": record.get("status"),
            "note": record.get("note"),
            "strategy": record.get("strategy") or {},
            "dependencies": record.get("dependencies") or {},
            "layout": record.get("prepared") or {},
            "metrics": record.get("metrics") or {},
            "resources": resources,
            "artifacts": record.get("artifacts") or {},
        }
    return document


def to_json(document):
    return json.dumps(document, indent=2, default=str)


def to_text(document):
    profile = document["profile"]
    lines = [
        "MiniMax H3 Ref2V experiment harness",
        "run %s" % document["run_id"],
        "",
        "profile   C=%(chunk_frames)d O=%(overlap_frames)d S=%(stride_frames)d "
        "T=%(target_latent_t)s at %(fps)d fps" % profile,
        "canvas    %dx%d" % tuple(profile["canvas"]),
        "overlap   latent positions %s-%s"
        % (profile["overlap_latent_start"],
           None if profile["overlap_latent_start"] is None
           else profile["overlap_latent_start"] + profile["overlap_latent_t"] - 1),
        "chunk A   %s" % ("reused from cache" if document["common_assets"]["chunk_a_reused"]
                          else "generated this run"),
        "runtime   %s" % _runtime_line(document.get("runtime")),
        "",
    ]
    warning = _runtime_warning(document.get("runtime"))
    if warning:
        lines.extend([warning, ""])
    for note in document.get("notes") or []:
        lines.append("  %s" % note)
    if document.get("notes"):
        lines.append("")

    for experiment_id, entry in document["experiments"].items():
        strategy = entry.get("strategy") or {}
        lines.append("%s  [%s]" % (experiment_id, entry.get("status")))
        lines.append("  %s" % strategy.get("display_name", ""))
        lines.append("  carry=%s prompt=%s position=%s source=%s target=%s" % (
            strategy.get("carry_strategy"), strategy.get("prompt_policy"),
            strategy.get("position_policy"), strategy.get("source_reference_policy"),
            strategy.get("target_policy")))
        layout = entry.get("layout") or {}
        if layout.get("carry"):
            lines.append("  carried: %s" % layout["carry"])
        for condition in layout.get("target_conditions") or []:
            lines.append("  condition: %s" % condition)
        if entry.get("note"):
            lines.append("  note: %s" % entry["note"])
        metrics_text = format_metrics(entry.get("metrics") or {})
        if metrics_text:
            lines.append("  metrics:")
            lines.append(metrics_text)
        resources = entry.get("resources") or {}
        if resources:
            lines.append("  resources: %s" % " ".join(
                "%s=%s" % (k, v) for k, v in sorted(resources.items())))
        lines.append("")

    lines.append(_ranking(document))
    ground_truth = _monolithic_ranking(document)
    if ground_truth:
        lines.extend(["", ground_truth])
    return "\n".join(lines)


def _monolithic_ranking(document):
    """Rank arms by agreement with a single run of the same length.

    Printed separately from the overlap ranking because it answers a different
    question. Overlap agreement asks whether chunk B followed chunk A; this asks
    whether chunking reproduced what one run would have produced - which is the
    question the production node exists to answer.
    """
    mono = (document.get("common_assets") or {}).get("monolithic_reference")
    if not mono:
        return None
    rows = []
    for experiment_id, entry in document["experiments"].items():
        metrics = entry.get("metrics") or {}
        value = metrics.get("monolithic_tail_mae")
        if value is None:
            continue
        rows.append((value, experiment_id,
                     metrics.get("monolithic_tail_motion_mae"),
                     metrics.get("monolithic_chunk_b_mae"),
                     metrics.get("monolithic_tail_mae_vs_baseline")))
    if not rows:
        return None

    a, b = mono["tail_range"]
    lines = [
        "agreement with one %d-frame run (ground truth), over global frames %d-%d:"
        % (mono["total_frames"], a, b - 1),
        "  %-34s %10s %11s %10s %9s"
        % ("experiment", "tail MAE", "tail motion", "chunk B", "vs base"),
    ]
    for value, experiment_id, motion, body, ratio in sorted(rows):
        lines.append("  %-34s %10.6f %11s %10s %9s" % (
            experiment_id, value,
            "-" if motion is None else "%.6f" % motion,
            "-" if body is None else "%.6f" % body,
            "-" if ratio is None else "%.3f" % ratio))
    lines.append("")
    lines.append("  This never reaches zero: the two runs denoise different latent")
    lines.append("  shapes and so consume different noise. Read it as a ranking - a")
    lines.append("  carry that works lands its tail closer to the single run than the")
    lines.append("  no-carry control does.")
    return "\n".join(lines)


def _runtime_line(status):
    if not status:
        return "memory optimizer not armed"
    from .memory import describe
    return describe(status).replace("memory optimizer: ", "", 1)


def _runtime_warning(status):
    """Say it plainly when the numbers are not the optimized path's numbers.

    All arms still share one backend, so the comparison between them holds. What
    stops holding is the absolute resource column - and that is exactly the
    figure someone will later quote into a headroom budget.
    """
    if not status:
        return None
    from .memory import ATTENTION_EXISTING
    if status.get("attention_selected") != ATTENTION_EXISTING:
        return None
    if status.get("armed_by") == "disabled":
        return ("NOTE: the memory optimizer was disabled for this run. Arm-to-arm "
                "comparison is valid; peak-VRAM and runtime describe the graph's "
                "own attention backend, not optimized Sage.")
    return ("WARNING: optimized attention was requested but fell back to the "
            "existing backend (%s). Arm-to-arm comparison is still valid; the "
            "absolute resource figures below are NOT attributable to optimized "
            "Sage." % status.get("attention_reason"))


def _ranking(document):
    """Order arms by overlap agreement, with the caveat that matters stated.

    A frozen frame scores a perfect overlap MAE, so the motion-energy ratio is
    printed next to it rather than folded into a single score.
    """
    rows = []
    for experiment_id, entry in document["experiments"].items():
        metrics = entry.get("metrics") or {}
        value = metrics.get("pixel_overlap_mae")
        if value is None:
            continue
        rows.append((value, experiment_id, metrics.get("latent_overlap_mae"),
                     metrics.get("motion_energy_ratio")))
    if not rows:
        return "no completed experiment produced overlap metrics."

    lines = ["overlap agreement (lower pixel MAE = closer to Chunk A):",
             "  %-34s %10s %12s %10s" % ("experiment", "pixel MAE", "latent MAE", "motion")]
    for value, experiment_id, latent, motion in sorted(rows):
        lines.append("  %-34s %10.6f %12s %10s" % (
            experiment_id, value,
            "-" if latent is None else "%.6f" % latent,
            "-" if motion is None else "%.3f" % motion))
    lines.append("")
    lines.append("  A motion ratio far below 1.0 means the arm stopped following the "
                 "source; a low overlap MAE bought that way is not a win.")
    return "\n".join(lines)
