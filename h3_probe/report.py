"""Trace persistence and human-readable candidate-mask reports."""

import json
import os

import numpy as np

from . import metrics


def _pct(x):
    return "%5.1f%%" % (100.0 * x)


def render_record(a):
    """One query block, in the form the mask decision is actually made from."""
    if a["kind"] == "video":
        head = "Layer %d, step %d, video query block t=%d (rows %d-%d)" % (
            a["layer"], a["step"], a["frame"], a["q_start"], a["q_stop"])
    else:
        head = "Layer %d, step %d, %s query block (rows %d-%d)" % (
            a["layer"], a["step"], a["kind"], a["q_start"], a["q_stop"])

    lines = [head]
    lines.append("  text/reference mandatory context: %s" % _pct(a["mandatory"]))
    lines.append("    text:                           %s" % _pct(a["text"]))
    lines.append("    references/keyframes:           %s" % _pct(a["references"]))
    lines.append("    target audio:                   %s" % _pct(a["target_audio"]))
    if a["kind"] == "video":
        lines.append("  current frame:                    %s" % _pct(a["current_frame"]))
        lines.append("  adjacent frames:                  %s" % _pct(a["adjacent_frames"]))
        lines.append("  other frames:                     %s" % _pct(a["other_frames"]))
        lines.append("")
        lines.append("  by temporal distance:             " + "  ".join(
            "%s:%s" % (k, _pct(v).strip()) for k, v in a["by_temporal_distance"].items()))
        lines.append("  same spatial region (r=%d):       %s  (elsewhere %s)" % (
            a["spatial_radius"], _pct(a["same_spatial_region"]), _pct(a["other_spatial"])))
    lines.append("")
    lines.append("  local mask retained:              %s   (exact tokens)" % _pct(a["local_exact"]))
    lines.append("  local blocks retained:            %s   (%d/%d blocks)" % (
        _pct(a["local_blocks"]), a["n_local_blocks"], a["n_blocks"]))
    for k in sorted(a["topk"]):
        lines.append("  local + top-%-2d distant:           %s" % (k, _pct(a["topk"][k])))
    return "\n".join(lines)


def render(run, analyses, summary):
    layout = run.layout
    lines = []
    lines.append("MiniMax H3 attention probe - %s" % run.tag)
    lines.append("=" * 72)
    lines.append("layout:      %s" % (layout.describe() if layout else "n/a"))
    lines.append("layers:      %s of %s" % (sorted(run.layers), run.notes.get("num_layers")))
    lines.append("steps:       %s of %s" % (sorted(run.steps), run.notes.get("total_steps")))
    lines.append("kv block:    %d tokens" % run.block)
    lines.append("records:     %d" % len(analyses))
    lines.append("")

    if summary:
        lines.append("SUMMARY (worst case over all probed query blocks)")
        lines.append("-" * 72)
        lines.append("  mandatory context:      min %s  median %s" % (
            _pct(summary["mandatory"]["min"]), _pct(summary["mandatory"]["median"])))
        lines.append("  local mask (exact):     min %s  median %s" % (
            _pct(summary["local_exact"]["min"]), _pct(summary["local_exact"]["median"])))
        lines.append("  local mask (blocks):    min %s  median %s" % (
            _pct(summary["local_blocks"]["min"]), _pct(summary["local_blocks"]["median"])))
        for k in sorted(summary["topk"]):
            s = summary["topk"][k]
            lines.append("  local + top-%-2d distant: min %s  median %s" % (
                k, _pct(s["min"]), _pct(s["median"])))
        if "video_only" in summary:
            v = summary["video_only"]
            lines.append("")
            lines.append("  video queries only (n=%d):" % v["n"])
            lines.append("    current frame:        min %s  median %s" % (
                _pct(v["current_frame"]["min"]), _pct(v["current_frame"]["median"])))
            lines.append("    adjacent frames:      min %s  median %s" % (
                _pct(v["adjacent_frames"]["min"]), _pct(v["adjacent_frames"]["median"])))
            lines.append("    other frames:         min %s  median %s" % (
                _pct(v["other_frames"]["min"]), _pct(v["other_frames"]["median"])))
            lines.append("    same spatial region:  min %s  median %s" % (
                _pct(v["same_spatial_region"]["min"]), _pct(v["same_spatial_region"]["median"])))
        rec = metrics.recommend(summary)
        lines.append("")
        lines.append("  smallest top-k clearing 99%% worst case: %s" % (
            ("top-%d" % rec) if rec else "none of %s" % sorted(summary["topk"])))
        lines.append("")

    lines.append("PER QUERY BLOCK")
    lines.append("-" * 72)
    for a in analyses:
        lines.append(render_record(a))
        lines.append("")
    return "\n".join(lines)


def write_run(run, adjacent=1, spatial_radius=4):
    """Re-render the trace and report. Cheap enough to run after every capture."""
    if not run.records:
        return None
    os.makedirs(run.out_dir, exist_ok=True)

    analyses = [metrics.analyze(r, run.layout, adjacent=adjacent, spatial_radius=spatial_radius)
                for r in run.records]
    summary = metrics.summarize(analyses)

    report_path = os.path.join(run.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render(run, analyses, summary))

    with open(os.path.join(run.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "tag": run.tag,
            "layout": run.layout.as_dict() if run.layout else None,
            "layers": sorted(run.layers),
            "steps": sorted(run.steps),
            "block": run.block,
            "notes": run.notes,
            "summary": summary,
            "records": [{k: v for k, v in a.items() if k != "cat"} for a in analyses],
        }, f, indent=2)

    arrays = {}
    for i, r in enumerate(run.records):
        arrays["block_mass_%d" % i] = r["block_mass"].numpy().astype(np.float16)
        arrays["cat_mass_%d" % i] = r["cat_mass"].numpy().astype(np.float32)
        arrays["frame_mass_%d" % i] = r["frame_mass"].numpy().astype(np.float32)
        arrays["spatial_mass_%d" % i] = r["spatial_mass"].numpy().astype(np.float32)
    arrays["index"] = np.array(json.dumps([
        {"i": i, "layer": r["layer"], "step": r["step"], "sigma": r["sigma"],
         "kind": r["kind"], "frame": r["frame"], "start": r["start"], "stop": r["stop"],
         "cond_or_uncond": r["cond_or_uncond"]}
        for i, r in enumerate(run.records)]))
    np.savez_compressed(os.path.join(run.out_dir, "trace.npz"), **arrays)
    return report_path
