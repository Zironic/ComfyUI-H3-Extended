"""Request-end structural reports for H3 hybrid sparse attention."""

import json
import os
import re

RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def validate_run_tag(value):
    tag = str(value).strip()
    if RUN_TAG_RE.fullmatch(tag) is None:
        raise ValueError(
            "run_tag must be 1-64 ASCII letters, digits, underscores, or hyphens"
        )
    return tag


def _mean(records, key):
    return sum(float(row[key]) for row in records) / len(records) if records else 0.0


def summarize(records, seconds=None):
    layers = sorted({int(row["layer"]) for row in records})
    steps = sorted({int(row["step"]) for row in records if int(row["step"]) >= 0})
    full = [float(row["full_mask_density"]) for row in records]
    video = [float(row["actual_video_tile_density"]) for row in records]
    return {
        "records": len(records),
        "layers_observed": layers,
        "layer_count": len(layers),
        "expected_layer_count": 50,
        "steps_observed": steps,
        "step_count": len(steps),
        "requested_video_budget": (
            float(records[0]["requested_video_budget"]) if records else None
        ),
        "mean_video_tile_density": _mean(records, "actual_video_tile_density"),
        "mean_full_mask_density": _mean(records, "full_mask_density"),
        "min_full_mask_density": min(full) if full else None,
        "max_full_mask_density": max(full) if full else None,
        "min_video_tile_density": min(video) if video else None,
        "max_video_tile_density": max(video) if video else None,
        "request_seconds": None if seconds is None else float(seconds),
    }


def render(payload):
    summary = payload["summary"]
    budget = summary["requested_video_budget"]
    lines = [
        "H3 Hybrid Sparse Attention",
        "mode: %s" % payload["mode"],
        "budget: %s" % ("unknown" if budget is None else "%.1f%%" % (100 * budget)),
        "layers: %d / %d" % (
            summary["layer_count"], summary["expected_layer_count"]),
        "steps observed: %d" % summary["step_count"],
        "",
        "mean video KV tile density: %.3f%%" % (
            100 * summary["mean_video_tile_density"]),
        "mean packed mask density: %.3f%%" % (
            100 * summary["mean_full_mask_density"]),
    ]
    if summary["min_full_mask_density"] is not None:
        lines.extend([
            "min packed mask density: %.3f%%" % (
                100 * summary["min_full_mask_density"]),
            "max packed mask density: %.3f%%" % (
                100 * summary["max_full_mask_density"]),
        ])
    lines.extend([
        "",
        "Phase A uses direct 128Q x 64KV Sparse Sage routing.",
        "Flex, compatibility fallback, Sol head dispatch, and timing are not enabled.",
    ])
    return "\n".join(lines) + "\n"


def write_request(output_root, run_tag, timestamp, request_id, records, seconds=None):
    tag = validate_run_tag(run_tag)
    directory = os.path.join(output_root, "%s_%s" % (tag, timestamp))
    os.makedirs(directory, exist_ok=False)
    payload = {
        "status": "complete",
        "mode": "sage128",
        "run_tag": tag,
        "request_id": int(request_id),
        "summary": summarize(records, seconds),
        "records": list(records),
    }
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(directory, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(render(payload))
    return directory
