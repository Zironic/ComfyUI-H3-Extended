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


def summarize(records, seconds=None, timing=None):
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
        "timing": timing or {
            "enabled": False,
            "call_count": 0,
            "stages": {},
            "total_measured_attention_cuda_seconds": 0.0,
            "request_wall_seconds": None if seconds is None else float(seconds),
            "attention_cuda_to_request_wall_ratio": None,
            "total_measured_dit_block_cuda_seconds": 0.0,
            "dit_block_cuda_to_request_wall_ratio": None,
            "model_forward_call_count": 0,
            "total_model_forward_cuda_seconds": 0.0,
            "model_forward_cuda_to_request_wall_ratio": None,
            "ratio_caveat": (
                "CUDA event timing was disabled; no attention CUDA time was measured."
            ),
        },
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
        "Flex, compatibility fallback, and Sol head dispatch are not enabled.",
        "Timing stages cover the DiT block, activation/MLP stages, attention "
        "projection stages (including fused QKV when selected), direct LUT "
        "construction, V FP8 preparation, Q/K "
        "int8 quantization, and the low-level Sparse Sage kernel. Compiled "
        "runs replace those internal events with one model-call event so "
        "timing does not split the compiled tensor graph.",
    ])
    timing = summary.get("timing") or {}
    if timing.get("enabled") and timing.get("stages"):
        lines.extend([
            "",
            "CUDA timing (deferred): %d attention calls" % timing.get("call_count", 0),
            *[
                "%s: %d calls, sum %.3f ms, mean %.3f ms" % (
                    stage, values["count"], values["sum_ms"], values["mean_ms"]
                )
                for stage, values in timing.get("stages", {}).items()
            ],
            "measured attention CUDA seconds: %.6f" % (
                timing["total_measured_attention_cuda_seconds"]),
            "request wall seconds: %s" % (
                "unknown" if timing.get("request_wall_seconds") is None
                else "%.6f" % timing["request_wall_seconds"]),
            "attention-CUDA/request-wall ratio: %s" % (
                "unknown" if timing.get("attention_cuda_to_request_wall_ratio") is None
                else "%.3f" % timing["attention_cuda_to_request_wall_ratio"]),
            "measured DiT block CUDA seconds: %.6f" % (
                timing.get("total_measured_dit_block_cuda_seconds", 0.0)),
            "DiT-block-CUDA/request-wall ratio: %s" % (
                "unknown" if timing.get("dit_block_cuda_to_request_wall_ratio") is None
                else "%.3f" % timing["dit_block_cuda_to_request_wall_ratio"]),
            "model forward CUDA seconds: %.6f (%d calls)" % (
                timing.get("total_model_forward_cuda_seconds", 0.0),
                timing.get("model_forward_call_count", 0)),
            "model-forward-CUDA/request-wall ratio: %s" % (
                "unknown" if timing.get("model_forward_cuda_to_request_wall_ratio") is None
                else "%.3f" % timing["model_forward_cuda_to_request_wall_ratio"]),
            timing.get("ratio_caveat", ""),
            "Stage times are nested/overlapping; do not sum stage totals.",
        ])
        for step in timing.get("per_step", ()):
            lines.append(
                "step %d (ordinal %d): attention %.6f s, DiT block %.6f s, "
                "model forward %.6f s"
                % (
                    int(step["step_index"]),
                    int(step["ordinal"]),
                    float(step.get("total_measured_attention_cuda_seconds", 0.0)),
                    float(step.get("total_measured_dit_block_cuda_seconds", 0.0)),
                    float(step.get("total_model_forward_cuda_seconds", 0.0)),
                )
            )
            for branch in step.get("branches", ()):
                lines.append(
                    "  branch %s: attention %.6f s, DiT block %.6f s, "
                    "model forward %.6f s"
                    % (
                        ",".join(str(value) for value in branch.get("branch", ())),
                        float(branch.get("total_measured_attention_cuda_seconds", 0.0)),
                        float(branch.get("total_measured_dit_block_cuda_seconds", 0.0)),
                        float(branch.get("total_model_forward_cuda_seconds", 0.0)),
                    )
                )
    return "\n".join(lines) + "\n"


def write_request(output_root, run_tag, timestamp, request_id, records, seconds=None,
                  timing=None):
    tag = validate_run_tag(run_tag)
    directory = os.path.join(output_root, "%s_%s" % (tag, timestamp))
    os.makedirs(directory, exist_ok=False)
    payload = {
        "status": "complete",
        "mode": "sage128",
        "run_tag": tag,
        "request_id": int(request_id),
        "summary": summarize(records, seconds, timing),
        "records": list(records),
    }
    with open(os.path.join(directory, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    with open(os.path.join(directory, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(render(payload))
    return directory
