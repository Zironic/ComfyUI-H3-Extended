"""Sampler-latent dynamics probe for MiniMax H3.

The MoBA probe measures how much attention output changes when video KV blocks
are removed. This module measures a different source of redundancy: how much
the target video latent itself is still moving from one sampler step to the
next. It observes the sampler callback rather than the DiT input so the metric
is not confounded by sigma-dependent ``calculate_input`` scaling.

Only target-video latents are retained between callbacks, in fp16, and metrics
are reduced at H3's native 1x2x2 DiT patch granularity. No sampling values are
modified.
"""

from __future__ import annotations

import math

import torch


EPS = 1e-12
STABLE_THRESHOLDS = (0.01, 0.02, 0.05)


def resolve_anchor_frames(payload, latent_t):
    """Map explicit H3 keyframes onto latent-time indices.

    Current H3 conditioning exposes pixel-frame ``resolved_frame_index`` plus
    ``frame_count``. Mapping proportionally keeps this future-proof if an
    intermediate keyframe is ever allowed; today's first/last anchors map to
    latent frames 0 and T-1 exactly.
    """
    keyframes = (payload or {}).get("keyframes") or ()
    if not keyframes or latent_t <= 0:
        return []

    frame_count = (payload or {}).get("frame_count")
    out = []
    for kf in keyframes:
        idx = kf.get("resolved_frame_index") if isinstance(kf, dict) else None
        if idx is None:
            continue
        idx = int(idx)
        if frame_count is not None and int(frame_count) > 1 and latent_t > 1:
            pos = round(idx * (latent_t - 1) / (int(frame_count) - 1))
        else:
            pos = 0 if idx <= 0 else latent_t - 1
        out.append(max(0, min(latent_t - 1, int(pos))))
    return sorted(set(out))


def anchor_distance(frame, anchors):
    if frame is None or not anchors:
        return None
    return min(abs(int(frame) - int(a)) for a in anchors)


def _video_from_callback(value, latent_shapes):
    """Return the target video stream from a sampler callback value."""
    if value is None:
        return None

    # A caller may already hand us the nested view.
    if getattr(value, "is_nested", False):
        parts = value.unbind()
        return parts[0] if parts else None

    if latent_shapes and len(latent_shapes) > 1:
        import comfy.utils

        return comfy.utils.unpack_latents(value, latent_shapes)[0]
    return value


def _pad_frame(frame):
    """Pad one [B,C,H,W] latent frame to H3's 2x2 spatial patch."""
    h, w = frame.shape[-2:]
    ph = (h + 1) // 2 * 2
    pw = (w + 1) // 2 * 2
    if ph == h and pw == w:
        return frame
    return torch.nn.functional.pad(frame, (0, pw - w, 0, ph - h))


def _ratio(delta_ss, base_ss):
    return torch.sqrt(delta_ss / torch.clamp(base_ss, min=EPS))


def _stable_fraction(rel, threshold):
    return float((rel <= threshold).float().mean().item()) if rel.numel() else 0.0


def _stream_update(previous, current, thresholds=STABLE_THRESHOLDS):
    """Measure one video stream update at 1x2x2 DiT-patch granularity.

    Parameters are [B,C,T,H,W]. Temporary fp32 work is bounded to one latent
    frame at a time. Returned ``patch_delta_ss`` / ``patch_base_ss`` remain on
    device only long enough for matching query-region reductions; callers
    remove them before serialising.
    """
    if previous is None or current is None:
        return None
    if previous.ndim != 5 or current.ndim != 5:
        raise ValueError("H3 latent dynamics expects video [B,C,T,H,W]")
    if tuple(previous.shape) != tuple(current.shape):
        raise ValueError(
            "latent shape changed between callbacks: %s -> %s"
            % (tuple(previous.shape), tuple(current.shape))
        )

    _b, _c, latent_t, h, w = current.shape
    patch_h = (h + 1) // 2
    patch_w = (w + 1) // 2
    patch_count = patch_h * patch_w

    patch_delta_rows = []
    patch_base_rows = []
    frames = []
    total_delta = 0.0
    total_base = 0.0
    total_values = 0
    stable_counts = {float(th): 0 for th in thresholds}

    for t in range(latent_t):
        prev = _pad_frame(previous[:, :, t].to(torch.float32))
        cur = _pad_frame(current[:, :, t].to(torch.float32))
        delta = cur - prev

        # [B,C,H,W] -> [B,C,patch_h,2,patch_w,2], then aggregate B/C/2/2.
        dss = delta.square().reshape(
            delta.shape[0], delta.shape[1], patch_h, 2, patch_w, 2
        ).sum(dim=(0, 1, 3, 5))
        bss = prev.square().reshape(
            prev.shape[0], prev.shape[1], patch_h, 2, patch_w, 2
        ).sum(dim=(0, 1, 3, 5))
        rel = _ratio(dss, bss)

        frame_delta = float(dss.sum().item())
        frame_base = float(bss.sum().item())
        n_values = int(previous[:, :, t].numel())
        frames.append(
            {
                "frame": int(t),
                "update_rel_l2": math.sqrt(frame_delta / max(frame_base, EPS)),
                "update_rms": math.sqrt(frame_delta / max(n_values, 1)),
                "base_rms": math.sqrt(frame_base / max(n_values, 1)),
                "stable_patch_fraction": {
                    _threshold_key(th): _stable_fraction(rel, th)
                    for th in thresholds
                },
            }
        )

        for th in thresholds:
            stable_counts[float(th)] += int((rel <= th).sum().item())
        total_delta += frame_delta
        total_base += frame_base
        total_values += n_values
        patch_delta_rows.append(dss.reshape(-1))
        patch_base_rows.append(bss.reshape(-1))
        del prev, cur, delta, rel

    patch_delta_ss = (
        torch.cat(patch_delta_rows) if patch_delta_rows else torch.empty(0)
    )
    patch_base_ss = torch.cat(patch_base_rows) if patch_base_rows else torch.empty(0)
    total_patches = latent_t * patch_count
    return {
        "frames": frames,
        "global": {
            "update_rel_l2": math.sqrt(total_delta / max(total_base, EPS)),
            "update_rms": math.sqrt(total_delta / max(total_values, 1)),
            "base_rms": math.sqrt(total_base / max(total_values, 1)),
            "stable_patch_fraction": {
                _threshold_key(th): stable_counts[float(th)] / max(total_patches, 1)
                for th in thresholds
            },
        },
        "patch_delta_ss": patch_delta_ss,
        "patch_base_ss": patch_base_ss,
        "patch_shape": (int(latent_t), int(patch_h), int(patch_w)),
    }


def _threshold_key(threshold):
    return "%g%%" % (100.0 * float(threshold))


def _region_metrics(stream, layout, start, stop, thresholds=STABLE_THRESHOLDS):
    """Reduce an update over the same target-video rows sampled by MoBA."""
    if stream is None or layout is None:
        return None
    v0, v1 = layout.video_range
    start = max(v0, int(start))
    stop = min(v1, int(stop))
    if stop <= start:
        return None

    local_start = start - v0
    local_stop = stop - v0
    dss = stream["patch_delta_ss"][local_start:local_stop]
    bss = stream["patch_base_ss"][local_start:local_stop]
    if not dss.numel():
        return None
    delta = float(dss.sum().item())
    base = float(bss.sum().item())
    rel = _ratio(dss, bss)
    return {
        "update_rel_l2": math.sqrt(delta / max(base, EPS)),
        "stable_patch_fraction": {
            _threshold_key(th): _stable_fraction(rel, th) for th in thresholds
        },
    }


def _strip_transient(stream):
    if stream is None:
        return None
    return {
        "frames": stream["frames"],
        "global": stream["global"],
        "patch_shape": list(stream["patch_shape"]),
    }


def _pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None
        and y is not None
        and math.isfinite(float(x))
        and math.isfinite(float(y))
    ]
    if len(pairs) < 2:
        return None
    ax = sum(x for x, _ in pairs) / len(pairs)
    ay = sum(y for _, y in pairs) / len(pairs)
    dx = [x - ax for x, _ in pairs]
    dy = [y - ay for _, y in pairs]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if den <= EPS:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / den


def _match_region(dynamics_by_step, rec):
    dyn = dynamics_by_step.get(int(rec.get("step", -1)))
    if not dyn:
        return None
    key = (int(rec.get("start", -1)), int(rec.get("stop", -1)))
    for region in dyn.get("query_regions", ()):
        if (int(region.get("start", -2)), int(region.get("stop", -2))) == key:
            return region
    return None


def summarize_dynamics(dynamics, moba_records=()):
    """Compact convergence and attention-sensitivity correlations."""
    if not dynamics:
        return {}

    by_step = {}
    for rec in dynamics:
        frame_rows = rec.get("frames", [])
        row = {
            "anchor_frames": list(rec.get("anchor_frames", [])),
            "sample_update_rel_l2": (
                rec.get("global", {}).get("sample") or {}
            ).get("update_rel_l2"),
            "prediction_update_rel_l2": (
                rec.get("global", {}).get("prediction") or {}
            ).get("update_rel_l2"),
        }
        for stream_name in ("sample", "prediction"):
            distances, updates = [], []
            for fr in frame_rows:
                value = (fr.get(stream_name) or {}).get("update_rel_l2")
                distance = fr.get("anchor_distance")
                if value is not None and distance is not None:
                    distances.append(distance)
                    updates.append(value)
            row["anchor_distance_vs_%s_update_pearson" % stream_name] = _pearson(
                distances, updates
            )
        by_step[str(int(rec["step"]))] = row

    dyn_by_step = {int(r["step"]): r for r in dynamics}
    grouped = {}
    for attn in moba_records or ():
        region = _match_region(dyn_by_step, attn)
        if region is None:
            continue
        layer = int(attn["layer"])
        for budget in attn.get("moba3d", {}).get("budgets", ()):
            bkey = "%.6f" % float(budget["budget"])
            bucket = grouped.setdefault(
                (layer, bkey),
                {
                    "sample": [],
                    "prediction": [],
                    "sparse": [],
                    "oracle": [],
                    "anchor_distance": [],
                },
            )
            bucket["sample"].append(
                (region.get("sample") or {}).get("update_rel_l2")
            )
            bucket["prediction"].append(
                (region.get("prediction") or {}).get("update_rel_l2")
            )
            bucket["sparse"].append(budget.get("sparse_output_rel_l2_mean_head"))
            bucket["oracle"].append(budget.get("oracle_output_rel_l2_mean_head"))
            bucket["anchor_distance"].append(region.get("anchor_distance"))

    correlations = {}
    for (layer, bkey), bucket in grouped.items():
        correlations.setdefault(str(layer), {})[bkey] = {
            "budget": float(bkey),
            "samples": len(bucket["sparse"]),
            "sample_update_vs_sparse_error_pearson": _pearson(
                bucket["sample"], bucket["sparse"]
            ),
            "prediction_update_vs_sparse_error_pearson": _pearson(
                bucket["prediction"], bucket["sparse"]
            ),
            "prediction_update_vs_oracle_error_pearson": _pearson(
                bucket["prediction"], bucket["oracle"]
            ),
            "anchor_distance_vs_sparse_error_pearson": _pearson(
                bucket["anchor_distance"], bucket["sparse"]
            ),
        }

    return {"by_step": by_step, "attention_correlations": correlations}


class LatentDynamicsTracker:
    """Stateful sampler-callback observer for one generation."""

    def __init__(self):
        self.previous_sample = None
        self.previous_prediction = None
        self.previous_step = None

    def close(self):
        self.previous_sample = None
        self.previous_prediction = None
        self.previous_step = None

    def capture(self, run, step, x0, x, total_steps, latent_shapes, queries):
        step = int(step)
        sample = _video_from_callback(x, latent_shapes)
        prediction = _video_from_callback(x0, latent_shapes)
        if sample is None:
            return None

        # Some samplers/callback wrappers can restart their local step counter.
        # Treat that as a new trajectory rather than comparing unrelated states.
        if self.previous_step is not None and step <= self.previous_step:
            self.close()

        sample_now = sample.detach()
        prediction_now = prediction.detach() if prediction is not None else None
        sample_update = _stream_update(self.previous_sample, sample_now)
        prediction_update = _stream_update(self.previous_prediction, prediction_now)

        self.previous_sample = sample_now.to(dtype=torch.float16).clone()
        self.previous_prediction = (
            prediction_now.to(dtype=torch.float16).clone()
            if prediction_now is not None
            else None
        )
        self.previous_step = step

        # First callback establishes the baseline, so there is no step-to-step
        # delta to report yet.
        if sample_update is None and prediction_update is None:
            return None

        anchors = list(getattr(run, "anchor_frames", []) or [])
        frame_rows = []
        latent_t = int(sample_now.shape[2])
        for t in range(latent_t):
            frame_rows.append(
                {
                    "frame": t,
                    "anchor_distance": anchor_distance(t, anchors),
                    "sample": sample_update["frames"][t] if sample_update else None,
                    "prediction": (
                        prediction_update["frames"][t]
                        if prediction_update
                        else None
                    ),
                }
            )

        region_rows = []
        for spec in queries or ():
            if spec.get("kind") != "video":
                continue
            region_rows.append(
                {
                    "kind": "video",
                    "frame": spec.get("frame"),
                    "spatial_offset": spec.get("spatial_offset"),
                    "start": int(spec["start"]),
                    "stop": int(spec["stop"]),
                    "anchor_distance": anchor_distance(spec.get("frame"), anchors),
                    "sample": _region_metrics(
                        sample_update, run.layout, spec["start"], spec["stop"]
                    ),
                    "prediction": _region_metrics(
                        prediction_update, run.layout, spec["start"], spec["stop"]
                    ),
                }
            )

        patch_shape = (
            sample_update["patch_shape"]
            if sample_update
            else prediction_update["patch_shape"]
        )
        return {
            "step": step,
            "total_steps": int(total_steps),
            "anchor_frames": anchors,
            "frames": frame_rows,
            "query_regions": region_rows,
            "global": {
                "sample": _strip_transient(sample_update).get("global")
                if sample_update
                else None,
                "prediction": _strip_transient(prediction_update).get("global")
                if prediction_update
                else None,
            },
            "patch_shape": list(patch_shape),
        }
