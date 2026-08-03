"""Decision-oriented metrics over captured H3 attention aggregates.

Everything here is pure: it consumes the aggregates produced by `capture` plus a
`TokenLayout`, and answers the design question directly — for this query block,
how much attention mass does a candidate block mask retain, and how much does a
Top-k dynamic budget add on top.

Two granularities are reported side by side, deliberately:

* *exact* masses come from segment/frame slices and are what the model actually
  attends to;
* *block* masses come from the BLOCK-token KV grid a sparse kernel would work
  on, so a block straddling the mask boundary is retained whole.

The block figures are the ones a real kernel can deliver; the exact figures say
how much of that is genuinely needed.
"""

import numpy as np

from . import layout as h3_layout

KINDS = (h3_layout.KIND_TEXT, h3_layout.KIND_COND, h3_layout.KIND_REF_IMG,
         h3_layout.KIND_REF_AUDIO, h3_layout.KIND_AUDIO, h3_layout.KIND_VIDEO)

# text + keyframe/reference conditioning + the target audio stream: context a
# sparse mask must always keep, on the assumption that dropping cross-modal
# conditioning breaks prompt adherence and AV sync outright
MANDATORY_KINDS = (h3_layout.KIND_TEXT, h3_layout.KIND_COND, h3_layout.KIND_REF_IMG,
                   h3_layout.KIND_REF_AUDIO, h3_layout.KIND_AUDIO)

TOP_K = (4, 8, 16, 32)


def _np(x):
    return x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x)


def local_token_mask(layout, q_frame, adjacent=1):
    """Boolean token mask for the candidate fixed pattern.

    mandatory context + the query's own latent frame + `adjacent` frames either
    side. For non-video queries only the mandatory context is local.
    """
    mask = np.zeros(layout.seq_len, dtype=bool)
    for a, b, kind in layout.segments:
        if kind in MANDATORY_KINDS:
            mask[a:b] = True
    if q_frame is not None:
        lo = max(0, q_frame - adjacent)
        hi = min(layout.latent_t - 1, q_frame + adjacent)
        for t in range(lo, hi + 1):
            a, b = layout.video_frame_range(t)
            mask[a:b] = True
    return mask


def analyze(rec, layout, adjacent=1, spatial_radius=4, top_k=TOP_K):
    """Full metric set for one captured query block."""
    cat = _np(rec["cat_mass"]).mean(0)                 # [n_kinds] head-mean
    frame = _np(rec["frame_mass"]).mean(0)             # [latent_t]
    blocks = _np(rec["block_mass"]).mean(0)            # [n_blocks]
    spatial = _np(rec["spatial_mass"])                 # [frame_rows]
    block = int(rec["block"])
    q_frame = rec.get("frame")

    cat_d = {k: float(cat[i]) for i, k in enumerate(KINDS)}
    mandatory = float(sum(cat_d[k] for k in MANDATORY_KINDS))
    total_video = float(cat_d[h3_layout.KIND_VIDEO])

    out = {
        "layer": rec["layer"], "step": rec["step"], "sigma": rec["sigma"],
        "cond_or_uncond": rec["cond_or_uncond"], "kind": rec["kind"],
        "frame": q_frame, "spatial_offset": rec.get("spatial_offset"),
        "q_start": rec["start"], "q_stop": rec["stop"],
        "cat": cat_d,
        "mandatory": mandatory,
        "text": cat_d[h3_layout.KIND_TEXT],
        "references": float(cat_d[h3_layout.KIND_COND] + cat_d[h3_layout.KIND_REF_IMG]
                            + cat_d[h3_layout.KIND_REF_AUDIO]),
        "target_audio": cat_d[h3_layout.KIND_AUDIO],
        "target_video": total_video,
    }

    # ---- temporal structure -------------------------------------------------
    if q_frame is not None:
        cur = float(frame[q_frame])
        adj_lo = max(0, q_frame - adjacent)
        adj_hi = min(layout.latent_t - 1, q_frame + adjacent)
        adj = float(frame[adj_lo:adj_hi + 1].sum()) - cur
        out["current_frame"] = cur
        out["adjacent_frames"] = adj
        out["other_frames"] = total_video - cur - adj

        dists = np.arange(layout.latent_t) - q_frame
        by_dist = {}
        for lo, hi, label in ((0, 0, "0"), (1, 1, "+/-1"), (2, 2, "+/-2"),
                              (3, 5, "+/-3..5"), (6, 10**6, "> +/-5")):
            sel = (np.abs(dists) >= lo) & (np.abs(dists) <= hi)
            by_dist[label] = float(frame[sel].sum())
        out["by_temporal_distance"] = by_dist

        # ---- spatial structure ---------------------------------------------
        _, ph, pw = layout.video_shape
        rows = np.arange(rec["start"], rec["stop"]) - layout.video_range[0]
        pos = rows % (ph * pw)
        cy, cx = float(np.mean(pos // pw)), float(np.mean(pos % pw))
        yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
        near = (np.abs(yy - cy) <= spatial_radius) & (np.abs(xx - cx) <= spatial_radius)
        near_mass = float(spatial[near.reshape(-1)].sum())
        out["same_spatial_region"] = near_mass
        out["other_spatial"] = total_video - near_mass
        out["spatial_centroid"] = (cy, cx)
        out["spatial_radius"] = spatial_radius
    else:
        out["current_frame"] = out["adjacent_frames"] = 0.0
        out["other_frames"] = total_video

    # ---- candidate mask coverage -------------------------------------------
    tok_mask = local_token_mask(layout, q_frame, adjacent=adjacent)
    exact_local = float(mandatory + out["current_frame"] + out["adjacent_frames"])

    n_blocks = blocks.shape[0]
    pad = n_blocks * block - layout.seq_len
    padded = np.pad(tok_mask, (0, pad)) if pad else tok_mask
    block_is_local = padded.reshape(n_blocks, block).any(axis=1)

    block_local = float(blocks[block_is_local].sum())
    distant = np.sort(blocks[~block_is_local])[::-1]
    cum = np.cumsum(distant)

    out["local_exact"] = exact_local
    out["local_blocks"] = block_local
    out["n_local_blocks"] = int(block_is_local.sum())
    out["n_blocks"] = n_blocks
    out["topk"] = {int(k): float(block_local + (cum[min(k, len(cum)) - 1] if len(cum) else 0.0))
                   for k in top_k}
    out["distant_total"] = float(distant.sum())
    return out


def summarize(analyses, top_k=TOP_K):
    """Worst-case and median coverage across every probed (layer, step, block).

    The minimum is the number that matters: a mask is only safe if its *worst*
    query block retains enough mass, not its average one.
    """
    if not analyses:
        return {}
    def stat(vals):
        v = np.asarray(vals, dtype=np.float64)
        return {"min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}

    out = {
        "n": len(analyses),
        "mandatory": stat([a["mandatory"] for a in analyses]),
        "local_exact": stat([a["local_exact"] for a in analyses]),
        "local_blocks": stat([a["local_blocks"] for a in analyses]),
        "topk": {int(k): stat([a["topk"][int(k)] for a in analyses]) for k in top_k},
    }
    vid = [a for a in analyses if a["kind"] == "video"]
    if vid:
        out["video_only"] = {
            "n": len(vid),
            "current_frame": stat([a["current_frame"] for a in vid]),
            "adjacent_frames": stat([a["adjacent_frames"] for a in vid]),
            "other_frames": stat([a["other_frames"] for a in vid]),
            "same_spatial_region": stat([a["same_spatial_region"] for a in vid]),
        }
    return out


def recommend(summary, target=0.99, top_k=TOP_K):
    """Smallest Top-k budget whose worst-case coverage clears `target`."""
    if not summary:
        return None
    for k in sorted(top_k):
        if summary["topk"][int(k)]["min"] >= target:
            return int(k)
    return None
