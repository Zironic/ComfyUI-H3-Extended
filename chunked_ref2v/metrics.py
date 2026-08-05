"""Diagnostic metrics for one Chunk B result.

These are not quality scores. They measure whether the carried state arrived -
how closely the new overlap follows the old one in pixels and in latent space,
whether the motion matches, and whether the arm still obeys the source. A
strategy that scores a perfect overlap MAE by freezing the frame is a failure,
which is why the source-adherence and motion numbers sit next to it.

Everything runs on CPU float32 and never allocates a full N x N anything.
"""

import torch


def _f(x):
    return None if x is None else float(x)


def _cpu(t):
    return None if t is None else t.detach().to("cpu", torch.float32)


def mae(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    return _f((a - b).abs().mean())


def per_frame_mae(a, b):
    if a is None or b is None or a.shape != b.shape:
        return None
    dims = tuple(range(1, a.ndim))
    return [_f(v) for v in (a - b).abs().mean(dim=dims)]


def pixel_overlap_metrics(chunk_a_pixels, chunk_b_pixels, geometry):
    """Compare the two renderings of the same global frames.

    Chunk A's frames `S..C-1` and Chunk B's frames `0..O-1` cover identical
    global timestamps, so they are directly comparable.
    """
    a = _cpu(chunk_a_pixels)
    b = _cpu(chunk_b_pixels)
    if a is None or b is None:
        return {}
    s, c = geometry.stride_frames, geometry.chunk_frames
    if a.shape[0] < c or b.shape[0] < geometry.overlap_frames:
        return {"error": "not enough decoded frames to measure the overlap"}
    a_overlap = a[s:c]
    b_overlap = b[:geometry.overlap_frames]
    return {
        "pixel_overlap_mae": mae(a_overlap, b_overlap),
        "pixel_overlap_mae_per_frame": per_frame_mae(a_overlap, b_overlap),
        "first_frame_pixel_mae": mae(a[s:s + 1], b[:1]),
    }


def latent_overlap_metrics(chunk_a_latent, chunk_b_latent, geometry):
    """Does the new generated overlap actually follow the carried latent state?

    Only defined on a profile whose overlap lands on latent boundaries; an
    unaligned profile reports the reason instead of an approximate number.
    """
    from .geometry import UnalignedProfileError

    a = _cpu(chunk_a_latent)
    b = _cpu(chunk_b_latent)
    if a is None or b is None:
        return {}
    try:
        start, count = geometry.overlap_slice()
    except UnalignedProfileError as exc:
        return {"latent_overlap_mae": None, "latent_overlap_note": str(exc)}
    if a.shape[2] < start + count or b.shape[2] < count:
        return {"latent_overlap_mae": None,
                "latent_overlap_note": "latent shorter than the overlap slice"}
    a_overlap = a[:, :, start:start + count]
    b_overlap = b[:, :, :count]
    return {
        "latent_overlap_mae": mae(a_overlap, b_overlap),
        "latent_overlap_mae_per_position": per_frame_mae(
            a_overlap[0].movedim(1, 0), b_overlap[0].movedim(1, 0)),
        "first_position_latent_mae": mae(a[:, :, start:start + 1], b[:, :, :1]),
    }


def motion_metrics(chunk_a_pixels, chunk_b_pixels, geometry):
    """Frame-to-frame deltas over the shared window.

    Position agreement without motion agreement is the failure mode a single
    carried frame is expected to have - it fixes where the subject is, not where
    it was going.
    """
    a = _cpu(chunk_a_pixels)
    b = _cpu(chunk_b_pixels)
    if a is None or b is None:
        return {}
    s, c, o = geometry.stride_frames, geometry.chunk_frames, geometry.overlap_frames
    if a.shape[0] < c or b.shape[0] < o or o < 2:
        return {}
    motion_a = a[s + 1:c] - a[s:c - 1]
    motion_b = b[1:o] - b[:o - 1]
    return {
        "motion_delta_mae": mae(motion_a, motion_b),
        "motion_magnitude_a": _f(motion_a.abs().mean()),
        "motion_magnitude_b": _f(motion_b.abs().mean()),
    }


def source_adherence(chunk_b_pixels, source_chunk_b_pixels):
    """Does the arm still track the source's motion structure?

    Compares frame-difference energy, not appearance - the whole point of a
    ref2v edit is that appearance differs. A near-zero ratio means the output
    stopped moving; a ratio near 1 means it moves like the source does.
    """
    b = _cpu(chunk_b_pixels)
    src = _cpu(source_chunk_b_pixels)
    if b is None or src is None:
        return {}
    n = min(b.shape[0], src.shape[0])
    if n < 2:
        return {}
    d_out = (b[1:n] - b[:n - 1]).abs().mean()
    d_src = (src[1:n] - src[:n - 1]).abs().mean()
    ratio = None if float(d_src) == 0.0 else _f(d_out / d_src)
    return {
        "output_motion_energy": _f(d_out),
        "source_motion_energy": _f(d_src),
        "motion_energy_ratio": ratio,
    }


def collect(*, geometry, chunk_a_latent, chunk_a_pixels,
            chunk_b_latent, chunk_b_pixels, source_chunk_b_pixels=None):
    out = {}
    out.update(pixel_overlap_metrics(chunk_a_pixels, chunk_b_pixels, geometry))
    out.update(latent_overlap_metrics(chunk_a_latent, chunk_b_latent, geometry))
    out.update(motion_metrics(chunk_a_pixels, chunk_b_pixels, geometry))
    out.update(source_adherence(chunk_b_pixels, source_chunk_b_pixels))
    return out


def format_metrics(metrics):
    lines = []
    for key in ("pixel_overlap_mae", "first_frame_pixel_mae",
                "latent_overlap_mae", "first_position_latent_mae",
                "motion_delta_mae", "motion_energy_ratio"):
        value = metrics.get(key)
        if value is not None:
            lines.append("    %-28s %.6f" % (key, value))
    for key in ("latent_overlap_note", "error"):
        if metrics.get(key):
            lines.append("    %-28s %s" % (key, metrics[key]))
    return "\n".join(lines)


def compare_to_baseline(metrics, baseline_metrics):
    """Relative improvement over the control, where both numbers exist."""
    out = {}
    for key in ("pixel_overlap_mae", "first_frame_pixel_mae",
                "latent_overlap_mae", "motion_delta_mae"):
        value, base = metrics.get(key), (baseline_metrics or {}).get(key)
        if value is None or base in (None, 0):
            continue
        out[key + "_vs_baseline"] = _f(value / base)
    return out
