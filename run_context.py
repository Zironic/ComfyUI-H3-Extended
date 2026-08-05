"""What the H3 nodes were asked to produce, so the VRAM guard can say why it cancelled.

The guard fires inside the DiT forward, where the only things in scope are
tensors — by then the interesting numbers (requested frame length, source
resolutions of every reference image and video, the canvas each was resized to)
have already been consumed and thrown away by the conditioning nodes. A bare
"cancelled at 612 MB free" tells you nothing about which knob to turn.

So the conditioning nodes deposit their inputs here as they run, and the guard
prints them alongside the memory picture.

Records are keyed by node id and overwritten on re-execution, so a workflow that
runs H3 twice reports the second run's numbers, not both. Since records outlive
the prompt that wrote them, each one also stores the video latent shape it
produced; the guard compares that against the tensor actually being denoised and
labels any record that does not match, rather than presenting a leftover from a
previous workflow as if it described the current run.
"""

import threading
import time

_lock = threading.Lock()
_records = {}

MAX_RECORDS = 32


def record(node, unique_id, fields, video_latent_shape=None):
    """Store one node's inputs.

    `fields` is a list of (label, value) pairs, kept in the order given so the
    log reads the way the node's widgets do.
    """
    key = (node, str(unique_id))
    with _lock:
        if len(_records) >= MAX_RECORDS and key not in _records:
            oldest = min(_records, key=lambda k: _records[k]["when"])
            del _records[oldest]
        _records[key] = {
            "node": node,
            "unique_id": unique_id,
            "when": time.time(),
            "fields": list(fields),
            "video_latent_shape": tuple(video_latent_shape) if video_latent_shape else None,
        }


def clear():
    with _lock:
        _records.clear()


def _same_latent(got, want):
    """Compare latent shapes ignoring batch.

    The sampler batches cond and uncond together, so the tensor in the forward
    pass routinely has a larger batch dim than the latent the node created.
    Everything after it — channels, time, spatial — must match.
    """
    if got is None or want is None:
        return True
    return got[1:] == want[1:]


def describe(video_latent_shape=None, indent="  "):
    """Format every record as log lines, newest first.

    When `video_latent_shape` is given, records that produced a different shape
    are flagged: they are almost certainly from an earlier run.
    """
    with _lock:
        records = sorted(_records.values(), key=lambda r: r["when"], reverse=True)

    want = tuple(video_latent_shape) if video_latent_shape else None
    lines = []
    for rec in records:
        got = rec["video_latent_shape"]
        note = ""
        if not _same_latent(got, want):
            note = "  [stale: produced latent %s, not the %s being sampled]" % (
                list(got), list(want))
        lines.append("%s%s [node %s]:%s" % (indent, rec["node"], rec["unique_id"], note))
        for label, value in rec["fields"]:
            lines.append("%s%s%s: %s" % (indent, indent, label, value))
    return lines


def node_id(node_cls):
    """The executing node's graph id, or None.

    `ComfyNode.hidden` is only populated on the clone the executor runs, so this
    stays None-tolerant: a missing id costs a label in a log line, and must never
    be able to fail a generation.
    """
    hidden = getattr(node_cls, "hidden", None)
    return getattr(hidden, "unique_id", None) if hidden is not None else None


def image_res(image):
    """`WxH` for a ComfyUI IMAGE batch, with the frame count when it is a video."""
    if image is None:
        return "none"
    try:
        n, h, w = image.shape[0], image.shape[1], image.shape[2]
    except (AttributeError, IndexError):
        return "unreadable"
    return "%dx%d" % (w, h) if n == 1 else "%dx%d x%d frames" % (w, h, n)


def audio_desc(audio):
    """`Ns @ SR Hz` for a ComfyUI AUDIO dict."""
    if audio is None:
        return "none"
    try:
        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        return "%.2fs @ %d Hz, %d ch" % (
            waveform.shape[-1] / float(sr), sr, waveform.shape[1])
    except (AttributeError, IndexError, KeyError, TypeError, ZeroDivisionError):
        return "unreadable"
