"""Comparison outputs.

Two views, because they answer different questions.

The **overlap comparison** puts source / Chunk A / baseline / experiment in
columns over the same global frames, which is how you see whether the carried
state arrived at all.

The **boundary playback** is the one that actually settles seam quality: Chunk A
running into Chunk B as continuous playback. Motion discontinuity is obvious in
time and nearly invisible in a side-by-side still, so judging seams from the
column view is a good way to ship a visible seam.
"""

import torch

import comfy.utils


def _match(frames, height, width):
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1)


def _pad_to(frames, count):
    if frames.shape[0] >= count:
        return frames[:count]
    tail = frames[-1:].repeat(count - frames.shape[0], 1, 1, 1)
    return torch.cat([frames, tail])


def columns(clips, height=None, width=None):
    """Stack IMAGE batches side by side into one batch.

    `clips` is a list of (label, frames). Labels are returned rather than drawn -
    burning text into the pixels would corrupt the same frames the metrics read.
    """
    clips = [(label, f) for label, f in clips if f is not None and f.shape[0] > 0]
    if not clips:
        return None, []
    height = height or clips[0][1].shape[1]
    width = width or clips[0][1].shape[2]
    count = max(f.shape[0] for _, f in clips)
    tiles = [_pad_to(_match(f.to("cpu", torch.float32), height, width), count)
             for _, f in clips]
    return torch.cat(tiles, dim=2), [label for label, _ in clips]


def overlap_comparison(*, source_pixels, chunk_a_pixels, baseline_pixels,
                       experiment_pixels, geometry):
    """Columns over the shared global frames `S..C-1`."""
    s, c, o = geometry.stride_frames, geometry.chunk_frames, geometry.overlap_frames
    clips = [
        ("source", _slice(source_pixels, s, c)),
        ("chunk A", _slice(chunk_a_pixels, s, c)),
        ("baseline B", _slice(baseline_pixels, 0, o)),
        ("experiment B", _slice(experiment_pixels, 0, o)),
    ]
    return columns(clips)


def boundary_playback(*, chunk_a_pixels, chunk_b_pixels, geometry, lead=12, trail=12):
    """Chunk A running into Chunk B as continuous playback across the seam.

    Chunk A frames `S-lead .. S-1` are global frames before the seam; Chunk B
    frame 0 is global frame `S`, so concatenating them is a continuous timeline
    with the cut exactly at the join.
    """
    if chunk_a_pixels is None or chunk_b_pixels is None:
        return None
    s = geometry.stride_frames
    a = _slice(chunk_a_pixels, max(0, s - lead), s)
    b = _slice(chunk_b_pixels, 0, min(trail, chunk_b_pixels.shape[0]))
    if a is None or b is None:
        return None
    b = _match(b.to("cpu", torch.float32), a.shape[1], a.shape[2])
    return torch.cat([a.to("cpu", torch.float32), b])


def _slice(frames, start, stop):
    if frames is None:
        return None
    stop = min(stop, frames.shape[0])
    if start >= stop:
        return None
    return frames[start:stop]


def contact_sheet(results, geometry, max_experiments=6):
    """One row per experiment, over the overlap window - a quick visual index."""
    rows = []
    labels = []
    for result in list(results)[:max_experiments]:
        pixels = result.get("pixels")
        if pixels is None:
            continue
        rows.append(_pad_to(pixels[:geometry.overlap_frames].to("cpu", torch.float32),
                            geometry.overlap_frames))
        labels.append(result["experiment_id"])
    if not rows:
        return None, []
    height, width = rows[0].shape[1], rows[0].shape[2]
    rows = [_match(r, height, width) for r in rows]
    return torch.cat(rows, dim=1), labels
