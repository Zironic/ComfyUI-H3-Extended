"""Comparison outputs.

Two views, because they answer different questions.

The **overlap comparison** puts source / Chunk A / baseline / experiment in
columns over the same global frames, which is how you see whether the carried
state arrived at all.

The **boundary playback** is the one that actually settles seam quality: Chunk A
running into Chunk B as continuous playback. Motion discontinuity is obvious in
time and nearly invisible in a side-by-side still, so judging seams from the
column view is a good way to ship a visible seam.

Comfy IMAGE outputs are float32, so returning source-resolution previews can
consume gigabytes of RAM. Preview builders cap one tile's longest edge; complete
frames remain in the artifact directory.
"""

import torch

import comfy.utils

DEFAULT_PREVIEW_LONG_EDGE = 512


def _match(frames, height, width):
    if frames.shape[1] == height and frames.shape[2] == width:
        return frames
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1)


def _preview_size(height, width, max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
    if max_long_edge is None or max_long_edge <= 0 or max(height, width) <= max_long_edge:
        return int(height), int(width)
    scale = max_long_edge / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def preview_clip(frames, max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
    """Return a CPU float32 clip bounded for in-memory node outputs."""
    if frames is None or frames.shape[0] == 0:
        return frames
    height, width = _preview_size(frames.shape[1], frames.shape[2], max_long_edge)
    return _match(frames.to("cpu", torch.float32), height, width)


def _pad_to(frames, count):
    if frames.shape[0] >= count:
        return frames[:count]
    tail = frames[-1:].repeat(count - frames.shape[0], 1, 1, 1)
    return torch.cat([frames, tail])


def columns(clips, height=None, width=None, max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
    """Stack IMAGE batches side by side into one bounded preview batch."""
    clips = [(label, f) for label, f in clips if f is not None and f.shape[0] > 0]
    if not clips:
        return None, []
    base_h = int(height or clips[0][1].shape[1])
    base_w = int(width or clips[0][1].shape[2])
    height, width = _preview_size(base_h, base_w, max_long_edge)
    count = max(f.shape[0] for _, f in clips)
    tiles = [_pad_to(_match(f.to("cpu", torch.float32), height, width), count)
             for _, f in clips]
    return torch.cat(tiles, dim=2), [label for label, _ in clips]


def overlap_comparison(*, source_pixels, chunk_a_pixels, baseline_pixels,
                       experiment_pixels, geometry,
                       max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
    s, c, o = geometry.stride_frames, geometry.chunk_frames, geometry.overlap_frames
    clips = [
        ("source", _slice(source_pixels, s, c)),
        ("chunk A", _slice(chunk_a_pixels, s, c)),
        ("baseline B", _slice(baseline_pixels, 0, o)),
        ("experiment B", _slice(experiment_pixels, 0, o)),
    ]
    return columns(clips, max_long_edge=max_long_edge)


def boundary_playback(*, chunk_a_pixels, chunk_b_pixels, geometry, lead=12, trail=12,
                      max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
    if chunk_a_pixels is None or chunk_b_pixels is None:
        return None
    s = geometry.stride_frames
    a = _slice(chunk_a_pixels, max(0, s - lead), s)
    b = _slice(chunk_b_pixels, 0, min(trail, chunk_b_pixels.shape[0]))
    if a is None or b is None:
        return None
    height, width = _preview_size(a.shape[1], a.shape[2], max_long_edge)
    a = _match(a.to("cpu", torch.float32), height, width)
    b = _match(b.to("cpu", torch.float32), height, width)
    return torch.cat([a, b])


def _slice(frames, start, stop):
    if frames is None:
        return None
    stop = min(stop, frames.shape[0])
    if start >= stop:
        return None
    return frames[start:stop]


def contact_sheet(results, geometry, max_experiments=6,
                  max_long_edge=DEFAULT_PREVIEW_LONG_EDGE):
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
    height, width = _preview_size(rows[0].shape[1], rows[0].shape[2], max_long_edge)
    rows = [_match(r, height, width) for r in rows]
    return torch.cat(rows, dim=1), labels
