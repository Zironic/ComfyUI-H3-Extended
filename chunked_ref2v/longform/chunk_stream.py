"""Overlapping source-chunk iterator with bounded memory.

`long-form plan.md` section 6.3. Each chunk shares its first `O` frames with the
previous chunk's last `O`, so the iterator retains only that tail and reads one
stride of new frames per step. Resident source frames stay at `C + read-ahead`
regardless of how long the video is - which is the whole point, since 3 minutes
at 24 fps is 4320 frames and materializing them as float32 would be ~11 GB.

Frames stay uint8 until a consumer converts one chunk at a time.
"""

from dataclasses import dataclass

import torch

from .frame_source import FFmpegFrameSource, FrameSourceError


@dataclass
class SourceChunk:
    index: int
    global_start: int
    model_frames: int          # always C - what the model is given
    actual_frames: int         # real source frames; < C only on the final chunk
    frames_u8: torch.Tensor
    is_final: bool

    @property
    def padding_frames(self):
        return self.model_frames - self.actual_frames

    def describe(self):
        text = "chunk %d: global %d-%d (%d frames)" % (
            self.index, self.global_start,
            self.global_start + self.actual_frames - 1, self.actual_frames)
        if self.padding_frames:
            text += " + %d padded" % self.padding_frames
        return text


def plan_chunks(total_frames, chunk_frames, stride_frames):
    """How many chunks cover `total_frames`, and where each starts."""
    if total_frames < chunk_frames:
        return [0]
    extra = total_frames - chunk_frames
    count = 1 + (extra + stride_frames - 1) // stride_frames
    return [i * stride_frames for i in range(count)]


def chunk_count_for(total_frames, chunk_frames, stride_frames):
    return len(plan_chunks(total_frames, chunk_frames, stride_frames))


def frames_needed_for(chunk_count, chunk_frames, stride_frames):
    """Source frames a given chunk count consumes."""
    return (chunk_count - 1) * stride_frames + chunk_frames


def iter_source_chunks(video_path, *, chunk_frames, stride_frames, canvas=None,
                       start_frame=0, total_frames=None, fps=24,
                       ffmpeg_location=None):
    """Yield `SourceChunk`s, retaining only the overlap between them.

    `total_frames` bounds the run; without it the iterator runs to EOF. The last
    chunk repeat-pads its final real frame up to `chunk_frames` so the model
    always receives a legal length, and records `actual_frames` so the padding
    can be discarded after sampling.
    """
    overlap = chunk_frames - stride_frames
    if overlap < 0:
        raise ValueError("stride %d exceeds chunk %d" % (stride_frames, chunk_frames))

    source = FFmpegFrameSource(video_path, canvas=canvas, fps=fps,
                               start_frame=start_frame,
                               ffmpeg_location=ffmpeg_location)
    source.open()
    try:
        emitted = 0
        index = 0
        retained = None
        while True:
            if retained is None:
                need = chunk_frames
            else:
                need = stride_frames
            budget = None if total_frames is None else max(0, total_frames - emitted)
            if budget == 0:
                return
            if budget is not None:
                need = min(need, budget)

            fresh = source.read_frames(need) if need > 0 else None
            if fresh is not None and fresh.shape[0] == 0:
                fresh = None

            if retained is None:
                if fresh is None:
                    return
                frames = fresh
            elif fresh is None:
                frames = retained
            else:
                frames = torch.cat([retained, fresh])

            actual = frames.shape[0]
            if actual == 0:
                return
            # A chunk that is only the retained overlap adds no new frames; the
            # previous chunk already covered them.
            if retained is not None and (fresh is None or fresh.shape[0] == 0):
                return

            is_final = (
                source.exhausted and (fresh is None or fresh.shape[0] < need)
            ) or (total_frames is not None and emitted + actual >= total_frames)

            if actual < chunk_frames:
                pad = frames[-1:].repeat(chunk_frames - actual, 1, 1, 1)
                model_frames = torch.cat([frames, pad])
            else:
                model_frames = frames

            # Global frames are numbered from the run's start, not the file's,
            # so chunk geometry matches the harness everywhere else.
            global_start = index * stride_frames
            yield SourceChunk(
                index=index,
                global_start=global_start,
                model_frames=chunk_frames,
                actual_frames=actual,
                frames_u8=model_frames,
                is_final=bool(is_final),
            )

            if is_final:
                return
            retained = frames[-overlap:] if overlap else frames[:0]
            emitted = global_start + stride_frames
            index += 1
    finally:
        source.close()


def verify_stream(video_path, *, chunk_frames, stride_frames, total_frames,
                  canvas=None, start_frame=0, fps=24):
    """Walk the iterator counting frames only - the plan's Stage 0 gate.

    Returns `(chunks, covered_frames)`. Covered frames must equal the requested
    total, or the chunk bookkeeping is wrong in a way that would silently drop
    or duplicate output later.
    """
    chunks = 0
    covered = 0
    for chunk in iter_source_chunks(
            video_path, chunk_frames=chunk_frames, stride_frames=stride_frames,
            canvas=canvas, start_frame=start_frame, total_frames=total_frames,
            fps=fps):
        chunks += 1
        covered = chunk.global_start + chunk.actual_frames
        del chunk
    return chunks, covered
