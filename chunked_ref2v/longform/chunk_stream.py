"""Overlapping source-chunk iterator with bounded memory and exact accounting."""

from dataclasses import dataclass

import torch

from .frame_source import FFmpegFrameSource


@dataclass
class SourceChunk:
    index: int
    global_start: int
    model_frames: int
    actual_frames: int
    frames_u8: torch.Tensor
    is_final: bool

    @property
    def padding_frames(self):
        return self.model_frames - self.actual_frames

    def describe(self):
        text = "chunk %d: global %d-%d (%d frames)" % (
            self.index,
            self.global_start,
            self.global_start + self.actual_frames - 1,
            self.actual_frames,
        )
        if self.padding_frames:
            text += " + %d padded" % self.padding_frames
        return text


def plan_chunks(total_frames, chunk_frames, stride_frames):
    if total_frames <= 0:
        return []
    if total_frames <= chunk_frames:
        return [0]
    extra = total_frames - chunk_frames
    count = 1 + (extra + stride_frames - 1) // stride_frames
    return [i * stride_frames for i in range(count)]


def chunk_count_for(total_frames, chunk_frames, stride_frames):
    return len(plan_chunks(total_frames, chunk_frames, stride_frames))


def frames_needed_for(chunk_count, chunk_frames, stride_frames):
    if chunk_count <= 0:
        return 0
    return (chunk_count - 1) * stride_frames + chunk_frames


def actual_frames_for_chunk(index, *, total_frames, chunk_frames, stride_frames):
    """Real source frames visible in chunk ``index`` before repeat padding."""
    start = index * stride_frames
    return max(0, min(chunk_frames, total_frames - start))


def iter_source_chunks(video_path, *, chunk_frames, stride_frames, canvas=None,
                       start_frame=0, total_frames=None, fps=24,
                       ffmpeg_location=None):
    """Yield overlapping uint8 chunks while reading each source frame once.

    ``total_frames`` is the exact normalized output extent. It counts unique
    timeline frames, not bytes read by each overlapping model window. The old
    implementation subtracted only the emitted stride and could over-read the
    final window by the overlap length.
    """
    overlap = chunk_frames - stride_frames
    if overlap < 0:
        raise ValueError("stride %d exceeds chunk %d" % (stride_frames, chunk_frames))
    if chunk_frames <= 0 or stride_frames <= 0:
        raise ValueError("chunk and stride must be positive")
    if total_frames is not None and total_frames <= 0:
        return

    source = FFmpegFrameSource(
        video_path,
        canvas=canvas,
        fps=fps,
        start_frame=start_frame,
        ffmpeg_location=ffmpeg_location,
    )
    source.open()
    try:
        index = 0
        retained = None
        while True:
            global_start = index * stride_frames
            if total_frames is not None and global_start >= total_frames:
                return

            desired_actual = (
                chunk_frames
                if total_frames is None
                else min(chunk_frames, total_frames - global_start)
            )
            retained_count = 0 if retained is None else int(retained.shape[0])
            need = max(0, desired_actual - retained_count)

            fresh = source.read_frames(need) if need else None
            if fresh is not None and fresh.shape[0] == 0:
                fresh = None

            if retained is None:
                if fresh is None:
                    return
                frames = fresh
            elif fresh is None:
                frames = retained
            else:
                frames = torch.cat([retained, fresh], dim=0)

            actual = min(int(frames.shape[0]), int(desired_actual))
            frames = frames[:actual]
            if actual <= 0:
                return
            if retained is not None and need > 0 and fresh is None:
                return

            hit_requested_end = total_frames is not None and global_start + actual >= total_frames
            hit_source_end = source.exhausted and (fresh is None or fresh.shape[0] < need)
            is_final = bool(hit_requested_end or hit_source_end or actual < chunk_frames)

            if actual < chunk_frames:
                pad = frames[-1:].repeat(chunk_frames - actual, 1, 1, 1)
                model_frames = torch.cat([frames, pad], dim=0)
            else:
                model_frames = frames

            yield SourceChunk(
                index=index,
                global_start=global_start,
                model_frames=chunk_frames,
                actual_frames=actual,
                frames_u8=model_frames,
                is_final=is_final,
            )

            if is_final:
                return
            retained = frames[-overlap:] if overlap else frames[:0]
            index += 1
    finally:
        source.close()


def verify_stream(video_path, *, chunk_frames, stride_frames, total_frames,
                  canvas=None, start_frame=0, fps=24):
    chunks = 0
    covered = 0
    for chunk in iter_source_chunks(
        video_path,
        chunk_frames=chunk_frames,
        stride_frames=stride_frames,
        canvas=canvas,
        start_frame=start_frame,
        total_frames=total_frames,
        fps=fps,
    ):
        chunks += 1
        covered = min(total_frames, chunk.global_start + chunk.actual_frames)
        del chunk
    return chunks, covered
