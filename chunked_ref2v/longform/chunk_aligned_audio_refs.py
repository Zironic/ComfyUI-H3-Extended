"""Optional chunk-aligned audio references for LongFormReferenceVideo.

The generic long-form reference node normally encodes every audio reference once
and reuses the complete latent in every model invocation.  This module adds an
opt-in mode that keeps the same Qwen reference structure but replaces each DiT
audio reference with the chronological slice matching the current video chunk.
The slice includes the same leading overlap as the video chunk and never wraps
to sample zero.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass

import torch

from .. import ref_builder
from . import audio_runtime, reference_runner

_ACTIVE = contextvars.ContextVar(
    "h3_longform_chunk_aligned_audio_refs", default=False
)
_INSTALLED = False


def slice_audio_reference(audio, *, start_frame, frame_count, fps=24):
    """Return an exact-duration chronological audio slice, padding with silence.

    The returned waveform keeps the source batch/channel layout.  Negative starts
    are rejected and slices beyond the end are zero-padded rather than wrapped.
    """

    if audio is None:
        return None
    start_frame = int(start_frame)
    frame_count = int(frame_count)
    fps = int(fps)
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count <= 0 or fps <= 0:
        raise ValueError("frame_count and fps must be positive")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    start = round(start_frame / float(fps) * sample_rate)
    stop = round((start_frame + frame_count) / float(fps) * sample_rate)
    wanted = max(0, stop - start)

    clipped = waveform[..., start:min(stop, waveform.shape[-1])]
    if clipped.shape[-1] < wanted:
        clipped = torch.nn.functional.pad(
            clipped, (0, wanted - clipped.shape[-1])
        )
    return {"waveform": clipped, "sample_rate": sample_rate}


@dataclass(frozen=True)
class _AudioSlot:
    block_index: int
    audio: dict
    paired_video: bool


class _ChunkReferenceBlocks:
    """Sequence whose single iteration yields refs for the next chunk.

    The installed sampler calls ``list(self.static_blocks)`` exactly once per
    chunk.  Advancing here avoids replacing the sampler implementation and keeps
    preview/audio-carry behavior owned by ``audio_runtime``.
    """

    def __init__(self, run_obj, base_blocks, slots, audio_vae, cond_cache):
        self.run_obj = run_obj
        self.base_blocks = list(base_blocks)
        self.slots = list(slots)
        self.audio_vae = audio_vae
        self.cond_cache = cond_cache
        self.index = run_obj._first_invalid("samples", 1_000_000)

    def __iter__(self):
        chunk_index = self.index
        self.index += 1
        blocks = [dict(block) for block in self.base_blocks]
        start_frame = chunk_index * self.run_obj.geometry.stride_frames

        for slot in self.slots:
            audio = slice_audio_reference(
                slot.audio,
                start_frame=start_frame,
                frame_count=self.run_obj.geometry.chunk_frames,
                fps=self.run_obj.geometry.fps,
            )
            latent, ref_audio_t = ref_builder.encode_ref_audio(
                self.audio_vae, audio, cond_cache=self.cond_cache
            )
            block = blocks[slot.block_index]
            block["audio_latent"] = latent
            block["ref_audio_t"] = int(ref_audio_t)
            block["kind"] = "video_audio" if slot.paired_video else "audio"

        return iter(blocks)

    def __len__(self):
        return len(self.base_blocks)


def _audio_slots(blocks, ref_videos, ref_video_audios, ref_audios):
    slots = []
    block_index = 0

    # Image-reference blocks come first.
    block_index += sum(
        1 for _ in reference_runner._ordered_values(
            getattr(_CURRENT_INPUTS.get(), "ref_images", None)
        )
    )

    for name, _frames in reference_runner._ordered_values(ref_videos):
        audio = reference_runner._paired_audio(ref_video_audios, name)
        if audio is not None:
            slots.append(_AudioSlot(block_index, audio, True))
        block_index += 1

    for _name, audio in reference_runner._ordered_values(ref_audios):
        slots.append(_AudioSlot(block_index, audio, False))
        block_index += 1

    return slots


@dataclass
class _Inputs:
    ref_images: object = None


_CURRENT_INPUTS = contextvars.ContextVar(
    "h3_longform_chunk_aligned_audio_inputs", default=_Inputs()
)


def _prepare_references(self, *args, **kwargs):
    conditioning, notes = _ORIGINAL_PREPARE(self, *args, **kwargs)
    if not _ACTIVE.get():
        return conditioning, notes

    ref_videos = kwargs.get("ref_videos")
    ref_video_audios = kwargs.get("ref_video_audios")
    ref_audios = kwargs.get("ref_audios")
    slots = _audio_slots(
        self.static_blocks, ref_videos, ref_video_audios, ref_audios
    )
    if not slots:
        return conditioning, notes

    self.static_blocks = _ChunkReferenceBlocks(
        self,
        self.static_blocks,
        slots,
        kwargs["audio_vae"],
        kwargs.get("cond_cache", "auto"),
    )
    return conditioning, notes + [
        "audio references: chunk-aligned chronological slices"
    ]


def install():
    global _INSTALLED, _ORIGINAL_PREPARE
    if _INSTALLED:
        return
    _ORIGINAL_PREPARE = reference_runner.LongFormReferenceRun.prepare_references
    reference_runner.LongFormReferenceRun.prepare_references = _prepare_references
    _INSTALLED = True


def run(*, chunk_align_audio_references=False, **kwargs):
    """Run LongFormReferenceVideo with optional chronological audio slicing."""

    install()
    active_token = _ACTIVE.set(bool(chunk_align_audio_references))
    input_token = _CURRENT_INPUTS.set(
        _Inputs(ref_images=kwargs.get("ref_images"))
    )
    try:
        return audio_runtime.run(**kwargs)
    finally:
        _CURRENT_INPUTS.reset(input_token)
        _ACTIVE.reset(active_token)


_ORIGINAL_PREPARE = None
install()


__all__ = ["install", "run", "slice_audio_reference"]
