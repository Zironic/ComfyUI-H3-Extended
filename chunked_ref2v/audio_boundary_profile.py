"""Shared 24-fps video / 40-Hz audio boundary enforcement.

H3 video frames live on a 24 Hz presentation grid while audio latents live on a
40 Hz grid. Their boundaries coincide every three video frames (five audio
latents). Long-form chunking also has to respect H3's 17k+5 video generation
lengths and exact video-latent overlap boundaries.

This module keeps that policy independent from Comfy node code so the timeline
and both long-form nodes can resolve the same C/O/S profile.
"""

from __future__ import annotations

from .geometry import (
    AUDIO_LATENT_FPS,
    FPS,
    HarnessGeometry,
    UnalignedProfileError,
    aligned_overlap_frames,
    audio_boundary_is_exact,
)

MIN_LONGFORM_CHUNK_FRAMES = 22
MAX_LONGFORM_CHUNK_FRAMES = 362


def audio_aligned_chunk_frames(
    *,
    min_frames=MIN_LONGFORM_CHUNK_FRAMES,
    max_frames=MAX_LONGFORM_CHUNK_FRAMES,
    fps=FPS,
):
    """Legal H3 chunk lengths whose end also lands on an audio boundary."""

    min_frames = int(min_frames)
    max_frames = int(max_frames)
    if min_frames > max_frames:
        raise ValueError("min_frames must not exceed max_frames")
    return [
        frames
        for frames in range(min_frames, max_frames + 1)
        if frames % 17 == 5 and audio_boundary_is_exact(frames, fps=fps)
    ]


def profile_audio_boundaries_are_exact(
    chunk_frames,
    overlap_frames,
    *,
    fps=FPS,
    audio_latent_fps=AUDIO_LATENT_FPS,
):
    """Whether C, O and S=C-O all land on whole audio-latent boundaries."""

    chunk_frames = int(chunk_frames)
    overlap_frames = int(overlap_frames)
    stride_frames = chunk_frames - overlap_frames
    if not 0 < overlap_frames < chunk_frames:
        return False
    return all(
        audio_boundary_is_exact(
            value,
            fps=fps,
            audio_latent_fps=audio_latent_fps,
        )
        for value in (chunk_frames, overlap_frames, stride_frames)
    )


def _nearest(value, candidates, *, prefer_larger_on_tie=True):
    if not candidates:
        raise UnalignedProfileError("no legal audio-aligned candidates")
    value = int(value)
    if prefer_larger_on_tie:
        return min(candidates, key=lambda item: (abs(item - value), -item))
    return min(candidates, key=lambda item: (abs(item - value), item))


def resolve_audio_boundary_profile(
    chunk_frames,
    overlap_frames,
    enabled,
    *,
    min_chunk_frames=MIN_LONGFORM_CHUNK_FRAMES,
    max_chunk_frames=MAX_LONGFORM_CHUNK_FRAMES,
    fps=FPS,
):
    """Return ``(C, O, note)`` with exact video/audio boundaries when enabled.

    ``C`` is first snapped to the nearest value that is legal on both H3's
    17k+5 generation grid and the 24/40 shared time grid. ``O`` is then snapped
    to the nearest exact video-latent suffix whose audio boundary is also exact.
    Because C and O are exact on the shared grid, S=C-O is exact as well.

    Ties prefer the larger value. For overlap this preserves more carried
    context; for chunk length it avoids shortening a requested model window when
    the two legal choices are equally distant.
    """

    chunk_frames = int(chunk_frames)
    overlap_frames = int(overlap_frames)
    if not enabled:
        return chunk_frames, overlap_frames, None

    chunks = audio_aligned_chunk_frames(
        min_frames=min_chunk_frames,
        max_frames=max_chunk_frames,
        fps=fps,
    )
    aligned_chunk = _nearest(chunk_frames, chunks)
    aligned_overlap = aligned_overlap_frames(
        aligned_chunk,
        overlap_frames,
        fps=fps,
    )

    geometry = HarnessGeometry(
        chunk_frames=aligned_chunk,
        overlap_frames=aligned_overlap,
        fps=int(fps),
    ).validate()
    if not profile_audio_boundaries_are_exact(
        geometry.chunk_frames,
        geometry.overlap_frames,
        fps=fps,
    ):
        raise UnalignedProfileError(
            "resolved profile is not exact on all audio boundaries: C=%d O=%d S=%d"
            % (
                geometry.chunk_frames,
                geometry.overlap_frames,
                geometry.stride_frames,
            )
        )

    if (
        aligned_chunk == chunk_frames
        and aligned_overlap == overlap_frames
    ):
        return aligned_chunk, aligned_overlap, None

    return aligned_chunk, aligned_overlap, (
        "align_audio_chunks: profile C=%d O=%d S=%d -> C=%d O=%d S=%d so "
        "chunk, overlap, and stride boundaries all land on whole %d Hz audio "
        "latents at %d fps"
        % (
            chunk_frames,
            overlap_frames,
            chunk_frames - overlap_frames,
            aligned_chunk,
            aligned_overlap,
            aligned_chunk - aligned_overlap,
            AUDIO_LATENT_FPS,
            fps,
        )
    )


__all__ = [
    "MAX_LONGFORM_CHUNK_FRAMES",
    "MIN_LONGFORM_CHUNK_FRAMES",
    "audio_aligned_chunk_frames",
    "profile_audio_boundaries_are_exact",
    "resolve_audio_boundary_profile",
]
