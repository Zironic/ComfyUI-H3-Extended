"""Chunk persistence and the resume scan for N+1 AV continuation.

A 10-chunk run costs about an hour. Editing prompt 7 should cost the four
chunks from 7 onward, not all ten. That requires three things this module
provides: every completed chunk's latents on disk, a rule for which stored
chunks are still valid, and a way to rebuild chunk k-1's reference from storage
that matches what an uninterrupted run would have used.

Validity is deliberately three-part:

    valid(k) = prompt digest matches
             ∧ seed matches
             ∧ recorded parent digest == digest of the chunk on disk at k-1

The parent digest is what makes this safe. Prompt and seed alone say a chunk
*could* have been generated from the current plan; the parent digest says it
actually was generated from the chunk that is still sitting at k-1. Without it,
a directory edited twice in different ways can splice chunk 7 onto a chunk 6 it
never saw.

N+1 is causal, so invalidation is always a suffix: chunks after a changed chunk
inherited its world state and are stale even when their own prompt is untouched.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os

try:
    from ..geometry import (
        VAE_GROUP_FRAMES,
        VAE_GROUP_POSITIONS,
        latent_frame_spans,
        video_latent_t,
    )
except ImportError:  # the self-tests import this file as a top-level module
    from geometry import (
        VAE_GROUP_FRAMES,
        VAE_GROUP_POSITIONS,
        latent_frame_spans,
        video_latent_t,
    )

LOG_PREFIX = "[H3 Extended] n+1 resume"

# Bump when the stored payload or the validity rule changes in a way that makes
# older directories unsafe to reuse.
CHUNK_SCHEMA = 3

SAMPLES_DIR = "samples"

# The model's own floor for a reference video (`encode_video_ref`). This reports
# geometric legality only - the planner applies the practical 2-15 s bounds.
MIN_REFERENCE_FRAMES = 5


# ------------------------------------------------------------------ geometry

def legal_reference_frames(chunk_frames):
    """Reference tails whose decode is exact: a whole number of VAE groups.

    The VAE decodes independent 17-frame groups, so a tail can be decoded on its
    own only when it begins on a group boundary - i.e. when `C - R` is a whole
    number of groups. That makes the legal R values exactly those satisfying
    `R % 17 == C % 17`, which for a legal chunk length is the familiar `17k+5`.
    """
    chunk_frames = int(chunk_frames)
    return [
        r for r in range(MIN_REFERENCE_FRAMES, chunk_frames)
        if (chunk_frames - r) % VAE_GROUP_FRAMES == 0
    ]


def group_aligned_slice(chunk_frames, video_reference_frames):
    """`(latent_start, latent_count)` decoding to exactly `video_reference_frames`.

    Raises rather than silently decoding a partial group, because the failure is
    invisible: a mid-group start still returns frames, just not the frames a
    full decode would have produced at those positions.
    """
    chunk_frames = int(chunk_frames)
    video_reference_frames = int(video_reference_frames)
    if not 0 < video_reference_frames <= chunk_frames:
        raise ValueError("video_reference_frames must be in (0, chunk_frames]")
    if (chunk_frames - video_reference_frames) % VAE_GROUP_FRAMES:
        raise ValueError(
            "R=%d is not VAE-group aligned for C=%d; the tail would start "
            "mid-group and decode incorrectly. Legal: %s"
            % (video_reference_frames, chunk_frames,
               ", ".join(str(v) for v in legal_reference_frames(chunk_frames))))

    latent_t = video_latent_t(chunk_frames)
    groups_before = (chunk_frames - video_reference_frames) // VAE_GROUP_FRAMES
    latent_start = groups_before * VAE_GROUP_POSITIONS
    spans = latent_frame_spans(latent_t)
    covered = sum(spans[latent_start:])
    if covered != video_reference_frames:
        raise ValueError(
            "internal: positions %d-%d cover %d frames, expected %d"
            % (latent_start, latent_t - 1, covered, video_reference_frames))
    return latent_start, latent_t - latent_start


# ------------------------------------------------------------------ digests

def prompt_digest(text):
    """Digest of a *compiled* chunk prompt.

    Compiled, not the raw timeline entry: the compiled string embeds the global
    prompt, so editing the global block invalidates from chunk 0 for free.
    """
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def latent_digest(tensor):
    if tensor is None:
        return None
    try:
        from .manifest import tensor_digest
    except ImportError:
        from manifest import tensor_digest
    return tensor_digest(tensor)


def chunk_digest_from_shas(video_sha, audio_sha):
    return hashlib.sha256(
        (str(video_sha) + "\n" + str(audio_sha)).encode("ascii")
    ).hexdigest()


# ------------------------------------------------------------------ storage

def chunk_path(root, index, suffix=".safetensors"):
    return os.path.join(root, SAMPLES_DIR, "%06d%s" % (int(index), suffix))


def _meta_path(root, index):
    return chunk_path(root, index, ".json")


def save_chunk(root, index, *, video_latent, audio_latent, seed, prompt_sha,
               parent_sha, video_reference_frames,
               audio_reference_latents, chunk_frames):
    """Persist one completed chunk plus everything the scan needs to judge it."""
    try:
        from .runner import _save
    except ImportError:
        from runner import _save

    os.makedirs(os.path.join(root, SAMPLES_DIR), exist_ok=True)
    _save(chunk_path(root, index), {
        "video_latent": video_latent,
        "audio_latent": audio_latent,
    })
    video_sha = latent_digest(video_latent)
    audio_sha = latent_digest(audio_latent)
    if video_reference_frames is None or audio_reference_latents is None:
        raise ValueError("both resolved N+1 reference lengths are required")
    meta = {
        "schema": CHUNK_SCHEMA,
        "index": int(index),
        "seed": int(seed),
        "prompt_sha256": prompt_sha,
        "parent_sha256": parent_sha,
        "video_reference_frames": int(video_reference_frames),
        "audio_reference_latents": int(audio_reference_latents),
        "chunk_frames": int(chunk_frames),
        "video_sha256": video_sha,
        "audio_sha256": audio_sha,
        "chunk_sha256": chunk_digest_from_shas(video_sha, audio_sha),
    }
    tmp = _meta_path(root, index) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    os.replace(tmp, _meta_path(root, index))
    return meta


def load_chunk(root, index):
    """`(tensors, meta)` for a stored chunk, or `(None, None)`."""
    try:
        from .runner import _load
    except ImportError:
        from runner import _load

    meta_path = _meta_path(root, index)
    if not os.path.exists(meta_path):
        return None, None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return None, None
    tensors = _load(chunk_path(root, index))
    if (
        tensors is None
        or "video_latent" not in tensors
        or "audio_latent" not in tensors
    ):
        return None, None
    return tensors, meta


def invalidate_from(root, start, chunk_count):
    """Drop the whole suffix. N+1 is causal; a later chunk inherits earlier state."""
    removed = []
    for index in range(int(start), int(chunk_count)):
        for suffix in (".safetensors", ".json"):
            path = chunk_path(root, index, suffix)
            try:
                os.remove(path)
                removed.append(path)
            except OSError:
                pass
    if removed:
        logging.info("%s invalidated %d chunk artifact(s) from index %d",
                     LOG_PREFIX, len(removed), start)
    return removed


# ------------------------------------------------------------------ the scan

def resume_point(root, *, chunk_count, chunk_digests, chunk_seeds,
                 video_reference_frames, audio_reference_latents,
                 chunk_frames):
    """Index of the first chunk that must be regenerated.

    Returns `chunk_count` when every chunk on disk is still valid, in which case
    the run has nothing to sample and only needs reassembly.
    """
    if video_reference_frames is None or audio_reference_latents is None:
        raise ValueError("both resolved N+1 reference lengths are required")
    chunk_count = int(chunk_count)
    if len(chunk_digests) != chunk_count or len(chunk_seeds) != chunk_count:
        raise ValueError("resume identity must contain one digest and seed per chunk")
    parent_sha = None
    for index in range(chunk_count):
        tensors, meta = load_chunk(root, index)
        if meta is None:
            return index
        if int(meta.get("schema", -1)) != CHUNK_SCHEMA:
            return index
        if int(meta.get("index", -1)) != index:
            return index
        if int(meta.get("chunk_frames", -1)) != int(chunk_frames):
            return index
        if int(meta.get("video_reference_frames", -1)) != int(video_reference_frames):
            return index
        if int(meta.get("audio_reference_latents", -1)) != int(audio_reference_latents):
            return index
        if meta.get("prompt_sha256") != chunk_digests[index]:
            return index
        if int(meta.get("seed", -1)) != int(chunk_seeds[index]):
            return index
        if meta.get("parent_sha256") != parent_sha:
            # Chunk k records the digest of the chunk it actually continued
            # from. A mismatch means the prefix was rewritten underneath it.
            return index
        video_sha = latent_digest(tensors["video_latent"])
        audio_sha = latent_digest(tensors["audio_latent"])
        if meta.get("video_sha256") != video_sha:
            return index
        if meta.get("audio_sha256") != audio_sha:
            return index
        chunk_sha = chunk_digest_from_shas(video_sha, audio_sha)
        if meta.get("chunk_sha256") != chunk_sha:
            return index
        parent_sha = chunk_sha
    return chunk_count


def describe(root, *, chunk_count, resume_from):
    reused = max(0, int(resume_from))
    return ("%s reusing %d/%d stored chunk(s); sampling %d onward (%s)"
            % (LOG_PREFIX, reused, chunk_count, chunk_count - reused, root))
