"""Per-chunk dumps of what each model invocation actually generated.

``save_frames`` writes the *assembled* output: every chunk contributes only its
stride, so the generated overlap - the frames a seam investigation is about - is
trimmed away before anything reaches disk. This module writes the complete
decoded chunk instead, before any trimming, into a parallel ``diagnostics``
tree:

    diagnostics/chunk_000001/frames/frame_000000.png ... frame_000140.png
    diagnostics/chunk_000001/audio.wav
    diagnostics/chunk_000001/metadata.json

Frame numbering is local to the chunk, so chunk 1 frame 0 is the first carried
frame and chunk 1 frame ``overlap_frames`` is the first frame past the carry.

Nothing here changes generation, so the toggle deliberately stays out of the run
identity: enabling it against a finished run directory replays the stored chunk
latents and dumps them without resampling.
"""

from __future__ import annotations

import json
import os
import struct

import torch

WAVE_FORMAT_IEEE_FLOAT = 3


class DiagnosticSink:
    """Stand-in for a ``LongFormRun`` for callers that do not have one.

    The AV-continuation node runs its own loop instead of driving a
    ``LongFormRun``, but the dumps only ever need a root, a geometry and the
    toggle. ``metadata_overrides`` lets a caller correct fields the shared
    geometry cannot express - a runner with no overlap at all, for instance.
    """

    def __init__(self, root, geometry, *, carry=None, enabled=True,
                 metadata_overrides=None):
        self.root = root
        self.geometry = geometry
        self.carry = carry
        self.diagnostic_dump_chunks = bool(enabled)
        self.metadata_overrides = dict(metadata_overrides or {})


def enabled(run):
    return bool(getattr(run, "diagnostic_dump_chunks", False))


def chunk_dir(run, index):
    path = os.path.join(run.root, "diagnostics", "chunk_%06d" % index)
    os.makedirs(path, exist_ok=True)
    return path


def dump_chunk_video(run, index, pixels):
    """Write every decoded frame of one chunk. Returns the frame count.

    ``pixels`` is the untrimmed float decode, so this includes the overlap tail
    that assembly discards and, on chunk >= 1, the regenerated carry region.
    """
    from PIL import Image

    frames_dir = os.path.join(chunk_dir(run, index), "frames")
    os.makedirs(frames_dir, exist_ok=True)
    frames_u8 = (pixels.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    for i in range(int(frames_u8.shape[0])):
        Image.fromarray(frames_u8[i].numpy()).save(
            os.path.join(frames_dir, "frame_%06d.png" % i)
        )
    count = int(frames_u8.shape[0])
    del frames_u8
    return count


def emit_video(run, index, pixels):
    """Dump the untrimmed decode from inside ``_emit_chunk``, if enabled.

    The frame count is remembered so the audio runtime can record it in the
    chunk metadata without decoding anything a second time.
    """
    if not enabled(run):
        return 0
    count = dump_chunk_video(run, index, pixels)
    store = getattr(run, "_h3_diag_video_frames", None)
    if store is None:
        store = {}
        run._h3_diag_video_frames = store
    store[index] = count
    return count


def video_frames_dumped(run, index):
    return (getattr(run, "_h3_diag_video_frames", None) or {}).get(index, 0)


def dump_chunk_audio(run, index, waveform, sample_rate):
    """Write the complete decoded chunk waveform as float32 PCM.

    Float PCM rather than the AAC the final mux uses: codec artifacts have no
    place in the thing being diagnosed. Written directly instead of through
    ``FFmpegAudioWriter`` so a diagnostic dump never depends on ffmpeg.
    """
    if waveform is None:
        return 0
    path = os.path.join(chunk_dir(run, index), "audio.wav")
    return _write_float_wav(path, waveform, sample_rate)


def _write_float_wav(path, waveform, sample_rate):
    data = waveform.detach().to("cpu", torch.float32)
    if data.ndim == 3:
        data = data[0]
    if data.ndim != 2:
        raise ValueError(
            "unexpected waveform shape %r" % (tuple(waveform.shape),)
        )
    channels = int(data.shape[0])
    samples = int(data.shape[1])
    payload = data.transpose(0, 1).contiguous().numpy().tobytes()

    byte_rate = int(sample_rate) * channels * 4
    header = b"".join([
        b"RIFF",
        struct.pack("<I", 36 + len(payload)),
        b"WAVEfmt ",
        struct.pack(
            "<IHHIIHH", 16, WAVE_FORMAT_IEEE_FLOAT, channels,
            int(sample_rate), byte_rate, channels * 4, 32,
        ),
        b"data",
        struct.pack("<I", len(payload)),
    ])
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "wb") as fh:
        fh.write(header)
        fh.write(payload)
    os.replace(tmp, path)
    return samples


def dump_chunk_metadata(run, index, *, global_start_frame, committed_frames,
                        video_frames, video_latent=None, audio_latent=None,
                        audio_samples=0, audio_sample_rate=None):
    """Write the numbers an analysis script needs to locate the seam regions."""
    geometry = run.geometry
    payload = {
        "chunk": int(index),
        "fps": int(geometry.fps),
        "chunk_frames": int(geometry.chunk_frames),
        "overlap_frames": int(geometry.overlap_frames),
        "stride_frames": int(geometry.stride_frames),
        "carry": getattr(run, "carry", None),
        "global_start_frame": int(global_start_frame),
        "committed_frames": int(committed_frames),
        "video_frames": int(video_frames),
        "video_latent_t": (
            int(video_latent.shape[2]) if video_latent is not None else None
        ),
        "audio_latent_t": (
            int(audio_latent.shape[-1]) if audio_latent is not None else None
        ),
        "audio_sample_rate": (
            int(audio_sample_rate) if audio_sample_rate else None
        ),
        "audio_samples": int(audio_samples),
        # Local frame indices inside this chunk.
        "carry_local_frames": [0, int(geometry.overlap_frames)],
        "previous_chunk_overlap_local_frames": [
            int(geometry.stride_frames), int(geometry.chunk_frames),
        ],
    }
    try:
        start, count = geometry.overlap_slice()
        payload["video_overlap_latent"] = [start, start + count]
    except Exception:
        payload["video_overlap_latent"] = None
    payload.update(getattr(run, "metadata_overrides", None) or {})
    # Written here rather than through runner._atomic_json so this module stays
    # importable without Comfy loaded - the dumps are also read by standalone
    # analysis scripts.
    path = os.path.join(chunk_dir(run, index), "metadata.json")
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    os.replace(tmp, path)
    return payload


__all__ = [
    "DiagnosticSink",
    "chunk_dir",
    "emit_video",
    "video_frames_dumped",
    "dump_chunk_audio",
    "dump_chunk_metadata",
    "dump_chunk_video",
    "enabled",
]
