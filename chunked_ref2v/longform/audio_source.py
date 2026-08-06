"""Bounded FFmpeg PCM extraction for per-chunk Ref2V audio conditioning."""

from __future__ import annotations

import subprocess

import torch

from .frame_source import resolve_ffmpeg


class AudioSourceError(RuntimeError):
    pass


def _missing_audio_error(stderr):
    """Whether FFmpeg is reporting that the optional input has no audio stream."""

    text = (stderr or "").lower()
    return any(
        marker in text
        for marker in (
            "does not contain any stream",
            "matches no streams",
            "stream map '0:a:0' matches no streams",
            "output file #0 does not contain any stream",
        )
    )


def read_audio_window(
    path,
    *,
    start_frame,
    frame_count,
    fps=24,
    sample_rate=32000,
    channels=2,
    ffmpeg_location=None,
    missing_ok=False,
):
    """Return a Comfy AUDIO dict for one exact video-frame interval.

    The process seeks directly to each chunk. Pass A is VAE-bound, so process
    startup is negligible and this avoids retaining duration-proportional PCM.
    Short source audio is zero-padded to keep video/audio reference geometry
    deterministic. When ``missing_ok`` is true, a video with no audio stream
    returns ``None`` rather than failing the complete long-form run.
    """

    ffmpeg = resolve_ffmpeg(ffmpeg_location)
    start_seconds = start_frame / float(fps)
    sample_count = round(frame_count * sample_rate / float(fps))
    duration = sample_count / float(sample_rate)
    args = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_seconds:.9f}",
        "-i",
        path,
        "-map",
        "0:a:0?",
        "-vn",
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        "-t",
        f"{duration:.9f}",
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = proc.stderr.decode("utf-8", "replace")
    if proc.returncode != 0:
        if missing_ok and _missing_audio_error(stderr):
            return None
        raise AudioSourceError(
            "ffmpeg audio extraction failed: %s" % stderr[-2000:]
        )
    if not proc.stdout and missing_ok:
        return None

    values = torch.frombuffer(bytearray(proc.stdout), dtype=torch.float32)
    complete = (values.numel() // channels) * channels
    values = values[:complete]
    if complete:
        waveform = values.reshape(-1, channels).transpose(0, 1).contiguous()
    else:
        waveform = torch.zeros(channels, 0, dtype=torch.float32)
    if waveform.shape[-1] < sample_count:
        waveform = torch.nn.functional.pad(
            waveform, (0, sample_count - waveform.shape[-1])
        )
    else:
        waveform = waveform[..., :sample_count]
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


__all__ = [
    "AudioSourceError",
    "_missing_audio_error",
    "read_audio_window",
]
