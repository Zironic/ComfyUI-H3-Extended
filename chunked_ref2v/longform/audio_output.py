"""Bounded generated-audio decode, streaming output, and final mux."""

from __future__ import annotations

import os
import subprocess

import torch

from .frame_source import resolve_ffmpeg
from .writer import VideoWriterError, _partial_path


def audio_sample_rate(audio_vae):
    """Return the decoded waveform rate used by Comfy's audio VAE helper."""

    return int(
        getattr(
            audio_vae,
            "audio_sample_rate_output",
            getattr(audio_vae, "audio_sample_rate", 32000),
        )
    )


def audio_samples_for_frames(frame_count, sample_rate, fps=24):
    """Exact cumulative sample boundary for a video-frame boundary."""

    return round(int(frame_count) / float(fps) * int(sample_rate))


@torch.no_grad()
def decode_audio_chunk(audio_vae, audio_latent):
    """Decode one H3 audio latent into a normalized [1, channels, samples] tensor.

    Comfy's normal audio decode moves the VAE's final channel axis into channel
    position and applies a conservative standard-deviation normalization. The
    same behavior is retained here, chunk by chunk.
    """

    decoded = audio_vae.decode(audio_latent)
    if decoded.ndim != 3:
        raise ValueError(
            "audio VAE returned %s; expected [B, samples, channels]"
            % (tuple(decoded.shape),)
        )

    # Released audio VAEs return [B, samples, channels]. The second branch keeps
    # this tolerant of wrappers that already return [B, channels, samples].
    if decoded.shape[-1] <= 8:
        waveform = decoded.movedim(-1, 1)
    elif decoded.shape[1] <= 8:
        waveform = decoded
    else:
        waveform = decoded.movedim(-1, 1)

    waveform = waveform.to("cpu", torch.float32)
    scale = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    scale[scale < 1.0] = 1.0
    waveform = waveform / scale
    return waveform[:1].contiguous()


class FFmpegAudioWriter:
    """Keep one FFmpeg process open for interleaved float32 PCM batches."""

    def __init__(
        self,
        path,
        *,
        sample_rate,
        channels,
        ffmpeg_location=None,
        codec="pcm_s16le",
    ):
        self.path = path
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.ffmpeg = resolve_ffmpeg(ffmpeg_location)
        self.codec = codec
        self.samples_written = 0
        self._process = None
        self._partial = _partial_path(path)

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        args = [
            self.ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "f32le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            str(self.channels),
            "-i",
            "-",
            "-vn",
            "-c:a",
            self.codec,
            "-y",
            self._partial,
        ]
        self._process = subprocess.Popen(
            args, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return self

    def write(self, waveform):
        if self._process is None or self._process.stdin is None:
            raise VideoWriterError("audio writer is not open")
        if waveform.ndim == 3:
            if waveform.shape[0] != 1:
                raise VideoWriterError(
                    "audio writer only supports batch size 1"
                )
            waveform = waveform[0]
        if waveform.ndim != 2 or waveform.shape[0] != self.channels:
            raise VideoWriterError(
                "unexpected waveform shape %r for %d channels"
                % (tuple(waveform.shape), self.channels)
            )
        interleaved = (
            waveform.detach()
            .to("cpu", torch.float32)
            .transpose(0, 1)
            .contiguous()
        )
        try:
            self._process.stdin.write(interleaved.numpy().tobytes())
        except BrokenPipeError:
            stderr = self._process.stderr.read().decode(
                "utf-8", "replace"
            )
            raise VideoWriterError(
                "ffmpeg audio encoder stopped: %s" % stderr[-2000:]
            ) from None
        self.samples_written += int(waveform.shape[-1])

    def close(self, *, commit=True):
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        stderr = (
            process.stderr.read().decode("utf-8", "replace")
            if process.stderr
            else ""
        )
        code = process.wait()
        if code != 0:
            raise VideoWriterError(
                "ffmpeg audio encoder exited with %d: %s"
                % (code, stderr[-2000:])
            )
        if commit:
            os.replace(self._partial, self.path)
        else:
            try:
                os.remove(self._partial)
            except FileNotFoundError:
                pass

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close(commit=exc_type is None)
        return False


def mux_generated_audio(
    video_path,
    audio_path,
    output_path,
    *,
    frame_count,
    fps=24,
    ffmpeg_location=None,
):
    """Mux generated audio with exact video-duration trimming."""

    ffmpeg = resolve_ffmpeg(ffmpeg_location)
    duration = int(frame_count) / float(fps)
    partial = _partial_path(output_path)
    args = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-t",
        "%.9f" % duration,
        "-shortest",
        "-y",
        partial,
    ]
    process = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        try:
            os.remove(partial)
        except FileNotFoundError:
            pass
        raise VideoWriterError(
            "generated-audio mux failed: %s"
            % process.stderr.decode("utf-8", "replace")[-2000:]
        )
    os.replace(partial, output_path)
    return output_path


__all__ = [
    "FFmpegAudioWriter",
    "audio_sample_rate",
    "audio_samples_for_frames",
    "decode_audio_chunk",
    "mux_generated_audio",
]
