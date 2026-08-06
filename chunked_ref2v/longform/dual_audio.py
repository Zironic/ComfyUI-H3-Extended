"""Final mux for generated and original source audio as separate tracks."""

from __future__ import annotations

import os
import subprocess

from .frame_source import resolve_ffmpeg
from .writer import VideoWriterError, _partial_path


def build_dual_audio_mux_args(
    ffmpeg,
    video_path,
    generated_audio_path,
    source_path,
    output_path,
    *,
    start_frame,
    frame_count,
    fps=24,
):
    """Build an FFmpeg command with generated audio first and source second.

    Track 0 is generated and marked default. Track 1 is the synchronized source
    audio and remains selectable. Both tracks are trimmed to the exact normalized
    video interval.
    """

    start = int(start_frame) / float(fps)
    duration = int(frame_count) / float(fps)
    partial = _partial_path(output_path)
    return [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-i",
        generated_audio_path,
        "-ss",
        "%.9f" % start,
        "-i",
        source_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-map",
        "2:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a:0",
        "192k",
        "-b:a:1",
        "192k",
        "-metadata:s:a:0",
        "title=Generated audio",
        "-metadata:s:a:1",
        "title=Source audio",
        "-disposition:a:0",
        "default",
        "-disposition:a:1",
        "0",
        "-t",
        "%.9f" % duration,
        "-shortest",
        "-movflags",
        "+faststart",
        "-y",
        partial,
    ]


def mux_generated_and_source_audio(
    video_path,
    generated_audio_path,
    source_path,
    output_path,
    *,
    start_frame,
    frame_count,
    fps=24,
    ffmpeg_location=None,
):
    """Mux generated/default and source/selectable audio into one MP4."""

    ffmpeg = resolve_ffmpeg(ffmpeg_location)
    partial = _partial_path(output_path)
    args = build_dual_audio_mux_args(
        ffmpeg,
        video_path,
        generated_audio_path,
        source_path,
        output_path,
        start_frame=start_frame,
        frame_count=frame_count,
        fps=fps,
    )
    process = subprocess.run(
        args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if process.returncode != 0:
        try:
            os.remove(partial)
        except FileNotFoundError:
            pass
        raise VideoWriterError(
            "dual-audio mux failed: %s"
            % process.stderr.decode("utf-8", "replace")[-2000:]
        )
    os.replace(partial, output_path)
    return output_path


__all__ = [
    "build_dual_audio_mux_args",
    "mux_generated_and_source_audio",
]
