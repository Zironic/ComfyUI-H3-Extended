"""Bounded-memory FFmpeg output writer and source-audio mux."""

from __future__ import annotations

import os
import subprocess

from .frame_source import resolve_ffmpeg


class VideoWriterError(RuntimeError):
    pass


def _partial_path(path, marker="partial"):
    root, ext = os.path.splitext(path)
    return "%s.%s%s" % (root, marker, ext or ".mkv")


class FFmpegVideoWriter:
    """Keep one encoder open and accept finalized uint8 RGB frame batches."""

    def __init__(self, path, *, width, height, fps=24, ffmpeg_location=None,
                 codec="libx264", crf=16, preset="medium"):
        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.ffmpeg = resolve_ffmpeg(ffmpeg_location)
        self.codec = codec
        self.crf = int(crf)
        self.preset = preset
        self.frames_written = 0
        self._process = None
        self._partial = _partial_path(path)

    def open(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        args = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v",
            f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", self.codec, "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", "yuv420p", "-y", self._partial,
        ]
        self._process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        return self

    def write(self, frames_u8):
        if self._process is None or self._process.stdin is None:
            raise VideoWriterError("writer is not open")
        if frames_u8.ndim != 4 or tuple(frames_u8.shape[1:3]) != (self.height, self.width):
            raise VideoWriterError("unexpected frame batch shape %r" % (tuple(frames_u8.shape),))
        payload = frames_u8.detach().to("cpu").contiguous().numpy().tobytes()
        try:
            self._process.stdin.write(payload)
        except BrokenPipeError:
            stderr = self._process.stderr.read().decode("utf-8", "replace")
            raise VideoWriterError("ffmpeg encoder stopped: %s" % stderr[-2000:]) from None
        self.frames_written += int(frames_u8.shape[0])

    def close(self, *, commit=True):
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin:
            process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", "replace") if process.stderr else ""
        code = process.wait()
        if code != 0:
            raise VideoWriterError("ffmpeg encoder exited with %d: %s" % (code, stderr[-2000:]))
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


def mux_source_audio(video_path, source_path, output_path, *, start_frame,
                     frame_count, fps=24, ffmpeg_location=None):
    """Trim the source audio to the exact normalized video interval and mux it."""
    ffmpeg = resolve_ffmpeg(ffmpeg_location)
    start = start_frame / float(fps)
    duration = frame_count / float(fps)
    partial = _partial_path(output_path)
    args = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-ss", f"{start:.9f}", "-t", f"{duration:.9f}",
        "-i", source_path, "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-shortest", "-y", partial,
    ]
    proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        try:
            os.remove(partial)
        except FileNotFoundError:
            pass
        raise VideoWriterError(
            "audio mux failed: %s" % proc.stderr.decode("utf-8", "replace")[-2000:]
        )
    os.replace(partial, output_path)
    return output_path
