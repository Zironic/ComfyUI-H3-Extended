"""Bounded, seekable source decoding for long-form chunked Ref2V.

Stage 0 of `long-form plan.md`. The existing harness takes the whole source as
one materialized Comfy `IMAGE`, which is fine for a two-chunk laboratory and
impossible for real input: 41,552 frames of 1080p float32 is ~500 GB, and even
at 608x352 it is ~105 GB. This decodes sequentially instead, handing back one
bounded window at a time.

It also fixes a second problem measured on the way here. `VHS_LoadVideoPath`
asked for 130 frames at the three-minute mark of a 29-minute AV1 file and had
not returned after two minutes, because it walks the file rather than seeking.
FFmpeg's `-ss` before `-i` seeks on the container, so the same request is
near-instant regardless of how far in the window sits.

Two deliberate deviations from the plan as written:

* **ffprobe is not used.** `imageio_ffmpeg` bundles `ffmpeg` but not `ffprobe`,
  and requiring a system FFmpeg install would be a new dependency for no gain.
  Metadata comes from PyAV, which is already a hard dependency of this stack and
  reads the same libav containers.
* **The canvas is optional.** `canvas=None` emits source-resolution frames, so
  existing scaling nodes keep working unchanged during the transition. Pinning
  is still available and still required for real long-form runs.

The authoritative frame count is what this emits, never the container's estimate.
"""

import json
import os
import shutil
import subprocess
import sys
import threading

FPS = 24
_BYTES_PER_PIXEL = 3


class FrameSourceError(RuntimeError):
    """Decoding failed in a way that must not be mistaken for end of stream."""


def resolve_ffmpeg(explicit=None):
    """Locate an ffmpeg binary: explicit, then PATH, then the bundled copy."""
    if explicit:
        if not os.path.exists(explicit):
            raise FrameSourceError("ffmpeg not found at %r" % explicit)
        return explicit
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FrameSourceError(
            "no ffmpeg on PATH and imageio_ffmpeg is unavailable (%s); pass "
            "ffmpeg_location explicitly" % exc) from None


class VideoMetadata:
    """Container facts. `estimated_frames` is an estimate and named as one."""

    def __init__(self, path, width, height, source_fps, duration, has_audio,
                 rotation=0):
        self.path = path
        self.source_width = width
        self.source_height = height
        self.source_fps = source_fps
        self.duration = duration
        self.has_audio = has_audio
        self.rotation = rotation

    @property
    def estimated_frames(self):
        if not self.duration:
            return None
        return int(self.duration * FPS)

    def as_dict(self):
        return {
            "path": self.path,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_fps": self.source_fps,
            "duration": self.duration,
            "has_audio": self.has_audio,
            "rotation": self.rotation,
            "estimated_frames_at_24fps": self.estimated_frames,
        }

    def __repr__(self):
        return ("VideoMetadata(%dx%d @ %.3f fps, %.1fs, audio=%s)"
                % (self.source_width, self.source_height, self.source_fps,
                   self.duration or -1, self.has_audio))


def probe(path):
    """Read container metadata via PyAV. Cheap - no frames are decoded."""
    import av

    if not os.path.exists(path):
        raise FrameSourceError("source does not exist: %r" % path)
    with av.open(path) as container:
        if not container.streams.video:
            raise FrameSourceError("no video stream in %r" % path)
        stream = container.streams.video[0]
        duration = None
        if container.duration:
            duration = float(container.duration) / 1_000_000.0
        elif stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        rotation = 0
        try:
            rotation = int(stream.side_data.get("DISPLAYMATRIX", 0) or 0)
        except Exception:
            pass
        return VideoMetadata(
            path=path,
            width=stream.codec_context.width,
            height=stream.codec_context.height,
            source_fps=float(stream.average_rate or FPS),
            duration=duration,
            has_audio=bool(container.streams.audio),
            rotation=rotation,
        )


class FFmpegFrameSource:
    """Sequential rgb24 frames at a normalized rate, optionally canvas-pinned.

    `start_frame` counts in *output* frames at `fps`, matching how chunk
    geometry is expressed everywhere else in this package.
    """

    def __init__(self, path, *, canvas=None, fps=FPS, start_frame=0,
                 ffmpeg_location=None, read_ahead_frames=0):
        self.path = path
        self.canvas = tuple(canvas) if canvas else None
        self.fps = int(fps)
        self.start_frame = int(start_frame)
        self.ffmpeg = resolve_ffmpeg(ffmpeg_location)
        self.read_ahead_frames = int(read_ahead_frames)
        self.metadata = None
        self._process = None
        self._stderr_tail = []
        self._stderr_thread = None
        self._frames_emitted = 0
        self._eof = False

    # -- lifecycle -------------------------------------------------------

    def open(self):
        self.metadata = probe(self.path)
        width, height = self.canvas or (self.metadata.source_width,
                                        self.metadata.source_height)
        self._width, self._height = int(width), int(height)
        self._frame_bytes = self._width * self._height * _BYTES_PER_PIXEL

        filters = ["fps=%d" % self.fps]
        if self.canvas:
            # Cover-crop to the pinned canvas rather than letterboxing, so the
            # aspect matches what the reference builder expects.
            filters.append(
                "scale=%d:%d:force_original_aspect_ratio=increase"
                % (self._width, self._height))
            filters.append("crop=%d:%d" % (self._width, self._height))

        args = [self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if self.start_frame:
            # -ss before -i seeks the container; after -i it decodes and
            # discards, which is the behaviour that made VHS unusable here.
            args += ["-ss", "%.6f" % (self.start_frame / float(self.fps))]
        args += ["-i", self.path, "-map", "0:v:0", "-vf", ",".join(filters),
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

        self._process = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=self._frame_bytes * max(1, self.read_ahead_frames or 1))
        # stderr must be drained continuously or a chatty decoder fills the
        # pipe buffer and the process blocks forever mid-run.
        self._stderr_thread = threading.Thread(target=self._drain, daemon=True)
        self._stderr_thread.start()
        return self.metadata

    def _drain(self):
        try:
            for line in self._process.stderr:
                text = line.decode("utf-8", "replace").rstrip()
                if text:
                    self._stderr_tail.append(text)
                    del self._stderr_tail[:-40]
        except Exception:
            pass

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        try:
            if process.stdout:
                process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if process.stderr:
                try:
                    process.stderr.close()
                except Exception:
                    pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- reading ---------------------------------------------------------

    def read_frames(self, count):
        """Up to `count` frames as CPU uint8 [n, H, W, 3]. Short only at EOF."""
        import numpy as np
        import torch

        if self._process is None:
            raise FrameSourceError("read_frames() before open()")
        if count <= 0 or self._eof:
            return torch.empty(0, self._height, self._width, 3, dtype=torch.uint8)

        wanted = self._frame_bytes * count
        buffer = bytearray()
        stdout = self._process.stdout
        while len(buffer) < wanted:
            block = stdout.read(wanted - len(buffer))
            if not block:
                self._eof = True
                break
            buffer.extend(block)

        whole, remainder = divmod(len(buffer), self._frame_bytes)
        if remainder:
            # A partial frame is decoder failure, never a short final read.
            self.close()
            raise FrameSourceError(
                "truncated frame from ffmpeg: %d trailing bytes of a %d-byte "
                "frame after %d frames%s"
                % (remainder, self._frame_bytes, self._frames_emitted,
                   self._stderr_note()))
        if whole == 0:
            return torch.empty(0, self._height, self._width, 3, dtype=torch.uint8)

        array = np.frombuffer(bytes(buffer[:whole * self._frame_bytes]),
                              dtype=np.uint8)
        array = array.reshape(whole, self._height, self._width, 3)
        self._frames_emitted += whole
        return torch.from_numpy(array.copy())

    def _stderr_note(self):
        if not self._stderr_tail:
            return ""
        return "; ffmpeg said: " + " | ".join(self._stderr_tail[-3:])

    @property
    def frames_emitted(self):
        return self._frames_emitted

    @property
    def exhausted(self):
        return self._eof


def read_window(path, start_frame, count, *, canvas=None, fps=FPS,
                ffmpeg_location=None):
    """One bounded window. The common case, and what the harness needs today."""
    with FFmpegFrameSource(path, canvas=canvas, fps=fps, start_frame=start_frame,
                           ffmpeg_location=ffmpeg_location) as source:
        frames = source.read_frames(count)
    if frames.shape[0] < count:
        raise FrameSourceError(
            "source ran out at %d frames; %d requested from start_frame %d"
            % (frames.shape[0], count, start_frame))
    return frames


def _cli(argv):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--count", type=int, default=124)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--out", default=None, help="write PNGs here")
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args(argv)

    metadata = probe(args.video)
    print(json.dumps(metadata.as_dict(), indent=2))
    if args.probe_only:
        return 0

    canvas = (args.width, args.height) if args.width and args.height else None
    import time
    started = time.time()
    frames = read_window(args.video, args.start_frame, args.count,
                         canvas=canvas, fps=args.fps)
    elapsed = time.time() - started
    print("\nread %d frames %s in %.2fs (%.1f fps)"
          % (frames.shape[0], tuple(frames.shape[1:]), elapsed,
             frames.shape[0] / max(elapsed, 1e-6)))

    if args.out:
        from PIL import Image
        os.makedirs(args.out, exist_ok=True)
        for i in range(frames.shape[0]):
            Image.fromarray(frames[i].numpy()).save(
                os.path.join(args.out, "frame_%05d.png" % i))
        print("wrote %d PNGs to %s" % (frames.shape[0], args.out))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
