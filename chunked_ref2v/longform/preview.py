"""Live previews for the disk-backed long-form MiniMax H3 runners.

Two independent channels are published while a node is running:

* ``current_chunk`` is a lightweight approximation of the current denoised H3
  latent. It uses Comfy's normal latent previewer by default, so it does not try
  to load the full video VAE from inside the sampler callback. If an explicit
  ``preview_vae`` is connected, an exact bounded VAE animation is attempted
  first and the latent preview remains the failure-safe fallback.
* ``completed_chunk`` writes the already-decoded committed frames of each
  completed chunk as a small MP4 segment, then stream-copies every segment
  produced so far into one growing ``completed_all`` MP4. The browser gets that
  single stitched file, so the pane shows the whole finished output instead of
  the newest chunk alone, and the main still-open output container never has to
  be readable mid-run. If the concat step fails the per-chunk segment is
  published on its own and the browser falls back to playlist playback.

  Runtimes that generate audio hand each chunk's committed samples over after
  ``_emit_chunk`` returns, so the frames are staged rather than published
  immediately (see ``stage_completed_chunk``). The samples accumulate as raw
  PCM and are encoded once per stitch, which keeps the pane in sync instead of
  drifting by an AAC frame per chunk.

The production runner predates these callbacks. This module patches its two
narrow seams at import time rather than duplicating the complete runner:
``harness.sample`` accepts a sampler callback, ``LongFormRun.pass_c`` installs
one for each new chunk, and ``LongFormRun._emit_chunk`` publishes the exact
frames that were committed after overlap trimming.
"""

from __future__ import annotations

import contextvars
import gc
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import harness
from ..geometry import FRAME_PER_TOKEN
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import runner
from .frame_source import resolve_ffmpeg
from .writer import FFmpegVideoWriter

LOG = "[H3 Extended] longform preview"
EVENT = "h3_longform_preview"
# One temporal latent position stands for this many pixel frames, so a latent
# preview image has to be held that many frame periods to run at real speed.
LATENT_FRAME_STRIDE = max(FRAME_PER_TOKEN)
_ACTIVE = contextvars.ContextVar("h3_longform_preview_publisher", default=None)
_PATCHED = False


def should_emit_step(step: int, total_steps: int, every: int) -> bool:
    """Return whether a one-based sampler step should publish a preview."""
    step = int(step)
    total_steps = int(total_steps)
    every = max(1, int(every))
    return step == total_steps or step % every == 0


def _safe_component(value) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "node"))[:96]


def _video_latent(value):
    """Extract the video stream from a callback NestedTensor."""
    if value is None:
        raise ValueError("sampler callback did not provide a latent")
    tensors = getattr(value, "tensors", None)
    if tensors is not None:
        return tensors[0]
    unbind = getattr(value, "unbind", None)
    if callable(unbind):
        values = unbind()
        if values:
            return values[0]
    if isinstance(value, (tuple, list)) and value:
        return value[0]
    if isinstance(value, torch.Tensor) and value.ndim == 5:
        return value
    raise TypeError("unsupported H3 preview latent %s" % type(value).__name__)


def _resize_frames_u8(frames_u8: torch.Tensor, max_width: int) -> torch.Tensor:
    """Downscale an NHWC uint8 batch while preserving aspect and even sizes."""
    if frames_u8.ndim != 4 or frames_u8.shape[-1] < 3:
        raise ValueError("expected NHWC RGB frames, got %r" % (tuple(frames_u8.shape),))
    frames_u8 = frames_u8[..., :3].detach().to("cpu", torch.uint8)
    height, width = int(frames_u8.shape[1]), int(frames_u8.shape[2])
    max_width = max(16, int(max_width))
    scale = min(1.0, max_width / max(1, width))
    out_w = max(16, int(round(width * scale)) // 2 * 2)
    out_h = max(16, int(round(height * scale)) // 2 * 2)
    if (out_h, out_w) == (height, width):
        return frames_u8.contiguous()
    nchw = frames_u8.permute(0, 3, 1, 2).to(torch.float32)
    resized = F.interpolate(
        nchw,
        size=(out_h, out_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return (
        resized.round_()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
    )


def _resize_pil(image, max_width):
    from PIL import Image

    image = image.convert("RGB")
    max_width = max(16, int(max_width))
    if image.width <= max_width:
        return image
    height = max(1, round(image.height * max_width / image.width))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return image.resize((max_width, height), resampling)


def _send_event(payload) -> None:
    try:
        from server import PromptServer

        server = getattr(PromptServer, "instance", None)
        if server is not None:
            server.send_sync(EVENT, payload)
    except Exception as exc:  # preview must never kill a generation
        logging.debug("%s event unavailable: %s", LOG, exc)


def _asset_payload(path: str, folder_type: str) -> dict:
    import folder_paths

    base = (
        folder_paths.get_temp_directory()
        if folder_type == "temp"
        else folder_paths.get_output_directory()
    )
    relative = os.path.relpath(path, base)
    subfolder, filename = os.path.split(relative)
    return {
        "filename": filename,
        "subfolder": "" if subfolder == "." else subfolder.replace("\\", "/"),
        "type": folder_type,
    }


def _build_latent_previewer(model):
    """Build Comfy's normal no-VAE previewer for the incoming model."""
    if model is None:
        return None
    try:
        import latent_preview

        return latent_preview.get_previewer(
            model.load_device,
            model.model.latent_format,
        )
    except Exception as exc:
        logging.warning("%s latent previewer unavailable: %s", LOG, exc)
        return None


def _latent_process_out(model):
    """Best-effort conversion from sampler x0 space to VAE latent space."""
    base = getattr(model, "model", None)
    processor = getattr(base, "process_latent_out", None)
    return processor if callable(processor) else None


@dataclass
class PreviewOptions:
    current_enabled: bool = True
    completed_enabled: bool = True
    every_steps: int = 2
    current_frames: int = 17
    width: int = 512


def resolve_unique_id(cls, fallback=None):
    """Return the executing node id for a V3 node.

    Hidden inputs declared in the schema are never passed to ``execute`` as
    keyword arguments. Comfy binds them onto a per-execution class clone
    (``execution.get_input_data`` fills ``v3_data["hidden_inputs"]``, and
    ``ComfyNode.PREPARE_CLASS_CLONE`` turns that into ``cls.hidden``). Reading
    the ``unique_id=None`` signature default therefore published every preview
    event under the literal node id ``"None"``, which no browser node matches.
    """

    hidden = getattr(cls, "hidden", None)
    resolved = getattr(hidden, "unique_id", None) if hidden is not None else None
    if resolved is None:
        resolved = fallback
    return resolved


class LongFormPreviewPublisher:
    """Persist and announce both live-preview channels.

    ``video_vae`` is deliberately optional. A disconnected preview VAE means
    "use the lightweight latent preview", not "reuse the production VAE inside
    the active sampler". Loading the production VAE during a DiT callback is
    exactly the model-switching path that left the panel stuck at waiting.
    """

    def __init__(
        self,
        *,
        node_id,
        model=None,
        video_vae=None,
        root,
        fps=24,
        ffmpeg_location=None,
        options=None,
        audio_expected=False,
    ):
        self.node_id = str(node_id)
        # Set by runtimes that decode a waveform per chunk. It makes the
        # completed pane wait for that decode instead of publishing silence.
        self.audio_expected = bool(audio_expected)
        self.pending_chunk = None
        self.video_vae = video_vae
        self.latent_previewer = _build_latent_previewer(model)
        self.latent_process_out = _latent_process_out(model)
        self.root = root
        self.fps = int(fps)
        self.ffmpeg_location = ffmpeg_location
        self.options = options or PreviewOptions()
        self.revision = 0
        self.completed_frames = 0
        self.segment_paths = []
        self.stitched_path = None
        self.stitch_index = 0
        self.audio_channels = None
        self.audio_rate = None

        import folder_paths

        self.temp_root = os.path.join(
            folder_paths.get_temp_directory(),
            "h3_longform_preview",
            _safe_component(self.node_id),
        )
        os.makedirs(self.temp_root, exist_ok=True)
        # Appended to per chunk. A previous run of the same node left its own
        # track here, and appending to that would prefix the preview with stale
        # audio, so start every run from an empty file.
        self.audio_path = os.path.join(self.temp_root, "completed_audio.f32le")
        try:
            os.remove(self.audio_path)
        except OSError:
            pass

    def _announce(self, kind, **fields):
        self.revision += 1
        payload = {
            "node_id": self.node_id,
            "kind": kind,
            "revision": self.revision,
        }
        payload.update(fields)
        _send_event(payload)

    def sampler_callback(self, chunk_index: int):
        if not self.options.current_enabled:
            return None

        def callback(step, denoised, current, total_steps):
            completed_step = int(step) + 1
            if not should_emit_step(
                completed_step,
                total_steps,
                self.options.every_steps,
            ):
                return
            try:
                self.publish_current_chunk(
                    chunk_index=int(chunk_index),
                    step=completed_step,
                    total_steps=int(total_steps),
                    denoised=denoised,
                    current=current,
                )
            except Exception as exc:
                message = "%s: %s" % (type(exc).__name__, exc)
                logging.warning(
                    "%s current preview failed at chunk %d step %d: %s",
                    LOG,
                    chunk_index,
                    completed_step,
                    message,
                )
                self._announce(
                    "current_chunk_error",
                    chunk_index=int(chunk_index),
                    step=completed_step,
                    total_steps=int(total_steps),
                    message=message[:500],
                )

        return callback

    @torch.inference_mode()
    def publish_current_chunk(
        self,
        *,
        chunk_index,
        step,
        total_steps,
        denoised=None,
        current=None,
    ):
        """Publish the current denoised latent without risking the generation.

        An explicitly connected preview VAE gets first refusal. Any VAE load,
        decode or animation error falls back to the model's lightweight latent
        previewer. The default path never loads a VAE inside the sampler.
        """
        source = denoised if denoised is not None else current
        video = _video_latent(source)
        fallback_reason = None
        images = None
        mode = "latent"

        if self.video_vae is not None:
            try:
                images = self._exact_vae_images(video)
                mode = "vae"
            except Exception as exc:
                fallback_reason = "%s: %s" % (type(exc).__name__, exc)
                logging.warning(
                    "%s exact current preview failed; using latent preview: %s",
                    LOG,
                    fallback_reason,
                )

        if images is None:
            images = self._latent_images(video)

        # The VAE path decodes real consecutive frames, so it plays at the
        # production rate. A latent image covers LATENT_FRAME_STRIDE frames and
        # has to be held that much longer, or the pane runs at 4x speed.
        preview_fps = (
            self.fps
            if mode == "vae"
            else max(1, round(self.fps / LATENT_FRAME_STRIDE))
        )
        path = os.path.join(self.temp_root, "current.gif")
        self._write_animation(path, images, fps=preview_fps)
        fields = {
            "chunk_index": int(chunk_index),
            "step": int(step),
            "total_steps": int(total_steps),
            "frames": len(images),
            "mode": mode,
            "preview_fps": int(preview_fps),
            "asset": _asset_payload(path, "temp"),
        }
        if fallback_reason:
            fields["fallback_reason"] = fallback_reason[:500]
        self._announce("current_chunk", **fields)
        del images, video
        gc.collect()

    def _exact_vae_images(self, video):
        """Decode the first five H3 temporal positions to at most 17 frames.

        Five is the free tier, not a display preference. The VAE decodes a whole
        1+4*4 group at once, so one frame and seventeen frames cost the same
        wall-clock; a sixth latent position starts a second group and doubles
        the decode inside the sampler callback. Keep this at five.
        """
        latent_t = min(5, int(video.shape[2]))
        preview_latent = video[:, :, :latent_t].detach().to(torch.float32)
        if self.latent_process_out is not None:
            preview_latent = self.latent_process_out(preview_latent)
        pixels = runner.decode_chunk(self.video_vae, preview_latent).to(
            "cpu", torch.float32
        )
        requested = max(1, int(self.options.current_frames))
        pixels = pixels[:requested]
        frames_u8 = (pixels.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
        frames_u8 = _resize_frames_u8(frames_u8, self.options.width)
        from PIL import Image

        images = [
            Image.fromarray(frame.numpy(), "RGB")
            for frame in frames_u8
        ]
        del preview_latent, pixels, frames_u8
        if not images:
            raise ValueError("preview VAE decoded no frames")
        return images

    def _latent_images(self, video):
        """Render up to five temporal positions with Comfy's latent previewer."""
        count = min(
            max(1, int(self.options.current_frames)),
            5,
            int(video.shape[2]),
        )
        images = []
        preview_error = None
        if self.latent_previewer is not None:
            try:
                for index in range(count):
                    image = self.latent_previewer.decode_latent_to_preview(
                        video[:, :, index:index + 1]
                    )
                    images.append(_resize_pil(image, self.options.width))
            except Exception as exc:
                preview_error = exc
                images = []

        if not images:
            if preview_error is not None:
                logging.warning(
                    "%s model latent preview failed; using channel fallback: %s",
                    LOG,
                    preview_error,
                )
            images = self._channel_fallback_images(video, count)
        return images

    def _channel_fallback_images(self, video, count):
        """Always-available structural preview when no model previewer exists."""
        from PIL import Image

        images = []
        for index in range(count):
            frame = video[0, :3, index].detach().to("cpu", torch.float32)
            low = frame.amin(dim=(1, 2), keepdim=True)
            high = frame.amax(dim=(1, 2), keepdim=True)
            frame = (frame - low) / (high - low).clamp_min(1e-6)
            rgb = (
                frame.permute(1, 2, 0)
                .mul(255.0)
                .round_()
                .clamp_(0, 255)
                .to(torch.uint8)
                .numpy()
            )
            images.append(
                _resize_pil(Image.fromarray(rgb, "RGB"), self.options.width)
            )
        return images

    def stage_completed_chunk(self, *, chunk_index, frames_u8, completed_frames):
        """Hold a chunk's committed frames until its audio has been decoded.

        The audio runtime decodes a chunk's waveform only after ``_emit_chunk``
        returns, so publishing from inside that call could never produce a
        segment with sound. Staging costs one resized frame batch (a few tens of
        MB at preview width) and the wait is the length of one audio decode.
        """
        if not self.options.completed_enabled or int(frames_u8.shape[0]) == 0:
            return
        # A previous stage that never got its audio must not be lost.
        self.flush_completed_chunk()
        self.pending_chunk = {
            "chunk_index": int(chunk_index),
            "frames": _resize_frames_u8(frames_u8, self.options.width),
            "completed_frames": int(completed_frames),
        }

    def flush_completed_chunk(self, *, waveform=None, sample_rate=None):
        """Publish the staged chunk, with its audio when one was decoded."""
        pending = getattr(self, "pending_chunk", None)
        self.pending_chunk = None
        if pending is None:
            return
        audio = None
        if waveform is not None and sample_rate:
            audio = (waveform, int(sample_rate))
        # Route back through the public method so the GIF fallback still wraps it.
        self.publish_completed_chunk(
            chunk_index=pending["chunk_index"],
            frames_u8=pending["frames"],
            completed_frames=pending["completed_frames"],
            audio=audio,
        )

    def publish_completed_chunk(
        self,
        *,
        chunk_index,
        frames_u8,
        completed_frames,
        audio=None,
    ):
        """Publish only frames actually committed after overlap trimming."""
        if not self.options.completed_enabled or int(frames_u8.shape[0]) == 0:
            return
        frames = _resize_frames_u8(frames_u8, self.options.width)
        path = os.path.join(
            self.temp_root,
            "completed_%06d.mp4" % int(chunk_index),
        )
        writer = FFmpegVideoWriter(
            path,
            width=int(frames.shape[2]),
            height=int(frames.shape[1]),
            fps=self.fps,
            ffmpeg_location=self.ffmpeg_location,
            crf=24,
            preset="ultrafast",
        ).open()
        try:
            writer.write(frames)
        finally:
            writer.close(commit=True)
        self.completed_frames = int(completed_frames)

        # Sound is a bonus on a preview: a failed append keeps the silent
        # segment rather than costing the pane a chunk.
        audio_error = None
        if audio is not None:
            try:
                self._append_preview_audio(*audio)
            except Exception as exc:
                audio_error = ("%s: %s" % (type(exc).__name__, exc))[:500]
                logging.warning(
                    "%s completed preview audio failed at chunk %d: %s",
                    LOG,
                    int(chunk_index),
                    audio_error,
                )

        segments = self._segment_paths()
        if path not in segments:
            segments.append(path)
        fields = {
            "chunk_index": int(chunk_index),
            "chunk_frames": int(frames.shape[0]),
            "completed_frames": self.completed_frames,
            "fps": self.fps,
            "segments": len(segments),
            # Only the stitched file carries sound; segments stay video-only.
            "has_audio": self._has_preview_audio(),
            "segment_asset": _asset_payload(path, "temp"),
        }
        if audio_error:
            fields["audio_error"] = audio_error

        # The pane should show everything finished so far, not just this chunk.
        # A concat failure is never fatal: publishing the bare segment leaves the
        # browser on its older playlist path.
        stitched = None
        try:
            stitched = self._stitch_segments(segments)
        except Exception as exc:
            fields["stitch_error"] = ("%s: %s" % (type(exc).__name__, exc))[:500]
            logging.warning(
                "%s stitched completed preview failed at chunk %d: %s",
                LOG,
                int(chunk_index),
                fields["stitch_error"],
            )

        if stitched is None:
            fields["stitched"] = False
            fields["asset"] = fields["segment_asset"]
        else:
            fields["stitched"] = True
            fields["asset"] = _asset_payload(stitched, "temp")
        self._announce("completed_chunk", **fields)
        del frames

    def _append_preview_audio(self, waveform, sample_rate):
        """Append a chunk's committed samples to the running preview track.

        Audio is accumulated as raw PCM and encoded once per stitch rather than
        per segment. Encoding each segment separately would pad every one of
        them out to an AAC frame boundary, and concatenating those pads drifts
        the preview out of sync by ~30 ms per chunk -- across a 50-chunk run,
        more than a second of false lip-sync error.
        """
        audio = waveform.detach().to("cpu", torch.float32)
        if audio.ndim == 3:
            audio = audio[0]
        if audio.ndim != 2:
            raise ValueError(
                "expected [channels, samples] audio, got %r" % (tuple(audio.shape),)
            )
        channels = int(audio.shape[0])
        if channels < 1 or int(audio.shape[1]) < 1:
            raise ValueError("empty audio chunk")
        if self.audio_channels is None:
            self.audio_channels = channels
            self.audio_rate = int(sample_rate)
        elif channels != self.audio_channels or int(sample_rate) != self.audio_rate:
            raise ValueError(
                "preview audio format changed: %d ch @ %d -> %d ch @ %d"
                % (self.audio_channels, self.audio_rate, channels, int(sample_rate))
            )
        # ffmpeg wants interleaved samples; the decoder hands over planar.
        with open(self.audio_path, "ab") as handle:
            handle.write(audio.transpose(0, 1).contiguous().numpy().tobytes())
        return self.audio_path

    def _mux_stitched_audio(self, ffmpeg, video_path, output):
        """Attach the whole accumulated track to the stitched video."""
        args = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "f32le", "-ar", str(self.audio_rate),
            "-ac", str(self.audio_channels), "-i", self.audio_path,
            "-i", video_path,
            "-map", "1:v:0", "-map", "0:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "96k",
            "-movflags", "+faststart", "-shortest", "-y", output,
        ]
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(
                "ffmpeg audio mux exited with %d: %s"
                % (proc.returncode, proc.stderr.decode("utf-8", "replace")[-2000:])
            )
        return output

    def _has_preview_audio(self):
        return (
            self.audio_channels is not None
            and os.path.exists(self.audio_path)
            and os.path.getsize(self.audio_path) > 0
        )

    def _segment_paths(self):
        # Tests and older callers may build a publisher without __init__.
        if not hasattr(self, "segment_paths"):
            self.segment_paths = []
        return self.segment_paths

    def _stitch_segments(self, segments):
        """Stream-copy every finished segment into one growing preview MP4.

        The segments all come from the same encoder settings and the same
        resized geometry, so the concat demuxer can copy them without
        re-encoding. Each stitch writes a fresh filename: the browser may still
        be streaming the previous one, and on Windows overwriting a file that is
        open for reading fails.
        """
        if not segments:
            return None
        ffmpeg = resolve_ffmpeg(self.ffmpeg_location)
        listing = os.path.join(self.temp_root, "completed_segments.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            for segment in segments:
                escaped = segment.replace("\\", "/").replace("'", r"'\''")
                handle.write("file '%s'\n" % escaped)

        self.stitch_index = int(getattr(self, "stitch_index", 0)) + 1
        output = os.path.join(
            self.temp_root,
            "completed_all_%06d.mp4" % self.stitch_index,
        )
        # FFmpeg picks the muxer from the extension, so the in-progress name has
        # to keep ".mp4" last.
        partial = output[: -len(".mp4")] + ".partial.mp4"
        args = [
            ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", listing,
            "-c", "copy", "-movflags", "+faststart", "-y", partial,
        ]
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            try:
                os.remove(partial)
            except OSError:
                pass
            raise RuntimeError(
                "ffmpeg concat exited with %d: %s"
                % (proc.returncode, proc.stderr.decode("utf-8", "replace")[-2000:])
            )

        # One AAC encode over the whole accumulated track, so the pane's audio
        # lines up with its video at every chunk instead of drifting.
        if self._has_preview_audio():
            sounded = output[: -len(".mp4")] + ".sound.mp4"
            try:
                self._mux_stitched_audio(ffmpeg, partial, sounded)
                os.replace(sounded, partial)
            except Exception as exc:
                logging.warning(
                    "%s stitched preview audio failed; staying silent: %s",
                    LOG,
                    exc,
                )
                try:
                    os.remove(sounded)
                except OSError:
                    pass
        os.replace(partial, output)

        previous = getattr(self, "stitched_path", None)
        self.stitched_path = output
        if previous and previous != output:
            try:
                os.remove(previous)
            except OSError:
                # The browser can still hold the old file open; it is temp data.
                pass
        return output

    def _write_animation(self, path, images, *, fps=None):
        """Write an animated GIF; GIF support is mandatory in Pillow."""
        if not images:
            raise ValueError("preview produced no images")
        images = [image.convert("RGB") for image in images]
        partial = path + ".tmp"
        # Browsers clamp anything under ~20 ms up to 100 ms, so never ask for it.
        duration = max(20, round(1000 / max(1, int(fps or self.fps))))
        images[0].save(
            partial,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=False,
        )
        os.replace(partial, path)


def activate(publisher):
    return _ACTIVE.set(publisher)


def deactivate(token):
    _ACTIVE.reset(token)


def current_publisher():
    return _ACTIVE.get()


def _sample_with_callback(
    *,
    model,
    conditioning,
    latent,
    sampler,
    sigmas,
    seed,
    callback=None,
):
    """The existing harness sampler with Comfy's callback seam exposed."""
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils

    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": conditioning})
    guider.set_cfg(1.0)
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, seed)
    samples = guider.sample(
        noise,
        latent_image,
        sampler,
        sigmas,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=seed,
    )
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples.unbind()


def _pass_c_with_preview(
    self,
    *,
    model,
    sampler,
    sigmas,
    chunk_count,
    on_sampled=None,
):
    geometry = self.geometry
    started = time.time()
    overlap_start, overlap_count = (None, None)
    if self.carry != runner.CARRY_NONE:
        overlap_start, overlap_count = geometry.overlap_slice()

    prefix = self._first_invalid("samples", chunk_count)
    self._remove_suffix("samples", prefix, chunk_count)
    carry_latent = None
    if prefix:
        previous = runner._load(self.path("samples", prefix - 1))
        carry_latent = previous["video_latent"]

    for index in range(prefix, chunk_count):
        stored = runner._load(self.path("precompute", index))
        cond_store = runner._load(self.path("conditioning", index))
        cond_meta_path = self.path("conditioning", index, ".json")
        if stored is None or cond_store is None or not os.path.exists(cond_meta_path):
            raise RuntimeError("pass C: chunk %d missing inputs" % index)
        with open(cond_meta_path, encoding="utf-8") as fh:
            cond_meta = json.load(fh)

        block = {
            "kind": "video",
            "latent_t": int(stored["source_latent"].shape[2]),
            "latent_h": self.canvas[1] // 16,
            "latent_w": self.canvas[0] // 16,
            "ref_audio_t": 0,
            "latent": stored["source_latent"],
            "audio_latent": None,
        }
        conditioning = runner._unpack_conditioning(cond_store, cond_meta)
        conditioning = harness.attach_refs(
            conditioning,
            self.static_blocks + [block],
        )

        conditions = []
        if carry_latent is not None and self.carry != runner.CARRY_NONE:
            count = 1 if self.carry == runner.CARRY_FRAME else overlap_count
            conditions = [
                TargetAlignedCondition(
                    latent=carry_latent[
                        :,
                        :,
                        overlap_start:overlap_start + count,
                    ],
                    target_latent_start=0,
                    label="carry from chunk %d" % (index - 1),
                )
            ]
        arm_model = patch_target_conditions(model, conditions)
        latent = harness.empty_av_latent(self.canvas, geometry)
        publisher = current_publisher()
        callback = (
            None
            if publisher is None
            else publisher.sampler_callback(index)
        )
        video_latent, audio_latent = harness.sample(
            model=arm_model,
            conditioning=conditioning,
            latent=latent,
            sampler=sampler,
            sigmas=sigmas,
            seed=self.chunk_seed(index),
            callback=callback,
        )
        video_latent = video_latent.to("cpu", torch.float32)
        runner._save(
            self.path("samples", index),
            {
                "video_latent": video_latent,
                "audio_latent": audio_latent.to("cpu", torch.float32),
            },
        )
        runner._atomic_json(
            self.path("samples", index, ".json"),
            {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
            },
        )
        carry_latent = video_latent
        self.event(
            pass_="C",
            chunk=index,
            carry=self.carry,
            conditions=len(conditions),
        )
        if on_sampled is not None:
            on_sampled(index, video_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        del stored, cond_store, conditioning, arm_model, latent, audio_latent
        logging.info(
            "%s pass C: chunk %d/%d done (%.1f s)",
            runner.LOG,
            index + 1,
            chunk_count,
            time.time() - started,
        )
    return chunk_count


def _emit_chunk_with_preview(
    self,
    index,
    latent,
    *,
    video_vae,
    chunk_count,
    writer,
    save_frames,
    written,
    out_dir,
    pbar=None,
):
    geometry = self.geometry
    pixels = runner.decode_chunk(video_vae, latent).to("cpu", torch.float32)
    remaining = self.target_frames - written
    if remaining <= 0:
        return 0
    take = min(
        int(pixels.shape[0])
        if index == chunk_count - 1
        else geometry.stride_frames,
        remaining,
    )
    frames_u8 = (pixels[:take].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    if writer is not None:
        writer.write(frames_u8)
    if save_frames:
        from PIL import Image

        for i in range(take):
            Image.fromarray(frames_u8[i].numpy()).save(
                os.path.join(out_dir, "frame_%06d.png" % (written + i))
            )

    publisher = current_publisher()
    if publisher is not None:
        try:
            # With audio the caller publishes once the waveform is decoded; see
            # LongFormPreviewPublisher.stage_completed_chunk.
            publish = (
                publisher.stage_completed_chunk
                if getattr(publisher, "audio_expected", False)
                else publisher.publish_completed_chunk
            )
            publish(
                chunk_index=index,
                frames_u8=frames_u8,
                completed_frames=written + take,
            )
        except Exception as exc:
            logging.warning(
                "%s completed preview failed for chunk %d: %s",
                LOG,
                index,
                exc,
            )
        if pbar is not None:
            pbar.update_absolute(index + 1, chunk_count)
    elif pbar is not None:
        runner._send_preview(
            pbar,
            frames_u8[-1],
            index + 1,
            chunk_count,
        )

    self.event(
        pass_="D",
        chunk=index,
        frames=take,
        total=written + take,
    )
    if self.manifest:
        self.manifest.update_state(
            pass_d_chunks=index + 1,
            frames_written=written + take,
        )
    del pixels, frames_u8
    gc.collect()
    return take


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    harness.sample = _sample_with_callback
    runner.LongFormRun.pass_c = _pass_c_with_preview
    runner.LongFormRun._emit_chunk = _emit_chunk_with_preview
    _PATCHED = True


install()


__all__ = [
    "EVENT",
    "PreviewOptions",
    "LongFormPreviewPublisher",
    "activate",
    "deactivate",
    "current_publisher",
    "should_emit_step",
    "install",
]
