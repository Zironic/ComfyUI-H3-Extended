"""Live preview support for the long-form Ref2V runner.

Two independent channels are published while the node is running:

* ``current_chunk`` decodes a bounded 17-frame window from the denoised H3
  video latent every N sampler steps.
* ``completed_chunk`` writes the already-decoded committed frames of each
  completed chunk as a small MP4 segment.  The browser plays the finalized
  segments as a playlist, so the main still-open output container never has to
  be readable mid-run.

The production runner predates these callbacks.  This module patches its two
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
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import harness
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import runner
from .writer import FFmpegVideoWriter

LOG = "[H3 Extended] longform preview"
EVENT = "h3_longform_preview"
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
    """Extract the video stream from a callback NestedTensor.

    Comfy packs H3's video and audio tensors for sampling, then reconstructs a
    NestedTensor before invoking callbacks.  The fallbacks make this helper
    tolerant of older NestedTensor implementations used by local Comfy builds.
    """
    if value is None:
        raise ValueError("sampler callback did not provide a denoised latent")
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
    resized = F.interpolate(nchw, size=(out_h, out_w), mode="bilinear",
                            align_corners=False, antialias=True)
    return resized.round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()


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

    base = (folder_paths.get_temp_directory() if folder_type == "temp"
            else folder_paths.get_output_directory())
    relative = os.path.relpath(path, base)
    subfolder, filename = os.path.split(relative)
    return {
        "filename": filename,
        "subfolder": "" if subfolder == "." else subfolder.replace("\\", "/"),
        "type": folder_type,
    }


@dataclass
class PreviewOptions:
    current_enabled: bool = True
    completed_enabled: bool = True
    every_steps: int = 2
    current_frames: int = 17
    width: int = 512


class LongFormPreviewPublisher:
    """Decode, persist and announce both live-preview channels."""

    def __init__(self, *, node_id, video_vae, root, fps=24,
                 ffmpeg_location=None, options=None):
        self.node_id = str(node_id)
        self.video_vae = video_vae
        self.root = root
        self.fps = int(fps)
        self.ffmpeg_location = ffmpeg_location
        self.options = options or PreviewOptions()
        self.revision = 0
        self.completed_frames = 0

        import folder_paths
        self.temp_root = os.path.join(
            folder_paths.get_temp_directory(), "h3_longform_preview",
            _safe_component(self.node_id),
        )
        os.makedirs(self.temp_root, exist_ok=True)

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
            if not should_emit_step(completed_step, total_steps,
                                    self.options.every_steps):
                return
            try:
                self.publish_current_chunk(
                    chunk_index=int(chunk_index), step=completed_step,
                    total_steps=int(total_steps), denoised=denoised)
            except Exception as exc:
                logging.warning(
                    "%s current preview failed at chunk %d step %d: %s",
                    LOG, chunk_index, completed_step, exc)
        return callback

    @torch.inference_mode()
    def publish_current_chunk(self, *, chunk_index, step, total_steps, denoised):
        """Decode a bounded temporal window and publish it as animated WebP."""
        video = _video_latent(denoised)
        # The released H3 VAE maps five temporal latent positions to 17 frames.
        # For unusual short targets, decode what exists rather than failing.
        latent_t = min(5, int(video.shape[2]))
        preview_latent = video[:, :, :latent_t].detach()
        pixels = runner.decode_chunk(self.video_vae, preview_latent).to("cpu", torch.float32)
        requested = max(1, int(self.options.current_frames))
        pixels = pixels[:requested]
        frames_u8 = (pixels.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
        frames_u8 = _resize_frames_u8(frames_u8, self.options.width)
        path = os.path.join(self.temp_root, "current.webp")
        self._write_webp(path, frames_u8)
        self._announce(
            "current_chunk", chunk_index=int(chunk_index), step=int(step),
            total_steps=int(total_steps), frames=int(frames_u8.shape[0]),
            asset=_asset_payload(path, "temp"))
        del preview_latent, pixels, frames_u8
        gc.collect()

    def publish_completed_chunk(self, *, chunk_index, frames_u8, completed_frames):
        """Publish only frames actually committed after overlap trimming."""
        if not self.options.completed_enabled or int(frames_u8.shape[0]) == 0:
            return
        frames = _resize_frames_u8(frames_u8, self.options.width)
        path = os.path.join(self.temp_root, "completed_%06d.mp4" % int(chunk_index))
        writer = FFmpegVideoWriter(
            path, width=int(frames.shape[2]), height=int(frames.shape[1]),
            fps=self.fps, ffmpeg_location=self.ffmpeg_location,
            crf=24, preset="ultrafast").open()
        try:
            writer.write(frames)
        finally:
            writer.close(commit=True)
        self.completed_frames = int(completed_frames)
        self._announce(
            "completed_chunk", chunk_index=int(chunk_index),
            chunk_frames=int(frames.shape[0]),
            completed_frames=self.completed_frames,
            fps=self.fps, asset=_asset_payload(path, "temp"))
        del frames

    def _write_webp(self, path, frames_u8):
        from PIL import Image

        images = [Image.fromarray(frame.numpy(), "RGB") for frame in frames_u8]
        if not images:
            raise ValueError("preview decode produced no frames")
        partial = path + ".tmp"
        duration = max(20, round(1000 / max(1, self.fps)))
        images[0].save(
            partial, format="WEBP", save_all=True,
            append_images=images[1:], duration=duration, loop=0,
            quality=76, method=0)
        os.replace(partial, path)


def activate(publisher):
    return _ACTIVE.set(publisher)


def deactivate(token):
    _ACTIVE.reset(token)


def current_publisher():
    return _ACTIVE.get()


def _sample_with_callback(*, model, conditioning, latent, sampler, sigmas, seed,
                          callback=None):
    """The existing harness sampler with Comfy's callback seam exposed."""
    import comfy.model_management
    import comfy.sample
    import comfy.samplers

    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": conditioning})
    guider.set_cfg(1.0)
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, seed)
    samples = guider.sample(
        noise, latent_image, sampler, sigmas, callback=callback,
        disable_pbar=not runner.__import__("comfy.utils").utils.PROGRESS_BAR_ENABLED
        if False else not __import__("comfy.utils", fromlist=["PROGRESS_BAR_ENABLED"]).PROGRESS_BAR_ENABLED,
        seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples.unbind()


def _pass_c_with_preview(self, *, model, sampler, sigmas, chunk_count,
                         on_sampled=None):
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
        conditioning = harness.attach_refs(conditioning, self.static_blocks + [block])

        conditions = []
        if carry_latent is not None and self.carry != runner.CARRY_NONE:
            count = 1 if self.carry == runner.CARRY_FRAME else overlap_count
            conditions = [TargetAlignedCondition(
                latent=carry_latent[:, :, overlap_start:overlap_start + count],
                target_latent_start=0,
                label="carry from chunk %d" % (index - 1),
            )]
        arm_model = patch_target_conditions(model, conditions)
        latent = harness.empty_av_latent(self.canvas, geometry)
        publisher = current_publisher()
        callback = None if publisher is None else publisher.sampler_callback(index)
        video_latent, audio_latent = harness.sample(
            model=arm_model, conditioning=conditioning, latent=latent,
            sampler=sampler, sigmas=sigmas, seed=self.chunk_seed(index),
            callback=callback)
        video_latent = video_latent.to("cpu", torch.float32)
        runner._save(self.path("samples", index), {
            "video_latent": video_latent,
            "audio_latent": audio_latent.to("cpu", torch.float32),
        })
        runner._atomic_json(self.path("samples", index, ".json"), {
            "index": index,
            "seed": self.chunk_seed(index),
            "previous_sample": index - 1 if index else None,
            "carry": self.carry,
        })
        carry_latent = video_latent
        self.event(pass_="C", chunk=index, carry=self.carry,
                   conditions=len(conditions))
        if on_sampled is not None:
            on_sampled(index, video_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        del stored, cond_store, conditioning, arm_model, latent, audio_latent
        logging.info("%s pass C: chunk %d/%d done (%.1f s)", runner.LOG,
                     index + 1, chunk_count, time.time() - started)
    return chunk_count


def _emit_chunk_with_preview(self, index, latent, *, video_vae, chunk_count,
                             writer, save_frames, written, out_dir, pbar=None):
    geometry = self.geometry
    pixels = runner.decode_chunk(video_vae, latent).to("cpu", torch.float32)
    remaining = self.target_frames - written
    if remaining <= 0:
        return 0
    take = min(
        int(pixels.shape[0]) if index == chunk_count - 1 else geometry.stride_frames,
        remaining,
    )
    frames_u8 = (pixels[:take].clamp(0, 1) * 255.0 + 0.5).to(torch.uint8)
    if writer is not None:
        writer.write(frames_u8)
    if save_frames:
        from PIL import Image
        for i in range(take):
            Image.fromarray(frames_u8[i].numpy()).save(
                os.path.join(out_dir, "frame_%06d.png" % (written + i)))

    publisher = current_publisher()
    if publisher is not None:
        try:
            publisher.publish_completed_chunk(
                chunk_index=index, frames_u8=frames_u8,
                completed_frames=written + take)
        except Exception as exc:
            logging.warning("%s completed preview failed for chunk %d: %s",
                            LOG, index, exc)
        if pbar is not None:
            pbar.update_absolute(index + 1, chunk_count)
    elif pbar is not None:
        runner._send_preview(pbar, frames_u8[-1], index + 1, chunk_count)

    self.event(pass_="D", chunk=index, frames=take, total=written + take)
    if self.manifest:
        self.manifest.update_state(pass_d_chunks=index + 1,
                                   frames_written=written + take)
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
    "EVENT", "PreviewOptions", "LongFormPreviewPublisher", "activate",
    "deactivate", "current_publisher", "should_emit_step", "install",
]
