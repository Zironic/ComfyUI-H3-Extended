"""Sequential, disk-backed, bounded-memory long-form Ref2V engine."""

from __future__ import annotations

import gc
import json
import logging
import os
import time

import torch

try:
    from .. import harness, ref_builder
    from ..geometry import HarnessGeometry
    from ..layout_ops import TargetAlignedCondition
    from ..model_patch import patch_target_conditions
except ImportError:  # pragma: no cover
    import harness
    import ref_builder
    from geometry import HarnessGeometry
    from layout_ops import TargetAlignedCondition
    from model_patch import patch_target_conditions

from . import diagnostics
from .chunk_stream import iter_source_chunks, actual_frames_for_chunk
from .manifest import (
    RunManifest,
    file_identity,
    object_fingerprint,
    tensor_digest,
)
from .writer import FFmpegVideoWriter, close_writers, mux_source_audio

LOG = "[H3 Extended] longform"
CARRY_NONE = "none"
CARRY_FRAME = "direct_latent_frame"
CARRY_OVERLAP = "direct_latent_overlap"
CARRY_MODES = (CARRY_OVERLAP, CARRY_FRAME, CARRY_NONE)

_DECODED_PIXELS_HOOKS = []


def register_decoded_pixels_hook(hook):
    """Observe every chunk's untrimmed decode from inside ``_emit_chunk``.

    Features that only need to look at the decoded pixels must register here
    rather than replacing ``_emit_chunk``.  A replacement has to re-implement
    the diagnostics dump and the audio-aware completed-preview staging, and a
    copy that silently falls behind those contracts drops chunk audio and chunk
    frame dumps without any error.

    Called as ``hook(run, index, pixels, chunk_count)`` after the diagnostics
    dump and before any overlap trimming, so ``pixels`` still holds the whole
    chunk.  Exceptions propagate: a hook that cannot do its job means the run
    is producing wrong output.
    """
    _DECODED_PIXELS_HOOKS.append(hook)
    return hook


def on_decoded_pixels(run, index, pixels, chunk_count):
    for hook in _DECODED_PIXELS_HOOKS:
        hook(run, index, pixels, chunk_count)


def _save(path, tensors):
    import comfy.utils
    tmp = "%s.%d.tmp" % (path, os.getpid())
    payload = {
        k: v.detach().to("cpu").contiguous()
        for k, v in tensors.items()
        if v is not None
    }
    try:
        comfy.utils.save_torch_file(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _load(path):
    import comfy.utils
    if not os.path.exists(path):
        return None
    try:
        return comfy.utils.load_torch_file(path, safe_load=True)
    except Exception as exc:
        logging.warning("%s unreadable artifact %s (%s)", LOG, path, exc)
        return None


def _atomic_json(path, payload):
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@torch.no_grad()
def decode_chunk(video_vae, latent):
    images = video_vae.decode(latent)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _progress_bar(total):
    """A node progress bar, or None outside a running prompt."""
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(total)
    except Exception:
        return None


def _send_preview(pbar, frame_u8, done, total):
    """Push one finished frame to the node's progress preview.

    A long run is otherwise a blank wait: at ~78 s per chunk a 3 minute output
    is over an hour before anything is visible. The frame is already decoded, so
    this only costs a JPEG encode.
    """
    if pbar is None:
        return
    try:
        from PIL import Image
        image = Image.fromarray(frame_u8.numpy())
        pbar.update_absolute(done, total, ("JPEG", image, 768))
    except Exception as exc:
        logging.debug("%s preview frame unavailable: %s", LOG, exc)


def _pack_conditioning(conditioning):
    tensor_payload = {"cond": conditioning[0][0]}
    meta = {"tensor_keys": [], "none_keys": [], "scalar": {}}
    for key, value in conditioning[0][1].items():
        if isinstance(value, torch.Tensor):
            name = "extra__" + key
            tensor_payload[name] = value
            meta["tensor_keys"].append(key)
        elif value is None:
            meta["none_keys"].append(key)
        elif isinstance(value, (str, int, float, bool, list, dict)):
            meta["scalar"][key] = value
        else:
            raise TypeError("conditioning extra %r is not persistable: %s" % (key, type(value)))
    return tensor_payload, meta


def _unpack_conditioning(tensors, meta):
    extra = dict(meta.get("scalar", {}))
    for key in meta.get("tensor_keys", []):
        extra[key] = tensors["extra__" + key]
    for key in meta.get("none_keys", []):
        extra[key] = None
    return [[tensors["cond"], extra]]


class LongFormRun:
    def __init__(self, root, geometry, canvas, *, carry=CARRY_OVERLAP, seed=0,
                 target_frames=None, manifest=None):
        self.root = root
        self.geometry = geometry
        self.canvas = canvas
        self.carry = carry
        self.seed = int(seed)
        self.target_frames = int(target_frames) if target_frames is not None else None
        self.manifest = manifest
        for name in ("precompute", "conditioning", "samples", "frames", "output"):
            os.makedirs(os.path.join(root, name), exist_ok=True)
        self.events = os.path.join(root, "events.jsonl")
        self.static_blocks = []
        self.static_items = []
        # Set by the caller; see longform/diagnostics.py.
        self.diagnostic_dump_chunks = False

    def path(self, kind, index, suffix=".safetensors"):
        return os.path.join(self.root, kind, "chunk_%06d%s" % (index, suffix))

    def event(self, **fields):
        fields["t"] = time.time()
        with open(self.events, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, default=str) + "\n")

    def chunk_seed(self, index):
        return harness.splitmix64(self.seed + 1000 + index)

    def _valid_pair(self, kind, index):
        return _load(self.path(kind, index)) is not None and os.path.exists(
            self.path(kind, index, ".json")
        )

    def _first_invalid(self, kind, chunk_count, require_meta=False):
        for index in range(chunk_count):
            if _load(self.path(kind, index)) is None:
                return index
            if require_meta and not os.path.exists(self.path(kind, index, ".json")):
                return index
        return chunk_count

    def _remove_suffix(self, kind, start, chunk_count):
        for index in range(start, chunk_count):
            for suffix in (".safetensors", ".json"):
                try:
                    os.remove(self.path(kind, index, suffix))
                except FileNotFoundError:
                    pass

    def pass_a(self, *, video_path, start_frame, chunk_count, video_vae,
               audio_vae, ref_images, ref_image_size, cond_cache, fps=24,
               ffmpeg_location=None):
        geometry = self.geometry

        static_items, static_blocks, notes = [], [], []
        for name, image in sorted((ref_images or {}).items()):
            if image is None:
                continue
            item, block, note = ref_builder.encode_image_ref(
                video_vae, image, self.canvas, ref_image_size, cond_cache=cond_cache
            )
            static_items.append(item)
            static_blocks.append(block)
            notes.append("%s: %s" % (name, note))
        self.static_blocks = static_blocks
        self.static_items = static_items

        started = time.time()
        done = 0
        for chunk in iter_source_chunks(
            video_path,
            chunk_frames=geometry.chunk_frames,
            stride_frames=geometry.stride_frames,
            canvas=self.canvas,
            start_frame=start_frame,
            total_frames=self.target_frames,
            fps=fps,
            ffmpeg_location=ffmpeg_location,
        ):
            if chunk.index >= chunk_count:
                break
            target = self.path("precompute", chunk.index)
            meta_path = self.path("precompute", chunk.index, ".json")
            existing = _load(target)
            if existing is not None and os.path.exists(meta_path):
                done += 1
                continue

            frames = chunk.frames_u8.to(torch.float32).div_(255.0)
            items, block, note = ref_builder.encode_video_ref(
                video_vae, frames, self.canvas, cond_cache=cond_cache
            )
            qwen_frames = next(i["data"] for i in items if i.get("type") == "video")
            _save(
                target,
                {
                    "source_latent": block["latent"],
                    "qwen_frames": (qwen_frames * 255.0).round().clamp(0, 255).to(torch.uint8),
                },
            )
            _atomic_json(
                meta_path,
                {
                    "index": chunk.index,
                    "global_start": chunk.global_start,
                    "actual_frames": chunk.actual_frames,
                    "model_frames": chunk.model_frames,
                    "is_final": chunk.is_final,
                    "latent_t": int(block["latent"].shape[2]),
                    "note": note,
                },
            )
            self.event(pass_="A", chunk=chunk.index, actual=chunk.actual_frames)
            done += 1
            del frames, items, block, qwen_frames, chunk
        if done != chunk_count:
            raise RuntimeError("source ended after %d/%d chunks" % (done, chunk_count))
        if self.manifest:
            self.manifest.update_state(pass_a_chunks=done)
        logging.info("%s pass A complete: %d chunks in %.1f s; %s", LOG, done,
                     time.time() - started, "; ".join(notes) or "no static refs")
        return done

    def pass_b(self, *, clip, prompt, chunk_count, cond_cache):
        started = time.time()
        done = 0
        for index in range(chunk_count):
            target = self.path("conditioning", index)
            meta_path = self.path("conditioning", index, ".json")
            if _load(target) is not None and os.path.exists(meta_path):
                done += 1
                continue
            stored = _load(self.path("precompute", index))
            if stored is None:
                raise RuntimeError("pass B: missing precompute for chunk %d" % index)
            qwen_frames = stored["qwen_frames"].to(torch.float32).div_(255.0)
            items = list(self.static_items) + [{
                "type": "video",
                "data": qwen_frames,
                "timestamps": [i / 2.0 for i in range(qwen_frames.shape[0])],
            }]
            conditioning = harness._encode(clip, prompt, items, cond_cache)
            payload, metadata = _pack_conditioning(conditioning)
            _save(target, payload)
            _atomic_json(meta_path, metadata)
            self.event(pass_="B", chunk=index)
            done += 1
            del stored, qwen_frames, items, conditioning
        if self.manifest:
            self.manifest.update_state(pass_b_chunks=done)
        logging.info("%s pass B complete: %d chunks in %.1f s", LOG, done, time.time() - started)
        return done

    def pass_c(self, *, model, sampler, sigmas, chunk_count, on_sampled=None):
        geometry = self.geometry
        started = time.time()
        overlap_start, overlap_count = (None, None)
        if self.carry != CARRY_NONE:
            overlap_start, overlap_count = geometry.overlap_slice()

        prefix = self._first_invalid("samples", chunk_count)
        self._remove_suffix("samples", prefix, chunk_count)
        carry_latent = None
        if prefix:
            previous = _load(self.path("samples", prefix - 1))
            carry_latent = previous["video_latent"]

        for index in range(prefix, chunk_count):
            stored = _load(self.path("precompute", index))
            cond_store = _load(self.path("conditioning", index))
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
            conditioning = _unpack_conditioning(cond_store, cond_meta)
            conditioning = harness.attach_refs(conditioning, self.static_blocks + [block])

            conditions = []
            if carry_latent is not None and self.carry != CARRY_NONE:
                count = 1 if self.carry == CARRY_FRAME else overlap_count
                conditions = [TargetAlignedCondition(
                    latent=carry_latent[:, :, overlap_start:overlap_start + count],
                    target_latent_start=0,
                    label="carry from chunk %d" % (index - 1),
                )]
            arm_model = patch_target_conditions(model, conditions)
            latent = harness.empty_av_latent(self.canvas, geometry)
            video_latent, audio_latent = harness.sample(
                model=arm_model,
                conditioning=conditioning,
                latent=latent,
                sampler=sampler,
                sigmas=sigmas,
                seed=self.chunk_seed(index),
            )
            video_latent = video_latent.to("cpu", torch.float32)
            _save(self.path("samples", index), {
                "video_latent": video_latent,
                "audio_latent": audio_latent.to("cpu", torch.float32),
            })
            _atomic_json(self.path("samples", index, ".json"), {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
            })
            carry_latent = video_latent
            self.event(pass_="C", chunk=index, carry=self.carry, conditions=len(conditions))
            # Decoding here rather than in a later pass costs nothing extra -
            # the same decode happens either way - and it is the only point at
            # which finished pixels exist while the run is still going.
            if on_sampled is not None:
                on_sampled(index, video_latent)
            if self.manifest:
                self.manifest.update_state(pass_c_chunks=index + 1)
            del stored, cond_store, conditioning, arm_model, latent, audio_latent
            elapsed = time.time() - started
            logging.info("%s pass C: chunk %d/%d done (%.1f s)", LOG, index + 1, chunk_count, elapsed)
        return chunk_count

    def _emit_chunk(self, index, latent, *, video_vae, chunk_count, writer,
                    save_frames, written, out_dir, pbar=None):
        """Decode one chunk, append it to the video, return frames written.

        Shared by the batched and interleaved decode paths so both cut at the
        stride identically - the seam behaviour must not depend on when the
        decode happened.
        """
        geometry = self.geometry
        pixels = decode_chunk(video_vae, latent).to("cpu", torch.float32)
        # Before any trimming: the assembled output never contains the overlap.
        diagnostics.emit_video(self, index, pixels)
        on_decoded_pixels(self, index, pixels, chunk_count)
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
                    os.path.join(out_dir, "frame_%06d.png" % (written + i))
                )
        if pbar is not None:
            _send_preview(pbar, frames_u8[-1], index + 1, chunk_count)
        self.event(pass_="D", chunk=index, frames=take, total=written + take)
        if self.manifest:
            self.manifest.update_state(pass_d_chunks=index + 1,
                                       frames_written=written + take)
        del pixels, frames_u8
        gc.collect()
        return take

    def pass_d(self, *, video_vae, chunk_count, save_frames=True,
               source_video=None, start_frame=0,
               preserve_audio=True, ffmpeg_location=None):
        geometry = self.geometry
        out_dir = os.path.join(self.root, "frames")
        raw_video = os.path.join(self.root, "output", "video_only.mkv")
        final_video = os.path.join(self.root, "output", "final.mp4")
        written = 0
        started = time.time()
        pbar = _progress_bar(chunk_count)

        writer = FFmpegVideoWriter(
            raw_video,
            width=self.canvas[0],
            height=self.canvas[1],
            fps=geometry.fps,
            ffmpeg_location=ffmpeg_location,
        ).open()
        completed = False
        try:
            for index in range(chunk_count):
                stored = _load(self.path("samples", index))
                if stored is None:
                    raise RuntimeError("pass D: chunk %d has no sampled latent" % index)
                take = self._emit_chunk(
                    index, stored["video_latent"], video_vae=video_vae,
                    chunk_count=chunk_count, writer=writer,
                    save_frames=save_frames, written=written, out_dir=out_dir,
                    pbar=pbar)
                if take == 0:
                    break
                written += take
                del stored
            self._check_frame_total(written)
            completed = True
        finally:
            # Only a validated run may promote raw_video to its final name; a
            # short or failed pass leaves nothing that looks finished.
            close_writers(writer, commit=completed)

        output_path = raw_video
        if preserve_audio and source_video:
            try:
                output_path = mux_source_audio(
                    raw_video,
                    source_video,
                    final_video,
                    start_frame=start_frame,
                    frame_count=written,
                    fps=geometry.fps,
                    ffmpeg_location=ffmpeg_location,
                )
            except Exception as exc:
                logging.warning("%s source-audio mux failed; keeping video-only output: %s", LOG, exc)
                output_path = raw_video
        logging.info("%s pass D complete: %d frames in %.1f s", LOG, written, time.time() - started)
        return written, output_path

    def _check_frame_total(self, written):
        """Assert the assembled length, before anything is committed.

        Callers run this inside the writer's ``try`` rather than after it: a
        run that assembled the wrong number of frames must not leave a video
        behind under the name that means "complete".
        """
        if written != self.target_frames:
            raise RuntimeError("assembled %d frames, expected exactly %d"
                               % (written, self.target_frames))

    def _finalize_video(self, *, written, preserve_audio,
                        source_video, start_frame, ffmpeg_location, raw_video,
                        final_video):
        self._check_frame_total(written)
        output_path = raw_video
        if preserve_audio and source_video:
            try:
                output_path = mux_source_audio(
                    raw_video, source_video, final_video,
                    start_frame=start_frame, frame_count=written,
                    fps=self.geometry.fps, ffmpeg_location=ffmpeg_location)
            except Exception as exc:
                logging.warning("%s source-audio mux failed; keeping video-only "
                                "output: %s", LOG, exc)
                output_path = raw_video
        return output_path

    def pass_cd(self, *, model, sampler, sigmas, video_vae, chunk_count,
                save_frames=True, source_video=None,
                start_frame=0, preserve_audio=True, ffmpeg_location=None):
        """Sample and decode chunk by chunk, writing the video as it goes.

        The decode is the same work either way, so doing it here rather than in
        a trailing pass costs nothing and makes the run watchable: each chunk's
        last frame goes to the node preview as soon as it exists. The trade is
        model residency - the DiT and the VAE alternate every chunk instead of
        each staying put for a whole pass - which is why `pass_d` remains for
        the batched path.
        """
        out_dir = os.path.join(self.root, "frames")
        raw_video = os.path.join(self.root, "output", "video_only.mkv")
        final_video = os.path.join(self.root, "output", "final.mp4")
        started = time.time()
        pbar = _progress_bar(chunk_count)
        state = {"written": 0}

        writer = FFmpegVideoWriter(
            raw_video, width=self.canvas[0], height=self.canvas[1],
            fps=self.geometry.fps, ffmpeg_location=ffmpeg_location).open()

        def emit(index, latent):
            take = self._emit_chunk(
                index, latent, video_vae=video_vae, chunk_count=chunk_count,
                writer=writer, save_frames=save_frames,
                written=state["written"], out_dir=out_dir, pbar=pbar)
            state["written"] += take

        completed = False
        try:
            # Chunks sampled by an earlier interrupted run are skipped by
            # pass_c, so their frames would never reach the writer. Decode them
            # from disk first, in order, before sampling resumes.
            resume_from = self._first_invalid("samples", chunk_count)
            if resume_from:
                logging.info("%s resuming: decoding %d already-sampled chunk(s)",
                             LOG, resume_from)
            for index in range(resume_from):
                stored = _load(self.path("samples", index))
                if stored is None:
                    raise RuntimeError("pass CD: chunk %d has no sampled latent" % index)
                emit(index, stored["video_latent"])
                del stored

            self.pass_c(model=model, sampler=sampler, sigmas=sigmas,
                        chunk_count=chunk_count, on_sampled=emit)
            self._check_frame_total(state["written"])
            completed = True
        finally:
            close_writers(writer, commit=completed)

        written = state["written"]
        output_path = self._finalize_video(
            written=written,
            preserve_audio=preserve_audio, source_video=source_video,
            start_frame=start_frame, ffmpeg_location=ffmpeg_location,
            raw_video=raw_video, final_video=final_video)
        logging.info("%s passes C+D complete: %d frames in %.1f s",
                     LOG, written, time.time() - started)
        return written, output_path


def run(*, video_path, start_frame, chunk_frames, overlap_frames, chunk_count,
        target_frames, model, clip, video_vae, audio_vae, prompt, sampler,
        sigmas, seed, carry, canvas, root, ref_images=None,
        ref_image_size="match", cond_cache="auto", save_frames=True, fps=24,
        preserve_audio=True, ffmpeg_location=None,
        runtime_config=None, interleave_decode=True):
    geometry = HarnessGeometry(
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
    ).validate()
    if carry not in CARRY_MODES:
        raise ValueError("unknown carry %r" % carry)
    if chunk_count <= 0 or target_frames <= 0:
        raise ValueError("chunk_count and target_frames must be positive")

    os.makedirs(root, exist_ok=True)
    identity = {
        "source": file_identity(video_path),
        "start_frame": int(start_frame),
        "target_frames": int(target_frames),
        "chunk_frames": int(chunk_frames),
        "overlap_frames": int(overlap_frames),
        "chunk_count": int(chunk_count),
        "carry": carry,
        "audio_carry_policy": "video_floor_v1",
        "canvas": list(canvas),
        "fps": int(fps),
        "seed": int(seed),
        "prompt_sha256": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
        "sigmas_sha256": tensor_digest(sigmas),
        "model": object_fingerprint(model),
        "clip": object_fingerprint(clip),
        "video_vae": object_fingerprint(video_vae),
        "audio_vae": object_fingerprint(audio_vae),
        "runtime": runtime_config or {},
        "ref_image_size": ref_image_size,
        "cond_cache": cond_cache,
    }
    manifest = RunManifest(root, identity)
    manifest.ensure()
    run_obj = LongFormRun(
        root,
        geometry,
        canvas,
        carry=carry,
        seed=seed,
        target_frames=target_frames,
        manifest=manifest,
    )

    # `execution.py:751` runs every prompt inside `torch.inference_mode()`, so
    # nodes never think about autograd. Calling the passes directly skips that.
    # The samplers carry their own `@torch.no_grad()` and `decode_chunk` is
    # decorated, but pass A runs the video/audio VAE encoders and pass B runs
    # the text encoder with no guard at all - the same exposure that let pass D
    # build an autograd graph over 5 temporal clips x 6 spatial tiles and pin
    # 22.5 GiB on a 12 GB card. Covering the whole run closes the rest.
    # Nesting this under the node path is a no-op.
    with torch.inference_mode():
        run_obj.pass_a(
            video_path=video_path,
            start_frame=start_frame,
            chunk_count=chunk_count,
            video_vae=video_vae,
            audio_vae=audio_vae,
            ref_images=ref_images,
            ref_image_size=ref_image_size,
            cond_cache=cond_cache,
            fps=fps,
            ffmpeg_location=ffmpeg_location,
        )
        run_obj.pass_b(clip=clip, prompt=prompt, chunk_count=chunk_count, cond_cache=cond_cache)
        if interleave_decode:
            frames, output_path = run_obj.pass_cd(
                model=model, sampler=sampler, sigmas=sigmas,
                video_vae=video_vae, chunk_count=chunk_count,
                save_frames=save_frames,
                source_video=video_path, start_frame=start_frame,
                preserve_audio=preserve_audio, ffmpeg_location=ffmpeg_location,
            )
        else:
            run_obj.pass_c(model=model, sampler=sampler, sigmas=sigmas,
                           chunk_count=chunk_count)
            frames, output_path = run_obj.pass_d(
                video_vae=video_vae,
                chunk_count=chunk_count,
                save_frames=save_frames,
                source_video=video_path,
                start_frame=start_frame,
                preserve_audio=preserve_audio,
                ffmpeg_location=ffmpeg_location,
            )
    manifest.update_state(complete=True, frames_written=frames, output_path=output_path)
    return {
        "root": root,
        "chunks": chunk_count,
        "frames": frames,
        "carry": carry,
        "profile": geometry.describe(),
        "output_path": output_path,
    }
