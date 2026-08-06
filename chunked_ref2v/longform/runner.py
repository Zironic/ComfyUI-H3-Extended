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

from .chunk_stream import iter_source_chunks, actual_frames_for_chunk
from .manifest import (
    RunManifest,
    file_identity,
    object_fingerprint,
    tensor_digest,
)
from .writer import FFmpegVideoWriter, mux_source_audio

LOG = "[H3 Extended] longform"
CARRY_NONE = "none"
CARRY_FRAME = "direct_latent_frame"
CARRY_OVERLAP = "direct_latent_overlap"
CARRY_MODES = (CARRY_OVERLAP, CARRY_FRAME, CARRY_NONE)


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

    def pass_c(self, *, model, sampler, sigmas, chunk_count):
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
            if self.manifest:
                self.manifest.update_state(pass_c_chunks=index + 1)
            del stored, cond_store, conditioning, arm_model, latent, audio_latent
            elapsed = time.time() - started
            logging.info("%s pass C: chunk %d/%d done (%.1f s)", LOG, index + 1, chunk_count, elapsed)
        return chunk_count

    def pass_d(self, *, video_vae, chunk_count, save_frames=True,
               output_video=True, source_video=None, start_frame=0,
               preserve_audio=True, ffmpeg_location=None):
        geometry = self.geometry
        out_dir = os.path.join(self.root, "frames")
        raw_video = os.path.join(self.root, "output", "video_only.mkv")
        final_video = os.path.join(self.root, "output", "final.mp4")
        written = 0
        started = time.time()

        writer = None
        if output_video:
            writer = FFmpegVideoWriter(
                raw_video,
                width=self.canvas[0],
                height=self.canvas[1],
                fps=geometry.fps,
                ffmpeg_location=ffmpeg_location,
            ).open()
        try:
            for index in range(chunk_count):
                stored = _load(self.path("samples", index))
                if stored is None:
                    raise RuntimeError("pass D: chunk %d has no sampled latent" % index)
                pixels = decode_chunk(video_vae, stored["video_latent"]).to("cpu", torch.float32)
                remaining = self.target_frames - written
                if remaining <= 0:
                    break
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
                written += take
                self.event(pass_="D", chunk=index, frames=take, total=written)
                if self.manifest:
                    self.manifest.update_state(pass_d_chunks=index + 1, frames_written=written)
                del stored, pixels, frames_u8
                gc.collect()
        finally:
            if writer is not None:
                writer.close(commit=True)

        if written != self.target_frames:
            raise RuntimeError("assembled %d frames, expected exactly %d" % (written, self.target_frames))
        output_path = raw_video if output_video else ""
        if output_video and preserve_audio and source_video:
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


def run(*, video_path, start_frame, chunk_frames, overlap_frames, chunk_count,
        target_frames, model, clip, video_vae, audio_vae, prompt, sampler,
        sigmas, seed, carry, canvas, root, ref_images=None,
        ref_image_size="match", cond_cache="auto", save_frames=True, fps=24,
        output_video=True, preserve_audio=True, ffmpeg_location=None,
        runtime_config=None):
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
    run_obj.pass_c(model=model, sampler=sampler, sigmas=sigmas, chunk_count=chunk_count)
    frames, output_path = run_obj.pass_d(
        video_vae=video_vae,
        chunk_count=chunk_count,
        save_frames=save_frames,
        output_video=output_video,
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
