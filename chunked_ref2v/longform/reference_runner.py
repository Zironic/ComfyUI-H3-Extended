"""Generic disk-backed long-form MiniMax H3 reference-video generation.

Unlike the Ref2V runner, this module has no source timeline. References are
encoded once, the same conditioning is reused for every chunk, and only the
generated overlap is carried forward.
"""

from __future__ import annotations

import gc
import hashlib
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

from .manifest import RunManifest, object_fingerprint, tensor_digest
from .runner import (
    CARRY_FRAME,
    CARRY_MODES,
    CARRY_NONE,
    CARRY_OVERLAP,
    LongFormRun,
    _atomic_json,
    _load,
    _pack_conditioning,
    _save,
    _unpack_conditioning,
)
from .writer import FFmpegVideoWriter

LOG = "[H3 Extended] longform reference"


def _ordered_values(values):
    for name, value in sorted((values or {}).items()):
        if value is not None:
            yield name, value


def _paired_audio(ref_video_audios, video_name):
    suffix = video_name.rsplit("_", 1)[-1]
    return (ref_video_audios or {}).get("ref_video_audio_" + suffix)


def _reference_identity(ref_images, ref_videos, ref_video_audios, ref_audios):
    identity = {"images": {}, "videos": {}, "video_audios": {}, "audios": {}}
    for name, value in _ordered_values(ref_images):
        identity["images"][name] = tensor_digest(value)
    for name, value in _ordered_values(ref_videos):
        identity["videos"][name] = tensor_digest(value)
    for name, value in _ordered_values(ref_video_audios):
        identity["video_audios"][name] = {
            "waveform": tensor_digest(value["waveform"]),
            "sample_rate": int(value["sample_rate"]),
        }
    for name, value in _ordered_values(ref_audios):
        identity["audios"][name] = {
            "waveform": tensor_digest(value["waveform"]),
            "sample_rate": int(value["sample_rate"]),
        }
    return identity


class LongFormReferenceRun(LongFormRun):
    """Long-form run whose conditioning contains only persistent references."""

    def prepare_references(self, *, clip, prompt, video_vae, audio_vae,
                           ref_images, ref_videos, ref_video_audios, ref_audios,
                           ref_image_size, cond_cache):
        target = os.path.join(self.root, "conditioning", "persistent.safetensors")
        meta_path = os.path.join(self.root, "conditioning", "persistent.json")
        refs_path = os.path.join(self.root, "precompute", "persistent.safetensors")
        refs_meta_path = os.path.join(self.root, "precompute", "persistent.json")

        cached_cond = _load(target)
        cached_refs = _load(refs_path)
        if (cached_cond is not None and cached_refs is not None and
                os.path.exists(meta_path) and os.path.exists(refs_meta_path)):
            with open(meta_path, encoding="utf-8") as fh:
                cond_meta = json.load(fh)
            with open(refs_meta_path, encoding="utf-8") as fh:
                refs_meta = json.load(fh)
            self.static_items = []
            self.static_blocks = _unpack_blocks(cached_refs, refs_meta)
            return _unpack_conditioning(cached_cond, cond_meta), refs_meta.get("notes", [])

        items, blocks, notes = [], [], []
        for name, image in _ordered_values(ref_images):
            item, block, note = ref_builder.encode_image_ref(
                video_vae, image, self.canvas, ref_image_size, cond_cache=cond_cache)
            items.append(item)
            blocks.append(block)
            notes.append("%s: %s" % (name, note))

        for name, frames in _ordered_values(ref_videos):
            audio = _paired_audio(ref_video_audios, name)
            video_items, block, note = ref_builder.encode_video_ref(
                video_vae, frames, self.canvas, audio=audio,
                audio_vae=audio_vae, cond_cache=cond_cache)
            items.extend(video_items)
            blocks.append(block)
            notes.append("%s: %s" % (name, note))

        for name, audio in _ordered_values(ref_audios):
            audio_latent, ref_audio_t = ref_builder.encode_ref_audio(
                audio_vae, audio, cond_cache=cond_cache)
            items.append({"type": "audio"})
            blocks.append({
                "kind": "audio",
                "ref_audio_t": int(ref_audio_t),
                "audio_latent": audio_latent,
            })
            notes.append("%s: audio latent t=%d" % (name, ref_audio_t))

        conditioning = harness._encode(clip, prompt, items, cond_cache)
        payload, cond_meta = _pack_conditioning(conditioning)
        _save(target, payload)
        _atomic_json(meta_path, cond_meta)

        refs_payload, refs_meta = _pack_blocks(blocks)
        refs_meta["notes"] = notes
        _save(refs_path, refs_payload)
        _atomic_json(refs_meta_path, refs_meta)
        self.static_items = items
        self.static_blocks = blocks
        if self.manifest:
            self.manifest.update_state(references_prepared=True)
        return conditioning, notes

    def sample_chunks(self, *, model, conditioning, sampler, sigmas, chunk_count,
                      on_sampled=None):
        geometry = self.geometry
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
            chunk_conditioning = harness.attach_refs(conditioning, self.static_blocks)
            conditions = []
            if carry_latent is not None and self.carry != CARRY_NONE:
                count = 1 if self.carry == CARRY_FRAME else overlap_count
                conditions.append(TargetAlignedCondition(
                    latent=carry_latent[:, :, overlap_start:overlap_start + count],
                    target_latent_start=0,
                    label="carry from chunk %d" % (index - 1),
                ))
            armed = patch_target_conditions(model, conditions)
            latent = harness.empty_av_latent(self.canvas, geometry)
            video_latent, audio_latent = harness.sample(
                model=armed,
                conditioning=chunk_conditioning,
                latent=latent,
                sampler=sampler,
                sigmas=sigmas,
                seed=self.chunk_seed(index),
            )
            video_latent = video_latent.to("cpu", torch.float32)
            audio_latent = audio_latent.to("cpu", torch.float32)
            _save(self.path("samples", index), {
                "video_latent": video_latent,
                "audio_latent": audio_latent,
            })
            _atomic_json(self.path("samples", index, ".json"), {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
            })
            carry_latent = video_latent
            if on_sampled is not None:
                on_sampled(index, video_latent)
            if self.manifest:
                self.manifest.update_state(pass_c_chunks=index + 1)
            self.event(pass_="C", chunk=index, carry=self.carry,
                       conditions=len(conditions))
            del chunk_conditioning, armed, latent, audio_latent
            gc.collect()
            logging.info("%s chunk %d/%d sampled", LOG, index + 1, chunk_count)
        return chunk_count

    def sample_and_write(self, *, model, conditioning, sampler, sigmas,
                         video_vae, chunk_count, output_video, save_frames,
                         ffmpeg_location):
        out_dir = os.path.join(self.root, "frames")
        output_path = os.path.join(self.root, "output", "final.mp4")
        writer = None
        state = {"written": 0}
        pbar = None
        try:
            from .runner import _progress_bar
            pbar = _progress_bar(chunk_count)
        except Exception:
            pass

        if output_video:
            writer = FFmpegVideoWriter(
                output_path, width=self.canvas[0], height=self.canvas[1],
                fps=self.geometry.fps, ffmpeg_location=ffmpeg_location).open()

        def emit(index, latent):
            take = self._emit_chunk(
                index, latent, video_vae=video_vae, chunk_count=chunk_count,
                writer=writer, save_frames=save_frames,
                written=state["written"], out_dir=out_dir, pbar=pbar)
            state["written"] += take

        try:
            resume_from = self._first_invalid("samples", chunk_count)
            for index in range(resume_from):
                stored = _load(self.path("samples", index))
                if stored is None:
                    raise RuntimeError("chunk %d has no stored sample" % index)
                emit(index, stored["video_latent"])
            self.sample_chunks(
                model=model, conditioning=conditioning, sampler=sampler,
                sigmas=sigmas, chunk_count=chunk_count, on_sampled=emit)
        finally:
            if writer is not None:
                writer.close(commit=True)

        if state["written"] != self.target_frames:
            raise RuntimeError("assembled %d frames, expected exactly %d" %
                               (state["written"], self.target_frames))
        return state["written"], output_path if output_video else ""


def _pack_blocks(blocks):
    tensors = {}
    metadata = {"blocks": []}
    for index, block in enumerate(blocks):
        record = {}
        for key, value in block.items():
            if isinstance(value, torch.Tensor):
                tensor_key = "block_%03d__%s" % (index, key)
                tensors[tensor_key] = value
                record[key] = {"tensor": tensor_key}
            else:
                record[key] = value
        metadata["blocks"].append(record)
    return tensors, metadata


def _unpack_blocks(tensors, metadata):
    blocks = []
    for record in metadata.get("blocks", []):
        block = {}
        for key, value in record.items():
            if isinstance(value, dict) and "tensor" in value:
                block[key] = tensors[value["tensor"]]
            else:
                block[key] = value
        blocks.append(block)
    return blocks


def run(*, chunk_frames, overlap_frames, chunk_count, target_frames, model,
        clip, video_vae, audio_vae, prompt, sampler, sigmas, seed, carry,
        canvas, root, ref_images=None, ref_videos=None,
        ref_video_audios=None, ref_audios=None, ref_image_size="native",
        cond_cache="auto", save_frames=False, output_video=True,
        ffmpeg_location=None, runtime_config=None):
    geometry = HarnessGeometry(
        chunk_frames=chunk_frames, overlap_frames=overlap_frames).validate()
    if carry not in CARRY_MODES:
        raise ValueError("unknown carry %r" % carry)
    if chunk_count <= 0 or target_frames <= 0:
        raise ValueError("chunk_count and target_frames must be positive")

    os.makedirs(root, exist_ok=True)
    identity = {
        "mode": "reference",
        "target_frames": int(target_frames),
        "chunk_frames": int(chunk_frames),
        "overlap_frames": int(overlap_frames),
        "chunk_count": int(chunk_count),
        "carry": carry,
        "canvas": list(canvas),
        "fps": int(geometry.fps),
        "seed": int(seed),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "sigmas_sha256": tensor_digest(sigmas),
        "model": object_fingerprint(model),
        "clip": object_fingerprint(clip),
        "video_vae": object_fingerprint(video_vae),
        "audio_vae": object_fingerprint(audio_vae),
        "runtime": runtime_config or {},
        "ref_image_size": ref_image_size,
        "cond_cache": cond_cache,
        "references": _reference_identity(
            ref_images, ref_videos, ref_video_audios, ref_audios),
    }
    manifest = RunManifest(root, identity)
    manifest.ensure()
    run_obj = LongFormReferenceRun(
        root, geometry, canvas, carry=carry, seed=seed,
        target_frames=target_frames, manifest=manifest)

    with torch.inference_mode():
        conditioning, notes = run_obj.prepare_references(
            clip=clip, prompt=prompt, video_vae=video_vae,
            audio_vae=audio_vae, ref_images=ref_images,
            ref_videos=ref_videos, ref_video_audios=ref_video_audios,
            ref_audios=ref_audios, ref_image_size=ref_image_size,
            cond_cache=cond_cache)
        frames, output_path = run_obj.sample_and_write(
            model=model, conditioning=conditioning, sampler=sampler,
            sigmas=sigmas, video_vae=video_vae, chunk_count=chunk_count,
            output_video=output_video, save_frames=save_frames,
            ffmpeg_location=ffmpeg_location)

    manifest.update_state(complete=True, frames_written=frames,
                          output_path=output_path)
    return {
        "root": root,
        "chunks": chunk_count,
        "frames": frames,
        "carry": carry,
        "profile": geometry.describe(),
        "output_path": output_path,
        "reference_notes": notes,
    }
