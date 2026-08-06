"""Sequential N-chunk Ref2V engine - Passes A-D of `long-form plan.md`.

The two-chunk harness answers "which carry strategy", and structurally cannot
answer "does it drift", because drift accumulates per boundary and two chunks
have one. This runs many chunks in sequence so drift, seam behaviour under
repetition, and within-chunk detail consistency become observable.

Phase ordering is the load-bearing part. Sixty-four chunks interleaving
DiT/VAE/Qwen would stage a 14.6 GB encoder and a 19.5 GB DiT sixty-four times
each, so each model is made resident exactly once:

    A  video+audio VAE   encode every source chunk reference
    B  Qwen              encode every chunk presentation
    C  DiT               sample every chunk, carrying latent state forward
    D  video VAE         decode every chunk, stitch, write frames

Every pass persists per chunk before moving on, so an interruption costs one
chunk rather than the run. Pass C is sequential by nature - chunk i needs chunk
i-1's sampled latent - so a missing middle chunk invalidates everything after it.
"""

import gc
import json
import logging
import os
import time

import torch

import comfy.model_management

try:
    from .. import harness, ref_builder
    from ..geometry import HarnessGeometry
    from ..layout_ops import TargetAlignedCondition
    from ..model_patch import patch_target_conditions
except ImportError:  # top-level import in tests
    import harness
    import ref_builder
    from geometry import HarnessGeometry
    from layout_ops import TargetAlignedCondition
    from model_patch import patch_target_conditions

from .chunk_stream import chunk_count_for, iter_source_chunks

LOG = "[H3 Extended] longform"
CARRY_NONE = "none"
CARRY_FRAME = "direct_latent_frame"
CARRY_OVERLAP = "direct_latent_overlap"
CARRY_MODES = (CARRY_OVERLAP, CARRY_FRAME, CARRY_NONE)


def _save(path, tensors):
    import comfy.utils
    tmp = "%s.%d.tmp" % (path, os.getpid())
    payload = {k: v.detach().to("cpu").contiguous()
               for k, v in tensors.items() if v is not None}
    try:
        comfy.utils.save_torch_file(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def decode_chunk(video_vae, latent):
    """Decode one chunk. The same call `VAEDecode` makes.

    There is no slab option here on purpose. The H3 video VAE already chunks
    itself - `MiniMaxH3VideoVAE` sets `handles_tiling`, splits into 17-frame
    temporal clips and 256px spatial tiles internally, and its `decode_tiled`
    is literally `return self.decode(z)`, discarding tile_t/overlap_t. An
    earlier slab path here looked like it worked around a memory cliff; it was
    routing to the identical call. The real cause was host allocator churn,
    fixed in `cli._bootstrap` by setting MIMALLOC_PURGE_DELAY as main.py does.
    """
    images = video_vae.decode(latent)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2],
                                images.shape[-1])
    return images


def _load(path):
    import comfy.utils
    if not os.path.exists(path):
        return None
    try:
        return comfy.utils.load_torch_file(path, safe_load=True)
    except Exception as exc:
        logging.warning("%s unreadable artifact %s (%s) - will regenerate",
                        LOG, os.path.basename(path), exc)
        return None


class LongFormRun:
    """One run directory plus the phase logic that fills it."""

    def __init__(self, root, geometry, canvas, *, carry=CARRY_OVERLAP, seed=0):
        self.root = root
        self.geometry = geometry
        self.canvas = canvas
        self.carry = carry
        self.seed = int(seed)
        for name in ("precompute", "conditioning", "samples", "frames"):
            os.makedirs(os.path.join(root, name), exist_ok=True)
        self.events = os.path.join(root, "events.jsonl")

    # -- helpers ---------------------------------------------------------

    def path(self, kind, index, suffix=".safetensors"):
        return os.path.join(self.root, kind, "chunk_%06d%s" % (index, suffix))

    def event(self, **fields):
        fields["t"] = time.time()
        try:
            with open(self.events, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(fields, default=str) + "\n")
        except OSError:
            pass

    def chunk_seed(self, index):
        # Every chunk gets its own noise but derived from one run seed, so a run
        # is reproducible and no two chunks share a noise tensor.
        return harness.splitmix64(self.seed + 1000 + index)

    # -- Pass A ----------------------------------------------------------

    def pass_a(self, *, video_path, start_frame, chunk_count, video_vae,
               audio_vae, ref_images, ref_image_size, cond_cache, fps=24):
        """Encode every source chunk's reference while the VAE is resident."""
        geometry = self.geometry
        total = (chunk_count - 1) * geometry.stride_frames + geometry.chunk_frames

        static_items, static_blocks, notes = [], [], []
        for name, image in sorted((ref_images or {}).items()):
            if image is None:
                continue
            item, block, note = ref_builder.encode_image_ref(
                video_vae, image, self.canvas, ref_image_size, cond_cache=cond_cache)
            static_items.append(item)
            static_blocks.append(block)
            notes.append("%s: %s" % (name, note))
        _save(os.path.join(self.root, "static_refs.safetensors"),
              {"ref_%d" % i: b["latent"] for i, b in enumerate(static_blocks)})
        self.static_blocks = static_blocks
        self.static_items = static_items

        done = 0
        started = time.time()
        for chunk in iter_source_chunks(
                video_path, chunk_frames=geometry.chunk_frames,
                stride_frames=geometry.stride_frames, canvas=self.canvas,
                start_frame=start_frame, total_frames=total, fps=fps):
            if chunk.index >= chunk_count:
                break
            target = self.path("precompute", chunk.index)
            meta_path = self.path("precompute", chunk.index, ".json")
            if os.path.exists(target) and os.path.exists(meta_path):
                done += 1
                continue

            frames = chunk.frames_u8.to(torch.float32).div_(255.0)
            items, block, note = ref_builder.encode_video_ref(
                video_vae, frames, self.canvas, cond_cache=cond_cache)
            qwen_frames = next(i["data"] for i in items if i.get("type") == "video")
            _save(target, {"source_latent": block["latent"],
                           "qwen_frames": (qwen_frames * 255.0).round().clamp(0, 255)
                           .to(torch.uint8)})
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump({"index": chunk.index, "global_start": chunk.global_start,
                           "actual_frames": chunk.actual_frames,
                           "model_frames": chunk.model_frames,
                           "is_final": chunk.is_final,
                           "latent_t": int(block["latent"].shape[2]),
                           "note": note}, fh)
            self.event(pass_="A", chunk=chunk.index, actual=chunk.actual_frames)
            done += 1
            del frames, items, block, qwen_frames, chunk
            if done % 8 == 0:
                logging.info("%s pass A: %d/%d chunks (%.1f s)", LOG, done,
                             chunk_count, time.time() - started)

        logging.info("%s pass A complete: %d chunks in %.1f s; %s", LOG, done,
                     time.time() - started, "; ".join(notes) or "no static refs")
        harness.log_memory("long-form pass A")
        return done

    # -- Pass B ----------------------------------------------------------

    def pass_b(self, *, clip, prompt, chunk_count, cond_cache):
        """Encode every chunk presentation while Qwen is resident."""
        started = time.time()
        done = 0
        for index in range(chunk_count):
            target = self.path("conditioning", index)
            if os.path.exists(target):
                done += 1
                continue
            stored = _load(self.path("precompute", index))
            if stored is None:
                raise RuntimeError("pass B: missing precompute for chunk %d" % index)
            qwen_frames = stored["qwen_frames"].to(torch.float32).div_(255.0)
            items = list(self.static_items) + [
                {"type": "video", "data": qwen_frames,
                 "timestamps": [i / 2.0 for i in range(qwen_frames.shape[0])]}]
            conditioning = harness._encode(clip, prompt, items, cond_cache)
            # Conditioning is [[tensor, dict]]; persist the tensor and rebuild
            # the wrapper on load rather than pickling Comfy structures.
            _save(target, {"cond": conditioning[0][0],
                           "pooled": conditioning[0][1].get("pooled_output")})
            extra = {k: v for k, v in conditioning[0][1].items()
                     if k not in ("pooled_output",)}
            with open(self.path("conditioning", index, ".json"), "w",
                      encoding="utf-8") as fh:
                json.dump({"keys": sorted(extra)}, fh)
            self.event(pass_="B", chunk=index)
            done += 1
            del stored, qwen_frames, items, conditioning
            if done % 8 == 0:
                logging.info("%s pass B: %d/%d (%.1f s)", LOG, done, chunk_count,
                             time.time() - started)
        logging.info("%s pass B complete: %d chunks in %.1f s", LOG, done,
                     time.time() - started)
        harness.log_memory("long-form pass B")
        return done

    # -- Pass C ----------------------------------------------------------

    def pass_c(self, *, model, sampler, sigmas, chunk_count):
        """Sample every chunk with the DiT resident, carrying latent state."""
        geometry = self.geometry
        started = time.time()
        carry_latent = None
        overlap_start, overlap_count = None, None
        if self.carry != CARRY_NONE:
            overlap_start, overlap_count = geometry.overlap_slice()

        for index in range(chunk_count):
            target = self.path("samples", index)
            existing = _load(target)
            if existing is not None and "video_latent" in existing:
                carry_latent = existing["video_latent"]
                continue

            stored = _load(self.path("precompute", index))
            cond_store = _load(self.path("conditioning", index))
            if stored is None or cond_store is None:
                raise RuntimeError("pass C: chunk %d missing inputs" % index)

            block = {"kind": "video", "latent_t": int(stored["source_latent"].shape[2]),
                     "latent_h": self.canvas[1] // 16, "latent_w": self.canvas[0] // 16,
                     "ref_audio_t": 0, "latent": stored["source_latent"],
                     "audio_latent": None}
            extra = {}
            if cond_store.get("pooled") is not None:
                extra["pooled_output"] = cond_store["pooled"]
            conditioning = [[cond_store["cond"], extra]]
            conditioning = harness.attach_refs(conditioning,
                                               self.static_blocks + [block])

            conditions = []
            if carry_latent is not None and self.carry != CARRY_NONE:
                count = 1 if self.carry == CARRY_FRAME else overlap_count
                conditions = [TargetAlignedCondition(
                    latent=carry_latent[:, :, overlap_start:overlap_start + count],
                    target_latent_start=0,
                    label="carry from chunk %d" % (index - 1))]
            arm_model = patch_target_conditions(model, conditions)

            latent = harness.empty_av_latent(self.canvas, geometry)
            video_latent, audio_latent = harness.sample(
                model=arm_model, conditioning=conditioning, latent=latent,
                sampler=sampler, sigmas=sigmas, seed=self.chunk_seed(index))
            video_latent = video_latent.to("cpu", torch.float32)
            _save(target, {"video_latent": video_latent,
                           "audio_latent": audio_latent.to("cpu", torch.float32)})
            carry_latent = video_latent
            self.event(pass_="C", chunk=index, carry=self.carry,
                       conditions=len(conditions))
            del stored, cond_store, conditioning, arm_model, latent, audio_latent
            elapsed = time.time() - started
            logging.info("%s pass C: chunk %d/%d done (%.1f s elapsed, %.1f s/chunk)",
                         LOG, index + 1, chunk_count, elapsed, elapsed / (index + 1))
        harness.log_memory("long-form pass C")
        return chunk_count

    # -- Pass D ----------------------------------------------------------

    def pass_d(self, *, video_vae, chunk_count, save_frames=True):
        """Decode sequentially and hard-cut at the stride.

        Every chunk contributes its own first `S` frames, so the output always
        switches to the newest rendering at a boundary. That makes any seam
        maximally visible, which is what a diagnostic run wants; a `best_cut`
        search would hide exactly the artifact we are looking for.
        """
        geometry = self.geometry
        out_dir = os.path.join(self.root, "frames")
        started = time.time()
        written = 0
        for index in range(chunk_count):
            stored = _load(self.path("samples", index))
            if stored is None:
                raise RuntimeError("pass D: chunk %d has no sampled latent" % index)
            meta_path = self.path("precompute", index, ".json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
            pixels = decode_chunk(
                video_vae, stored["video_latent"]).to("cpu", torch.float32)

            is_final = index == chunk_count - 1
            take = pixels.shape[0] if is_final else min(geometry.stride_frames,
                                                        pixels.shape[0])
            if is_final:
                take = min(take, meta.get("actual_frames", take))
            if save_frames:
                from PIL import Image
                import numpy as np
                for i in range(take):
                    arr = pixels[i].clamp(0, 1).numpy()
                    Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8)).save(
                        os.path.join(out_dir, "frame_%06d.png" % (written + i)))
            written += take
            self.event(pass_="D", chunk=index, frames=take, total=written)
            del stored, pixels
            gc.collect()
            if (index + 1) % 8 == 0:
                logging.info("%s pass D: %d/%d chunks, %d frames (%.1f s)", LOG,
                             index + 1, chunk_count, written, time.time() - started)
        logging.info("%s pass D complete: %d frames (%.1f s) in %s", LOG, written,
                     time.time() - started, out_dir)
        harness.log_memory("long-form pass D")
        return written


def run(*, video_path, start_frame, chunk_frames, overlap_frames, chunk_count,
        model, clip, video_vae, audio_vae, prompt, sampler, sigmas, seed,
        carry, canvas, root, ref_images=None, ref_image_size="match",
        cond_cache="auto", save_frames=True, fps=24):
    """Execute all four passes. Returns a summary dict."""
    geometry = HarnessGeometry(chunk_frames=chunk_frames,
                               overlap_frames=overlap_frames).validate()
    if carry not in CARRY_MODES:
        raise ValueError("unknown carry %r" % carry)
    os.makedirs(root, exist_ok=True)
    run_obj = LongFormRun(root, geometry, canvas, carry=carry, seed=seed)

    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"video_path": video_path, "start_frame": start_frame,
                   "chunk_frames": chunk_frames, "overlap_frames": overlap_frames,
                   "stride_frames": geometry.stride_frames,
                   "chunk_count": chunk_count, "carry": carry,
                   "canvas": list(canvas), "seed": seed,
                   "prompt_chars": len(prompt)}, fh, indent=2)

    logging.info("%s starting: %d chunks C=%d O=%d S=%d carry=%s canvas=%dx%d",
                 LOG, chunk_count, chunk_frames, overlap_frames,
                 geometry.stride_frames, carry, canvas[0], canvas[1])

    run_obj.pass_a(video_path=video_path, start_frame=start_frame,
                   chunk_count=chunk_count, video_vae=video_vae,
                   audio_vae=audio_vae, ref_images=ref_images,
                   ref_image_size=ref_image_size, cond_cache=cond_cache, fps=fps)
    run_obj.pass_b(clip=clip, prompt=prompt, chunk_count=chunk_count,
                   cond_cache=cond_cache)
    run_obj.pass_c(model=model, sampler=sampler, sigmas=sigmas,
                   chunk_count=chunk_count)
    frames = run_obj.pass_d(video_vae=video_vae, chunk_count=chunk_count,
                            save_frames=save_frames)
    return {"root": root, "chunks": chunk_count, "frames": frames,
            "carry": carry, "profile": geometry.describe()}
