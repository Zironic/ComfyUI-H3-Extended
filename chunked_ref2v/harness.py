"""Five-phase experiment engine for the two-chunk Ref2V harness.

A  common VAE preprocessing
B  common Qwen preprocessing
C  generate Chunk A once
D  prepare dynamic carry assets
E  sample every Chunk B arm, then decode every successful arm

Sampling results are persisted before VAE decode. A decode failure therefore
cannot destroy a completed multi-hour diffusion run, and running all samples
before all decodes avoids switching DiT <-> VAE once per experiment.
"""

import gc
import logging
import os
import time

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import node_helpers
from comfy.model_management import InterruptProcessingException

from . import artifacts, comparison, metrics as metrics_mod, prompts, ref_builder

try:
    from .. import latent_cache
except ImportError:  # the self-tests import this package as top-level
    import latent_cache

from .experiments import CATALOG, union_dependencies
from .geometry import UnalignedProfileError
from .model_patch import patch_target_conditions
from .strategies import StrategyUnavailable

LOG_PREFIX = "[H3 Extended] harness"


def splitmix64(seed):
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


class SeedSet:
    def __init__(self, seed):
        self.node_seed = int(seed)
        self.chunk_a_noise = splitmix64(self.node_seed)
        self.chunk_b_noise = splitmix64(self.node_seed + 1)
        self.conditioning_augmentation = splitmix64(self.node_seed + 2)
        self.clamp_noise = splitmix64(self.node_seed + 3)
        self.monolithic_noise = splitmix64(self.node_seed + 4)

    def as_dict(self):
        return {
            "node_seed": self.node_seed,
            "chunk_a_noise_seed": self.chunk_a_noise,
            "chunk_b_noise_seed": self.chunk_b_noise,
            "conditioning_augmentation_seed": self.conditioning_augmentation,
            "clamp_noise_seed": self.clamp_noise,
            "monolithic_noise_seed": self.monolithic_noise,
        }


class HarnessContext:
    """Reusable CPU-resident assets shared by every Chunk B experiment."""

    def __init__(self, geometry, seeds, canvas):
        self.geometry = geometry
        self.seeds = seeds
        self.canvas = canvas
        self.base_prompt = ""
        self.source_chunk_a_pixels = None
        self.source_chunk_b_pixels = None
        self.static_ref_items = []
        self.static_ref_blocks = []
        self.source_ref_block_a = None
        self.source_ref_block_b = None
        self.qwen_ref_items_a = []
        self.qwen_ref_items_b = []
        self.dit_ref_blocks_a = []
        self.dit_ref_blocks_b = []
        self.conditionings = {}
        self.chunk_a_output_latent = None
        self.chunk_a_output_audio = None
        self.chunk_a_output_pixels = None
        self.overlap_latent = None
        self.overlap_pixels = None
        self.direct_frame_latent = None
        self.dynamic_assets = {}
        # Monolithic ground truth: one run covering both chunks' span at once.
        self.source_full_pixels = None
        self.monolithic_ref_block = None
        self.monolithic_ref_items = []
        self.monolithic_pixels = None
        self.monolithic_latent = None

    def prompt_for(self, policy):
        return prompts.build_prompt(self.base_prompt, policy)

    def conditioning_for(self, policy, fallback=None):
        key = prompts.encode_key(policy)
        cond = self.conditionings.get(key)
        if cond is None and policy == "original" and fallback:
            cond = self.conditionings.get(fallback)
        if cond is None:
            raise StrategyUnavailable(
                "no conditioning prepared for prompt policy %r (required key %r)"
                % (policy, key))
        return cond

    def require(self, name):
        value = self.dynamic_assets.get(name, getattr(self, name, None))
        if value is None:
            raise StrategyUnavailable(
                "asset %r was not prepared - its strategy did not declare the "
                "dependency, or Phase D was skipped" % name)
        return value


# ------------------------------------------------------------------ Phase A

def phase_a_vae(context, *, video_vae, audio_vae, source_frames, ref_images,
                ref_image_size, source_audio=None, monolithic=False):
    geometry = context.geometry
    if source_frames.shape[0] < geometry.required_source_frames:
        raise ValueError(
            "source video has %d frames; the %s profile needs at least %d"
            % (source_frames.shape[0], geometry.describe(), geometry.required_source_frames))

    a0, a1 = geometry.chunk_a_range
    b0, b1 = geometry.chunk_b_range
    width, height = context.canvas
    context.source_chunk_a_pixels = ref_builder.resize(
        source_frames[a0:a1], width, height).to("cpu", torch.float32)
    context.source_chunk_b_pixels = ref_builder.resize(
        source_frames[b0:b1], width, height).to("cpu", torch.float32)

    notes = []
    for name, image in sorted((ref_images or {}).items()):
        if image is None:
            continue
        item, block, note = ref_builder.encode_image_ref(
            video_vae, image, context.canvas, ref_image_size, cond_cache=cond_cache)
        context.static_ref_items.append(item)
        context.static_ref_blocks.append(block)
        notes.append("%s: %s" % (name, note))

    audio_a = _slice_audio(source_audio, geometry, 0)
    audio_b = _slice_audio(source_audio, geometry, geometry.stride_frames)
    items_a, block_a, note_a = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_a_pixels, context.canvas,
        audio=audio_a, audio_vae=audio_vae, cond_cache=cond_cache)
    items_b, block_b, note_b = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_b_pixels, context.canvas,
        audio=audio_b, audio_vae=audio_vae, cond_cache=cond_cache)

    context.dynamic_assets["source_audio_b"] = audio_b
    context.source_ref_block_a = block_a
    context.source_ref_block_b = block_b
    context.qwen_ref_items_a = context.static_ref_items + items_a
    context.qwen_ref_items_b = context.static_ref_items + items_b
    context.dit_ref_blocks_a = context.static_ref_blocks + [block_a]
    context.dit_ref_blocks_b = context.static_ref_blocks + [block_b]

    if monolithic:
        if not geometry.supports_monolithic:
            raise ValueError(
                "profile C=%d O=%d spans %d frames, which is not on the 17k+5 "
                "generation grid (needs S %% 17 == 0, here S=%d). A monolithic "
                "reference would have to be a different length than the chunked "
                "run, so the comparison would not be like for like."
                % (geometry.chunk_frames, geometry.overlap_frames,
                   geometry.total_frames, geometry.stride_frames))
        total = geometry.total_frames
        context.source_full_pixels = ref_builder.resize(
            source_frames[:total], width, height).to("cpu", torch.float32)
        items_full, block_full, note_full = ref_builder.encode_video_ref(
            video_vae, context.source_full_pixels, context.canvas,
            audio=_slice_audio_span(source_audio, geometry, 0, total),
            audio_vae=audio_vae)
        context.monolithic_ref_block = block_full
        context.monolithic_ref_items = context.static_ref_items + items_full
        notes.append("monolithic source (%d frames): %s" % (total, note_full))

    notes.extend(("source chunk A: %s" % note_a, "source chunk B: %s" % note_b))
    logging.info("%s phase A: canvas %dx%d; %s", LOG_PREFIX,
                 width, height, "; ".join(notes))
    log_memory("phase A (VAE preprocessing)")
    return notes


def _slice_audio(source_audio, geometry, start_frame):
    return _slice_audio_span(source_audio, geometry, start_frame,
                             geometry.chunk_frames)


def _slice_audio_span(source_audio, geometry, start_frame, frames):
    if source_audio is None:
        return None
    waveform = source_audio["waveform"]
    sample_rate = source_audio["sample_rate"]
    start = int(start_frame / geometry.fps * sample_rate)
    stop = start + int(frames / geometry.fps * sample_rate)
    clipped = waveform[..., start:stop]
    if clipped.shape[-1] == 0:
        return None
    return {"waveform": clipped, "sample_rate": sample_rate}


# ------------------------------------------------------------------ Phase B

def phase_b_qwen(context, *, clip, prompt, cond_cache="auto", monolithic=False):
    context.base_prompt = prompt
    context.conditionings["chunk_a"] = _encode(
        clip, prompt, context.qwen_ref_items_a, cond_cache)
    context.conditionings["chunk_b"] = _encode(
        clip, prompt, context.qwen_ref_items_b, cond_cache)
    if monolithic:
        # The reference run sees the same prompt against the full-length source,
        # which is exactly what a user would run without any of this machinery.
        context.conditionings["monolithic"] = _encode(
            clip, prompt, context.monolithic_ref_items, cond_cache)
    logging.info("%s phase B: encoded original prompt for %s", LOG_PREFIX,
                 "both chunks and the monolithic reference" if monolithic
                 else "both chunks")
    log_memory("phase B (text encoder)")


def _encode(clip, prompt, ref_items, cond_cache):
    try:
        from ..cond_cache import encode as encode_conditioning
    except ImportError:
        from cond_cache import encode as encode_conditioning
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    return encode_conditioning(clip, tokens, mode=cond_cache, label=prompt)


def attach_refs(conditioning, ref_blocks):
    if not ref_blocks:
        return conditioning
    return node_helpers.conditioning_set_values(
        conditioning, {"minimax_refs": list(ref_blocks)})


# ------------------------------------------------------------------ Phase C

def phase_c_chunk_a(context, *, model, sampler, sigmas, store, identity,
                    video_vae, reuse=True):
    geometry = context.geometry
    if reuse:
        cached = store.load_tensors("common", "chunk_a_output", identity)
        if cached is not None and "video_latent" in cached and "pixels" in cached:
            context.chunk_a_output_latent = cached["video_latent"]
            context.chunk_a_output_audio = cached.get("audio_latent")
            context.chunk_a_output_pixels = cached["pixels"]
            _derive_carry_assets(context)
            logging.info("%s phase C: reusing stored Chunk A (%s)", LOG_PREFIX, identity[:8])
            return True

    conditioning = attach_refs(context.conditionings["chunk_a"], context.dit_ref_blocks_a)
    video_latent, audio_latent = sample(
        model=model, conditioning=conditioning,
        latent=empty_av_latent(context.canvas, geometry), sampler=sampler,
        sigmas=sigmas, seed=context.seeds.chunk_a_noise)

    context.chunk_a_output_latent = video_latent.to("cpu", torch.float32)
    context.chunk_a_output_audio = audio_latent.to("cpu", torch.float32)
    store.save_tensors("common", "chunk_a_sample", {
        "video_latent": context.chunk_a_output_latent,
        "audio_latent": context.chunk_a_output_audio,
    }, identity=identity)

    context.chunk_a_output_pixels = decode_video(
        video_vae, context.chunk_a_output_latent).to("cpu", torch.float32)
    _derive_carry_assets(context)
    store.save_tensors("common", "chunk_a_output", {
        "video_latent": context.chunk_a_output_latent,
        "audio_latent": context.chunk_a_output_audio,
        "pixels": context.chunk_a_output_pixels,
    }, identity=identity)
    logging.info("%s phase C: generated Chunk A, latent %s, %d decoded frames",
                 LOG_PREFIX, list(context.chunk_a_output_latent.shape),
                 context.chunk_a_output_pixels.shape[0])
    return False


def _derive_carry_assets(context):
    geometry = context.geometry
    latent = context.chunk_a_output_latent
    pixels = context.chunk_a_output_pixels
    try:
        start, count = geometry.overlap_slice()
    except UnalignedProfileError:
        context.overlap_latent = None
        context.direct_frame_latent = None
    else:
        context.overlap_latent = latent[:, :, start:start + count].detach().clone()
        context.direct_frame_latent = latent[:, :, start:start + 1].detach().clone()
    s, c = geometry.stride_frames, geometry.chunk_frames
    if pixels is not None and pixels.shape[0] >= c:
        context.overlap_pixels = pixels[s:c].detach().clone()


# ------------------------------------------------- Phase C2 - ground truth

def phase_c_monolithic(context, *, model, sampler, sigmas, store, identity,
                       video_vae, reuse=True):
    """Generate the same span in one run, as ground truth for the chunked ones.

    This is what the user would have got without any chunking: one Ref2VA pass
    over `total_frames`, same prompt, same references, same canvas. Every
    chunked arm is then scored against it on the tail, which is the only place
    in this harness that measures against something other than itself.

    Cached like Chunk A - it is the same order of expense and, like Chunk A, no
    Chunk B experiment can change it.
    """
    geometry = context.geometry
    if reuse:
        cached = store.load_tensors("common", "monolithic_output", identity)
        if cached is not None and "pixels" in cached:
            context.monolithic_pixels = cached["pixels"]
            context.monolithic_latent = cached.get("video_latent")
            logging.info("%s monolithic reference: reusing stored run (%s)",
                         LOG_PREFIX, identity[:8])
            return True

    conditioning = attach_refs(context.conditionings["monolithic"],
                               context.static_ref_blocks + [context.monolithic_ref_block])
    latent = empty_av_latent_frames(context.canvas, geometry.total_frames, geometry.fps)
    video_latent, audio_latent = sample(
        model=model, conditioning=conditioning, latent=latent,
        sampler=sampler, sigmas=sigmas, seed=context.seeds.monolithic_noise)

    context.monolithic_latent = video_latent.to("cpu", torch.float32)
    store.save_tensors("common", "monolithic_sample", {
        "video_latent": context.monolithic_latent,
        "audio_latent": audio_latent.to("cpu", torch.float32),
    }, identity=identity)

    context.monolithic_pixels = decode_video(
        video_vae, context.monolithic_latent).to("cpu", torch.float32)
    store.save_tensors("common", "monolithic_output", {
        "video_latent": context.monolithic_latent,
        "pixels": context.monolithic_pixels,
    }, identity=identity)
    logging.info("%s monolithic reference: generated %d frames, latent %s",
                 LOG_PREFIX, context.monolithic_pixels.shape[0],
                 list(context.monolithic_latent.shape))
    return False


# ------------------------------------------------------------------ Phase D

def phase_d_dynamic(context, *, experiment_ids, video_vae, audio_vae, clip,
                    cond_cache, store, identity):
    deps = union_dependencies(experiment_ids)
    policies = {CATALOG[i].prompt_policy for i in experiment_ids}
    notes = []

    if deps.needs_dynamic_video_vae:
        notes += _phase_d_vae(
            context, deps, video_vae, audio_vae, experiment_ids, store, identity,
            cond_cache)

    required_keys = {prompts.encode_key(policy) for policy in policies}
    missing_keys = required_keys.difference(context.conditionings)
    if deps.needs_dynamic_qwen or missing_keys:
        notes += _phase_d_qwen(context, policies, clip, cond_cache)

    if notes:
        logging.info("%s phase D: %s", LOG_PREFIX, "; ".join(notes))
    log_memory("phase D (dynamic carry assets)")
    return deps, notes


def _phase_d_vae(context, deps, video_vae, audio_vae, experiment_ids, store, identity,
                 cond_cache="auto"):
    geometry = context.geometry
    notes = []

    if deps.needs_anchor_reencode:
        frame = context.chunk_a_output_pixels[
            geometry.stride_frames:geometry.stride_frames + 1]
        context.dynamic_assets["reencoded_frame_latent"] = (
            latent_cache.encode(video_vae, frame, mode=cond_cache,
                                label="anchor frame").to("cpu", torch.float32))
        notes.append("re-encoded anchor frame %d" % geometry.stride_frames)

    if "generated_overlap_video2" in experiment_ids:
        items, block, note = ref_builder.encode_video_ref(
            video_vae, context.overlap_pixels, context.canvas, cond_cache=cond_cache)
        context.dynamic_assets["video2_ref_block"] = block
        context.dynamic_assets["video2_ref_items"] = items
        notes.append("generated overlap as <Video 2> (%s)" % note)

    if "composite_source" in experiment_ids:
        frames = ref_builder.composite_frames(
            context.overlap_pixels, context.source_chunk_b_pixels,
            geometry.overlap_frames)
        audio_b = context.dynamic_assets.get("source_audio_b")
        items, block, note = ref_builder.encode_video_ref(
            video_vae, frames, context.canvas,
            audio=audio_b, audio_vae=audio_vae)
        context.dynamic_assets["composite_ref_block"] = block
        # The original Chunk B presentation already contains the source audio
        # item. Replacing the video with [audio, video] would duplicate it and
        # silently create a second <Audio N>. Keep only the replacement video.
        context.dynamic_assets["composite_ref_items"] = items[-1:] if audio_b else items
        notes.append("composite source reference (%s)" % note)

    payload = {}
    for key, value in (
        ("reencoded_frame_latent", context.dynamic_assets.get("reencoded_frame_latent")),
        ("video2_latent", (context.dynamic_assets.get("video2_ref_block") or {}).get("latent")),
        ("composite_latent", (context.dynamic_assets.get("composite_ref_block") or {}).get("latent")),
    ):
        if value is not None:
            payload[key] = value
    if payload:
        store.save_tensors("dynamic", "carry_assets", payload, identity=identity)
    return notes


def _phase_d_qwen(context, policies, clip, cond_cache):
    notes = []
    for policy in sorted(policies):
        key = prompts.encode_key(policy)
        if key in context.conditionings:
            continue
        if policy == "video2":
            items = context.qwen_ref_items_b + context.dynamic_assets.get(
                "video2_ref_items", [])
        elif policy == "composite":
            items = _replace_source_item(
                context, context.dynamic_assets.get("composite_ref_items"))
        else:
            items = context.qwen_ref_items_b
        context.conditionings[key] = _encode(
            clip, context.prompt_for(policy), items, cond_cache)
        notes.append("Qwen encode for prompt policy %r" % policy)
    return notes


def _replace_source_item(context, composite_items):
    if not composite_items:
        return context.qwen_ref_items_b
    source_items = [i for i in context.qwen_ref_items_b if i.get("type") == "video"]
    if not source_items:
        return context.qwen_ref_items_b + composite_items
    target = source_items[-1]
    out = []
    for item in context.qwen_ref_items_b:
        if item is target:
            out.extend(composite_items)
        else:
            out.append(item)
    return out


# ------------------------------------------------------------------ Phase E

def phase_e_experiments(context, *, experiment_ids, model, sampler, sigmas,
                        video_vae, store, continue_after_failure=True,
                        save_latents=True, save_frames=True, tail_frames=17):
    """Sample all arms first, then perform one grouped VAE decode phase."""
    results = []

    for experiment_id in experiment_ids:
        spec = CATALOG[experiment_id]
        record = {
            "experiment_id": experiment_id,
            "status": "pending",
            "strategy": spec.as_dict(),
            "dependencies": spec.dependencies().as_dict(),
        }
        started = time.time()
        _reset_peak_memory()
        try:
            outcome = sample_experiment(
                context, spec, model=model, sampler=sampler, sigmas=sigmas)
            record.update(outcome)
            recovery_path = store.save_experiment_tensors(
                experiment_id, "sampled_output", {
                    "video_latent": record["latent"],
                    "audio_latent": record.get("audio_latent"),
                })
            record["recovery_latent"] = recovery_path
            record["status"] = "sampled"
        except InterruptProcessingException:
            logging.warning("%s %s interrupted", LOG_PREFIX, experiment_id)
            raise
        except torch.cuda.OutOfMemoryError as exc:
            record["status"] = "oom"
            record["note"] = str(exc)
            _recover()
            logging.error("%s %s ran out of memory", LOG_PREFIX, experiment_id)
            if not continue_after_failure:
                results.append(record)
                raise
        except StrategyUnavailable as exc:
            record["status"] = "unavailable"
            record["note"] = str(exc)
            logging.warning("%s %s unavailable: %s", LOG_PREFIX, experiment_id, exc)
            if not continue_after_failure:
                results.append(record)
                raise
        except Exception as exc:
            record["status"] = "failed"
            record["note"] = "%s: %s" % (type(exc).__name__, exc)
            logging.exception("%s %s failed", LOG_PREFIX, experiment_id)
            if not continue_after_failure:
                results.append(record)
                raise
        finally:
            record["resources"] = _resource_snapshot(started)
        results.append(record)

    for record in results:
        if record.get("status") != "sampled":
            continue
        experiment_id = record["experiment_id"]
        try:
            pixels = decode_video(video_vae, record["latent"]).to("cpu", torch.float32)
            record["pixels"] = pixels
            record["metrics"] = metrics_mod.collect(
                geometry=context.geometry,
                chunk_a_latent=context.chunk_a_output_latent,
                chunk_a_pixels=context.chunk_a_output_pixels,
                chunk_b_latent=record["latent"],
                chunk_b_pixels=pixels,
                source_chunk_b_pixels=context.source_chunk_b_pixels,
                monolithic_pixels=context.monolithic_pixels,
                tail_frames=tail_frames,
            )
            record["boundary"] = comparison.boundary_playback(
                chunk_a_pixels=context.chunk_a_output_pixels,
                chunk_b_pixels=pixels, geometry=context.geometry)
            record["status"] = "completed"
        except InterruptProcessingException:
            raise
        except Exception as exc:
            record["status"] = "decode_failed"
            record["note"] = "%s: %s" % (type(exc).__name__, exc)
            logging.exception("%s %s decode failed; sampled latent preserved at %s",
                              LOG_PREFIX, experiment_id, record.get("recovery_latent"))
            if not continue_after_failure:
                raise
        _persist_decoded(
            store, record, geometry=context.geometry,
            save_latents=save_latents, save_frames=save_frames)

    baseline = next((r.get("metrics") for r in results
                     if r["experiment_id"] == "baseline_none"), None)
    for record in results:
        if record.get("metrics"):
            record["metrics"].update(
                metrics_mod.compare_to_baseline(record["metrics"], baseline))
    return results


def sample_experiment(context, spec, *, model, sampler, sigmas):
    strategy = spec.strategy()
    prepared = strategy.prepare(context, spec)
    conditioning = attach_refs(prepared.conditioning, prepared.dit_ref_blocks)
    arm_model = patch_target_conditions(
        model, prepared.target_conditions,
        position_policy=prepared.position_policy)
    video_latent, audio_latent = sample(
        model=arm_model, conditioning=conditioning,
        latent=empty_av_latent(context.canvas, context.geometry),
        sampler=sampler, sigmas=sigmas, seed=context.seeds.chunk_b_noise)
    return {
        "latent": video_latent.to("cpu", torch.float32),
        "audio_latent": audio_latent.to("cpu", torch.float32),
        "prepared": {
            "prompt_chars": len(prepared.prompt or ""),
            "target_conditions": [c.describe() for c in prepared.target_conditions],
            "reference_blocks": len(prepared.dit_ref_blocks),
            "position_policy": prepared.position_policy,
            **prepared.metadata,
        },
    }


def run_experiment(context, spec, *, model, sampler, sigmas, video_vae):
    outcome = sample_experiment(context, spec, model=model, sampler=sampler, sigmas=sigmas)
    pixels = decode_video(video_vae, outcome["latent"]).to("cpu", torch.float32)
    outcome["pixels"] = pixels
    outcome["metrics"] = metrics_mod.collect(
        geometry=context.geometry,
        chunk_a_latent=context.chunk_a_output_latent,
        chunk_a_pixels=context.chunk_a_output_pixels,
        chunk_b_latent=outcome["latent"],
        chunk_b_pixels=pixels,
        source_chunk_b_pixels=context.source_chunk_b_pixels,
        monolithic_pixels=context.monolithic_pixels,
    )
    outcome["boundary"] = comparison.boundary_playback(
        chunk_a_pixels=context.chunk_a_output_pixels,
        chunk_b_pixels=pixels, geometry=context.geometry)
    return outcome


def _persist_decoded(store, record, *, geometry, save_latents, save_frames):
    experiment_id = record["experiment_id"]
    record["artifacts"] = {}
    recovery = record.get("recovery_latent")
    if save_latents and recovery:
        record["artifacts"]["latent"] = recovery
    elif record.get("status") == "completed" and recovery:
        try:
            os.remove(recovery)
        except OSError:
            pass
    elif recovery:
        record["artifacts"]["recovery_latent"] = recovery

    if save_frames and record.get("pixels") is not None:
        directory = store.experiment_dir(experiment_id)
        frames_dir = os.path.join(directory, "frames")
        artifacts.save_frames(frames_dir, record["pixels"])
        record["artifacts"]["frames"] = frames_dir
        if record.get("boundary") is not None:
            boundary_dir = os.path.join(directory, "boundary")
            artifacts.save_frames(boundary_dir, record["boundary"], "seam")
            record["artifacts"]["boundary"] = boundary_dir

    # Metrics and full-resolution artifacts are complete. Retain only the
    # overlap-sized bounded preview needed by node outputs, not 73 full frames
    # per experiment. The sampled latent is already persisted as well.
    if record.get("pixels") is not None:
        record["pixels"] = comparison.preview_clip(
            record["pixels"][:geometry.overlap_frames])
    record.pop("latent", None)
    record.pop("audio_latent", None)


# ---------------------------------------------------------------- sampling

def empty_av_latent(canvas, geometry, batch_size=1):
    width, height = canvas
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, geometry.target_latent_t,
                         height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, geometry.audio_latent_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def empty_av_latent_frames(canvas, frame_count, fps, batch_size=1):
    """Target for an arbitrary legal length - used by the monolithic reference."""
    from .geometry import AUDIO_LATENT_FPS, video_latent_t

    width, height = canvas
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, video_latent_t(frame_count),
                         height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2,
                         round(frame_count / fps * AUDIO_LATENT_FPS)], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def sample(*, model, conditioning, latent, sampler, sigmas, seed):
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": conditioning})
    guider.set_cfg(1.0)
    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, seed)
    samples = guider.sample(
        noise, latent_image, sampler, sigmas,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    return samples.unbind()


def decode_video(video_vae, video_latent):
    images = video_vae.decode(video_latent)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


# ---------------------------------------------------------------- resources

def _reset_peak_memory():
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass


def _resource_snapshot(started):
    snapshot = {"runtime_seconds": round(time.time() - started, 2)}
    if not torch.cuda.is_available():
        return snapshot
    try:
        free, total = torch.cuda.mem_get_info()
        snapshot["free_physical_mb"] = free // (1 << 20)
        snapshot["total_physical_mb"] = total // (1 << 20)
        snapshot["peak_reserved_mb"] = torch.cuda.max_memory_reserved() // (1 << 20)
        snapshot["peak_allocated_mb"] = torch.cuda.max_memory_allocated() // (1 << 20)
    except Exception:
        pass
    return snapshot


def _recover():
    gc.collect()
    try:
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass


def log_memory(label):
    """Record the full memory picture at a phase boundary. Observation only.

    Four numbers, because the interesting question is which of them moves:

      torch reserved/allocated  the compute working set - activations, the
                                packed sequence, whatever the forward holds
      AIMDO reclaimable         weight pages faulted in by dynamic VRAM loading,
                                evictable on demand and therefore not really
                                "used" even though the driver reports them so
      free physical             what the driver will hand out right now

    A run whose torch reserved stays small while free physical collapses is
    AIMDO filling spare VRAM with its page cache, which is what it is for.
    A run where torch reserved itself climbs is a genuine working-set problem.
    Nothing here allocates, frees, or evicts.
    """
    if not torch.cuda.is_available():
        return None
    try:
        free, total = torch.cuda.mem_get_info()
        reserved = torch.cuda.memory_reserved()
        allocated = torch.cuda.memory_allocated()
        reclaimable = 0
        try:
            import comfy.memory_management
            if comfy.memory_management.aimdo_enabled:
                import comfy_aimdo.model_vbar
                reclaimable = int(comfy_aimdo.model_vbar.vbars_analyze(
                    torch.cuda.current_device()) or 0)
        except Exception:
            pass
        mb = 1 << 20
        snapshot = {
            "free_physical_mb": free // mb,
            "aimdo_reclaimable_mb": reclaimable // mb,
            "available_mb": (free + reclaimable) // mb,
            "torch_reserved_mb": reserved // mb,
            "torch_allocated_mb": allocated // mb,
            "peak_reserved_mb": torch.cuda.max_memory_reserved() // mb,
        }
        logging.info(
            "%s memory after %s: free %d MB + AIMDO reclaimable %d MB = %d MB "
            "available; torch reserved %d MB (peak %d MB), allocated %d MB, "
            "total %d MB", LOG_PREFIX, label,
            snapshot["free_physical_mb"], snapshot["aimdo_reclaimable_mb"],
            snapshot["available_mb"], snapshot["torch_reserved_mb"],
            snapshot["peak_reserved_mb"], snapshot["torch_allocated_mb"],
            total // mb)
        return snapshot
    except Exception as exc:
        logging.warning("%s could not read memory after %s: %s", LOG_PREFIX, label, exc)
        return None


def release_models(label):
    """Drop every resident model at a phase boundary.

    The phase split only bounds VRAM if the previous stage actually leaves.
    ComfyUI keeps a model resident until a *new prompt* is queued or it is told
    to release, and the harness switches DiT -> VAE -> DiT inside a single
    prompt execution - so that trigger never fires and the log reads
    "0 models unloaded" at exactly the transitions that matter. Without this the
    video VAE is still holding VRAM when the next DiT forward starts, and the
    guard correctly cancels a run that had plenty of headroom on paper.

    Best-effort: a failure here costs memory, and must never fail a run.
    """
    before = _free_mb()
    try:
        comfy.model_management.unload_all_models()
    except Exception as exc:
        logging.warning("%s could not unload models at %s: %s", LOG_PREFIX, label, exc)
    gc.collect()
    try:
        comfy.model_management.soft_empty_cache(force=True)
    except Exception:
        pass
    after = _free_mb()
    if before is not None and after is not None:
        logging.info("%s released models after %s: free physical %d -> %d MB",
                     LOG_PREFIX, label, before, after)
    return after


def _free_mb():
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()[0] // (1 << 20)
    except Exception:
        return None
