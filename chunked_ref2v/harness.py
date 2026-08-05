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

    def as_dict(self):
        return {
            "node_seed": self.node_seed,
            "chunk_a_noise_seed": self.chunk_a_noise,
            "chunk_b_noise_seed": self.chunk_b_noise,
            "conditioning_augmentation_seed": self.conditioning_augmentation,
            "clamp_noise_seed": self.clamp_noise,
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

    def prompt_for(self, policy):
        return prompts.build_prompt(self.base_prompt, policy)

    def conditioning_for(self, policy, fallback=None):
        """Return the exact Qwen encode required by `policy`.

        Non-original arms must never silently consume the original conditioning;
        doing so produces a plausible output while invalidating the experiment.
        `fallback` exists only for legacy callers and is accepted for the
        original policy.
        """
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
                ref_image_size, source_audio=None):
    geometry = context.geometry
    if source_frames.shape[0] < geometry.required_source_frames:
        raise ValueError(
            "source video has %d frames; the %s profile needs at least %d"
            % (source_frames.shape[0], geometry.describe(), geometry.required_source_frames))

    a0, a1 = geometry.chunk_a_range
    b0, b1 = geometry.chunk_b_range
    width, height = context.canvas
    # Store the exact canvas-space source used by both the VAE and metrics.
    context.source_chunk_a_pixels = ref_builder.resize(
        source_frames[a0:a1], width, height).to("cpu", torch.float32)
    context.source_chunk_b_pixels = ref_builder.resize(
        source_frames[b0:b1], width, height).to("cpu", torch.float32)

    notes = []
    for name, image in sorted((ref_images or {}).items()):
        if image is None:
            continue
        item, block, note = ref_builder.encode_image_ref(
            video_vae, image, context.canvas, ref_image_size)
        context.static_ref_items.append(item)
        context.static_ref_blocks.append(block)
        notes.append("%s: %s" % (name, note))

    audio_a = _slice_audio(source_audio, geometry, 0)
    audio_b = _slice_audio(source_audio, geometry, geometry.stride_frames)
    items_a, block_a, note_a = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_a_pixels, context.canvas,
        audio=audio_a, audio_vae=audio_vae)
    items_b, block_b, note_b = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_b_pixels, context.canvas,
        audio=audio_b, audio_vae=audio_vae)

    context.dynamic_assets["source_audio_b"] = audio_b
    context.source_ref_block_a = block_a
    context.source_ref_block_b = block_b
    context.qwen_ref_items_a = context.static_ref_items + items_a
    context.qwen_ref_items_b = context.static_ref_items + items_b
    context.dit_ref_blocks_a = context.static_ref_blocks + [block_a]
    context.dit_ref_blocks_b = context.static_ref_blocks + [block_b]

    notes.extend(("source chunk A: %s" % note_a, "source chunk B: %s" % note_b))
    logging.info("%s phase A: canvas %dx%d; %s", LOG_PREFIX,
                 width, height, "; ".join(notes))
    return notes


def _slice_audio(source_audio, geometry, start_frame):
    if source_audio is None:
        return None
    waveform = source_audio["waveform"]
    sample_rate = source_audio["sample_rate"]
    start = int(start_frame / geometry.fps * sample_rate)
    stop = start + int(geometry.chunk_frames / geometry.fps * sample_rate)
    clipped = waveform[..., start:stop]
    if clipped.shape[-1] == 0:
        return None
    return {"waveform": clipped, "sample_rate": sample_rate}


# ------------------------------------------------------------------ Phase B

def phase_b_qwen(context, *, clip, prompt, cond_cache="auto"):
    context.base_prompt = prompt
    context.conditionings["chunk_a"] = _encode(
        clip, prompt, context.qwen_ref_items_a, cond_cache)
    context.conditionings["chunk_b"] = _encode(
        clip, prompt, context.qwen_ref_items_b, cond_cache)
    logging.info("%s phase B: encoded original prompt for both chunks", LOG_PREFIX)


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
    # Persist the completed sample before attempting the much larger VAE decode.
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


# ------------------------------------------------------------------ Phase D

def phase_d_dynamic(context, *, experiment_ids, video_vae, audio_vae, clip,
                    cond_cache, store, identity):
    deps = union_dependencies(experiment_ids)
    policies = {CATALOG[i].prompt_policy for i in experiment_ids}
    notes = []

    if deps.needs_dynamic_video_vae:
        notes += _phase_d_vae(
            context, deps, video_vae, audio_vae, experiment_ids, store, identity)

    required_keys = {prompts.encode_key(policy) for policy in policies}
    missing_keys = required_keys.difference(context.conditionings)
    if deps.needs_dynamic_qwen or missing_keys:
        notes += _phase_d_qwen(context, policies, clip, cond_cache)

    if notes:
        logging.info("%s phase D: %s", LOG_PREFIX, "; ".join(notes))
    return deps, notes


def _phase_d_vae(context, deps, video_vae, audio_vae, experiment_ids, store, identity):
    geometry = context.geometry
    notes = []

    if deps.needs_anchor_reencode:
        frame = context.chunk_a_output_pixels[
            geometry.stride_frames:geometry.stride_frames + 1]
        context.dynamic_assets["reencoded_frame_latent"] = (
            video_vae.encode(frame).to("cpu", torch.float32))
        notes.append("re-encoded anchor frame %d" % geometry.stride_frames)

    if "generated_overlap_video2" in experiment_ids:
        items, block, note = ref_builder.encode_video_ref(
            video_vae, context.overlap_pixels, context.canvas)
        context.dynamic_assets["video2_ref_block"] = block
        context.dynamic_assets["video2_ref_items"] = items
        notes.append("generated overlap as <Video 2> (%s)" % note)

    if "composite_source" in experiment_ids:
        frames = ref_builder.composite_frames(
            context.overlap_pixels, context.source_chunk_b_pixels,
            geometry.overlap_frames)
        # Preserve exactly the same source-audio condition used by the original
        # Chunk B block; otherwise the composite arm changes two variables.
        audio_b = context.dynamic_assets.get("source_audio_b")
        items, block, note = ref_builder.encode_video_ref(
            video_vae, frames, context.canvas,
            audio=audio_b, audio_vae=audio_vae)
        context.dynamic_assets["composite_ref_block"] = block
        context.dynamic_assets["composite_ref_items"] = items
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
                        save_latents=True, save_frames=True):
    """Sample all arms first, then perform one grouped VAE decode phase."""
    results = []

    # E1: DiT sampling. Every successful sample is persisted immediately.
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
            # User cancellation and the VRAM guard intentionally use this same
            # exception. Never swallow it: the executor must unwind immediately.
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

    # E2: VAE decode and pixel diagnostics. This causes one DiT -> VAE switch.
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
            store, record, save_latents=save_latents, save_frames=save_frames)

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
    """Compatibility helper for one-off callers: sample, decode, and measure."""
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
    )
    outcome["boundary"] = comparison.boundary_playback(
        chunk_a_pixels=context.chunk_a_output_pixels,
        chunk_b_pixels=pixels, geometry=context.geometry)
    return outcome


def _persist_decoded(store, record, *, save_latents, save_frames):
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
    record.pop("audio_latent", None)


# ---------------------------------------------------------------- sampling

def empty_av_latent(canvas, geometry, batch_size=1):
    width, height = canvas
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, geometry.target_latent_t,
                         height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, geometry.audio_latent_t], device=device)
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
