"""The five-phase experiment engine.

    A  common VAE preprocessing   - canvas, source slices, static refs
    B  common Qwen preprocessing  - the unmodified prompt, both presentations
    C  generate Chunk A           - sampled once, ever
    D  dynamic carry preprocessing- only what the selected strategies asked for
    E  Chunk B experiment runs    - one payload per arm, same seed and sigmas

The phase split exists to bound model residency. Each of the three stages -
14.6 GB text encoder, 19.5 GB DiT, 4.9 GB VAE - is loaded at most once per run
rather than once per experiment, which is the difference between a suite that
fits in 12 GB and one that thrashes.

Chunk A is the expensive asset and every arm shares it. It is generated once,
stored losslessly, and reused across runs whose identity matches; adding an
experiment never regenerates it.
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
    """Deterministic sub-seed derivation. Python's hash() is salted per process."""
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


class SeedSet:
    """Independent seeds, derived so a node seed reproduces the whole run.

    `payload["seed"]` drives conditioning noise augmentation as well as sampling
    noise (`comfy/ldm/minimax/model.py:473-481`), so every Chunk B arm shares one
    augmentation seed. Varying it per arm would perturb the source and carried
    conditioning and make the comparison unclean at 0.1% amplitude - small, but
    exactly the size of the effect being measured.
    """

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
    """Reusable assets, all CPU-resident between model stages."""

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

        self.conditionings = {}          # encode key -> conditioning
        self.chunk_a_output_latent = None
        self.chunk_a_output_audio = None
        self.chunk_a_output_pixels = None

        self.overlap_latent = None
        self.overlap_pixels = None
        self.direct_frame_latent = None
        self.dynamic_assets = {}

    # --- strategy-facing accessors --------------------------------------

    def prompt_for(self, policy):
        return prompts.build_prompt(self.base_prompt, policy)

    def conditioning_for(self, policy, fallback="chunk_b"):
        key = prompts.encode_key(policy)
        cond = self.conditionings.get(key)
        if cond is None:
            cond = self.conditionings.get(fallback)
        if cond is None:
            raise StrategyUnavailable(
                "no conditioning prepared for prompt policy %r (key %r)" % (policy, key))
        return cond

    def require(self, name):
        value = self.dynamic_assets.get(name, getattr(self, name, None))
        if value is None:
            raise StrategyUnavailable(
                "asset %r was not prepared - its strategy did not declare the "
                "dependency, or Phase D was skipped" % name)
        return value


# ---------------------------------------------------------------------------
# Phase A - common VAE preprocessing
# ---------------------------------------------------------------------------

def phase_a_vae(context, *, video_vae, audio_vae, source_frames, ref_images,
                ref_image_size, source_audio=None):
    """Pin the canvas, cut both source chunks, encode everything static."""
    geometry = context.geometry
    if source_frames.shape[0] < geometry.required_source_frames:
        raise ValueError(
            "source video has %d frames; the %s profile needs at least %d "
            "(Chunk B ends at global frame %d)"
            % (source_frames.shape[0], geometry.describe(),
               geometry.required_source_frames, geometry.chunk_b_range[1] - 1))

    a0, a1 = geometry.chunk_a_range
    b0, b1 = geometry.chunk_b_range
    context.source_chunk_a_pixels = source_frames[a0:a1].to("cpu", torch.float32)
    context.source_chunk_b_pixels = source_frames[b0:b1].to("cpu", torch.float32)

    notes = []
    for name, image in sorted((ref_images or {}).items()):
        if image is None:
            continue
        item, block, note = ref_builder.encode_image_ref(
            video_vae, image, context.canvas, ref_image_size)
        context.static_ref_items.append(item)
        context.static_ref_blocks.append(block)
        notes.append("%s: %s" % (name, note))

    items_a, block_a, note_a = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_a_pixels, context.canvas,
        audio=_slice_audio(source_audio, geometry, 0), audio_vae=audio_vae)
    items_b, block_b, note_b = ref_builder.encode_video_ref(
        video_vae, context.source_chunk_b_pixels, context.canvas,
        audio=_slice_audio(source_audio, geometry, geometry.stride_frames),
        audio_vae=audio_vae)

    context.source_ref_block_a = block_a
    context.source_ref_block_b = block_b
    context.qwen_ref_items_a = context.static_ref_items + items_a
    context.qwen_ref_items_b = context.static_ref_items + items_b
    context.dit_ref_blocks_a = context.static_ref_blocks + [block_a]
    context.dit_ref_blocks_b = context.static_ref_blocks + [block_b]

    notes.append("source chunk A: %s" % note_a)
    notes.append("source chunk B: %s" % note_b)
    logging.info("%s phase A: canvas %dx%d; %s", LOG_PREFIX,
                 context.canvas[0], context.canvas[1], "; ".join(notes))
    return notes


def _slice_audio(source_audio, geometry, start_frame):
    """The source soundtrack over one chunk's window, at the source rate."""
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


# ---------------------------------------------------------------------------
# Phase B - common Qwen preprocessing
# ---------------------------------------------------------------------------

def phase_b_qwen(context, *, clip, prompt, cond_cache="auto"):
    """Encode the unmodified prompt against both chunk presentations.

    Reused by the baseline, every direct-latent arm, and any other
    unmodified-prompt strategy - which is most of the catalog.
    """
    context.base_prompt = prompt
    context.conditionings["chunk_a"] = _encode(
        clip, prompt, context.qwen_ref_items_a, cond_cache)
    context.conditionings["chunk_b"] = _encode(
        clip, prompt, context.qwen_ref_items_b, cond_cache)
    logging.info("%s phase B: encoded the original prompt for both chunks", LOG_PREFIX)


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


# ---------------------------------------------------------------------------
# Phase C - generate Chunk A
# ---------------------------------------------------------------------------

def phase_c_chunk_a(context, *, model, sampler, sigmas, store, identity,
                    video_vae, reuse=True):
    """Sample Chunk A once and derive every carry asset from its output.

    Reused across runs whose identity matches, because regenerating it is the
    single most expensive thing the harness can do and nothing about a Chunk B
    experiment can change it.
    """
    geometry = context.geometry
    if reuse:
        cached = store.load_tensors("common", "chunk_a_output", identity)
        if cached is not None and "video_latent" in cached and "pixels" in cached:
            context.chunk_a_output_latent = cached["video_latent"]
            context.chunk_a_output_audio = cached.get("audio_latent")
            context.chunk_a_output_pixels = cached["pixels"]
            _derive_carry_assets(context)
            logging.info("%s phase C: reusing stored Chunk A (%s)",
                         LOG_PREFIX, identity[:8])
            return True

    conditioning = attach_refs(context.conditionings["chunk_a"], context.dit_ref_blocks_a)
    latent = empty_av_latent(context.canvas, geometry)
    video_latent, audio_latent = sample(
        model=model, conditioning=conditioning, latent=latent,
        sampler=sampler, sigmas=sigmas, seed=context.seeds.chunk_a_noise)

    context.chunk_a_output_latent = video_latent.to("cpu", torch.float32)
    context.chunk_a_output_audio = audio_latent.to("cpu", torch.float32)
    context.chunk_a_output_pixels = decode_video(video_vae, video_latent).to("cpu", torch.float32)
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
    """Slice the direct-latent and pixel carry assets out of Chunk A's output.

    Losslessly: a `.clone()` off the sampler's own output, on CPU. Anything
    lossy here would silently become the thing every arm is compared against.
    """
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


# ---------------------------------------------------------------------------
# Phase D - dynamic carry preprocessing
# ---------------------------------------------------------------------------

def phase_d_dynamic(context, *, experiment_ids, video_vae, audio_vae, clip,
                    cond_cache, store, identity):
    """Prepare only what the selected strategies asked for, grouped by model.

    Reading the union of dependencies first is what keeps Qwen from being
    staged once per experiment. Nothing here runs when the selected suite is all
    direct-latent arms, which is the common case.
    """
    deps = union_dependencies(experiment_ids)
    policies = {CATALOG[i].prompt_policy for i in experiment_ids}
    notes = []

    if deps.needs_dynamic_video_vae:
        notes += _phase_d_vae(context, deps, video_vae, experiment_ids, store, identity)
    if deps.needs_dynamic_qwen:
        notes += _phase_d_qwen(context, policies, clip, cond_cache)

    if notes:
        logging.info("%s phase D: %s", LOG_PREFIX, "; ".join(notes))
    return deps, notes


def _phase_d_vae(context, deps, video_vae, experiment_ids, store, identity):
    geometry = context.geometry
    notes = []

    if deps.needs_anchor_reencode:
        frame = context.chunk_a_output_pixels[geometry.stride_frames:geometry.stride_frames + 1]
        context.dynamic_assets["reencoded_frame_latent"] = (
            video_vae.encode(frame).to("cpu", torch.float32))
        notes.append("re-encoded anchor frame %d" % geometry.stride_frames)

    wants_video2 = "generated_overlap_video2" in experiment_ids
    wants_composite = "composite_source" in experiment_ids

    if wants_video2:
        items, block, note = ref_builder.encode_video_ref(
            video_vae, context.overlap_pixels, context.canvas)
        context.dynamic_assets["video2_ref_block"] = block
        context.dynamic_assets["video2_ref_items"] = items
        notes.append("generated overlap as <Video 2> (%s)" % note)

    if wants_composite:
        frames = ref_builder.composite_frames(
            context.overlap_pixels, context.source_chunk_b_pixels, geometry.overlap_frames)
        items, block, note = ref_builder.encode_video_ref(
            video_vae, frames, context.canvas)
        context.dynamic_assets["composite_ref_block"] = block
        context.dynamic_assets["composite_ref_items"] = items
        notes.append("composite source reference (%s)" % note)

    store.save_tensors("dynamic", "carry_assets", {
        k: v for k, v in (
            ("reencoded_frame_latent", context.dynamic_assets.get("reencoded_frame_latent")),
            ("video2_latent", (context.dynamic_assets.get("video2_ref_block") or {}).get("latent")),
            ("composite_latent", (context.dynamic_assets.get("composite_ref_block") or {}).get("latent")),
        ) if v is not None
    }, identity=identity)
    return notes


def _phase_d_qwen(context, policies, clip, cond_cache):
    notes = []
    for policy in sorted(policies):
        key = prompts.encode_key(policy)
        if key in context.conditionings:
            continue
        if policy == "video2":
            items = context.qwen_ref_items_b + context.dynamic_assets.get("video2_ref_items", [])
        elif policy == "composite":
            items = _replace_source_item(context, context.dynamic_assets.get("composite_ref_items"))
        else:
            items = context.qwen_ref_items_b
        context.conditionings[key] = _encode(
            clip, context.prompt_for(policy), items, cond_cache)
        notes.append("Qwen encode for prompt policy %r" % policy)
    return notes


def _replace_source_item(context, composite_items):
    """Swap the source video presentation for the composite one, in place."""
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


# ---------------------------------------------------------------------------
# Phase E - Chunk B experiment runs
# ---------------------------------------------------------------------------

def phase_e_experiments(context, *, experiment_ids, model, sampler, sigmas,
                        video_vae, store, continue_after_failure=True,
                        save_latents=True, save_frames=True):
    """Run each arm in isolation and record what happened either way.

    Every arm gets its own payload, its own fresh empty target and the same
    Chunk B noise seed and sigma schedule. Results are written before the next
    arm starts, so an OOM at experiment five cannot destroy experiments one
    through four.
    """
    results = []
    for experiment_id in experiment_ids:
        spec = CATALOG[experiment_id]
        record = {"experiment_id": experiment_id, "status": "pending",
                  "strategy": spec.as_dict(),
                  "dependencies": spec.dependencies().as_dict()}
        started = time.time()
        _reset_peak_memory()
        try:
            outcome = run_experiment(
                context, spec, model=model, sampler=sampler, sigmas=sigmas,
                video_vae=video_vae)
            record.update(outcome)
            record["status"] = "completed"
        except InterruptProcessingException:
            record["status"] = "cancelled"
            record["note"] = ("cancelled - a VRAM-guard cancellation is a resource "
                              "result, not a model-quality result")
            logging.warning("%s %s cancelled", LOG_PREFIX, experiment_id)
            if not continue_after_failure:
                results.append(record)
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

        _persist(store, record, save_latents=save_latents, save_frames=save_frames)
        results.append(record)

    baseline = next((r.get("metrics") for r in results
                     if r["experiment_id"] == "baseline_none"), None)
    for record in results:
        if record.get("metrics"):
            record["metrics"].update(
                metrics_mod.compare_to_baseline(record["metrics"], baseline))
    return results


def run_experiment(context, spec, *, model, sampler, sigmas, video_vae):
    """One arm: prepare, patch, sample, decode, measure."""
    strategy = spec.strategy()
    prepared = strategy.prepare(context, spec)

    conditioning = attach_refs(prepared.conditioning, prepared.dit_ref_blocks)
    arm_model = patch_target_conditions(
        model, prepared.target_conditions,
        position_policy=prepared.position_policy)

    latent = empty_av_latent(context.canvas, context.geometry)
    video_latent, audio_latent = sample(
        model=arm_model, conditioning=conditioning, latent=latent,
        sampler=sampler, sigmas=sigmas, seed=context.seeds.chunk_b_noise)

    video_latent = video_latent.to("cpu", torch.float32)
    pixels = decode_video(video_vae, video_latent).to("cpu", torch.float32)

    measured = metrics_mod.collect(
        geometry=context.geometry,
        chunk_a_latent=context.chunk_a_output_latent,
        chunk_a_pixels=context.chunk_a_output_pixels,
        chunk_b_latent=video_latent,
        chunk_b_pixels=pixels,
        source_chunk_b_pixels=context.source_chunk_b_pixels,
    )

    return {
        "latent": video_latent,
        "audio_latent": audio_latent.to("cpu", torch.float32),
        "pixels": pixels,
        "metrics": measured,
        "prepared": {
            "prompt_chars": len(prepared.prompt or ""),
            "target_conditions": [c.describe() for c in prepared.target_conditions],
            "reference_blocks": len(prepared.dit_ref_blocks),
            "position_policy": prepared.position_policy,
            **prepared.metadata,
        },
        "boundary": comparison.boundary_playback(
            chunk_a_pixels=context.chunk_a_output_pixels,
            chunk_b_pixels=pixels, geometry=context.geometry),
    }


def _persist(store, record, *, save_latents, save_frames):
    """Write this arm's output before the next one starts, then free the latents.

    The metrics have already read the latents, and they are on disk, so holding
    them for the rest of the run buys nothing. The decoded pixels stay - the
    comparison outputs need them at the end.
    """
    experiment_id = record["experiment_id"]
    record["artifacts"] = {}
    if save_latents and record.get("latent") is not None:
        record["artifacts"]["latent"] = store.save_experiment_tensors(
            experiment_id, "output",
            {"video_latent": record["latent"], "audio_latent": record.get("audio_latent")})
    if save_frames and record.get("pixels") is not None:
        directory = store.experiment_dir(experiment_id)
        frames_dir = os.path.join(directory, "frames")
        artifacts.save_frames(frames_dir, record["pixels"])
        record["artifacts"]["frames"] = frames_dir
        if record.get("boundary") is not None:
            boundary_dir = os.path.join(directory, "boundary")
            artifacts.save_frames(boundary_dir, record["boundary"], "seam")
            record["artifacts"]["boundary"] = boundary_dir
    record.pop("latent", None)
    record.pop("audio_latent", None)


# ---------------------------------------------------------------------------
# sampling helpers
# ---------------------------------------------------------------------------

def empty_av_latent(canvas, geometry, batch_size=1):
    width, height = canvas
    device = comfy.model_management.intermediate_device()
    video = torch.zeros([batch_size, 24, geometry.target_latent_t,
                         height // 16, width // 16], device=device)
    audio = torch.zeros([batch_size, 32, 2, geometry.audio_latent_t], device=device)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def sample(*, model, conditioning, latent, sampler, sigmas, seed):
    """Sample one chunk and return `(video_latent, audio_latent)`.

    cfg is fixed at 1.0 with a single conditioning - the H3 workflow's basic
    guider - so the sampler never doubles the batch, which matters at a chunk
    size chosen for its headroom.
    """
    guider = comfy.samplers.CFGGuider(model)
    guider.inner_set_conds({"positive": conditioning})
    guider.set_cfg(1.0)

    latent_image = latent["samples"]
    noise = comfy.sample.prepare_noise(latent_image, seed)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    samples = guider.sample(noise, latent_image, sampler, sigmas,
                            disable_pbar=disable_pbar, seed=seed)
    samples = samples.to(comfy.model_management.intermediate_device())
    video_latent, audio_latent = samples.unbind()
    return video_latent, audio_latent


def decode_video(video_vae, video_latent):
    images = video_vae.decode(video_latent)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

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
        # Under cudaMallocAsync the Tot Alloc/Freed columns are meaningless;
        # reserved is the reading that tracks what the driver actually holds.
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
