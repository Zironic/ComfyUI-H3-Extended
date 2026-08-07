"""Running the bridge arms.

Phase 1  encode C1 and C3 whole (both legal 17k+5 lengths), plus a
         counterfactual context if the suite needs one
Phase 2  slice the reference intervals out of those latents
Phase 3  sample each arm against the same target geometry and seed
Phase 4  measure both seams and the signed landing

Every arm shares one noise seed. The arms differ in what <Video 2> contains and
in nothing else, so any difference in the output is attributable.
"""

import json
import logging
import os
import time

import torch

import comfy.model_management

from .. import artifacts, harness, ref_builder
from ..geometry import FPS
from . import seam_metrics
from .plan import ARMS, assert_shared_wording, build_prompt

LOG_PREFIX = "[H3 Extended] bridge"


def _encode_context(video_vae, audio_vae, frames, audio, canvas, cond_cache):
    """Encode one whole context segment. No snapping: `frames` is already legal."""
    width, height = canvas
    pixels = ref_builder.resize(frames, width, height)
    latent = harness_encode(video_vae, pixels, cond_cache, "bridge context video")
    audio_latent = None
    if audio is not None and audio_vae is not None:
        audio_latent, _ = ref_builder.encode_ref_audio(audio_vae, audio, cond_cache)
    return pixels, latent, audio_latent


def harness_encode(vae, pixels, cond_cache, label):
    try:
        from ... import latent_cache
    except ImportError:
        import latent_cache
    return latent_cache.encode(vae, pixels, mode=cond_cache, label=label)


def slice_audio_span(source_audio, start_frame, frames, fps=FPS):
    """Waveform for an exact frame interval, cut at the source timestamps."""
    if source_audio is None:
        return None
    waveform = source_audio["waveform"]
    sample_rate = source_audio["sample_rate"]
    start = int(start_frame / fps * sample_rate)
    stop = start + int(frames / fps * sample_rate)
    clipped = waveform[..., start:stop]
    if clipped.shape[-1] == 0:
        return None
    return {"waveform": clipped, "sample_rate": sample_rate}


def _av_reference(pixels, video_latent, audio_latent, canvas):
    """Qwen items + DiT block for an already-sliced AV interval.

    Mirrors `_dynamic_av_reference` in the N+1 path deliberately: the reference
    the experiment tests should be built the same way the production
    continuation builds one.
    """
    items = [{"type": "audio"}, ref_builder.qwen_video_item(pixels)]
    block = {
        "kind": "video_audio" if audio_latent is not None else "video",
        "latent_t": int(video_latent.shape[2]),
        "latent_h": int(canvas[1] // 16),
        "latent_w": int(canvas[0] // 16),
        "ref_audio_t": 0 if audio_latent is None else int(audio_latent.shape[-1]),
        "latent": video_latent,
        "audio_latent": audio_latent,
    }
    return items, block


class BridgeContext:
    """Everything sliced and encoded once, shared by every arm."""

    def __init__(self, plan, canvas, cond_cache="auto"):
        self.plan = plan
        self.canvas = canvas
        self.cond_cache = cond_cache
        self.left = None
        self.right_natural = None
        self.right_counterfactual = None
        self.ground_truth = None
        self.notes = []

    def _slice_side(self, pixels, latent, audio_latent, *, side):
        plan = self.plan
        if side == "left":
            v_start, v_count = plan.left_video_slice()
            a_start, a_count = plan.left_audio_slice()
            px = pixels[plan.chunk_frames - plan.ref_frames:]
        else:
            v_start, v_count = plan.right_video_slice()
            a_start, a_count = plan.right_audio_slice()
            px = pixels[:plan.ref_frames]
        sliced_audio = None
        if audio_latent is not None:
            sliced_audio = audio_latent[..., a_start:a_start + a_count].contiguous()
        self.notes.append(
            "%s reference: %d pixels, video latent [%d:%d], audio latent [%d:%d]"
            % (side, int(px.shape[0]), v_start, v_start + v_count,
               a_start, a_start + a_count))
        return {
            "pixels": px.contiguous(),
            "latent": latent[:, :, v_start:v_start + v_count].contiguous(),
            "audio_latent": sliced_audio,
        }

    def prepare(self, *, video_vae, audio_vae, source_frames, source_audio,
                counterfactual_frames=None, counterfactual_audio=None,
                need_counterfactual=True):
        plan = self.plan
        c = plan.chunk_frames

        c1 = source_frames[plan.left_start:plan.bridge_start]
        c3 = source_frames[plan.right_start:plan.right_start + c]
        held_out = source_frames[plan.bridge_start:plan.right_start]

        c1_audio = slice_audio_span(source_audio, plan.left_start, c, plan.fps)
        c3_audio = slice_audio_span(source_audio, plan.right_start, c, plan.fps)

        px, lat, aud = _encode_context(video_vae, audio_vae, c1, c1_audio,
                                       self.canvas, self.cond_cache)
        self.left = self._slice_side(px, lat, aud, side="left")

        px, lat, aud = _encode_context(video_vae, audio_vae, c3, c3_audio,
                                       self.canvas, self.cond_cache)
        self.right_natural = self._slice_side(px, lat, aud, side="right")

        width, height = self.canvas
        self.ground_truth = ref_builder.resize(held_out, width, height)

        if need_counterfactual:
            cf_frames, cf_audio = self._counterfactual_source(
                source_frames, source_audio, counterfactual_frames,
                counterfactual_audio)
            px, lat, aud = _encode_context(video_vae, audio_vae, cf_frames,
                                           cf_audio, self.canvas, self.cond_cache)
            self.right_counterfactual = self._slice_side(px, lat, aud, side="right")

        logging.info("%s prepared: %s", LOG_PREFIX, "; ".join(self.notes))
        return self.notes

    def _counterfactual_source(self, source_frames, source_audio,
                               counterfactual_frames, counterfactual_audio):
        """A future the model would not naturally reach, on the same lattice."""
        c = self.plan.chunk_frames
        if counterfactual_frames is not None:
            if int(counterfactual_frames.shape[0]) < c:
                raise ValueError(
                    "counterfactual clip has %d frames, needs at least %d"
                    % (int(counterfactual_frames.shape[0]), c))
            return counterfactual_frames[:c], counterfactual_audio
        start = self.plan.counterfactual_start
        if start < 0:
            raise ValueError(
                "arm C needs either a counterfactual clip or a valid "
                "counterfactual_start into the source")
        if start % 3:
            raise ValueError(
                "counterfactual_start=%d is not on the shared audio grid" % start)
        if start + c > int(source_frames.shape[0]):
            raise ValueError(
                "counterfactual_start=%d + %d frames exceeds the source" % (start, c))
        return (source_frames[start:start + c],
                slice_audio_span(source_audio, start, c, self.plan.fps))


def _arm_references(context, arm):
    items, blocks = [], []
    left_items, left_block = _av_reference(
        context.left["pixels"], context.left["latent"],
        context.left["audio_latent"], context.canvas)
    items.extend(left_items)
    blocks.append(left_block)

    right = None
    if arm.right_ref == "natural":
        right = context.right_natural
    elif arm.right_ref == "counterfactual":
        right = context.right_counterfactual
    if right is not None:
        right_items, right_block = _av_reference(
            right["pixels"], right["latent"], right["audio_latent"],
            context.canvas)
        items.extend(right_items)
        blocks.append(right_block)
    return items, blocks


def _encode_conditioning(clip, prompt, ref_items, cond_cache):
    try:
        from ...cond_cache import encode as encode_conditioning
    except ImportError:
        from cond_cache import encode as encode_conditioning
    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    return encode_conditioning(clip, tokens, mode=cond_cache, label=prompt)


def run_arms(context, arm_ids, *, model, clip, video_vae, sampler, sigmas, seed,
             base_prompt, cond_cache="auto", continue_after_failure=True,
             save_to=None):
    """Sample every selected arm and measure it. One seed across all arms."""
    plan = context.plan
    prompts = {arm_id: build_prompt(base_prompt, ARMS[arm_id].prompt_policy)
               for arm_id in arm_ids}
    assert_shared_wording(prompts)

    results = {}
    for arm_id in arm_ids:
        arm = ARMS[arm_id]
        record = {"arm_id": arm_id, "spec": arm.as_dict(), "status": "pending"}
        results[arm_id] = record
        started = time.time()
        try:
            items, blocks = _arm_references(context, arm)
            conditioning = harness.attach_refs(
                _encode_conditioning(clip, prompts[arm_id], items, cond_cache),
                blocks)
            latent = harness.empty_av_latent_frames(
                context.canvas, plan.chunk_frames, plan.fps)
            video_latent, audio_latent = harness.sample(
                model=model, conditioning=conditioning, latent=latent,
                sampler=sampler, sigmas=sigmas, seed=seed)
            pixels = harness.decode_video(
                video_vae, video_latent).to("cpu", torch.float32)

            record["metrics"] = seam_metrics.collect(
                generated=pixels,
                left_context=context.left["pixels"],
                right_natural=context.right_natural["pixels"],
                right_counterfactual=(
                    None if context.right_counterfactual is None
                    else context.right_counterfactual["pixels"]),
                ground_truth=context.ground_truth,
                generated_audio=audio_latent,
                right_natural_audio=context.right_natural["audio_latent"],
            )
            record["reference_blocks"] = len(blocks)
            record["prompt_chars"] = len(prompts[arm_id])
            record["pixels"] = pixels
            record["status"] = "completed"
            if save_to:
                record["artifacts"] = _persist(save_to, arm_id, pixels,
                                               video_latent, audio_latent)
        except Exception as exc:  # noqa: BLE001 - an arm failing must not lose the rest
            record["status"] = "failed"
            record["error"] = "%s: %s" % (type(exc).__name__, exc)
            logging.exception("%s arm %s failed", LOG_PREFIX, arm_id)
            if not continue_after_failure:
                record["elapsed"] = time.time() - started
                break
        record["elapsed"] = time.time() - started
        comfy.model_management.soft_empty_cache()

    results["_decisive"] = seam_metrics.decisive_comparison(results)
    return results


def _persist(directory, arm_id, pixels, video_latent, audio_latent):
    arm_dir = os.path.join(directory, arm_id)
    os.makedirs(arm_dir, exist_ok=True)
    torch.save({"video": video_latent.to("cpu", torch.float32),
                "audio": audio_latent.to("cpu", torch.float32)},
               os.path.join(arm_dir, "sample.pt"))
    artifacts.save_frames(arm_dir, pixels, prefix="frame")
    return {"directory": arm_dir}


def new_run_directory(plan, seed, output_directory=None):
    root = os.path.join(artifacts.resolve_root(output_directory), "bridge")
    name = "C%d_R%d_start%d_seed%d_%s" % (
        plan.chunk_frames, plan.ref_frames, plan.bridge_start, seed,
        time.strftime("%Y%m%d_%H%M%S"))
    directory = os.path.join(root, name)
    os.makedirs(directory, exist_ok=True)
    return directory


def write_manifest(directory, *, plan, arm_ids, seed, base_prompt, results, notes):
    payload = {
        "experiment": "two_sided_av_bridge",
        "seed": seed,
        "plan": {
            "chunk_frames": plan.chunk_frames,
            "ref_frames": plan.ref_frames,
            "bridge_start": plan.bridge_start,
            "counterfactual_start": plan.counterfactual_start,
            "fps": plan.fps,
            "left_video_slice": plan.left_video_slice(),
            "right_video_slice": plan.right_video_slice(),
            "left_audio_slice": plan.left_audio_slice(),
            "right_audio_slice": plan.right_audio_slice(),
        },
        "base_prompt": base_prompt,
        "notes": notes,
        "arms": {
            arm_id: {k: v for k, v in record.items() if k != "pixels"}
            for arm_id, record in results.items()
            if arm_id != "_decisive" and isinstance(record, dict)
        },
        "decisive": results.get("_decisive"),
    }
    path = os.path.join(directory, "manifest.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path
