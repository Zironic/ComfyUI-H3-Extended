"""Audiovisual runner seams for long-form dual prompts.

Both AV runtimes install optimized sampling methods after the base runners are
imported.  These replacements must select the first or continuation text while
retaining source-audio references, generated-audio carry, sampler previews, and
dual-track output.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import time

import torch

from .. import harness
from ..geometry import AUDIO_LATENT_FPS, latent_frame_spans
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import audio_runtime, reference_runner, runner, v2v_audio_runtime
from .audio_conditions import (
    TargetAlignedAudioCondition,
    patch_target_audio_conditions,
)
from .dual_prompt_runtime import prompt_for_index

_INSTALLED = False


def _pass_b_av(self, *, clip, prompt, chunk_count, cond_cache):
    """Ref2V text encoding with source-audio items and prompt selection."""
    started = time.time()
    done = 0
    for index in range(chunk_count):
        target = self.path("conditioning", index)
        meta_path = self.path("conditioning", index, ".json")
        if runner._load(target) is not None and os.path.exists(meta_path):
            done += 1
            continue
        stored = runner._load(self.path("precompute", index))
        if stored is None:
            raise RuntimeError("pass B: missing precompute for chunk %d" % index)
        qwen_frames = stored["qwen_frames"].to(torch.float32).div_(255.0)
        items = list(self.static_items)
        if stored.get("source_audio_latent") is not None:
            items.append({"type": "audio"})
        items.append({
            "type": "video",
            "data": qwen_frames,
            "timestamps": [i / 2.0 for i in range(qwen_frames.shape[0])],
        })
        role = (
            "continuation"
            if index > 0 and self.carry != runner.CARRY_NONE
            else "initial"
        )
        conditioning = harness._encode(
            clip, prompt_for_index(prompt, index, self.carry), items, cond_cache
        )
        payload, metadata = runner._pack_conditioning(conditioning)
        runner._save(target, payload)
        runner._atomic_json(meta_path, metadata)
        self.event(
            pass_="B",
            chunk=index,
            source_audio=stored.get("source_audio_latent") is not None,
            prompt=role,
        )
        done += 1
        del stored, qwen_frames, items, conditioning
    if self.manifest:
        self.manifest.update_state(pass_b_chunks=done)
    logging.info(
        "%s pass B complete: %d chunks in %.1f s",
        v2v_audio_runtime.LOG,
        done,
        time.time() - started,
    )
    return done


def _select_conditioning(conditioning, index, carry):
    if not isinstance(conditioning, dict):
        return conditioning, "initial"
    role = (
        "continuation"
        if index > 0 and carry != runner.CARRY_NONE
        else "initial"
    )
    return conditioning[role], role


def _sample_chunks_av(
    self,
    *,
    model,
    conditioning,
    sampler,
    sigmas,
    chunk_count,
    on_sampled=None,
):
    """Reference generation with dual prompts and synchronized AV carry."""
    geometry = self.geometry
    video_overlap_start = video_overlap_count = None
    audio_overlap_start = audio_overlap_count = None
    if self.carry != runner.CARRY_NONE:
        video_overlap_start, video_overlap_count = geometry.overlap_slice()
        audio_overlap_start, audio_overlap_count = audio_runtime.audio_overlap_slice(
            geometry
        )

    prefix = self._first_invalid("samples", chunk_count)
    self._remove_suffix("samples", prefix, chunk_count)
    carry_video = carry_audio = None
    if prefix:
        previous = reference_runner._load(self.path("samples", prefix - 1))
        carry_video = previous["video_latent"]
        carry_audio = previous["audio_latent"]
        del previous

    for index in range(prefix, chunk_count):
        active_conditioning, role = _select_conditioning(
            conditioning, index, self.carry
        )
        chunk_conditioning = harness.attach_refs(
            active_conditioning, self.static_blocks
        )
        video_conditions = []
        audio_conditions = []
        if (
            carry_video is not None
            and carry_audio is not None
            and self.carry != runner.CARRY_NONE
        ):
            if self.carry == runner.CARRY_FRAME:
                video_count = 1
                shared_frames = latent_frame_spans(
                    geometry.target_latent_t
                )[video_overlap_start]
                audio_start = audio_overlap_start
                audio_stop = audio_runtime.audio_latent_boundary(
                    geometry.stride_frames + shared_frames,
                    fps=geometry.fps,
                    audio_latent_fps=AUDIO_LATENT_FPS,
                )
                audio_count = audio_stop - audio_start
            else:
                video_count = video_overlap_count
                audio_start = audio_overlap_start
                audio_count = audio_overlap_count

            video_conditions.append(TargetAlignedCondition(
                latent=carry_video[
                    :,
                    :,
                    video_overlap_start:video_overlap_start + video_count,
                ],
                target_latent_start=0,
                label="video carry from chunk %d" % (index - 1),
            ))
            audio_conditions.append(TargetAlignedAudioCondition(
                latent=carry_audio[..., audio_start:audio_start + audio_count],
                target_latent_start=0,
                label="audio carry from chunk %d" % (index - 1),
            ))

        armed = patch_target_conditions(model, video_conditions)
        armed = patch_target_audio_conditions(armed, audio_conditions)
        latent = harness.empty_av_latent(self.canvas, geometry)

        callback = None
        try:
            from .preview import current_publisher
            publisher = current_publisher()
            if publisher is not None:
                callback = publisher.sampler_callback(index)
        except Exception:
            callback = None

        sample_kwargs = {
            "model": armed,
            "conditioning": chunk_conditioning,
            "latent": latent,
            "sampler": sampler,
            "sigmas": sigmas,
            "seed": self.chunk_seed(index),
        }
        if callback is not None:
            sample_kwargs["callback"] = callback
        video_latent, audio_latent = harness.sample(**sample_kwargs)
        video_latent = video_latent.to("cpu", torch.float32)
        audio_latent = audio_latent.to("cpu", torch.float32)
        reference_runner._save(self.path("samples", index), {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
        })
        reference_runner._atomic_json(self.path("samples", index, ".json"), {
            "index": index,
            "seed": self.chunk_seed(index),
            "previous_sample": index - 1 if index else None,
            "carry": self.carry,
            "video_conditions": len(video_conditions),
            "audio_conditions": len(audio_conditions),
            "prompt": role,
        })
        carry_video = video_latent
        carry_audio = audio_latent
        if on_sampled is not None:
            on_sampled(index, video_latent, audio_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        self.event(
            pass_="C",
            chunk=index,
            carry=self.carry,
            video_conditions=len(video_conditions),
            audio_conditions=len(audio_conditions),
            prompt=role,
        )
        del chunk_conditioning, armed, latent
        gc.collect()
        logging.info(
            "%s chunk %d/%d sampled with %s prompt",
            audio_runtime.LOG,
            index + 1,
            chunk_count,
            role,
        )
    return chunk_count


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    # v2v_audio_runtime.install() runs at import and again at every run. Replace
    # its source function as well as the currently installed class method.
    v2v_audio_runtime._pass_b_av = _pass_b_av
    runner.LongFormRun.pass_b = _pass_b_av

    # audio_runtime.install() is idempotent, but replace both the source symbol
    # and class method so this remains correct regardless of import order.
    audio_runtime._sample_chunks_av = _sample_chunks_av
    reference_runner.LongFormReferenceRun.sample_chunks = _sample_chunks_av
    _INSTALLED = True


__all__ = ["install"]
