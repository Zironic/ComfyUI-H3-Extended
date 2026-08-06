"""Install first-sample and continuation prompts for both long-form nodes.

The nodes keep their existing ``prompt`` input for workflow compatibility.  A
new optional ``continuation_prompt`` input is added immediately after it.  The
first model invocation uses ``prompt``; later invocations use
``continuation_prompt`` only when generated latent state is actually carried
forward.  A blank continuation prompt falls back to the original prompt.

The two prompts are packed into the existing runner ``prompt`` argument.  This
keeps the public runner call sites stable and, importantly, makes the existing
manifest prompt hash cover both values.  The patched conditioning stages unpack
and select the appropriate prompt before calling the text encoder.
"""

from __future__ import annotations

import gc
import inspect
import json
import logging
import os
from dataclasses import is_dataclass, replace

import torch
from comfy_api.latest import io

from .. import harness, ref_builder
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import preview_nodes, reference_nodes, reference_preview_nodes
from . import reference_runner, runner

_MAGIC = "H3_EXTENDED_DUAL_PROMPT_V1:"
_INSTALLED = False


def pack_prompts(initial_prompt, continuation_prompt=""):
    """Return a manifest-safe envelope containing both prompt variants."""
    initial = str(initial_prompt or "")
    continuation = str(continuation_prompt or "").strip() or initial
    return _MAGIC + json.dumps(
        {"initial": initial, "continuation": continuation},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def unpack_prompts(prompt):
    """Decode a dual-prompt envelope, accepting legacy single prompts."""
    text = str(prompt or "")
    if not text.startswith(_MAGIC):
        return text, text
    try:
        payload = json.loads(text[len(_MAGIC):])
        initial = str(payload.get("initial", ""))
        continuation = str(payload.get("continuation", "")).strip() or initial
        return initial, continuation
    except (TypeError, ValueError, json.JSONDecodeError):
        logging.warning(
            "[H3 Extended] invalid dual-prompt envelope; using it as a legacy prompt"
        )
        return text, text


def prompt_for_index(prompt, index, carry):
    """Select the prompt matching the temporal state visible to the model."""
    initial, continuation = unpack_prompts(prompt)
    if int(index) > 0 and carry != runner.CARRY_NONE:
        return continuation
    return initial


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


def _patch_schema(node_class):
    original = node_class.define_schema.__func__

    @classmethod
    def define_schema(cls):
        schema = original(cls)
        inputs = list(schema.inputs)
        if any(getattr(item, "name", None) == "continuation_prompt" for item in inputs):
            return schema
        prompt_index = next(
            i for i, item in enumerate(inputs)
            if getattr(item, "name", None) == "prompt"
        )
        inputs.insert(
            prompt_index + 1,
            io.String.Input(
                "continuation_prompt",
                default="",
                multiline=True,
                dynamic_prompts=True,
                tooltip=(
                    "Used after the first model invocation when prior generated "
                    "video/audio state is carried into the next invocation. Leave "
                    "blank to reuse prompt."
                ),
            ),
        )
        return _replace_inputs(schema, inputs)

    node_class.define_schema = define_schema


def _patch_execute(node_class):
    original = node_class.execute.__func__
    signature = inspect.signature(original)
    parameter_names = list(signature.parameters)
    prompt_position = parameter_names.index("prompt") - 1  # remove cls

    @classmethod
    def execute(cls, *args, continuation_prompt="", **kwargs):
        if "prompt" in kwargs:
            kwargs["prompt"] = pack_prompts(kwargs["prompt"], continuation_prompt)
            return original(cls, *args, **kwargs)

        positional = list(args)
        if prompt_position >= len(positional):
            raise TypeError("long-form node execution did not receive prompt")
        positional[prompt_position] = pack_prompts(
            positional[prompt_position], continuation_prompt
        )
        return original(cls, *positional, **kwargs)

    node_class.execute = execute


def _ref2v_pass_b(self, *, clip, prompt, chunk_count, cond_cache):
    """Encode each source-aligned invocation with its selected prompt."""
    started = __import__("time").time()
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
        items = list(self.static_items) + [{
            "type": "video",
            "data": qwen_frames,
            "timestamps": [i / 2.0 for i in range(qwen_frames.shape[0])],
        }]
        active_prompt = prompt_for_index(prompt, index, self.carry)
        conditioning = harness._encode(clip, active_prompt, items, cond_cache)
        payload, metadata = runner._pack_conditioning(conditioning)
        runner._save(target, payload)
        runner._atomic_json(meta_path, metadata)
        self.event(pass_="B", chunk=index, prompt=(
            "continuation" if index > 0 and self.carry != runner.CARRY_NONE
            else "initial"
        ))
        done += 1
        del stored, qwen_frames, items, conditioning
    if self.manifest:
        self.manifest.update_state(pass_b_chunks=done)
    logging.info(
        "%s pass B complete: %d chunks in %.1f s",
        runner.LOG, done, __import__("time").time() - started,
    )
    return done


def _conditioning_paths(root, role):
    return (
        os.path.join(root, "conditioning", "%s.safetensors" % role),
        os.path.join(root, "conditioning", "%s.json" % role),
    )


def _load_persisted_conditioning(target, meta_path):
    tensors = runner._load(target)
    if tensors is None or not os.path.exists(meta_path):
        return None
    with open(meta_path, encoding="utf-8") as fh:
        return runner._unpack_conditioning(tensors, json.load(fh))


def _save_persisted_conditioning(target, meta_path, conditioning):
    payload, metadata = runner._pack_conditioning(conditioning)
    runner._save(target, payload)
    runner._atomic_json(meta_path, metadata)


def _reference_prepare_references(
    self, *, clip, prompt, video_vae, audio_vae, ref_images, ref_videos,
    ref_video_audios, ref_audios, ref_image_size, cond_cache,
):
    """Encode persistent references once and cache both text conditionings."""
    initial_prompt, continuation_prompt = unpack_prompts(prompt)
    initial_target, initial_meta = _conditioning_paths(self.root, "initial")
    continuation_target, continuation_meta = _conditioning_paths(
        self.root, "continuation"
    )
    refs_path = os.path.join(self.root, "precompute", "persistent.safetensors")
    refs_meta_path = os.path.join(self.root, "precompute", "persistent.json")

    initial_conditioning = _load_persisted_conditioning(
        initial_target, initial_meta
    )
    continuation_conditioning = _load_persisted_conditioning(
        continuation_target, continuation_meta
    )
    cached_refs = runner._load(refs_path)
    if (
        initial_conditioning is not None
        and continuation_conditioning is not None
        and cached_refs is not None
        and os.path.exists(refs_meta_path)
    ):
        with open(refs_meta_path, encoding="utf-8") as fh:
            refs_meta = json.load(fh)
        self.static_items = []
        self.static_blocks = reference_runner._unpack_blocks(cached_refs, refs_meta)
        return {
            "initial": initial_conditioning,
            "continuation": continuation_conditioning,
        }, refs_meta.get("notes", [])

    items, blocks, notes = [], [], []
    for name, image in reference_runner._ordered_values(ref_images):
        item, block, note = ref_builder.encode_image_ref(
            video_vae, image, self.canvas, ref_image_size, cond_cache=cond_cache
        )
        items.append(item)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, frames in reference_runner._ordered_values(ref_videos):
        audio = reference_runner._paired_audio(ref_video_audios, name)
        video_items, block, note = ref_builder.encode_video_ref(
            video_vae, frames, self.canvas, audio=audio,
            audio_vae=audio_vae, cond_cache=cond_cache,
        )
        items.extend(video_items)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, audio in reference_runner._ordered_values(ref_audios):
        audio_latent, ref_audio_t = ref_builder.encode_ref_audio(
            audio_vae, audio, cond_cache=cond_cache
        )
        items.append({"type": "audio"})
        blocks.append({
            "kind": "audio",
            "ref_audio_t": int(ref_audio_t),
            "audio_latent": audio_latent,
        })
        notes.append("%s: audio latent t=%d" % (name, ref_audio_t))

    initial_conditioning = harness._encode(
        clip, initial_prompt, items, cond_cache
    )
    continuation_conditioning = harness._encode(
        clip, continuation_prompt, items, cond_cache
    )
    _save_persisted_conditioning(
        initial_target, initial_meta, initial_conditioning
    )
    _save_persisted_conditioning(
        continuation_target, continuation_meta, continuation_conditioning
    )

    refs_payload, refs_meta = reference_runner._pack_blocks(blocks)
    refs_meta["notes"] = notes
    runner._save(refs_path, refs_payload)
    runner._atomic_json(refs_meta_path, refs_meta)
    self.static_items = items
    self.static_blocks = blocks
    if self.manifest:
        self.manifest.update_state(references_prepared=True)
    return {
        "initial": initial_conditioning,
        "continuation": continuation_conditioning,
    }, notes


def _reference_sample_chunks(
    self, *, model, conditioning, sampler, sigmas, chunk_count, on_sampled=None,
):
    geometry = self.geometry
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
        role = (
            "continuation"
            if index > 0 and self.carry != runner.CARRY_NONE
            else "initial"
        )
        base_conditioning = (
            conditioning[role] if isinstance(conditioning, dict) else conditioning
        )
        chunk_conditioning = harness.attach_refs(
            base_conditioning, self.static_blocks
        )
        conditions = []
        if carry_latent is not None and self.carry != runner.CARRY_NONE:
            count = 1 if self.carry == runner.CARRY_FRAME else overlap_count
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
        runner._save(self.path("samples", index), {
            "video_latent": video_latent,
            "audio_latent": audio_latent,
        })
        runner._atomic_json(self.path("samples", index, ".json"), {
            "index": index,
            "seed": self.chunk_seed(index),
            "previous_sample": index - 1 if index else None,
            "carry": self.carry,
            "prompt": role,
        })
        carry_latent = video_latent
        if on_sampled is not None:
            on_sampled(index, video_latent)
        if self.manifest:
            self.manifest.update_state(pass_c_chunks=index + 1)
        self.event(
            pass_="C", chunk=index, carry=self.carry,
            conditions=len(conditions), prompt=role,
        )
        del chunk_conditioning, armed, latent, audio_latent
        gc.collect()
        logging.info(
            "%s chunk %d/%d sampled with %s prompt",
            reference_runner.LOG, index + 1, chunk_count, role,
        )
    return chunk_count


def install():
    """Install the dual-prompt UI and conditioning selection once."""
    global _INSTALLED
    if _INSTALLED:
        return

    # Base schemas feed the registered preview variants through super().
    _patch_schema(reference_nodes.MiniMaxH3LongFormReferenceVideo)
    _patch_schema(preview_nodes.MiniMaxH3LongFormRef2VPreview)

    # The registered variants own execute(), so wrap both of them directly.
    _patch_execute(preview_nodes.MiniMaxH3LongFormRef2VPreview)
    _patch_execute(reference_preview_nodes.MiniMaxH3LongFormReferenceVideoPreview)

    runner.LongFormRun.pass_b = _ref2v_pass_b
    reference_runner.LongFormReferenceRun.prepare_references = (
        _reference_prepare_references
    )
    reference_runner.LongFormReferenceRun.sample_chunks = (
        _reference_sample_chunks
    )
    _INSTALLED = True


__all__ = [
    "install",
    "pack_prompts",
    "unpack_prompts",
    "prompt_for_index",
]
