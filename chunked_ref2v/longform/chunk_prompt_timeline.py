"""Chunk-aware prompt timeline for LongFormReferenceVideo.

The editor produces one prompt instruction per model chunk.  The long-form
reference node combines each instruction with an optional global prompt,
pre-encodes unique text conditionings once, stores them on disk, and loads only
the active chunk conditioning while sampling.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import logging
import os
from dataclasses import dataclass, is_dataclass, replace

import torch
from comfy_api.latest import ComfyExtension, io

from .. import harness, ref_builder
from ..geometry import HarnessGeometry, latent_frame_spans
from ..layout_ops import TargetAlignedCondition
from ..model_patch import patch_target_conditions
from . import (
    audio_runtime,
    chunk_aligned_audio_refs,
    reference_nodes,
    reference_preview_nodes,
    reference_runner,
    runner,
)
from .audio_conditions import (
    TargetAlignedAudioCondition,
    patch_target_audio_conditions,
)
from .chunk_stream import chunk_count_for

LOG = "[H3 Extended] chunk prompt timeline"
PLAN_VERSION = 1
FPS = 24
_ENVELOPE_MAGIC = "H3_EXTENDED_CHUNK_PROMPTS_V1:"
ChunkPromptPlan = io.Custom("H3_CHUNK_PROMPT_PLAN")
_INSTALLED = False
_LEGACY_PREPARE = None


def _parse_prompt_store(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("chunk prompt data is not valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("prompts", [])
    if not isinstance(value, list):
        raise ValueError("chunk prompt data must contain a prompts list")
    return [str(item or "") for item in value]


def _chunk_geometry(output_seconds, chunk_frames, overlap_frames):
    geometry = HarnessGeometry(
        chunk_frames=int(chunk_frames),
        overlap_frames=int(overlap_frames),
    ).validate()
    output_seconds = int(output_seconds)
    if output_seconds <= 0:
        raise ValueError("output_seconds must be positive")
    target_frames = output_seconds * geometry.fps
    chunk_count = chunk_count_for(
        target_frames,
        geometry.chunk_frames,
        geometry.stride_frames,
    )
    return geometry, target_frames, chunk_count


def build_chunk_prompt_plan(
    *,
    output_seconds,
    chunk_frames,
    overlap_frames,
    global_prompt="",
    chunk_prompts_json="",
):
    geometry, target_frames, chunk_count = _chunk_geometry(
        output_seconds, chunk_frames, overlap_frames
    )
    stored = _parse_prompt_store(chunk_prompts_json)
    prompts = stored[:chunk_count]
    prompts.extend("" for _ in range(chunk_count - len(prompts)))
    return {
        "version": PLAN_VERSION,
        "fps": int(geometry.fps),
        "output_seconds": int(output_seconds),
        "target_frames": int(target_frames),
        "chunk_frames": int(geometry.chunk_frames),
        "overlap_frames": int(geometry.overlap_frames),
        "stride_frames": int(geometry.stride_frames),
        "chunk_count": int(chunk_count),
        "global_prompt": str(global_prompt or ""),
        "chunk_prompts": prompts,
    }


def validate_chunk_prompt_plan(
    plan,
    *,
    output_seconds,
    chunk_frames,
    overlap_frames,
):
    if not isinstance(plan, dict):
        raise TypeError("chunk_prompt_plan must come from the H3 chunk timeline node")
    if int(plan.get("version", -1)) != PLAN_VERSION:
        raise ValueError("unsupported chunk prompt plan version")

    geometry, target_frames, chunk_count = _chunk_geometry(
        output_seconds, chunk_frames, overlap_frames
    )
    expected = {
        "fps": int(geometry.fps),
        "output_seconds": int(output_seconds),
        "target_frames": int(target_frames),
        "chunk_frames": int(geometry.chunk_frames),
        "overlap_frames": int(geometry.overlap_frames),
        "stride_frames": int(geometry.stride_frames),
        "chunk_count": int(chunk_count),
    }
    for key, value in expected.items():
        if int(plan.get(key, -1)) != value:
            raise ValueError(
                "chunk prompt plan %s=%r does not match LongFormReferenceVideo %s=%r"
                % (key, plan.get(key), key, value)
            )

    prompts = plan.get("chunk_prompts")
    if not isinstance(prompts, list) or len(prompts) != chunk_count:
        raise ValueError(
            "chunk prompt plan contains %d prompts; expected exactly %d"
            % (len(prompts) if isinstance(prompts, list) else 0, chunk_count)
        )
    normalized = dict(plan)
    normalized["global_prompt"] = str(plan.get("global_prompt") or "")
    normalized["chunk_prompts"] = [str(item or "") for item in prompts]
    return normalized


def compile_chunk_prompts(plan, fallback_prompt=""):
    global_prompt = str(plan.get("global_prompt") or "").strip()
    fallback = str(fallback_prompt or "")
    compiled = []
    for local in plan["chunk_prompts"]:
        local = str(local or "").strip()
        if global_prompt and local:
            prompt = global_prompt + "\n\n" + local
        else:
            prompt = global_prompt or local or fallback
        compiled.append(prompt)
    return compiled


def pack_chunk_prompt_plan(
    fallback_prompt,
    plan,
    *,
    output_seconds,
    chunk_frames,
    overlap_frames,
):
    if plan is None:
        return str(fallback_prompt or "")
    normalized = validate_chunk_prompt_plan(
        plan,
        output_seconds=output_seconds,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
    )
    payload = {
        "version": PLAN_VERSION,
        "plan": {
            key: normalized[key]
            for key in (
                "fps",
                "output_seconds",
                "target_frames",
                "chunk_frames",
                "overlap_frames",
                "stride_frames",
                "chunk_count",
            )
        },
        "prompts": compile_chunk_prompts(normalized, fallback_prompt),
    }
    return _ENVELOPE_MAGIC + json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def unpack_chunk_prompt_envelope(prompt):
    text = str(prompt or "")
    if not text.startswith(_ENVELOPE_MAGIC):
        return None
    try:
        payload = json.loads(text[len(_ENVELOPE_MAGIC):])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid internal chunk prompt envelope") from exc
    prompts = payload.get("prompts")
    plan = payload.get("plan")
    if (
        int(payload.get("version", -1)) != PLAN_VERSION
        or not isinstance(plan, dict)
        or not isinstance(prompts, list)
        or len(prompts) != int(plan.get("chunk_count", -1))
    ):
        raise ValueError("invalid internal chunk prompt envelope")
    return {
        "plan": dict(plan),
        "prompts": [str(item or "") for item in prompts],
    }


def _prompt_digest(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _conditioning_paths(root, digest):
    stem = os.path.join(root, "conditioning", "prompt_%s" % digest)
    return stem + ".safetensors", stem + ".json"


def _conditioning_available(root, digest):
    target, meta_path = _conditioning_paths(root, digest)
    if not os.path.exists(meta_path):
        return False
    tensors = reference_runner._load(target)
    if tensors is None:
        return False
    del tensors
    return True


def _save_conditioning(root, digest, conditioning):
    target, meta_path = _conditioning_paths(root, digest)
    payload, metadata = runner._pack_conditioning(conditioning)
    runner._save(target, payload)
    runner._atomic_json(meta_path, metadata)


@dataclass
class ChunkPromptConditionings:
    root: str
    digests: tuple[str, ...]
    _cached_digest: str | None = None
    _cached_conditioning: object = None

    def for_index(self, index):
        index = int(index)
        if index < 0 or index >= len(self.digests):
            raise IndexError("chunk prompt conditioning index is out of range")
        digest = self.digests[index]
        if digest == self._cached_digest and self._cached_conditioning is not None:
            return self._cached_conditioning, digest

        target, meta_path = _conditioning_paths(self.root, digest)
        tensors = reference_runner._load(target)
        if tensors is None or not os.path.exists(meta_path):
            raise RuntimeError(
                "missing cached conditioning for chunk %d (%s)" % (index, digest[:12])
            )
        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
        conditioning = runner._unpack_conditioning(tensors, metadata)
        self._cached_digest = digest
        self._cached_conditioning = conditioning
        return conditioning, digest

    @property
    def unique_count(self):
        return len(set(self.digests))


def conditioning_for_chunk(conditioning, index):
    if isinstance(conditioning, ChunkPromptConditionings):
        return conditioning.for_index(index)
    return conditioning, None


def _encode_reference_assets(
    self,
    *,
    video_vae,
    audio_vae,
    ref_images,
    ref_videos,
    ref_video_audios,
    ref_audios,
    ref_image_size,
    cond_cache,
):
    items, blocks, notes = [], [], []
    for name, image in reference_runner._ordered_values(ref_images):
        item, block, note = ref_builder.encode_image_ref(
            video_vae,
            image,
            self.canvas,
            ref_image_size,
            cond_cache=cond_cache,
        )
        items.append(item)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, frames in reference_runner._ordered_values(ref_videos):
        audio = reference_runner._paired_audio(ref_video_audios, name)
        video_items, block, note = ref_builder.encode_video_ref(
            video_vae,
            frames,
            self.canvas,
            audio=audio,
            audio_vae=audio_vae,
            cond_cache=cond_cache,
        )
        items.extend(video_items)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, audio in reference_runner._ordered_values(ref_audios):
        audio_latent, ref_audio_t = ref_builder.encode_ref_audio(
            audio_vae,
            audio,
            cond_cache=cond_cache,
        )
        items.append({"type": "audio"})
        blocks.append({
            "kind": "audio",
            "ref_audio_t": int(ref_audio_t),
            "audio_latent": audio_latent,
        })
        notes.append("%s: audio latent t=%d" % (name, ref_audio_t))
    return items, blocks, notes


def _prepare_references(
    self,
    *,
    clip,
    prompt,
    video_vae,
    audio_vae,
    ref_images,
    ref_videos,
    ref_video_audios,
    ref_audios,
    ref_image_size,
    cond_cache,
):
    envelope = unpack_chunk_prompt_envelope(prompt)
    if envelope is None:
        return _LEGACY_PREPARE(
            self,
            clip=clip,
            prompt=prompt,
            video_vae=video_vae,
            audio_vae=audio_vae,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            ref_image_size=ref_image_size,
            cond_cache=cond_cache,
        )

    prompts = envelope["prompts"]
    digests = tuple(_prompt_digest(text) for text in prompts)
    unique_prompts = {}
    for digest, text in zip(digests, prompts):
        unique_prompts.setdefault(digest, text)

    refs_path = os.path.join(self.root, "precompute", "persistent.safetensors")
    refs_meta_path = os.path.join(self.root, "precompute", "persistent.json")
    refs_payload = reference_runner._load(refs_path)
    refs_cached = refs_payload is not None and os.path.exists(refs_meta_path)
    missing = [
        digest
        for digest in unique_prompts
        if not _conditioning_available(self.root, digest)
    ]

    if refs_cached and not missing:
        with open(refs_meta_path, encoding="utf-8") as fh:
            refs_meta = json.load(fh)
        self.static_items = []
        self.static_blocks = reference_runner._unpack_blocks(
            refs_payload,
            refs_meta,
        )
        notes = list(refs_meta.get("notes", []))
    else:
        items, blocks, notes = _encode_reference_assets(
            self,
            video_vae=video_vae,
            audio_vae=audio_vae,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
            ref_image_size=ref_image_size,
            cond_cache=cond_cache,
        )
        for digest, text in unique_prompts.items():
            if digest not in missing:
                continue
            conditioning = harness._encode(clip, text, items, cond_cache)
            _save_conditioning(self.root, digest, conditioning)
            del conditioning
            gc.collect()

        packed_refs, refs_meta = reference_runner._pack_blocks(blocks)
        refs_meta["notes"] = notes
        runner._save(refs_path, packed_refs)
        runner._atomic_json(refs_meta_path, refs_meta)
        self.static_items = items
        self.static_blocks = blocks

    notes = list(notes) + [
        "chunk prompts: %d chunks, %d unique text conditionings"
        % (len(digests), len(unique_prompts))
    ]
    if self.manifest:
        self.manifest.update_state(references_prepared=True)
    return ChunkPromptConditionings(self.root, digests), notes


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
    geometry = self.geometry
    video_overlap_start = video_overlap_count = None
    audio_overlap_start = audio_overlap_count = None
    if self.carry != runner.CARRY_NONE:
        video_overlap_start, video_overlap_count = geometry.overlap_slice()
        video_carry_frames = geometry.overlap_frames
        if self.carry == runner.CARRY_FRAME:
            video_carry_frames = latent_frame_spans(
                geometry.target_latent_t
            )[video_overlap_start]
        audio_overlap_start, audio_overlap_count = audio_runtime.audio_overlap_slice(
            geometry, video_carry_frames
        )
        audio_runtime.log_audio_carry(
            geometry,
            self.carry,
            video_carry_frames,
            1 if self.carry == runner.CARRY_FRAME else video_overlap_count,
            audio_overlap_start,
            audio_overlap_count,
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
        active_conditioning, prompt_digest = conditioning_for_chunk(
            conditioning,
            index,
        )
        chunk_conditioning = harness.attach_refs(
            active_conditioning,
            self.static_blocks,
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
                audio_start = audio_overlap_start
                audio_count = audio_overlap_count
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
        reference_runner._save(
            self.path("samples", index),
            {
                "video_latent": video_latent,
                "audio_latent": audio_latent,
            },
        )
        reference_runner._atomic_json(
            self.path("samples", index, ".json"),
            {
                "index": index,
                "seed": self.chunk_seed(index),
                "previous_sample": index - 1 if index else None,
                "carry": self.carry,
                "video_conditions": len(video_conditions),
                "audio_conditions": len(audio_conditions),
                "prompt_sha256": prompt_digest,
            },
        )
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
            prompt_sha256=prompt_digest,
        )
        del chunk_conditioning, armed, latent
        gc.collect()
        logging.info(
            "%s chunk %d/%d sampled%s",
            LOG,
            index + 1,
            chunk_count,
            " with prompt %s" % prompt_digest[:12]
            if prompt_digest
            else "",
        )
    return chunk_count


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


def _patch_reference_node():
    original_schema = reference_nodes.MiniMaxH3LongFormReferenceVideo.define_schema.__func__

    @classmethod
    def define_schema(cls):
        schema = original_schema(cls)
        inputs = list(schema.inputs)
        if any(getattr(item, "id", getattr(item, "name", None)) == "chunk_prompt_plan" for item in inputs):
            return schema
        prompt_index = next(
            index
            for index, item in enumerate(inputs)
            if getattr(item, "id", getattr(item, "name", None)) == "prompt"
        )
        inputs.insert(
            prompt_index + 1,
            ChunkPromptPlan.Input(
                "chunk_prompt_plan",
                optional=True,
                tooltip=(
                    "Optional plan from MiniMax H3 Chunk Prompt Timeline (Zi). "
                    "When connected, each model chunk uses its matching timeline "
                    "prompt; the normal prompt is used only as a blank-entry fallback."
                ),
            ),
        )
        return _replace_inputs(schema, inputs)

    reference_nodes.MiniMaxH3LongFormReferenceVideo.define_schema = define_schema

    node_cls = reference_preview_nodes.MiniMaxH3LongFormReferenceVideoPreview
    original_execute = node_cls.execute.__func__
    signature = inspect.signature(original_execute)

    @classmethod
    def execute(cls, *args, chunk_prompt_plan=None, **kwargs):
        bound = signature.bind(cls, *args, **kwargs)
        bound.apply_defaults()
        bound.arguments["prompt"] = pack_chunk_prompt_plan(
            bound.arguments["prompt"],
            chunk_prompt_plan,
            output_seconds=bound.arguments["output_seconds"],
            chunk_frames=bound.arguments["chunk_frames"],
            overlap_frames=bound.arguments["overlap_frames"],
        )
        return original_execute(*bound.args, **bound.kwargs)

    node_cls.execute = execute


class MiniMaxH3ChunkPromptTimeline(io.ComfyNode):
    """Build a fixed one-prompt-per-model-chunk timeline."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChunkPromptTimelineZi",
            display_name="MiniMax H3 Chunk Prompt Timeline (Zi)",
            category="model/video/minimax/testing",
            description=(
                "Calculates the exact LongFormReferenceVideo chunk schedule and "
                "provides one editable prompt instruction for every model chunk."
            ),
            inputs=[
                io.Int.Input(
                    "output_seconds",
                    default=30,
                    min=1,
                    max=3600,
                    tooltip="Desired final duration at MiniMax H3's fixed 24 fps.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=90,
                    min=22,
                    max=362,
                    step=17,
                    tooltip="Per-sample H3 model length on the legal 17k+5 grid.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=4,
                    min=4,
                    max=180,
                    tooltip="Overlap shared by adjacent chunks.",
                ),
                io.String.Input(
                    "global_prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Optional text prepended to every chunk instruction. Use it "
                        "for persistent subject, style, scene, and continuity rules."
                    ),
                ),
                io.String.Input(
                    "chunk_prompts_json",
                    default='{"version":1,"prompts":[]}',
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip="Internal timeline storage managed by the node editor.",
                ),
            ],
            outputs=[
                ChunkPromptPlan.Output(
                    "chunk_prompt_plan",
                    display_name="chunk prompt plan",
                ),
                io.Int.Output("output_seconds", display_name="output seconds"),
                io.Int.Output("chunk_frames", display_name="chunk frames"),
                io.Int.Output("overlap_frames", display_name="overlap frames"),
                io.String.Output("report", display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        output_seconds=30,
        chunk_frames=90,
        overlap_frames=4,
        global_prompt="",
        chunk_prompts_json="",
    ) -> io.NodeOutput:
        plan = build_chunk_prompt_plan(
            output_seconds=output_seconds,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
            global_prompt=global_prompt,
            chunk_prompts_json=chunk_prompts_json,
        )
        lines = [
            "MiniMax H3 Chunk Prompt Timeline",
            "output    %d s / %d frames at %d fps"
            % (plan["output_seconds"], plan["target_frames"], plan["fps"]),
            "profile   C=%d O=%d S=%d"
            % (
                plan["chunk_frames"],
                plan["overlap_frames"],
                plan["stride_frames"],
            ),
            "chunks    %d" % plan["chunk_count"],
        ]
        for index, prompt in enumerate(plan["chunk_prompts"]):
            start = index * plan["stride_frames"]
            stop = min(start + plan["chunk_frames"], plan["target_frames"])
            preview = " ".join(prompt.strip().split())
            if len(preview) > 72:
                preview = preview[:69] + "..."
            lines.append(
                "chunk %03d frames %d-%d (%.3f-%.3f s): %s"
                % (
                    index,
                    start,
                    max(start, stop - 1),
                    start / plan["fps"],
                    stop / plan["fps"],
                    preview or "<global/fallback only>",
                )
            )
        return io.NodeOutput(
            plan,
            int(output_seconds),
            int(chunk_frames),
            int(overlap_frames),
            "\n".join(lines),
        )


class MiniMaxH3ChunkPromptTimelineExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ChunkPromptTimeline]


def install():
    global _INSTALLED, _LEGACY_PREPARE
    if _INSTALLED:
        return

    _LEGACY_PREPARE = chunk_aligned_audio_refs._ORIGINAL_PREPARE
    chunk_aligned_audio_refs._ORIGINAL_PREPARE = _prepare_references
    reference_runner.LongFormReferenceRun.prepare_references = (
        chunk_aligned_audio_refs._prepare_references
    )

    audio_runtime._sample_chunks_av = _sample_chunks_av
    reference_runner.LongFormReferenceRun.sample_chunks = _sample_chunks_av
    _patch_reference_node()
    _INSTALLED = True


install()


__all__ = [
    "ChunkPromptConditionings",
    "ChunkPromptPlan",
    "MiniMaxH3ChunkPromptTimeline",
    "MiniMaxH3ChunkPromptTimelineExtension",
    "build_chunk_prompt_plan",
    "compile_chunk_prompts",
    "conditioning_for_chunk",
    "install",
    "pack_chunk_prompt_plan",
    "unpack_chunk_prompt_envelope",
    "validate_chunk_prompt_plan",
]
