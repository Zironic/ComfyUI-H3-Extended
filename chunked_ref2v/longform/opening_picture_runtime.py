"""Generated opening-frame picture continuity for LongFormReferenceVideo.

When enabled, every chunk after the first receives the decoded frame from the
previous chunk that maps to its frame zero as one additional MiniMax picture
reference.  The picture augments the existing direct latent video/audio carry;
it does not replace either carry path.

This is deliberately an in-process runtime feature.  It keeps exactly one
opening-frame image on CPU between chunks and builds the next Qwen conditioning
just in time.  It does not add a stop/resume contract to ComfyUI or persist
continuation pictures/conditionings for later execution.
"""

from __future__ import annotations

import contextvars
import gc
import hashlib
import logging
from dataclasses import dataclass, is_dataclass, replace

from comfy_api.latest import io

from .. import harness, ref_builder
from . import (
    chunk_aligned_audio_refs,
    chunk_prompt_timeline,
    reference_preview_nodes,
    reference_runner,
    runner,
)
from .chunk_stream import chunk_count_for

LOG = "[H3 Extended] longform opening picture"
_ACTIVE = contextvars.ContextVar(
    "h3_longform_opening_picture", default=False
)
_INSTALLED = False
_ORIGINAL_PREPARE = None
_ORIGINAL_CONDITIONING_FOR_CHUNK = None
_ORIGINAL_NODE_SCHEMA = None
_ORIGINAL_NODE_EXECUTE = None


def opening_picture_prompt(prompt, picture_number):
    """Add an explicit frame-zero alignment instruction for a generated picture."""

    picture_number = int(picture_number)
    if picture_number <= 0:
        raise ValueError("picture_number must be positive")
    picture = "<Picture %d>" % picture_number
    instruction = (
        "For the target video, at 0.00 seconds into the target video, "
        "%s is fully referenced.\n\n"
        "%s is the opening frame of this continuing video segment. Preserve "
        "its exact subject state, pose, composition, lighting, environment, "
        "and motion phase as the action continues smoothly forward."
        % (picture, picture)
    )
    body = str(prompt or "").strip()
    return instruction if not body else instruction + "\n\n" + body


def _picture_number(items):
    return 1 + sum(
        1 for item in items if isinstance(item, dict) and item.get("type") == "image"
    )


def _dynamic_digest(prompt, frame):
    digest = hashlib.sha256(str(prompt).encode("utf-8"))
    try:
        digest.update(reference_runner.tensor_digest(frame).encode("ascii"))
    except Exception:
        digest.update(str(tuple(frame.shape)).encode("ascii"))
    return digest.hexdigest()


@dataclass
class DynamicOpeningPictureConditionings:
    """Build one Qwen conditioning at a time from the previous decoded frame."""

    run_obj: object
    clip: object
    prompts: tuple[str, ...]
    base_items: tuple[dict, ...]
    video_vae: object
    cond_cache: str
    picture_number: int
    initial_conditioning: object
    current_picture_block: object = None
    _current_conditioning: object = None

    def for_index(self, index):
        index = int(index)
        if index < 0 or index >= len(self.prompts):
            raise IndexError("chunk prompt conditioning index is out of range")

        if index == 0:
            self.current_picture_block = None
            self._current_conditioning = self.initial_conditioning
            return self.initial_conditioning, hashlib.sha256(
                self.prompts[0].encode("utf-8")
            ).hexdigest()

        frame = getattr(self.run_obj, "_h3_next_opening_picture", None)
        if frame is None:
            raise RuntimeError(
                "opening-picture continuity has no decoded frame for chunk %d; "
                "this mode requires one uninterrupted LongFormReferenceVideo execution"
                % index
            )

        item, block, note = ref_builder.encode_image_ref(
            self.video_vae,
            frame,
            self.run_obj.canvas,
            "match",
            cond_cache=self.cond_cache,
        )
        prompt = opening_picture_prompt(
            self.prompts[index], self.picture_number
        )
        conditioning = harness._encode(
            self.clip,
            prompt,
            list(self.base_items) + [item],
            self.cond_cache,
        )

        old = self._current_conditioning
        self._current_conditioning = conditioning
        self.current_picture_block = block
        digest = _dynamic_digest(prompt, frame)
        logging.info(
            "%s chunk %d injected <Picture %d> from previous chunk frame %d (%s)",
            LOG,
            index,
            self.picture_number,
            self.run_obj.geometry.stride_frames,
            note,
        )
        if old is not None and old is not self.initial_conditioning:
            del old
        gc.collect()
        return conditioning, digest


class _OpeningPictureBlocks:
    """Append the current generated picture to the chunk's normal DiT refs."""

    def __init__(self, base_blocks, conditionings):
        self.base_blocks = base_blocks
        self.conditionings = conditionings

    def __iter__(self):
        # base_blocks can itself be the chronological-audio reference sequence;
        # consume it exactly once per chunk before appending our visual anchor.
        blocks = list(self.base_blocks)
        block = self.conditionings.current_picture_block
        if block is not None:
            blocks.append(block)
        return iter(blocks)

    def __len__(self):
        base = len(self.base_blocks)
        return base + (1 if self.conditionings.current_picture_block is not None else 0)


def _prepare_references(self, *args, **kwargs):
    if not _ACTIVE.get():
        return _ORIGINAL_PREPARE(self, *args, **kwargs)

    prompt = str(kwargs.get("prompt") or "")
    envelope = chunk_prompt_timeline.unpack_chunk_prompt_envelope(prompt)
    if envelope is not None:
        prompts = tuple(envelope["prompts"])
    else:
        count = chunk_count_for(
            self.target_frames,
            self.geometry.chunk_frames,
            self.geometry.stride_frames,
        )
        prompts = tuple(prompt for _ in range(count))

    # This feature intentionally does not turn the runner's disk artifacts into
    # a supported interrupted-run continuation API.  A pre-populated sample set
    # cannot provide the in-memory decoded opening frame expected below.
    existing = self._first_invalid("samples", len(prompts))
    if existing:
        raise RuntimeError(
            "opening-picture continuity requires a fresh run directory; "
            "interrupted LongFormReferenceVideo executions are not resumable"
        )

    items, blocks, notes = chunk_prompt_timeline._encode_reference_assets(
        self,
        video_vae=kwargs["video_vae"],
        audio_vae=kwargs["audio_vae"],
        ref_images=kwargs.get("ref_images"),
        ref_videos=kwargs.get("ref_videos"),
        ref_video_audios=kwargs.get("ref_video_audios"),
        ref_audios=kwargs.get("ref_audios"),
        ref_image_size=kwargs.get("ref_image_size", "native"),
        cond_cache=kwargs.get("cond_cache", "auto"),
    )
    self.static_items = items
    self.static_blocks = blocks
    self._h3_next_opening_picture = None

    initial = harness._encode(
        kwargs["clip"],
        prompts[0],
        items,
        kwargs.get("cond_cache", "auto"),
    )
    picture_number = _picture_number(items)
    conditionings = DynamicOpeningPictureConditionings(
        run_obj=self,
        clip=kwargs["clip"],
        prompts=prompts,
        base_items=tuple(items),
        video_vae=kwargs["video_vae"],
        cond_cache=kwargs.get("cond_cache", "auto"),
        picture_number=picture_number,
        initial_conditioning=initial,
    )
    notes = list(notes) + [
        "opening picture: chunk 2+ inject previous frame %d as <Picture %d>"
        % (self.geometry.stride_frames, picture_number)
    ]
    if self.manifest:
        self.manifest.update_state(references_prepared=True)
    return conditionings, notes


def _conditioning_for_chunk(conditioning, index):
    if not isinstance(conditioning, DynamicOpeningPictureConditionings):
        return _ORIGINAL_CONDITIONING_FOR_CHUNK(conditioning, index)

    run_obj = conditioning.run_obj
    if not isinstance(run_obj.static_blocks, _OpeningPictureBlocks):
        # At this point chunk_aligned_audio_refs has already had the opportunity
        # to wrap static_blocks, so preserve that wrapper underneath ours.
        run_obj.static_blocks = _OpeningPictureBlocks(
            run_obj.static_blocks, conditioning
        )
    return conditioning.for_index(index)


def _retain_opening_picture(run, index, pixels, chunk_count):
    """Keep one CPU frame from this chunk to open the next one.

    Registered as a decoded-pixels hook so the canonical ``_emit_chunk`` keeps
    owning the diagnostics dump and the audio-aware completed-preview staging.
    """

    if not _ACTIVE.get() or index >= chunk_count - 1:
        return

    source_index = int(run.geometry.stride_frames)
    if source_index >= int(pixels.shape[0]):
        raise RuntimeError(
            "chunk %d decoded %d frames; cannot extract next chunk frame zero at %d"
            % (index, int(pixels.shape[0]), source_index)
        )
    run._h3_next_opening_picture = (
        pixels[source_index:source_index + 1].detach().clone()
    )
    logging.info(
        "%s chunk %d retained decoded frame %d for chunk %d frame zero",
        LOG,
        index,
        source_index,
        index + 1,
    )


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


def _patch_node():
    global _ORIGINAL_NODE_SCHEMA, _ORIGINAL_NODE_EXECUTE

    node_cls = reference_preview_nodes.MiniMaxH3LongFormReferenceVideoPreview
    _ORIGINAL_NODE_SCHEMA = node_cls.define_schema.__func__
    _ORIGINAL_NODE_EXECUTE = node_cls.execute.__func__

    @classmethod
    def define_schema(cls):
        schema = _ORIGINAL_NODE_SCHEMA(cls)
        inputs = list(schema.inputs)
        if any(
            getattr(item, "id", getattr(item, "name", None))
            == "inject_prev_frame_picture"
            for item in inputs
        ):
            return schema
        insert_at = next(
            (
                i + 1
                for i, item in enumerate(inputs)
                if getattr(item, "id", getattr(item, "name", None))
                == "chunk_prompt_plan"
            ),
            next(
                (
                    i + 1
                    for i, item in enumerate(inputs)
                    if getattr(item, "id", getattr(item, "name", None))
                    == "prompt"
                ),
                len(inputs),
            ),
        )
        inputs.insert(
            insert_at,
            io.Boolean.Input(
                "inject_prev_frame_picture",
                default=False,
                tooltip=(
                    "For every chunk after the first, decode the previous chunk "
                    "frame that becomes this chunk's frame zero and inject it as "
                    "the next <Picture N> reference. Qwen is run just in time for "
                    "each continuation chunk. Existing video/audio latent carry "
                    "remains enabled independently."
                ),
            ),
        )
        return _replace_inputs(schema, inputs)

    @classmethod
    def execute(cls, *args, inject_prev_frame_picture=False, **kwargs):
        token = _ACTIVE.set(bool(inject_prev_frame_picture))
        try:
            logging.info(
                "%s %s",
                LOG,
                "enabled" if inject_prev_frame_picture else "disabled",
            )
            return _ORIGINAL_NODE_EXECUTE(cls, *args, **kwargs)
        finally:
            _ACTIVE.reset(token)

    node_cls.define_schema = define_schema
    node_cls.execute = execute


def install():
    """Install after chunk-prompt, audio-reference, and preview patches."""

    global _INSTALLED, _ORIGINAL_PREPARE
    global _ORIGINAL_CONDITIONING_FOR_CHUNK
    if _INSTALLED:
        return

    _ORIGINAL_PREPARE = chunk_aligned_audio_refs._ORIGINAL_PREPARE
    chunk_aligned_audio_refs._ORIGINAL_PREPARE = _prepare_references

    _ORIGINAL_CONDITIONING_FOR_CHUNK = (
        chunk_prompt_timeline.conditioning_for_chunk
    )
    chunk_prompt_timeline.conditioning_for_chunk = _conditioning_for_chunk

    runner.register_decoded_pixels_hook(_retain_opening_picture)

    _patch_node()
    _INSTALLED = True


__all__ = [
    "DynamicOpeningPictureConditionings",
    "install",
    "opening_picture_prompt",
]
