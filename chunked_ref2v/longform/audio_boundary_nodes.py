"""Comfy-facing exact audio-boundary enforcement for long-form H3 nodes.

This is intentionally a small registration-time compatibility layer. It keeps
existing workflow widget positions stable while giving the timeline and both
long-form nodes the same ``align_audio_chunks`` policy.
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import is_dataclass, replace

from comfy_api.latest import io

from ..audio_boundary_profile import resolve_audio_boundary_profile
from . import (
    chunk_prompt_timeline,
    preview_nodes,
    reference_nodes,
    reference_preview_nodes,
)

LOG = "[H3 Extended] audio boundary profile"
_INSTALLED = False


def _name(item):
    return getattr(item, "id", None) or getattr(item, "name", None)


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


def _alignment_input():
    return io.Boolean.Input(
        "align_audio_chunks",
        default=True,
        tooltip=(
            "Enforce exact H3 audio/video chunk boundaries. When enabled, "
            "chunk_frames is snapped to the nearest H3 17k+5 length that also "
            "lands on the shared 24 fps / 40 Hz time grid, and overlap_frames "
            "is snapped to the nearest exact video-latent suffix on that same "
            "grid. This makes C, O, and S=C-O exact audio boundaries. Disable "
            "only for compatibility or diagnostics."
        ),
    )


def _patch_schema_add_toggle(node_cls):
    original = node_cls.define_schema.__func__

    @classmethod
    def define_schema(cls):
        schema = original(cls)
        inputs = list(schema.inputs)
        existing = next(
            (index for index, item in enumerate(inputs) if _name(item) == "align_audio_chunks"),
            None,
        )
        if existing is not None:
            # Replace instead of mutating the old Input object. Comfy has used
            # both dataclass and pydantic-backed schema models, and replacing the
            # item is stable across both while preserving widget position.
            inputs[existing] = _alignment_input()
            return _replace_inputs(schema, inputs)

        insert_at = next(
            (index for index, item in enumerate(inputs) if _name(item) == "ref_images"),
            len(inputs),
        )
        inputs.insert(insert_at, _alignment_input())
        return _replace_inputs(schema, inputs)

    node_cls.define_schema = define_schema


def _patch_timeline_execute():
    node_cls = chunk_prompt_timeline.MiniMaxH3ChunkPromptTimeline
    original = node_cls.execute.__func__
    signature = inspect.signature(original)

    @classmethod
    def execute(cls, *args, align_audio_chunks=True, **kwargs):
        bound = signature.bind(cls, *args, **kwargs)
        bound.apply_defaults()
        chunk_frames, overlap_frames, note = resolve_audio_boundary_profile(
            bound.arguments["chunk_frames"],
            bound.arguments["overlap_frames"],
            bool(align_audio_chunks),
        )
        bound.arguments["chunk_frames"] = chunk_frames
        bound.arguments["overlap_frames"] = overlap_frames
        if note:
            logging.warning("%s timeline %s", LOG, note)
        return original(*bound.args, **bound.kwargs)

    node_cls.execute = execute


def _patch_ref2v_execute():
    node_cls = preview_nodes.MiniMaxH3LongFormRef2VPreview
    original = node_cls.execute.__func__
    signature = inspect.signature(original)

    @classmethod
    def execute(cls, *args, align_audio_chunks=True, **kwargs):
        bound = signature.bind(cls, *args, **kwargs)
        bound.apply_defaults()
        chunk_frames, overlap_frames, note = resolve_audio_boundary_profile(
            bound.arguments["chunk_frames"],
            bound.arguments["overlap_frames"],
            bool(align_audio_chunks),
        )
        bound.arguments["chunk_frames"] = chunk_frames
        bound.arguments["overlap_frames"] = overlap_frames
        if note:
            logging.warning("%s %s", LOG, note)
        return original(*bound.args, **bound.kwargs)

    node_cls.execute = execute


def _patch_reference_execute():
    """Wrap the reference node after the chunk-prompt timeline wrapper.

    ``chunk_prompt_timeline.install()`` has already replaced this execute method
    with a generic ``*args/**kwargs`` adapter by the time the top-level extension
    imports us. Use the stable base-node signature only as a positional map, then
    forward into that adapter so per-chunk prompt packing remains intact.
    """

    node_cls = reference_preview_nodes.MiniMaxH3LongFormReferenceVideoPreview
    original = node_cls.execute.__func__
    positional_signature = inspect.signature(
        reference_nodes.MiniMaxH3LongFormReferenceVideo.execute.__func__
    )
    names = [name for name in positional_signature.parameters if name != "cls"]
    positions = {name: index for index, name in enumerate(names)}

    def get_value(args, kwargs, name, default):
        if name in kwargs:
            return kwargs[name]
        position = positions[name]
        if position < len(args):
            return args[position]
        return default

    def set_value(args, kwargs, name, value):
        position = positions[name]
        if position < len(args):
            args[position] = value
        else:
            kwargs[name] = value

    @classmethod
    def execute(cls, *args, align_audio_chunks=True, **kwargs):
        args = list(args)
        # A legacy direct caller may still provide align_audio_chunks positionally.
        # Positional input wins over the wrapper default; normal Comfy execution
        # supplies the new Boolean widget by name.
        enabled = bool(
            get_value(args, kwargs, "align_audio_chunks", align_audio_chunks)
        )
        requested_chunk = get_value(args, kwargs, "chunk_frames", 90)
        requested_overlap = get_value(args, kwargs, "overlap_frames", 4)
        chunk_frames, overlap_frames, note = resolve_audio_boundary_profile(
            requested_chunk,
            requested_overlap,
            enabled,
        )
        set_value(args, kwargs, "chunk_frames", chunk_frames)
        set_value(args, kwargs, "overlap_frames", overlap_frames)
        set_value(args, kwargs, "align_audio_chunks", enabled)
        if note:
            logging.warning("%s %s", LOG, note)
        return original(cls, *args, **kwargs)

    node_cls.execute = execute


def install():
    global _INSTALLED
    if _INSTALLED:
        return

    timeline_cls = chunk_prompt_timeline.MiniMaxH3ChunkPromptTimeline
    ref2v_cls = preview_nodes.MiniMaxH3LongFormRef2VPreview
    reference_cls = reference_preview_nodes.MiniMaxH3LongFormReferenceVideoPreview

    _patch_schema_add_toggle(timeline_cls)
    _patch_schema_add_toggle(ref2v_cls)
    _patch_schema_add_toggle(reference_cls)

    _patch_timeline_execute()
    _patch_ref2v_execute()
    _patch_reference_execute()

    _INSTALLED = True


__all__ = ["install"]
