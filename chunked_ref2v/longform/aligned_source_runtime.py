"""Opt-in same-time source AV conditioning for native N+1 continuation.

This extends ``MiniMaxH3LongFormAVContinuation`` without changing its legacy
behavior.  When an aligned source video is connected, every generated chunk
receives the chronologically corresponding source interval as an H3 reference.
The source reference can remain at source/native resolution while the target
canvas stays independently selectable, which makes reference-based regeneration
and upscaling possible without pixel-upscaling the target initialization.
"""

from __future__ import annotations

import contextvars
import inspect
from dataclasses import dataclass, is_dataclass, replace

import torch
from comfy_api.latest import io

from .. import ref_builder
from . import av_continuation_nodes as av
from .chunk_aligned_audio_refs import slice_audio_reference
from .reference_runner import _ordered_values, _paired_audio

_INSTALLED = False
_ORIGINAL_SCHEMA = None
_ORIGINAL_EXECUTE = None
_ORIGINAL_PREPARE = None
_ORIGINAL_PROMPTS = None
_ORIGINAL_COUNT_ITEMS = None
_ORIGINAL_RUN_IDENTITY = None


@dataclass(frozen=True)
class _AlignedInputs:
    video: object = None
    audio: object = None
    video_size: str = "source"
    source_video_number: int = 0
    source_audio_number: int = 0


_ACTIVE = contextvars.ContextVar(
    "h3_nplusone_aligned_source", default=_AlignedInputs()
)


def slice_video_reference(frames, *, start_frame, frame_count):
    """Return one chronological full generation window, holding the last frame.

    The caller separately validates that all frames which will be committed to
    the target exist in the source.  Padding is therefore only for the unused
    tail of the final legal H3 generation window and never loops to frame zero.
    """

    if frames is None:
        return None
    start_frame = int(start_frame)
    frame_count = int(frame_count)
    if start_frame < 0:
        raise ValueError("start_frame must be non-negative")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    total = int(frames.shape[0])
    if start_frame >= total:
        raise ValueError(
            "aligned source starts at frame %d but source has only %d frames"
            % (start_frame, total)
        )
    stop = min(total, start_frame + frame_count)
    clipped = frames[start_frame:stop]
    if int(clipped.shape[0]) < frame_count:
        pad = clipped[-1:].expand(
            frame_count - int(clipped.shape[0]), *clipped.shape[1:]
        )
        clipped = torch.cat([clipped, pad], dim=0)
    return clipped


def _reference_canvas(frames, target_canvas, mode):
    mode = str(mode or "source")
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if mode == "match":
        return tuple(target_canvas)
    if mode == "native":
        return ref_builder.adapt_canvas(w, h)
    if mode == "source":
        return ref_builder.canvas_for_source(w, h)
    raise ValueError("unknown aligned_source_video_size %r" % mode)


def encode_aligned_source_reference(
    *, video_vae, audio_vae, frames, audio, target_canvas, video_size, cond_cache
):
    """Encode a same-time source interval independently of the target canvas."""

    ref_canvas = _reference_canvas(frames, target_canvas, video_size)
    items, block, note = ref_builder.encode_video_ref(
        video_vae,
        frames,
        ref_canvas,
        audio=audio,
        audio_vae=audio_vae,
        cond_cache=cond_cache,
    )
    return items, block, note, ref_canvas


def aligned_source_prompt(prompt, video_number, audio_number=0):
    video_number = int(video_number)
    audio_number = int(audio_number or 0)
    if video_number <= 0:
        raise ValueError("source video reference number must be positive")
    video = "<Video %d>" % video_number
    if audio_number > 0:
        audio = "<Audio %d>" % audio_number
        audio_text = (
            " %s is the synchronized source audio for the same chronological "
            "interval and should preserve its timing and audiovisual phase."
            % audio
        )
        task = "[reference generation + audio reference]"
    else:
        audio_text = ""
        task = "[reference generation]"
    instruction = (
        "Reference relationship for this generation: %s %s shows the same "
        "chronological interval as the target video, not an earlier segment. "
        "Reconstruct the same subjects, actions, poses, expressions, camera "
        "motion, framing, scene state, lighting, and event timing on the target "
        "canvas while preserving the source video's temporal structure and "
        "adding spatial detail appropriate to the target resolution.%s"
        % (task, video, audio_text)
    )
    body = str(prompt or "").strip()
    return instruction if not body else instruction + "\n\n" + body


def _static_reference_numbers(ref_videos, ref_video_audios, ref_audios):
    video_count = 0
    audio_count = 0
    for name, _frames in _ordered_values(ref_videos):
        video_count += 1
        if _paired_audio(ref_video_audios, name) is not None:
            audio_count += 1
    audio_count += sum(1 for _ in _ordered_values(ref_audios))
    return video_count, audio_count


class _AlignedState:
    def __init__(
        self,
        *,
        static_items,
        static_blocks,
        source_video,
        source_audio,
        video_vae,
        audio_vae,
        target_canvas,
        chunk_frames,
        fps,
        video_size,
        cond_cache,
    ):
        self.static_items = list(static_items)
        self.static_blocks = list(static_blocks)
        self.source_video = source_video
        self.source_audio = source_audio
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.target_canvas = tuple(target_canvas)
        self.chunk_frames = int(chunk_frames)
        self.fps = int(fps)
        self.video_size = str(video_size)
        self.cond_cache = cond_cache
        self.index = 0
        self._prepared_index = None
        self._source_items = None
        self._source_block = None

    def count(self, kind):
        base = sum(
            1
            for item in self.static_items
            if isinstance(item, dict) and item.get("type") == kind
        )
        if kind == "video":
            return base + 1
        if kind == "audio" and self.source_audio is not None:
            return base + 1
        return base

    def prepare(self):
        if self._prepared_index == self.index:
            return
        start = self.index * self.chunk_frames
        frames = slice_video_reference(
            self.source_video,
            start_frame=start,
            frame_count=self.chunk_frames,
        )
        audio = (
            slice_audio_reference(
                self.source_audio,
                start_frame=start,
                frame_count=self.chunk_frames,
                fps=self.fps,
            )
            if self.source_audio is not None
            else None
        )
        items, block, _note, _canvas = encode_aligned_source_reference(
            video_vae=self.video_vae,
            audio_vae=self.audio_vae,
            frames=frames,
            audio=audio,
            target_canvas=self.target_canvas,
            video_size=self.video_size,
            cond_cache=self.cond_cache,
        )
        self._source_items = items
        self._source_block = block
        self._prepared_index = self.index

    def advance(self):
        self.index += 1
        self._prepared_index = None
        self._source_items = None
        self._source_block = None


class _ItemsView:
    def __init__(self, state):
        self.state = state

    def __iter__(self):
        self.state.prepare()
        yield from self.state.static_items
        yield from self.state._source_items

    def stable_count(self, kind):
        return self.state.count(kind)


class _BlocksView:
    def __init__(self, state):
        self.state = state

    def __iter__(self):
        self.state.prepare()
        try:
            yield from self.state.static_blocks
            yield self.state._source_block
        finally:
            self.state.advance()


def _prepare_references(*args, **kwargs):
    static_items, static_blocks, notes = _ORIGINAL_PREPARE(*args, **kwargs)
    active = _ACTIVE.get()
    if active.video is None:
        return static_items, static_blocks, notes

    canvas = kwargs["canvas"]
    # The N+1 runner's target geometry is fixed at 24 fps and its chunk length
    # is not passed into reference preparation.  It is stored on the active
    # context by the execute wrapper below.
    chunk_frames = getattr(active, "chunk_frames", None)
    fps = getattr(active, "fps", 24)
    if chunk_frames is None:
        raise RuntimeError("aligned source runtime is missing chunk geometry")
    state = _AlignedState(
        static_items=static_items,
        static_blocks=static_blocks,
        source_video=active.video,
        source_audio=active.audio,
        video_vae=kwargs["video_vae"],
        audio_vae=kwargs["audio_vae"],
        target_canvas=canvas,
        chunk_frames=chunk_frames,
        fps=fps,
        video_size=active.video_size,
        cond_cache=kwargs.get("cond_cache", "auto"),
    )
    return (
        _ItemsView(state),
        _BlocksView(state),
        list(notes) + [
            "aligned source: chronological per-chunk AV reference; video sizing=%s"
            % active.video_size
        ],
    )


def _count_items(items, kind):
    stable = getattr(items, "stable_count", None)
    if stable is not None:
        return int(stable(kind))
    return _ORIGINAL_COUNT_ITEMS(items, kind)


def _prompts_for_plan(*args, **kwargs):
    prompts = _ORIGINAL_PROMPTS(*args, **kwargs)
    active = _ACTIVE.get()
    if active.video is None:
        return prompts
    return [
        aligned_source_prompt(
            prompt,
            active.source_video_number,
            active.source_audio_number,
        )
        for prompt in prompts
    ]


def _run_identity(*args, **kwargs):
    identity = _ORIGINAL_RUN_IDENTITY(*args, **kwargs)
    active = _ACTIVE.get()
    if active.video is not None:
        identity["aligned_source"] = av._reference_identity({
            "video": active.video,
            "audio": active.audio,
            "video_size": active.video_size,
        })
    return identity


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


def install():
    global _INSTALLED, _ORIGINAL_SCHEMA, _ORIGINAL_EXECUTE
    global _ORIGINAL_PREPARE, _ORIGINAL_PROMPTS, _ORIGINAL_COUNT_ITEMS
    global _ORIGINAL_RUN_IDENTITY
    if _INSTALLED:
        return

    node_cls = av.MiniMaxH3LongFormAVContinuation
    _ORIGINAL_SCHEMA = node_cls.define_schema.__func__
    _ORIGINAL_EXECUTE = node_cls.execute.__func__
    execute_signature = inspect.signature(_ORIGINAL_EXECUTE)
    _ORIGINAL_PREPARE = av._encode_static_references
    _ORIGINAL_PROMPTS = av.prompts_for_av_continuation_plan
    _ORIGINAL_COUNT_ITEMS = av._count_items
    _ORIGINAL_RUN_IDENTITY = av._run_identity

    @classmethod
    def define_schema(cls):
        schema = _ORIGINAL_SCHEMA(cls)
        inputs = list(schema.inputs)
        # Append only: Comfy persists widget values positionally.
        inputs.extend([
            io.Image.Input(
                "aligned_source_video",
                optional=True,
                tooltip=(
                    "Optional original/source video for same-time regeneration. "
                    "Each N+1 invocation receives only the chronological source "
                    "interval matching that target chunk."
                ),
            ),
            io.Audio.Input(
                "aligned_source_audio",
                optional=True,
                tooltip=(
                    "Optional source audio paired with aligned_source_video. "
                    "It is sliced to the exact same chronological interval and "
                    "never wraps to the beginning."
                ),
            ),
            io.Combo.Input(
                "aligned_source_video_size",
                options=["source", "native", "match"],
                default="source",
                tooltip=(
                    "source keeps the reference at its own 32-aligned resolution; "
                    "native uses H3's normal reference canvas; match reproduces "
                    "the legacy target-canvas-pinned reference behavior."
                ),
            ),
        ])
        return _replace_inputs(schema, inputs)

    @classmethod
    def execute(
        cls,
        *args,
        aligned_source_video=None,
        aligned_source_audio=None,
        aligned_source_video_size="source",
        **kwargs,
    ):
        bound = execute_signature.bind(cls, *args, **kwargs)
        bound.apply_defaults()
        prompt_plan = bound.arguments.get("n_plus_one_prompt_plan")
        if prompt_plan is not None:
            prompt_plan = av.validate_nplusone_chunk_prompt_plan(prompt_plan)
            target_frames = int(prompt_plan["target_frames"])
            chunk_frames = int(prompt_plan["chunk_frames"])
        else:
            target_frames = int(bound.arguments["output_seconds"]) * 24
            chunk_frames = int(bound.arguments["chunk_frames"])

        if aligned_source_audio is not None and aligned_source_video is None:
            raise ValueError(
                "aligned_source_audio requires aligned_source_video"
            )
        if aligned_source_video is not None:
            source_frames = int(aligned_source_video.shape[0])
            if source_frames < target_frames:
                raise ValueError(
                    "aligned source has %d frames but requested output needs %d "
                    "frames at 24 fps" % (source_frames, target_frames)
                )
            static_videos, static_audios = _static_reference_numbers(
                bound.arguments.get("ref_videos"),
                bound.arguments.get("ref_video_audios"),
                bound.arguments.get("ref_audios"),
            )
            source_video_number = static_videos + 1
            source_audio_number = (
                static_audios + 1 if aligned_source_audio is not None else 0
            )
        else:
            source_video_number = source_audio_number = 0

        active = _AlignedInputs(
            video=aligned_source_video,
            audio=aligned_source_audio,
            video_size=str(aligned_source_video_size or "source"),
            source_video_number=source_video_number,
            source_audio_number=source_audio_number,
        )
        # Frozen dataclass: attach runtime-only geometry without exposing it as
        # part of the public configuration object.
        object.__setattr__(active, "chunk_frames", chunk_frames)
        object.__setattr__(active, "fps", 24)
        token = _ACTIVE.set(active)
        try:
            return _ORIGINAL_EXECUTE(*bound.args, **bound.kwargs)
        finally:
            _ACTIVE.reset(token)

    node_cls.define_schema = define_schema
    node_cls.execute = execute
    av._encode_static_references = _prepare_references
    av.prompts_for_av_continuation_plan = _prompts_for_plan
    av._count_items = _count_items
    av._run_identity = _run_identity
    _INSTALLED = True


__all__ = [
    "aligned_source_prompt",
    "encode_aligned_source_reference",
    "install",
    "slice_video_reference",
]
