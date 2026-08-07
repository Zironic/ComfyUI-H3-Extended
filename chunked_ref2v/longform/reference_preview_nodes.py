"""LongFormReferenceVideo variant with generated audio and dual live previews."""

from __future__ import annotations

import logging
import os
from dataclasses import is_dataclass, replace

import torch
from comfy_api.latest import ComfyExtension, io

from ..geometry import HarnessGeometry
from . import chunk_aligned_audio_refs, runner
from .audio_runtime import resolve_audio_aligned_overlap
from .chunk_stream import chunk_count_for
from .nodes import (
    _describe_ffmpeg,
    _preview,
    _preview_from_video,
    _video_from_path,
)
from .preview import (
    DECODER_AUTO,
    LongFormPreviewPublisher,
    PreviewOptions,
    activate,
    deactivate,
    resolve_unique_id,
)
from .preview_nodes import _decoder_input
from .reference_nodes import (
    MiniMaxH3LongFormReferenceVideo,
    _resolve_root,
    _validate_canvas,
)

try:
    from .. import memory
except ImportError:  # pragma: no cover
    import memory

LOG = "[H3 Extended] longform reference preview"


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


class MiniMaxH3LongFormReferenceVideoPreview(
    MiniMaxH3LongFormReferenceVideo
):
    """Generic long-form AV generation with current/completed previews."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        inputs = list(schema.inputs)
        preview_inputs = [
            io.Boolean.Input(
                "chunk_align_audio_references",
                default=False,
                tooltip=(
                    "When enabled, each model invocation receives only the "
                    "chronological slice of every audio reference matching that "
                    "video chunk, including the same overlap. The audio never "
                    "wraps or restarts at sample zero. Leave disabled to reuse "
                    "the complete audio reference in every chunk."
                ),
            ),
            io.Boolean.Input(
                "current_chunk_preview",
                default=True,
                tooltip=(
                    "Show the current denoised chunk while it is sampling. "
                    "TAEH3 decodes the whole chunk by default; nothing needs "
                    "to be connected for that."
                ),
            ),
            io.Int.Input(
                "preview_every_steps",
                default=1,
                min=1,
                max=100,
                step=1,
                tooltip=(
                    "Publish the current-chunk preview every this many "
                    "sampler steps. The final step is always published."
                ),
            ),
            io.Int.Input(
                "current_preview_frames",
                default=0,
                min=0,
                max=1024,
                step=1,
                tooltip=(
                    "TAEH3 only: frames in the current-chunk preview, 0 for "
                    "the whole chunk. The preview_vae and latent paths ignore "
                    "it because neither has a meaningful count to choose - the "
                    "VAE decodes a whole 17-frame group for the cost of one "
                    "frame, and the latent previewer emits one image per "
                    "latent position."
                ),
            ),
            io.Boolean.Input(
                "completed_chunks_preview",
                default=True,
                tooltip=(
                    "Publish every overlap-trimmed completed chunk as a "
                    "small finalized MP4 segment. The second pane plays "
                    "all available segments as a playlist."
                ),
            ),
            io.Int.Input(
                "live_preview_width",
                default=512,
                min=128,
                max=1024,
                step=32,
                tooltip="Maximum width of both live preview streams.",
            ),
            io.Vae.Input(
                "preview_vae",
                optional=True,
                tooltip=(
                    "Optional exact-VAE preview, used only when TAEH3 is "
                    "unavailable or its decode fails, or when "
                    "current_preview_decoder is set to preview_vae. TAEH3 is "
                    "the default backend and needs nothing connected here. "
                    "The production video VAE is never loaded from inside the "
                    "active sampler callback."
                ),
            ),
        ]

        # Anchor above align_audio_chunks so that widget stays last: Comfy maps
        # widgets_values by position, and the preview widgets already have saved
        # positions in existing workflows.
        anchors = ("align_audio_chunks", "ref_images")
        insert_at = next(
            (
                index
                for index, item in enumerate(inputs)
                if getattr(item, "name", None) in anchors
            ),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_inputs
        # Appended past the anchors on purpose. Inserting here would shift
        # align_audio_chunks and break every saved workflow's widgets_values;
        # appending leaves every existing index alone.
        inputs.append(_decoder_input())
        return _replace_inputs(schema, inputs)

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        prompt,
        sampler,
        sigmas,
        seed,
        output_seconds=30,
        width=1344,
        height=768,
        chunk_frames=90,
        overlap_frames=4,
        align_audio_chunks=False,
        carry=runner.CARRY_OVERLAP,
        ref_image_size="native",
        cond_cache="auto",
        attention="auto",
        activation="mlp_chunked_native",
        run_directory="",
        ffmpeg_location="",
        save_frames=False,
        diagnostic_dump_chunks=False,
        chunk_align_audio_references=False,
        current_chunk_preview=True,
        preview_every_steps=1,
        current_preview_frames=0,
        completed_chunks_preview=True,
        live_preview_width=512,
        preview_vae=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        current_preview_decoder=DECODER_AUTO,
        unique_id=None,
    ) -> io.NodeOutput:
        overlap_frames, alignment_note = resolve_audio_aligned_overlap(
            chunk_frames, overlap_frames, align_audio_chunks,
        )
        if alignment_note:
            logging.warning("%s %s", LOG, alignment_note)
        geometry = HarnessGeometry(
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        ).validate()
        canvas = _validate_canvas(width, height)
        target_frames = int(output_seconds * geometry.fps)
        chunk_count = chunk_count_for(
            target_frames,
            geometry.chunk_frames,
            geometry.stride_frames,
        )

        model, memory_status = memory.arm(
            model,
            attention=attention,
            activation=activation,
        )
        logging.info("%s %s", LOG, memory.describe(memory_status))

        root = _resolve_root(run_directory, carry, chunk_frames)
        # See resolve_unique_id: hidden inputs reach V3 nodes through the class
        # clone, not through execute() kwargs. Reading the signature default
        # addressed every preview event to node id "None".
        unique_id = resolve_unique_id(cls, unique_id)
        publisher = LongFormPreviewPublisher(
            node_id=unique_id,
            model=model,
            video_vae=preview_vae,
            root=root,
            fps=geometry.fps,
            ffmpeg_location=ffmpeg_location.strip() or None,
            options=PreviewOptions(
                current_enabled=bool(current_chunk_preview),
                completed_enabled=bool(completed_chunks_preview),
                every_steps=int(preview_every_steps),
                current_frames=int(current_preview_frames),
                width=int(live_preview_width),
                decoder=str(current_preview_decoder or DECODER_AUTO),
            ),
            # This runtime decodes a waveform per chunk, and the node always
            # writes one, so every completed segment carries audio.
            audio_expected=True,
        )
        token = activate(publisher)
        publisher._announce("reset")
        logging.info(
            "%s enabled for node %s: current=%s every=%d frames=%s decoder=%s "
            "taeh3=%s; completed=%s width=%d exact_vae=%s audio_refs=%s",
            LOG,
            unique_id,
            current_chunk_preview,
            preview_every_steps,
            current_preview_frames or "all",
            current_preview_decoder,
            publisher.taeh3 is not None,
            completed_chunks_preview,
            live_preview_width,
            preview_vae is not None,
            "chunk-aligned" if chunk_align_audio_references else "full-track",
        )
        try:
            summary = chunk_aligned_audio_refs.run(
                chunk_align_audio_references=chunk_align_audio_references,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                chunk_count=chunk_count,
                target_frames=target_frames,
                model=model,
                clip=clip,
                video_vae=video_vae,
                audio_vae=audio_vae,
                prompt=prompt,
                sampler=sampler,
                sigmas=sigmas,
                seed=seed,
                carry=carry,
                canvas=canvas,
                root=root,
                ref_images=ref_images,
                ref_videos=ref_videos,
                ref_video_audios=ref_video_audios,
                ref_audios=ref_audios,
                ref_image_size=ref_image_size,
                cond_cache=cond_cache,
                save_frames=save_frames,
                diagnostic_dump_chunks=diagnostic_dump_chunks,
                ffmpeg_location=ffmpeg_location.strip() or None,
                runtime_config={
                    "attention": attention,
                    "activation": activation,
                    "audio_carry_frames": overlap_frames,
                    "audio_output": True,
                    "audio_reference_mode": (
                        "chunk_aligned"
                        if chunk_align_audio_references
                        else "full_track"
                    ),
                },
            )
        finally:
            # An aborted or audio-less run can leave the last chunk staged;
            # publish it silently rather than dropping it from the pane.
            try:
                publisher.flush_completed_chunk()
            except Exception as exc:
                logging.warning("%s final preview flush failed: %s", LOG, exc)
            deactivate(token)

        audio_start, audio_count = summary["audio_overlap_latent"]
        report_lines = [
            "MiniMax H3 LongFormReferenceVideo",
            "profile   %s" % summary["profile"],
            "carry     %s (video + audio)" % carry,
            "canvas    %dx%d (%.2f MP, %.3f:1)"
            % (
                canvas[0],
                canvas[1],
                canvas[0] * canvas[1] / 1e6,
                canvas[0] / canvas[1],
            ),
            "chunks    %d" % chunk_count,
            "output    %d exact frames (%.3f s at %d fps)"
            % (
                summary["frames"],
                summary["frames"] / geometry.fps,
                geometry.fps,
            ),
            "audio refs %s"
            % (
                "chunk-aligned chronological slices"
                if chunk_align_audio_references
                else "complete references reused per chunk"
            ),
            "audio     %d samples at %d Hz; O=%d -> latent [%d:%d]"
            % (
                summary["audio_samples"],
                summary["audio_sample_rate"],
                overlap_frames,
                audio_start,
                audio_start + audio_count,
            ),
            "references %d"
            % len(summary.get("reference_notes", [])),
            "video     %s" % summary["output_path"],
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
            "ffmpeg    %s"
            % _describe_ffmpeg(
                ffmpeg_location.strip() or None
            ),
        ]
        if alignment_note:
            report_lines.append("align     %s" % alignment_note)
        report_lines.extend(
            "ref       %s" % note
            for note in summary.get("reference_notes", [])
        )
        report = "\n".join(report_lines)

        preview = None
        if save_frames:
            preview = _preview(os.path.join(root, "frames"))
        if preview is None and summary["output_path"]:
            preview = _preview_from_video(
                summary["output_path"],
                ffmpeg_location=ffmpeg_location.strip() or None,
            )
        if preview is None:
            preview = torch.zeros(1, 64, 64, 3)

        return io.NodeOutput(
            _video_from_path(summary["output_path"]),
            preview,
            root,
            summary["output_path"],
            report,
        )


class MiniMaxH3LongFormReferencePreviewExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormReferenceVideoPreview]


__all__ = [
    "MiniMaxH3LongFormReferenceVideoPreview",
    "MiniMaxH3LongFormReferencePreviewExtension",
]
