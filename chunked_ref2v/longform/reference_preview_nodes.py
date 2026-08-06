"""LongFormReferenceVideo variant with generated audio and dual live previews."""

from __future__ import annotations

import logging
import os
from dataclasses import is_dataclass, replace

import torch
from comfy_api.latest import ComfyExtension, io

from ..geometry import HarnessGeometry
from . import audio_runtime, runner
from .chunk_stream import chunk_count_for
from .nodes import (
    _describe_ffmpeg,
    _preview,
    _preview_from_video,
    _video_result,
)
from .preview import (
    LongFormPreviewPublisher,
    PreviewOptions,
    activate,
    deactivate,
)
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
                "current_chunk_preview",
                default=True,
                tooltip=(
                    "Show a lightweight approximation of the current "
                    "denoised chunk while it is sampling. Connect "
                    "preview_vae only for an exact bounded VAE animation."
                ),
            ),
            io.Int.Input(
                "preview_every_steps",
                default=2,
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
                default=17,
                min=1,
                max=17,
                step=1,
                tooltip=(
                    "Maximum decoded frames when preview_vae is connected. "
                    "The default lightweight mode shows up to five temporal "
                    "latent positions instead."
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
                    "Optional VAE used only for an exact every-N-step "
                    "animation. Leave disconnected for the lightweight "
                    "latent preview. The production video VAE is never "
                    "loaded from inside the active sampler callback."
                ),
            ),
        ]

        insert_at = next(
            (
                index
                for index, item in enumerate(inputs)
                if getattr(item, "name", None) == "ref_images"
            ),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_inputs
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
        carry=runner.CARRY_OVERLAP,
        ref_image_size="native",
        cond_cache="auto",
        attention="auto",
        activation="mlp_chunked_native",
        run_directory="",
        ffmpeg_location="",
        output_video=True,
        save_frames=False,
        current_chunk_preview=True,
        preview_every_steps=2,
        current_preview_frames=17,
        completed_chunks_preview=True,
        live_preview_width=512,
        preview_vae=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        unique_id=None,
    ) -> io.NodeOutput:
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
            ),
        )
        token = activate(publisher)
        publisher._announce("reset")
        logging.info(
            "%s enabled for node %s: current=%s every=%d frames=%d; "
            "completed=%s width=%d exact_vae=%s",
            LOG,
            unique_id,
            current_chunk_preview,
            preview_every_steps,
            current_preview_frames,
            completed_chunks_preview,
            live_preview_width,
            preview_vae is not None,
        )
        try:
            summary = audio_runtime.run(
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
                output_video=output_video,
                ffmpeg_location=ffmpeg_location.strip() or None,
                runtime_config={
                    "attention": attention,
                    "activation": activation,
                    "audio_carry_frames": overlap_frames,
                    "audio_output": bool(output_video),
                },
            )
        finally:
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
            "audio     %s"
            % (
                (
                    "%d samples at %d Hz; O=%d -> latent [%d:%d]"
                    % (
                        summary["audio_samples"],
                        summary["audio_sample_rate"],
                        overlap_frames,
                        audio_start,
                        audio_start + audio_count,
                    )
                )
                if output_video
                else "disabled with output_video"
            ),
            "references %d"
            % len(summary.get("reference_notes", [])),
            "video     %s"
            % (summary["output_path"] or "disabled"),
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
            "ffmpeg    %s"
            % _describe_ffmpeg(
                ffmpeg_location.strip() or None
            ),
        ]
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

        video, video_ui = _video_result(summary["output_path"])
        return io.NodeOutput(
            preview,
            root,
            summary["output_path"],
            report,
            video,
            ui=video_ui,
        )


class MiniMaxH3LongFormReferencePreviewExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormReferenceVideoPreview]


__all__ = [
    "MiniMaxH3LongFormReferenceVideoPreview",
    "MiniMaxH3LongFormReferencePreviewExtension",
]
