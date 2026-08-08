"""Comfy node for generic, bounded-memory long-form reference generation."""

import logging
import os
import time

import torch

import nodes
from comfy_api.latest import ComfyExtension, io

try:
    from .. import memory
    from ..geometry import HarnessGeometry
    from ...cond_cache import MODES as COND_CACHE_MODES
except ImportError:  # pragma: no cover
    import memory
    from geometry import HarnessGeometry
    COND_CACHE_MODES = ["auto", "off", "refresh"]

from . import reference_runner, runner
from .chunk_stream import chunk_count_for
from .nodes import (
    _describe_ffmpeg,
    _preview,
    _preview_from_video,
    _video_from_path,
)

LOG = "[H3 Extended] longform reference"
CANVAS_MULTIPLE = 32


class MiniMaxH3LongFormReferenceVideo(io.ComfyNode):
    """Generate an arbitrary-length clip from persistent H3 references."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongFormReferenceVideoZi",
            display_name="MiniMax H3 LongFormReferenceVideo (Zi)",
            category="model/video/minimax/testing",
            is_experimental=True,
            description=(
                "Generates an independently sized long clip by repeating the same "
                "prompt and reference payload for overlapping chunks. It does not "
                "treat any input as a source timeline, clamp output length to a "
                "reference, schedule shots, or rewrite prompts."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "output_seconds", default=30, min=1, max=3600,
                    tooltip=(
                        "Exact requested output duration at 24 fps. This is "
                        "authoritative and is never clamped to any reference length."
                    ),
                ),
                io.Int.Input(
                    "width", default=1344, min=32,
                    max=nodes.MAX_RESOLUTION, step=32,
                ),
                io.Int.Input(
                    "height", default=768, min=32,
                    max=nodes.MAX_RESOLUTION, step=32,
                ),
                io.Int.Input(
                    "chunk_frames", default=90, min=22, max=362, step=17,
                    tooltip="Per-sample model length; must satisfy the H3 17k+5 grid.",
                ),
                io.Int.Input(
                    "overlap_frames", default=4, min=4, max=180,
                    tooltip=(
                        "Generated overlap between consecutive chunks. Legal values "
                        "are validated against the latent grid; O=4 carries one "
                        "temporal latent position."
                    ),
                ),
                io.Combo.Input(
                    "carry", options=list(runner.CARRY_MODES),
                    default=runner.CARRY_OVERLAP,
                    tooltip=(
                        "direct_latent_frame carries one video latent; "
                        "direct_latent_overlap carries every video latent in "
                        "overlap_frames; none disables generated carry."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size", options=["native", "match", "max"],
                    default="native",
                ),
                io.Combo.Input(
                    "cond_cache", options=list(COND_CACHE_MODES), default="auto",
                ),
                io.Combo.Input(
                    "attention", options=list(memory.ATTENTION_MODES), default="auto",
                ),
                io.Combo.Input(
                    "activation", options=list(memory.ACTIVATION_MODES),
                    default="mlp_chunked_native",
                ),
                io.String.Input(
                    "run_directory", default="", multiline=False,
                    tooltip=(
                        "Leave blank to create an output/h3_longform_reference run. "
                        "Point at a matching existing run to resume from its first "
                        "incomplete chunk."
                    ),
                ),
                io.String.Input(
                    "ffmpeg_location", default="", multiline=False,
                    tooltip=(
                        "Leave blank to use ffmpeg on PATH or the bundled "
                        "imageio_ffmpeg binary."
                    ),
                ),
                io.Boolean.Input(
                    "save_frames", default=False,
                    tooltip="Save a diagnostic PNG sequence in addition to the video.",
                ),
                # Must stay the LAST widget. Comfy maps widgets_values by
                # position, so a new widget inserted anywhere earlier shifts
                # every saved value after it in existing workflows; appending
                # here leaves older workflows on the default.
                io.Boolean.Input(
                    "diagnostic_dump_chunks", default=False,
                    tooltip=(
                        "Dump every complete generated chunk before overlap "
                        "trimming - all decoded video frames plus the generated "
                        "audio - under diagnostics/chunk_NNNNNN/. Frame numbers "
                        "are local to the chunk, so frame 0 is the first carried "
                        "frame and frame overlap_frames is the first frame past "
                        "the carry. This does not change what is generated, so "
                        "it can be switched on against a finished run directory "
                        "to dump the stored chunks without resampling."
                    ),
                ),
                io.Autogrow.Input(
                    "ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_", min=0, max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input(
                            "ref_video",
                            tooltip="Reference video frames; not an output timeline.",
                        ),
                        prefix="ref_video_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input(
                            "ref_video_audio",
                            tooltip="Soundtrack paired with the same-numbered reference video.",
                        ),
                        prefix="ref_video_audio_", min=0, max=3,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_", min=0, max=3,
                    ),
                ),
            ],
            # Not an output node: the VIDEO goes to Save Video, which owns
            # final naming, container, codec, metadata, and the output preview.
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="run_directory"),
                io.String.Output(display_name="video_path"),
                io.String.Output(display_name="report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, model, clip, video_vae, audio_vae, prompt, sampler, sigmas, seed,
        output_seconds=30, width=1344, height=768, chunk_frames=90,
        overlap_frames=4, carry=runner.CARRY_OVERLAP,
        ref_image_size="native", cond_cache="auto", attention="auto",
        activation="mlp_chunked_native", run_directory="",
        ffmpeg_location="", save_frames=False,
        diagnostic_dump_chunks=False, ref_images=None, ref_videos=None, ref_video_audios=None,
        ref_audios=None,
    ) -> io.NodeOutput:
        geometry = HarnessGeometry(
            chunk_frames=chunk_frames, overlap_frames=overlap_frames,
        ).validate()
        canvas = _validate_canvas(width, height)
        target_frames = int(output_seconds * geometry.fps)
        chunk_count = chunk_count_for(
            target_frames, geometry.chunk_frames, geometry.stride_frames,
        )

        model, memory_status = memory.arm(
            model, attention=attention, activation=activation,
        )
        logging.info("%s %s", LOG, memory.describe(memory_status))

        root = _resolve_root(run_directory, carry, chunk_frames)
        summary = reference_runner.run(
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
                "audio_carry_policy": "video_floor_v1",
            },
        )

        report_lines = [
            "MiniMax H3 LongFormReferenceVideo",
            "profile   %s" % summary["profile"],
            "carry     %s" % carry,
            "canvas    %dx%d (%.2f MP, %.3f:1)" % (
                canvas[0], canvas[1], canvas[0] * canvas[1] / 1e6,
                canvas[0] / canvas[1],
            ),
            "chunks    %d" % chunk_count,
            "output    %d exact frames (%.3f s at %d fps)" % (
                summary["frames"], summary["frames"] / geometry.fps,
                geometry.fps,
            ),
            "references %d" % len(summary.get("reference_notes", [])),
            "audio     reference conditioning retained; stitched output is video-only",
            "video     %s" % summary["output_path"],
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
            "ffmpeg    %s" % _describe_ffmpeg(ffmpeg_location.strip() or None),
        ]
        report_lines.extend(
            "ref       %s" % note for note in summary.get("reference_notes", [])
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
            preview, root, summary["output_path"], report,
        )


def _validate_canvas(width, height):
    width = int(width)
    height = int(height)
    if width < CANVAS_MULTIPLE or height < CANVAS_MULTIPLE:
        raise ValueError("width and height must both be at least 32")
    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError("width and height must both be divisible by 32")
    return width, height


def _resolve_root(run_directory, carry, chunk_frames):
    if run_directory.strip():
        return run_directory.strip()
    import folder_paths
    return os.path.join(
        folder_paths.get_output_directory(),
        "h3_longform_reference",
        "%s_%s_c%d" % (
            time.strftime("%Y%m%d_%H%M%S"),
            carry.replace("direct_latent_", ""),
            chunk_frames,
        ),
    )


class MiniMaxH3LongFormReferenceExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormReferenceVideo]


async def comfy_entrypoint() -> MiniMaxH3LongFormReferenceExtension:
    return MiniMaxH3LongFormReferenceExtension()


__all__ = [
    "MiniMaxH3LongFormReferenceVideo",
    "MiniMaxH3LongFormReferenceExtension",
]
