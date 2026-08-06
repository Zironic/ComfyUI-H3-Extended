"""Long-form node variant with two independent live preview panes."""

from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace

from comfy_api.latest import ComfyExtension, io

from .nodes import MiniMaxH3LongFormRef2V
from .preview import (
    LongFormPreviewPublisher,
    PreviewOptions,
    activate,
    deactivate,
)

LOG = "[H3 Extended] longform preview"


def _replace_inputs(schema, inputs):
    if is_dataclass(schema):
        return replace(schema, inputs=inputs)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"inputs": inputs})
    schema.inputs = inputs
    return schema


class MiniMaxH3LongFormRef2VPreview(MiniMaxH3LongFormRef2V):
    """The production long-form node with current and completed live previews."""

    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        inputs = list(schema.inputs)
        preview_inputs = [
            io.Boolean.Input(
                "current_chunk_preview", default=True,
                tooltip=("Decode a bounded 17-frame view of the current denoised "
                         "chunk while it is sampling.")),
            io.Int.Input(
                "preview_every_steps", default=2, min=1, max=100, step=1,
                tooltip=("Publish the current-chunk preview every this many "
                         "sampler steps. The final step is always published.")),
            io.Int.Input(
                "current_preview_frames", default=17, min=1, max=17, step=1,
                tooltip=("Frames retained from the bounded five-latent H3 preview "
                         "decode. Five H3 temporal latents decode to 17 frames.")),
            io.Boolean.Input(
                "completed_chunks_preview", default=True,
                tooltip=("Publish every overlap-trimmed completed chunk as a "
                         "small finalized MP4 segment. The second preview pane "
                         "plays all available segments as a playlist.")),
            io.Int.Input(
                "live_preview_width", default=512, min=128, max=1024, step=32,
                tooltip="Maximum width of both live preview streams."),
            io.Vae.Input(
                "preview_vae", optional=True,
                tooltip=("Optional VAE used only for the every-N-step preview. "
                         "Leave disconnected to reuse video_vae. A smaller preview "
                         "VAE can reduce model switching and memory pressure.")),
        ]

        # Keep the dynamic reference-image sockets at the bottom of the node.
        insert_at = next(
            (i for i, item in enumerate(inputs)
             if getattr(item, "name", None) == "ref_images"),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_inputs
        return _replace_inputs(schema, inputs)

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, video_path, prompt,
                sampler, sigmas, seed, start_frame=0, output_seconds=180,
                chunk_frames=90, overlap_frames=4,
                carry="direct_latent_overlap", ref_image_size="native",
                cond_cache="auto", attention="auto",
                activation="mlp_chunked_native", canvas_mode="native",
                megapixels=1.0, width=0, height=0, run_directory="",
                ffmpeg_location="", output_video=True,
                preserve_source_audio=True, save_frames=False,
                current_chunk_preview=True, preview_every_steps=2,
                current_preview_frames=17, completed_chunks_preview=True,
                live_preview_width=512, preview_vae=None, ref_images=None,
                unique_id=None) -> io.NodeOutput:
        publisher = LongFormPreviewPublisher(
            node_id=unique_id,
            video_vae=preview_vae or video_vae,
            root=run_directory.strip() or "pending automatic run directory",
            fps=24,
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
            "completed=%s width=%d",
            LOG, unique_id, current_chunk_preview, preview_every_steps,
            current_preview_frames, completed_chunks_preview,
            live_preview_width)
        try:
            return super().execute(
                model=model, clip=clip, video_vae=video_vae,
                audio_vae=audio_vae, video_path=video_path, prompt=prompt,
                sampler=sampler, sigmas=sigmas, seed=seed,
                start_frame=start_frame, output_seconds=output_seconds,
                chunk_frames=chunk_frames, overlap_frames=overlap_frames,
                carry=carry, ref_image_size=ref_image_size,
                cond_cache=cond_cache, attention=attention,
                activation=activation, canvas_mode=canvas_mode,
                megapixels=megapixels, width=width, height=height,
                run_directory=run_directory, ffmpeg_location=ffmpeg_location,
                output_video=output_video,
                preserve_source_audio=preserve_source_audio,
                save_frames=save_frames, ref_images=ref_images)
        finally:
            deactivate(token)


class MiniMaxH3LongFormPreviewExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormRef2VPreview]


__all__ = [
    "MiniMaxH3LongFormRef2VPreview",
    "MiniMaxH3LongFormPreviewExtension",
]
