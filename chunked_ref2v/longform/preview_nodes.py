"""Long-form node variant with two independent live preview panes."""

from __future__ import annotations

import logging
from dataclasses import is_dataclass, replace

from comfy_api.latest import ComfyExtension, io

from .nodes import MiniMaxH3LongFormRef2V
from .preview import (
    CURRENT_FRAMES_TOOLTIP,
    DECODER_AUTO,
    LongFormPreviewPublisher,
    PreviewOptions,
    activate,
    deactivate,
    decoder_input,
    resolve_unique_id,
)

LOG = "[H3 Extended] longform preview"


#: Kept as a module-local alias: the definition moved to ``preview`` so every
#: node with a current-chunk pane can share it without importing this module.
_decoder_input = decoder_input


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
                    "small finalized MP4 segment. The second preview pane "
                    "plays all available segments as a playlist."
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

        # Keep the dynamic reference-image sockets at the bottom of the node.
        insert_at = next(
            (
                i
                for i, item in enumerate(inputs)
                if getattr(item, "name", None) == "ref_images"
            ),
            len(inputs),
        )
        inputs[insert_at:insert_at] = preview_inputs
        # Appended, never inserted: Comfy maps widgets_values positionally, so a
        # new widget is only safe at the very end where it cannot shift the
        # indices a saved workflow already stores.
        inputs.append(_decoder_input())
        return _replace_inputs(schema, inputs)

    @classmethod
    def execute(
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        video_path,
        prompt,
        sampler,
        sigmas,
        seed,
        start_frame=0,
        output_seconds=180,
        chunk_frames=90,
        overlap_frames=4,
        carry="direct_latent_overlap",
        ref_image_size="native",
        cond_cache="auto",
        attention="auto",
        activation="mlp_chunked_native",
        canvas_mode="native",
        megapixels=1.0,
        width=0,
        height=0,
        run_directory="",
        ffmpeg_location="",
        output_video=True,
        preserve_source_audio=True,
        save_frames=False,
        current_chunk_preview=True,
        preview_every_steps=2,
        current_preview_frames=0,
        completed_chunks_preview=True,
        live_preview_width=512,
        preview_vae=None,
        ref_images=None,
        current_preview_decoder=DECODER_AUTO,
        unique_id=None,
    ) -> io.NodeOutput:
        # See resolve_unique_id: hidden inputs reach V3 nodes through the class
        # clone, not through execute() kwargs.
        unique_id = resolve_unique_id(cls, unique_id)
        publisher = LongFormPreviewPublisher(
            node_id=unique_id,
            model=model,
            video_vae=preview_vae,
            root=run_directory.strip() or "pending automatic run directory",
            fps=24,
            ffmpeg_location=ffmpeg_location.strip() or None,
            options=PreviewOptions(
                current_enabled=bool(current_chunk_preview),
                completed_enabled=bool(completed_chunks_preview),
                every_steps=int(preview_every_steps),
                current_frames=int(current_preview_frames),
                width=int(live_preview_width),
                decoder=str(current_preview_decoder or DECODER_AUTO),
            ),
        )
        token = activate(publisher)
        publisher._announce("reset")
        logging.info(
            "%s enabled for node %s: current=%s every=%d frames=%s decoder=%s "
            "taeh3=%s; completed=%s width=%d exact_vae=%s",
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
        )
        try:
            return super().execute(
                model=model,
                clip=clip,
                video_vae=video_vae,
                audio_vae=audio_vae,
                video_path=video_path,
                prompt=prompt,
                sampler=sampler,
                sigmas=sigmas,
                seed=seed,
                start_frame=start_frame,
                output_seconds=output_seconds,
                chunk_frames=chunk_frames,
                overlap_frames=overlap_frames,
                carry=carry,
                ref_image_size=ref_image_size,
                cond_cache=cond_cache,
                attention=attention,
                activation=activation,
                canvas_mode=canvas_mode,
                megapixels=megapixels,
                width=width,
                height=height,
                run_directory=run_directory,
                ffmpeg_location=ffmpeg_location,
                output_video=output_video,
                preserve_source_audio=preserve_source_audio,
                save_frames=save_frames,
                ref_images=ref_images,
            )
        finally:
            deactivate(token)


class MiniMaxH3LongFormPreviewExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormRef2VPreview]


__all__ = [
    "MiniMaxH3LongFormRef2VPreview",
    "MiniMaxH3LongFormPreviewExtension",
]
