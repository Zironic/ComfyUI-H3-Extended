"""Experimental native AV-continuation long-form generation for MiniMax H3.

Every continuation chunk starts from fresh target latents. The immediately
preceding generated audiovisual chunk is fed back as a dynamic
<Video N+1>/<Audio M+1> reference pair:

* Qwen sees decoded previous-chunk video frames plus one audio reference item.
* The DiT reuses the already-generated video/audio latents directly as the
  matching reference block, avoiding a redundant VAE encode.

Every generated chunk contributes all of its frames to the final output. There
is no overlap window or target-latent transplantation. An optional N+1 prompt
plan supplies one user-authored base instruction per generated chunk; the
runtime continuation relationship is injected only after the previous chunk
exists.
"""

from __future__ import annotations

import gc
import logging
import math
import os
import time

import nodes
import torch
from comfy_api.latest import ComfyExtension, InputImpl, io

from .. import harness, memory, ref_builder
from ..audio_boundary_profile import (
    legal_reference_tail_frames,
    validate_av_continuation_chunk_frames,
)
from ..geometry import AUDIO_LATENT_FPS, HarnessGeometry
try:
    from ...cond_cache import MODES as COND_CACHE_MODES
except ImportError:  # the self-tests import this file as a top-level module
    from cond_cache import MODES as COND_CACHE_MODES
from .audio_output import (
    FFmpegAudioWriter,
    audio_sample_rate,
    audio_samples_for_frames,
    decode_audio_chunk,
    mux_generated_audio,
)
from . import diagnostics
from . import audio_runtime
from .nplusone_chunk_prompt_timeline import (
    NPlusOneChunkPromptPlan,
    prompts_for_av_continuation_plan,
)
from .preview import (
    CURRENT_FRAMES_TOOLTIP,
    DECODER_AUTO,
    LongFormPreviewPublisher,
    PreviewOptions,
    activate,
    deactivate,
    current_publisher,
    decoder_input,
    resolve_unique_id,
)
from .reference_nodes import _validate_canvas
from .reference_runner import _ordered_values, _paired_audio
from .runner import decode_chunk
from .writer import FFmpegVideoWriter, close_writers

LOG = "[H3 Extended] longform AV continuation"
REFERENCE_SECONDS_MIN = 2
REFERENCE_SECONDS_MAX = 15
REFERENCE_FRAMES_MIN = 24 * REFERENCE_SECONDS_MIN
REFERENCE_FRAMES_MAX = 24 * REFERENCE_SECONDS_MAX


def _chunk_count(target_frames, chunk_frames):
    target_frames = int(target_frames)
    chunk_frames = int(chunk_frames)
    if target_frames <= 0 or chunk_frames <= 0:
        raise ValueError("target_frames and chunk_frames must be positive")
    return int(math.ceil(target_frames / float(chunk_frames)))


def _nearest(value, candidates, *, prefer_larger_on_tie=True):
    if not candidates:
        raise ValueError("no legal reference-tail candidate")
    value = int(value)
    if prefer_larger_on_tie:
        return min(candidates, key=lambda item: (abs(item - value), -item))
    return min(candidates, key=lambda item: (abs(item - value), item))


def _resolve_reference_frames(chunk_frames, reference_frames):
    validate_av_continuation_chunk_frames(chunk_frames)
    candidates = legal_reference_tail_frames(chunk_frames)
    bounded_candidates = [
        value
        for value in candidates
        if REFERENCE_FRAMES_MIN <= value <= REFERENCE_FRAMES_MAX
    ]
    return _nearest(
        reference_frames,
        bounded_candidates or candidates,
        prefer_larger_on_tie=True,
    )


def _count_items(items, kind):
    return sum(
        1
        for item in items
        if isinstance(item, dict) and item.get("type") == kind
    )


def continuation_prompt(prompt, video_number, audio_number):
    """Describe the dynamic previous chunk using native continuation semantics."""

    video_number = int(video_number)
    audio_number = int(audio_number)
    if video_number <= 0 or audio_number <= 0:
        raise ValueError("reference numbers must be positive")
    video = "<Video %d>" % video_number
    audio = "<Audio %d>" % audio_number
    instruction = (
        "Reference relationship for this generation: [video continuation + audio reference]. "
        "%s is the immediately preceding generated video segment of the same continuous "
        "target video, and the target video begins immediately after its end. %s is the "
        "synchronized audio of %s and provides the immediately preceding audio state. "
        "Continue both picture and sound directly forward without restarting, repeating, "
        "or replaying the referenced segment. Preserve the final subject states, scene, "
        "camera state, motion direction and phase, lighting, and audiovisual continuity "
        "established at the end of %s."
        % (video, audio, video, video)
    )
    body = str(prompt or "").strip()
    return instruction if not body else instruction + "\n\n" + body


def _encode_static_references(
    *,
    video_vae,
    audio_vae,
    canvas,
    ref_images,
    ref_videos,
    ref_video_audios,
    ref_audios,
    ref_image_size,
    cond_cache,
):
    items, blocks, notes = [], [], []

    for name, image in _ordered_values(ref_images):
        item, block, note = ref_builder.encode_image_ref(
            video_vae, image, canvas, ref_image_size, cond_cache=cond_cache,
        )
        items.append(item)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, frames in _ordered_values(ref_videos):
        audio = _paired_audio(ref_video_audios, name)
        video_items, block, note = ref_builder.encode_video_ref(
            video_vae,
            frames,
            canvas,
            audio=audio,
            audio_vae=audio_vae,
            cond_cache=cond_cache,
        )
        items.extend(video_items)
        blocks.append(block)
        notes.append("%s: %s" % (name, note))

    for name, audio in _ordered_values(ref_audios):
        audio_latent, ref_audio_t = ref_builder.encode_ref_audio(
            audio_vae, audio, cond_cache=cond_cache,
        )
        items.append({"type": "audio"})
        blocks.append({
            "kind": "audio",
            "ref_audio_t": int(ref_audio_t),
            "audio_latent": audio_latent,
        })
        notes.append("%s: audio latent t=%d" % (name, ref_audio_t))

    return items, blocks, notes


def _dynamic_av_reference(previous_pixels, previous_video, previous_audio, canvas):
    """Return Qwen items + DiT block for the preceding generated AV segment."""

    if previous_pixels is None or previous_video is None or previous_audio is None:
        raise ValueError("dynamic continuation reference is incomplete")

    items = [
        {"type": "audio"},
        ref_builder.qwen_video_item(previous_pixels),
    ]
    block = {
        "kind": "video_audio",
        "latent_t": int(previous_video.shape[2]),
        "latent_h": int(canvas[1] // 16),
        "latent_w": int(canvas[0] // 16),
        "ref_audio_t": int(previous_audio.shape[-1]),
        "latent": previous_video,
        "audio_latent": previous_audio,
    }
    return items, block


def _slice_dynamic_av_reference(
    previous_pixels,
    previous_video,
    previous_audio,
    *,
    reference_frames,
    geometry,
):
    reference_frames = int(reference_frames)
    if reference_frames <= 0:
        raise ValueError("reference_frames must be positive")
    if previous_pixels is None or previous_audio is None or previous_video is None:
        raise ValueError("dynamic continuation reference is incomplete")
    if reference_frames > int(previous_pixels.shape[0]):
        raise ValueError(
            "reference_frames=%d exceeds previous decoded chunk length=%d"
            % (reference_frames, int(previous_pixels.shape[0]))
        )
    if int(reference_frames) == int(previous_pixels.shape[0]):
        return (
            previous_pixels,
            previous_video,
            previous_audio,
        )
    if geometry.overlap_frames != int(reference_frames):
        raise ValueError(
            "reference frames mismatch: geometry overlap=%d reference=%d"
            % (geometry.overlap_frames, int(reference_frames))
        )
    video_overlap_start, video_overlap_count = geometry.overlap_slice()
    audio_overlap_start, audio_overlap_count = audio_runtime.audio_overlap_slice(geometry)
    expected_audio_latents = (
        int(reference_frames) * AUDIO_LATENT_FPS // geometry.fps
    )
    if int(reference_frames) * AUDIO_LATENT_FPS % geometry.fps:
        raise RuntimeError(
            "reference duration is not audio exact: R=%d fps=%d latent_fps=%d"
            % (reference_frames, geometry.fps, AUDIO_LATENT_FPS)
        )
    if audio_overlap_count != expected_audio_latents:
        raise RuntimeError(
            "reference audio latent mismatch: expected %d from %d frames at %d fps, "
            "but geometry overlap gives %d"
            % (
                expected_audio_latents,
                int(reference_frames),
                geometry.fps,
                audio_overlap_count,
            )
        )
    return (
        previous_pixels[-int(reference_frames):],
        previous_video[
            :,
            :,
            video_overlap_start:video_overlap_start + video_overlap_count,
        ],
        previous_audio[
            ...,
            audio_overlap_start:audio_overlap_start + audio_overlap_count,
        ],
    )


def _resolve_root(run_directory, chunk_frames):
    if str(run_directory or "").strip():
        root = str(run_directory).strip()
        if os.path.exists(root) and os.listdir(root):
            raise RuntimeError(
                "LongFormAVContinuation requires a fresh run directory; "
                "interrupted executions are not resumable"
            )
        return root

    import folder_paths

    return os.path.join(
        folder_paths.get_output_directory(),
        "h3_longform_av_continuation",
        "%s_c%d" % (time.strftime("%Y%m%d_%H%M%S"), int(chunk_frames)),
    )


class MiniMaxH3LongFormAVContinuation(io.ComfyNode):
    """Long-form generation using H3's native audiovisual continuation references."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongFormAVContinuationZi",
            display_name="MiniMax H3 LongForm AV Continuation (Zi)",
            category="model/video/minimax/testing",
            is_experimental=True,
            description=(
                "Generates long video as sequential native H3 audiovisual continuations. "
                "Each chunk after the first receives the configured tail of the "
                "preceding generated chunk as dynamic <Video N+1> and <Audio M+1> "
                "references. Connect the "
                "N+1 prompt timeline for per-chunk action instructions."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                NPlusOneChunkPromptPlan.Input(
                    "n_plus_one_prompt_plan",
                    optional=True,
                    tooltip=(
                        "Optional plan from MiniMax H3 N+1 Chunk Prompt Timeline (Zi). "
                        "When connected, each generated chunk uses its matching base "
                        "instruction. The normal prompt is used only as a blank-entry "
                        "fallback; dynamic <Video N+1>/<Audio M+1> continuation text is "
                        "added by this node at runtime."
                    ),
                ),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input(
                    "output_seconds",
                    default=30,
                    min=1,
                    max=3600,
                    tooltip="Exact requested output duration at H3's fixed 24 fps.",
                ),
                io.Int.Input(
                    "width",
                    default=1344,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "height",
                    default=768,
                    min=32,
                    max=nodes.MAX_RESOLUTION,
                    step=32,
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=141,
                    min=22,
                    max=362,
                    step=17,
                    tooltip=(
                        "Generated frames per continuation invocation. Every chunk "
                        "contributes all of its frames; there is no overlap trimming."
                    ),
                ),
                io.Combo.Input(
                    "ref_image_size",
                    options=["native", "match", "max"],
                    default="native",
                ),
                io.Combo.Input(
                    "cond_cache",
                    options=list(COND_CACHE_MODES),
                    default="auto",
                ),
                io.Combo.Input(
                    "attention",
                    options=list(memory.ATTENTION_MODES),
                    default="auto",
                ),
                io.Combo.Input(
                    "activation",
                    options=list(memory.ACTIVATION_MODES),
                    default="mlp_chunked_native",
                ),
                io.String.Input(
                    "run_directory",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Leave blank to create a new output/h3_longform_av_continuation "
                        "directory. Existing non-empty directories are rejected."
                    ),
                ),
                io.String.Input(
                    "ffmpeg_location",
                    default="",
                    multiline=False,
                    tooltip="Leave blank to use ffmpeg on PATH or the bundled binary.",
                ),
                io.Boolean.Input(
                    "save_frames",
                    default=False,
                    tooltip="Save committed output frames as diagnostic PNGs.",
                ),
                io.Boolean.Input("current_chunk_preview", default=True),
                io.Int.Input(
                    "preview_every_steps",
                    default=2,
                    min=1,
                    max=100,
                    step=1,
                ),
                io.Int.Input(
                    "current_preview_frames",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=CURRENT_FRAMES_TOOLTIP,
                ),
                io.Boolean.Input("completed_chunks_preview", default=True),
                io.Int.Input(
                    "live_preview_width",
                    default=512,
                    min=128,
                    max=1024,
                    step=32,
                ),
                io.Vae.Input(
                    "preview_vae",
                    optional=True,
                    tooltip=(
                        "Optional exact-VAE preview, used only when TAEH3 is "
                        "unavailable or its decode fails, or when "
                        "current_preview_decoder is set to preview_vae. TAEH3 "
                        "is the default backend and needs nothing connected "
                        "here. The production video VAE is never loaded from "
                        "inside the active sampler callback."
                    ),
                ),
                io.Autogrow.Input(
                    "ref_images",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image"),
                        prefix="ref_image_",
                        min=0,
                        max=9,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_videos",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"),
                        prefix="ref_video_",
                        min=0,
                        max=2,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_video_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_",
                        min=0,
                        max=2,
                    ),
                ),
                io.Autogrow.Input(
                    "ref_audios",
                    optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"),
                        prefix="ref_audio_",
                        min=0,
                        max=3,
                    ),
                ),
                # Appended last on purpose. Comfy maps widgets_values by
                # position, so inserting anywhere earlier shifts every saved
                # value after it in existing workflows.
                io.Boolean.Input(
                    "diagnostic_dump_chunks",
                    default=False,
                    tooltip=(
                        "Dump every complete generated chunk under "
                        "diagnostics/chunk_NNNNNN/ - all decoded video frames "
                        "plus the full generated audio, numbered locally to the "
                        "chunk. This runner commits every frame it generates, so "
                        "the dump differs from save_frames only on the final "
                        "chunk, whose tail is truncated to the requested "
                        "duration. Nothing is stored to replay from, so this "
                        "captures only while the run is generating."
                    ),
                ),
                # Same rule as diagnostic_dump_chunks above: appended, never
                # inserted, so saved widgets_values keep their indices.
                decoder_input(),
                io.Int.Input(
                    "reference_frames",
                    default=90,
                    min=1,
                    max=362,
                    tooltip=(
                        "Tail frames from each generated chunk used as dynamic "
                        "<Video N+1>/<Audio M+1> reference."
                    ),
                ),
            ],
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
        cls,
        model,
        clip,
        video_vae,
        audio_vae,
        prompt,
        sampler,
        sigmas,
        seed,
        n_plus_one_prompt_plan=None,
        output_seconds=30,
        width=1344,
        height=768,
        chunk_frames=141,
        ref_image_size="native",
        cond_cache="auto",
        attention="auto",
        activation="mlp_chunked_native",
        run_directory="",
        ffmpeg_location="",
        save_frames=False,
        current_chunk_preview=True,
        preview_every_steps=2,
        current_preview_frames=0,
        completed_chunks_preview=True,
        live_preview_width=512,
        preview_vae=None,
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
        diagnostic_dump_chunks=False,
        current_preview_decoder=DECODER_AUTO,
        reference_frames=90,
        unique_id=None,
    ) -> io.NodeOutput:
        # H3 target construction only needs a legal chunk length. The geometry
        # helper requires an overlap value, but this node never consumes its
        # stride/overlap fields; O=4 is used only to obtain the target AV shape.
        geometry = HarnessGeometry(
            chunk_frames=int(chunk_frames),
            overlap_frames=4,
        ).validate()
        canvas = _validate_canvas(width, height)
        target_frames = int(output_seconds * geometry.fps)
        resolved_reference_frames = _resolve_reference_frames(
            chunk_frames=geometry.chunk_frames,
            reference_frames=reference_frames,
        )
        chunk_prompts = prompts_for_av_continuation_plan(
            n_plus_one_prompt_plan,
            prompt,
            output_seconds=output_seconds,
            chunk_frames=geometry.chunk_frames,
            reference_frames=resolved_reference_frames,
        )
        reference_geometry = (
            HarnessGeometry(
                chunk_frames=int(chunk_frames),
                overlap_frames=int(resolved_reference_frames),
            ).validate()
            if resolved_reference_frames < geometry.chunk_frames
            else None
        )
        chunk_count = _chunk_count(target_frames, geometry.chunk_frames)
        if len(chunk_prompts) != chunk_count:
            raise RuntimeError(
                "N+1 timeline resolved %d prompts for %d continuation chunks"
                % (len(chunk_prompts), chunk_count)
            )

        root = _resolve_root(run_directory, geometry.chunk_frames)
        for path in ("frames", "output"):
            os.makedirs(os.path.join(root, path), exist_ok=True)

        # geometry carries a placeholder O=4 that this runner never consumes, so
        # the overlap-shaped metadata fields are corrected here rather than
        # reported as a carry that does not exist.
        diagnostic_sink = diagnostics.DiagnosticSink(
            root,
            geometry,
            carry="none",
            enabled=diagnostic_dump_chunks,
            metadata_overrides={
                "mode": "av_continuation",
                "overlap_frames": 0,
                "stride_frames": int(geometry.chunk_frames),
                "carry_local_frames": None,
                "previous_chunk_overlap_local_frames": None,
                "video_overlap_latent": None,
            },
        )

        model, memory_status = memory.arm(
            model,
            attention=attention,
            activation=activation,
        )
        logging.info("%s %s", LOG, memory.describe(memory_status))

        static_video_count = sum(1 for _ in _ordered_values(ref_videos))
        if static_video_count > 2:
            raise ValueError(
                "AV continuation reserves one video-reference slot for the previous "
                "generated chunk; connect at most two static reference videos"
            )

        ffmpeg = ffmpeg_location.strip() or None
        raw_video = os.path.join(root, "output", "video_only.mkv")
        raw_audio = os.path.join(root, "output", "generated_audio.wav")
        final_video = os.path.join(root, "output", "final.mp4")
        out_dir = os.path.join(root, "frames")

        unique_id = resolve_unique_id(cls, unique_id)
        publisher = LongFormPreviewPublisher(
            node_id=unique_id,
            model=model,
            video_vae=preview_vae,
            root=root,
            fps=geometry.fps,
            ffmpeg_location=ffmpeg,
            options=PreviewOptions(
                current_enabled=bool(current_chunk_preview),
                completed_enabled=bool(completed_chunks_preview),
                every_steps=int(preview_every_steps),
                current_frames=int(current_preview_frames),
                width=int(live_preview_width),
                decoder=str(current_preview_decoder or DECODER_AUTO),
            ),
            # This runner always decodes a waveform per chunk, so the completed
            # pane must wait for it instead of publishing a silent segment.
            audio_expected=True,
        )
        preview_token = activate(publisher)
        publisher._announce("reset")
        logging.info(
            "%s preview: current=%s every=%d frames=%s decoder=%s taeh3=%s; "
            "completed=%s width=%d exact_vae=%s",
            LOG,
            current_chunk_preview,
            preview_every_steps,
            current_preview_frames or "all",
            current_preview_decoder,
            publisher.taeh3 is not None,
            completed_chunks_preview,
            live_preview_width,
            preview_vae is not None,
        )

        video_writer = FFmpegVideoWriter(
            raw_video,
            width=canvas[0],
            height=canvas[1],
            fps=geometry.fps,
            ffmpeg_location=ffmpeg,
        ).open()
        audio_writer = None

        written_frames = 0
        written_audio = 0
        sample_rate = audio_sample_rate(audio_vae)
        previous_video = previous_audio = previous_pixels = None
        reference_notes = []

        # Raw artifacts are committed only once every chunk has been written
        # and both totals check out; a run that dies mid-chunk leaves partials
        # that close_writers removes instead of a truncated final file.
        completed = False
        try:
            with torch.inference_mode():
                static_items, static_blocks, reference_notes = _encode_static_references(
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                    canvas=canvas,
                    ref_images=ref_images,
                    ref_videos=ref_videos,
                    ref_video_audios=ref_video_audios,
                    ref_audios=ref_audios,
                    ref_image_size=ref_image_size,
                    cond_cache=cond_cache,
                )
                dynamic_video_number = _count_items(static_items, "video") + 1
                dynamic_audio_number = _count_items(static_items, "audio") + 1

                for index, base_prompt in enumerate(chunk_prompts):
                    items = list(static_items)
                    blocks = list(static_blocks)
                    active_prompt = str(base_prompt or "")

                    if index > 0:
                        reference_pixels, reference_video, reference_audio = (
                            _slice_dynamic_av_reference(
                                previous_pixels,
                                previous_video,
                                previous_audio,
                                reference_frames=resolved_reference_frames,
                                geometry=reference_geometry,
                            )
                        )
                        dynamic_items, dynamic_block = _dynamic_av_reference(
                            reference_pixels,
                            reference_video,
                            reference_audio,
                            canvas,
                        )
                        items.extend(dynamic_items)
                        blocks.append(dynamic_block)
                        active_prompt = continuation_prompt(
                            active_prompt,
                            dynamic_video_number,
                            dynamic_audio_number,
                        )

                    conditioning = harness._encode(
                        clip,
                        active_prompt,
                        items,
                        cond_cache,
                    )
                    conditioning = harness.attach_refs(conditioning, blocks)
                    latent = harness.empty_av_latent(canvas, geometry)
                    active_publisher = current_publisher()
                    callback = (
                        active_publisher.sampler_callback(index)
                        if active_publisher is not None
                        else None
                    )

                    sample_kwargs = {
                        "model": model,
                        "conditioning": conditioning,
                        "latent": latent,
                        "sampler": sampler,
                        "sigmas": sigmas,
                        "seed": harness.splitmix64(int(seed) + 1000 + index),
                    }
                    if callback is not None:
                        sample_kwargs["callback"] = callback
                    video_latent, audio_latent = harness.sample(**sample_kwargs)
                    video_latent = video_latent.to("cpu", torch.float32)
                    audio_latent = audio_latent.to("cpu", torch.float32)

                    pixels = decode_chunk(video_vae, video_latent).to(
                        "cpu", torch.float32
                    )
                    # Before the final-chunk truncation below.
                    diagnostics.emit_video(diagnostic_sink, index, pixels)
                    remaining = target_frames - written_frames
                    take_frames = min(int(pixels.shape[0]), remaining)
                    if take_frames <= 0:
                        break
                    frames_u8 = (
                        pixels[:take_frames].clamp(0, 1) * 255.0 + 0.5
                    ).to(torch.uint8)
                    video_writer.write(frames_u8)

                    if save_frames:
                        from PIL import Image

                        for local_index in range(take_frames):
                            Image.fromarray(frames_u8[local_index].numpy()).save(
                                os.path.join(
                                    out_dir,
                                    "frame_%06d.png"
                                    % (written_frames + local_index),
                                )
                            )

                    if active_publisher is not None:
                        try:
                            # Staged, not published: this chunk's waveform is
                            # only decoded further down, so publishing here
                            # could never produce a segment with sound. The
                            # flush after the audio is trimmed releases it.
                            active_publisher.stage_completed_chunk(
                                chunk_index=index,
                                frames_u8=frames_u8,
                                completed_frames=written_frames + take_frames,
                            )
                        except Exception as exc:
                            logging.warning(
                                "%s completed preview failed for chunk %d: %s",
                                LOG,
                                index,
                                exc,
                            )

                    waveform = decode_audio_chunk(audio_vae, audio_latent)

                    if diagnostic_dump_chunks:
                        dumped_samples = diagnostics.dump_chunk_audio(
                            diagnostic_sink, index, waveform, sample_rate,
                        )
                        diagnostics.dump_chunk_metadata(
                            diagnostic_sink,
                            index,
                            global_start_frame=written_frames,
                            committed_frames=take_frames,
                            video_frames=diagnostics.video_frames_dumped(
                                diagnostic_sink, index,
                            ),
                            video_latent=video_latent,
                            audio_latent=audio_latent,
                            audio_samples=dumped_samples,
                            audio_sample_rate=sample_rate,
                        )

                    channels = int(waveform.shape[1])
                    if audio_writer is None:
                        audio_writer = FFmpegAudioWriter(
                            raw_audio,
                            sample_rate=sample_rate,
                            channels=channels,
                            ffmpeg_location=ffmpeg,
                        ).open()
                    desired_audio = audio_samples_for_frames(
                        written_frames + take_frames,
                        sample_rate,
                        fps=geometry.fps,
                    )
                    take_audio = desired_audio - written_audio
                    if int(waveform.shape[-1]) < take_audio:
                        raise RuntimeError(
                            "chunk %d decoded %d audio samples; need %d"
                            % (index, int(waveform.shape[-1]), take_audio)
                        )
                    committed_audio = waveform[..., :take_audio]
                    audio_writer.write(committed_audio)
                    written_audio += take_audio

                    # Release the staged frames now that their sound exists.
                    # Hand over exactly the committed samples so the preview
                    # pane stays in step with the video it just wrote.
                    if active_publisher is not None:
                        try:
                            active_publisher.flush_completed_chunk(
                                waveform=committed_audio,
                                sample_rate=sample_rate,
                            )
                        except Exception as exc:
                            logging.warning(
                                "%s completed preview audio failed for chunk %d: %s",
                                LOG,
                                index,
                                exc,
                            )
                    written_frames += take_frames

                    # Keep exactly one complete previous chunk on CPU. Qwen uses
                    # decoded pixels; the DiT uses the generated AV latents directly.
                    previous_video = video_latent
                    previous_audio = audio_latent
                    previous_pixels = pixels

                    logging.info(
                        "%s chunk %d/%d complete; committed %d frames; dynamic ref=%s",
                        LOG,
                        index + 1,
                        chunk_count,
                        take_frames,
                        "none"
                        if index == 0
                        else "<Video %d>/<Audio %d>"
                        % (dynamic_video_number, dynamic_audio_number),
                    )

                    del conditioning, latent, waveform, frames_u8
                    gc.collect()

            if written_frames != target_frames:
                raise RuntimeError(
                    "assembled %d frames, expected exactly %d"
                    % (written_frames, target_frames)
                )
            expected_audio = audio_samples_for_frames(
                target_frames,
                sample_rate,
                fps=geometry.fps,
            )
            if written_audio != expected_audio:
                raise RuntimeError(
                    "assembled %d audio samples, expected exactly %d"
                    % (written_audio, expected_audio)
                )
            completed = True
        finally:
            # A run that ended between staging a chunk and decoding its audio
            # would otherwise strand those frames; publish them silently rather
            # than lose the segment.
            try:
                publisher.flush_completed_chunk()
            except Exception as exc:
                logging.warning("%s final preview flush failed: %s", LOG, exc)
            deactivate(preview_token)
            close_writers(video_writer, audio_writer, commit=completed)

        output_path = mux_generated_audio(
            raw_video,
            raw_audio,
            final_video,
            frame_count=target_frames,
            fps=geometry.fps,
            ffmpeg_location=ffmpeg,
        )

        preview_image = (
            previous_pixels[-1:].clone()
            if previous_pixels is not None
            else torch.zeros(1, 64, 64, 3)
        )
        video = InputImpl.VideoFromFile(output_path)
        reference_note = (
            "complete previous chunk"
            if resolved_reference_frames == geometry.chunk_frames
            else "from previous generated tail"
        )
        reference_seconds = resolved_reference_frames / float(geometry.fps)
        report_lines = [
            "MiniMax H3 LongForm AV Continuation",
            "mode      native video + audio continuation; no latent overlap carry",
            "prompts   %s"
            % (
                "N+1 per-chunk timeline"
                if n_plus_one_prompt_plan is not None
                else "single prompt repeated per chunk"
            ),
            "canvas    %dx%d" % canvas,
            "chunk     %d frames; every chunk contributes all generated frames"
            % geometry.chunk_frames,
            "chunks    %d" % chunk_count,
            "output    %d frames (%.3f s at %d fps)"
            % (target_frames, target_frames / geometry.fps, geometry.fps),
            "reference R=%d frames (%.3f s) %s"
            % (
                resolved_reference_frames,
                reference_seconds,
                reference_note,
            ),
            "dynamic   <Video %d> + <Audio %d> from previous generated chunk"
            % (dynamic_video_number, dynamic_audio_number),
            "audio     %d samples at %d Hz" % (written_audio, sample_rate),
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
            "video     %s" % output_path,
        ]
        if diagnostic_dump_chunks:
            report_lines.append(
                "diag      untrimmed chunk dumps in %s"
                % os.path.join(root, "diagnostics")
            )
        report_lines.extend("ref       %s" % note for note in reference_notes)
        return io.NodeOutput(
            video,
            preview_image,
            root,
            output_path,
            "\n".join(report_lines),
        )


class MiniMaxH3LongFormAVContinuationExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormAVContinuation]


__all__ = [
    "MiniMaxH3LongFormAVContinuation",
    "MiniMaxH3LongFormAVContinuationExtension",
    "continuation_prompt",
]
