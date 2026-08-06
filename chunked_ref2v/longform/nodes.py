"""Comfy node for disk-backed, bounded-memory long-form Ref2V."""

import logging
import os

import torch

import nodes
from comfy_api.latest import ComfyExtension, io

try:
    from .. import memory
    from ..geometry import HarnessGeometry
    from ..ref_builder import pin_canvas
    from ...cond_cache import MODES as COND_CACHE_MODES
except ImportError:  # pragma: no cover
    import memory
    from geometry import HarnessGeometry
    from ref_builder import pin_canvas
    COND_CACHE_MODES = ["auto", "off", "refresh"]

from . import runner
from .chunk_stream import chunk_count_for, frames_needed_for
from .frame_source import probe

LOG = "[H3 Extended] longform"


class MiniMaxH3LongFormRef2V(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongFormRef2VZi",
            display_name="MiniMax H3 Long-Form Ref2V (Zi)",
            category="model/video/minimax/testing",
            is_experimental=True,
            description=(
                "Streams a source video, persists per-chunk model artifacts, samples "
                "with latent carry, then decodes into an FFmpeg writer without ever "
                "materializing the full input or output as a Comfy IMAGE batch."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("video_path", multiline=False),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True),
                io.Int.Input("start_frame", default=0, min=0, max=10_000_000),
                io.Int.Input("output_seconds", default=180, min=1, max=3600,
                             tooltip=("Exact output duration at 24 fps. Clamped "
                                      "down to what the source can supply from "
                                      "start_frame - a run consumes the full span "
                                      "of its chunks, which overruns the kept "
                                      "frames by up to chunk_frames-1. The clamp "
                                      "is logged and reported. Only a source too "
                                      "short for a single chunk is an error.")),
                io.Int.Input("chunk_frames", default=90, min=22, max=362, step=17),
                io.Int.Input("overlap_frames", default=4, min=4, max=180,
                             tooltip=("Source overlap, and the only compute knob: "
                                      "asymptotic overhead is O/(C-O), so O=4 costs "
                                      "4.7% over monolithic and O=22 costs 32.4%. "
                                      "H3 packs 4 frames per latent, so O=4 carries "
                                      "exactly one latent position - measured "
                                      "sufficient for full coherence. Legal values "
                                      "are quantized by the 17-frame grid: 4, 5, 9, "
                                      "13, 17, 21, 22, 26...; anything else raises "
                                      "UnalignedProfileError.")),
                io.Combo.Input("carry", options=list(runner.CARRY_MODES),
                               default="direct_latent_overlap"),
                io.Combo.Input("ref_image_size", options=["native", "match", "max"], default="native"),
                io.Combo.Input("cond_cache", options=list(COND_CACHE_MODES), default="auto"),
                io.Combo.Input("attention", options=list(memory.ATTENTION_MODES), default="auto"),
                io.Combo.Input("activation", options=list(memory.ACTIVATION_MODES),
                               default="mlp_chunked_native"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32),
                io.String.Input("run_directory", default="", multiline=False,
                                tooltip=("Leave blank to create "
                                         "output/h3_longform/<timestamp>_<carry>_c<chunk_frames>. "
                                         "Point at an existing run to resume it - "
                                         "it continues from the first incomplete "
                                         "chunk, and only when the immutable "
                                         "manifest matches this configuration.")),
                io.String.Input("ffmpeg_location", default="", multiline=False,
                                tooltip=("Leave blank to auto-detect: ffmpeg on "
                                         "PATH, otherwise the imageio_ffmpeg "
                                         "binary bundled with this environment. "
                                         "Accepts either an executable or a "
                                         "directory containing one. The resolved "
                                         "path is echoed in the report output.")),
                io.Boolean.Input("output_video", default=True),
                io.Boolean.Input("preserve_source_audio", default=True),
                io.Boolean.Input("save_frames", default=False,
                                 tooltip="Diagnostic PNG sequence; normally leave disabled."),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image"),
                                      prefix="ref_image_", min=0, max=9)),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="run_directory"),
                io.String.Output(display_name="video_path"),
                io.String.Output(display_name="report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, video_path, prompt,
                sampler, sigmas, seed, start_frame=0, output_seconds=180,
                chunk_frames=90, overlap_frames=4, carry="direct_latent_overlap",
                ref_image_size="native", cond_cache="auto", attention="auto",
                activation="mlp_chunked_native", width=0, height=0,
                run_directory="", ffmpeg_location="", output_video=True,
                preserve_source_audio=True, save_frames=False,
                ref_images=None) -> io.NodeOutput:
        video_path = video_path.strip().strip('"')
        if not os.path.exists(video_path):
            raise ValueError("source video not found: %r" % video_path)

        geometry = HarnessGeometry(
            chunk_frames=chunk_frames, overlap_frames=overlap_frames
        ).validate()
        target_frames = int(output_seconds * geometry.fps)
        chunk_count = chunk_count_for(
            target_frames, geometry.chunk_frames, geometry.stride_frames
        )

        # A run consumes the full span of its chunks, not just the frames it
        # keeps: the last chunk always overhangs the target by up to C-1 frames.
        # Checking `target_frames` here let a source land in that gap, pass
        # validation, and then die inside pass A with the DiT already staged.
        needed = frames_needed_for(
            chunk_count, geometry.chunk_frames, geometry.stride_frames
        )
        metadata = probe(video_path)
        available = None if metadata.estimated_frames is None else metadata.estimated_frames - start_frame

        clamp_note = None
        if available is not None and available < needed:
            # One chunk is the floor - below that there is nothing to generate,
            # so this is the only case that can still refuse.
            if available < geometry.chunk_frames:
                raise ValueError(
                    "source has about %d normalized frames after start_frame %d, "
                    "which cannot fill even one %d-frame chunk"
                    % (max(0, available), start_frame, geometry.chunk_frames)
                )
            requested_frames, requested_chunks = target_frames, chunk_count
            chunk_count = (available - geometry.chunk_frames) // geometry.stride_frames + 1
            target_frames = frames_needed_for(
                chunk_count, geometry.chunk_frames, geometry.stride_frames
            )
            clamp_note = (
                "requested %.2f s (%d frames, %d chunks) but the source only "
                "offers about %d frames after start_frame %d; clamped to %.2f s "
                "(%d frames, %d chunks)"
                % (requested_frames / geometry.fps, requested_frames,
                   requested_chunks, max(0, available), start_frame,
                   target_frames / geometry.fps, target_frames, chunk_count)
            )
            logging.warning("%s %s", LOG, clamp_note)

        if width and height:
            canvas = (max(32, width // 32 * 32), max(32, height // 32 * 32))
        else:
            canvas = pin_canvas(torch.zeros(1, metadata.source_height, metadata.source_width, 3))

        model, memory_status = memory.arm(
            model, attention=attention, activation=activation
        )
        logging.info("%s %s", LOG, memory.describe(memory_status))

        if run_directory.strip():
            root = run_directory.strip()
        else:
            import folder_paths
            import time
            root = os.path.join(
                folder_paths.get_output_directory(),
                "h3_longform",
                "%s_%s_c%d" % (
                    time.strftime("%Y%m%d_%H%M%S"),
                    carry.replace("direct_latent_", ""),
                    chunk_frames,
                ),
            )

        summary = runner.run(
            video_path=video_path,
            start_frame=start_frame,
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
            ref_image_size=ref_image_size,
            cond_cache=cond_cache,
            save_frames=save_frames,
            output_video=output_video,
            preserve_audio=preserve_source_audio,
            ffmpeg_location=ffmpeg_location.strip() or None,
            runtime_config={"attention": attention, "activation": activation},
        )

        report = "\n".join([
            "MiniMax H3 long-form Ref2V",
            "profile   %s" % summary["profile"],
            "carry     %s" % carry,
            "canvas    %dx%d" % canvas,
            "chunks    %d" % chunk_count,
            "output    %d exact frames (%.3f s at %d fps)" % (
                summary["frames"], summary["frames"] / geometry.fps, geometry.fps
            ),
            "video     %s" % (summary["output_path"] or "disabled"),
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
            # Both of these accept a blank widget and resolve themselves, so echo
            # what was actually used rather than leaving the user to guess.
            "ffmpeg    %s" % _describe_ffmpeg(ffmpeg_location.strip() or None),
        ] + (["clamped   %s" % clamp_note] if clamp_note else []))
        preview = None
        if save_frames:
            preview = _preview(os.path.join(root, "frames"))
        if preview is None and summary["output_path"]:
            preview = _preview_from_video(
                summary["output_path"], ffmpeg_location=ffmpeg_location.strip() or None)
        if preview is None:
            preview = torch.zeros(1, 64, 64, 3)
        return io.NodeOutput(preview, root, summary["output_path"], report)


def _describe_ffmpeg(explicit):
    """Resolved ffmpeg path for the report, never raising into a finished run."""
    from .frame_source import resolve_ffmpeg
    try:
        resolved = resolve_ffmpeg(explicit)
    except Exception as exc:
        return "unresolved (%s)" % exc
    return "%s%s" % (resolved, "" if explicit else "  (auto-detected)")


def _preview_from_video(video_path, limit=48, ffmpeg_location=None):
    """Sample the written video so the preview works without `save_frames`.

    `save_frames` defaults off - a PNG per frame is a diagnostic, not something
    a normal run should pay for - so without this the node's only visual output
    is a black square. Reads a bounded, evenly spaced sample straight from the
    file rather than the whole thing, which is the point of the node.
    """
    from .frame_source import probe as probe_video, read_window

    try:
        meta = probe_video(video_path)
        total = meta.estimated_frames or 0
        if total <= 0:
            return None
        step = max(1, total // limit)
        out = []
        for index in range(0, min(total, step * limit), step):
            try:
                out.append(read_window(video_path, index, 1,
                                       ffmpeg_location=ffmpeg_location)[0])
            except Exception:
                break
        if not out:
            return None
        return torch.stack(out)
    except Exception as exc:
        logging.warning("%s preview unavailable: %s", LOG, exc)
        return None


def _preview(frames_dir, limit=48, width=384):
    from PIL import Image
    import numpy as np

    if not os.path.isdir(frames_dir):
        return torch.zeros(1, 64, 64, 3)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not files:
        return torch.zeros(1, 64, 64, 3)
    step = max(1, len(files) // limit)
    out = []
    for name in files[::step][:limit]:
        img = Image.open(os.path.join(frames_dir, name)).convert("RGB")
        if img.width > width:
            img = img.resize((width, max(1, int(img.height * width / img.width))))
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(out))


class MiniMaxH3LongFormExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormRef2V]
