"""The long-form Ref2V node.

Takes a video *path*, not an IMAGE. Three minutes at 24 fps is 4320 frames, and
as a float32 Comfy tensor that is ~11 GB before any model loads - so the source
is streamed through `FFmpegFrameSource` and only one chunk is ever resident.

Outputs are a bounded preview and paths. Returning the whole result as an IMAGE
would reintroduce exactly the problem the streaming input solves.
"""

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
except ImportError:  # pragma: no cover - direct import in tests
    import memory
    from geometry import HarnessGeometry
    from ref_builder import pin_canvas
    COND_CACHE_MODES = ["auto", "off", "refresh"]

from . import runner
from .chunk_stream import chunk_count_for, frames_needed_for
from .frame_source import probe

LOG = "[H3 Extended] longform"


class MiniMaxH3LongFormRef2V(io.ComfyNode):
    """Sequential multi-chunk Ref2V over a streamed source."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LongFormRef2VZi",
            display_name="MiniMax H3 Long-Form Ref2V (Zi)",
            category="model/video/minimax/testing",
            is_experimental=True,
            description=(
                "Multi-chunk Ref2V over a streamed video file. Each of the three "
                "models is made resident exactly once (VAE, then Qwen, then DiT, "
                "then VAE again), and every chunk is persisted before the next "
                "starts, so an interruption costs one chunk rather than the run. "
                "Built to expose drift and seam behaviour across many boundaries, "
                "which a two-chunk experiment cannot show."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("video_path", multiline=False,
                                tooltip="Absolute path to the source video. Decoded with "
                                        "ffmpeg, normalized to 24 fps, canvas-pinned."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True,
                             tooltip="Each chunk derives its own noise from this via "
                                     "SplitMix64, so no two chunks share a noise tensor "
                                     "and the run stays reproducible."),
                io.Int.Input("start_frame", default=0, min=0, max=10_000_000,
                             tooltip="Offset into the source, counted in 24 fps frames."),
                io.Int.Input("output_seconds", default=180, min=1, max=3600,
                             tooltip="Approximate output duration; the chunk count is "
                                     "derived from this and the profile."),
                io.Int.Input("chunk_frames", default=90, min=22, max=362, step=17,
                             tooltip="Must satisfy n %% 17 == 5. Longer chunks mean fewer "
                                     "boundaries and more within-chunk context, at higher "
                                     "peak sequence length."),
                io.Int.Input("overlap_frames", default=22, min=5, max=180,
                             tooltip="Must map exactly onto latent positions for the "
                                     "carry to be sliceable; the run refuses otherwise."),
                io.Combo.Input("carry", options=list(runner.CARRY_MODES),
                               default="direct_latent_overlap",
                               tooltip="State carried from the previous chunk. 'none' is "
                                       "the control - independent chunks, no carry."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match"),
                io.Combo.Input("cond_cache", options=list(COND_CACHE_MODES), default="auto"),
                io.Combo.Input("attention", options=list(memory.ATTENTION_MODES),
                               default="auto"),
                io.Combo.Input("activation", options=list(memory.ACTIVATION_MODES),
                               default="mlp_chunked_native"),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="0 derives one canvas from the source and pins it."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32),
                io.String.Input("run_directory", default="", multiline=False,
                                tooltip="Blank creates one under output/h3_longform. An "
                                        "existing directory resumes: completed chunks are "
                                        "kept and only missing work is redone."),
                io.Boolean.Input("save_frames", default=True),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image"),
                                      prefix="ref_image_", min=0, max=9)),
            ],
            outputs=[
                io.Image.Output(display_name="preview"),
                io.String.Output(display_name="run_directory"),
                io.String.Output(display_name="report"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, video_path, prompt,
                sampler, sigmas, seed, start_frame=0, output_seconds=180,
                chunk_frames=90, overlap_frames=22, carry="direct_latent_overlap",
                ref_image_size="match", cond_cache="auto", attention="auto",
                activation="mlp_chunked_native", width=0, height=0,
                run_directory="", save_frames=True, ref_images=None) -> io.NodeOutput:

        video_path = video_path.strip().strip('"')
        if not os.path.exists(video_path):
            raise ValueError("source video not found: %r" % video_path)

        geometry = HarnessGeometry(chunk_frames=chunk_frames,
                                   overlap_frames=overlap_frames).validate()
        target_frames = int(output_seconds * geometry.fps)
        chunk_count = chunk_count_for(target_frames, geometry.chunk_frames,
                                      geometry.stride_frames)
        needed = frames_needed_for(chunk_count, geometry.chunk_frames,
                                   geometry.stride_frames)

        metadata = probe(video_path)
        available = (metadata.estimated_frames or 0) - start_frame
        if available < needed:
            raise ValueError(
                "source has about %d frames after start_frame %d, but %d chunks "
                "need %d. Reduce output_seconds or start_frame."
                % (max(0, available), start_frame, chunk_count, needed))

        if width and height:
            canvas = (width // 32 * 32, height // 32 * 32)
        else:
            canvas = pin_canvas(torch.zeros(1, metadata.source_height,
                                            metadata.source_width, 3))

        model, memory_status = memory.arm(model, attention=attention,
                                          activation=activation)
        logging.info("%s %s", LOG, memory.describe(memory_status))

        if run_directory.strip():
            root = run_directory.strip()
        else:
            import folder_paths
            import time
            root = os.path.join(folder_paths.get_output_directory(), "h3_longform",
                                "%s_%s_c%d" % (time.strftime("%Y%m%d_%H%M%S"),
                                               carry.replace("direct_latent_", ""),
                                               chunk_frames))

        summary = runner.run(
            video_path=video_path, start_frame=start_frame,
            chunk_frames=chunk_frames, overlap_frames=overlap_frames,
            chunk_count=chunk_count, model=model, clip=clip,
            video_vae=video_vae, audio_vae=audio_vae, prompt=prompt,
            sampler=sampler, sigmas=sigmas, seed=seed, carry=carry,
            canvas=canvas, root=root, ref_images=ref_images,
            ref_image_size=ref_image_size, cond_cache=cond_cache,
            save_frames=save_frames)

        report = "\n".join([
            "MiniMax H3 long-form Ref2V",
            "profile   %s" % summary["profile"],
            "carry     %s" % carry,
            "canvas    %dx%d" % canvas,
            "chunks    %d  (%d source frames from start_frame %d)"
            % (chunk_count, needed, start_frame),
            "output    %d frames (%.1f s at %d fps)"
            % (summary["frames"], summary["frames"] / geometry.fps, geometry.fps),
            "runtime   %s" % memory.describe(memory_status),
            "run dir   %s" % root,
        ])
        logging.info("%s finished: %d frames in %s", LOG, summary["frames"], root)

        preview = _preview(os.path.join(root, "frames"))
        return io.NodeOutput(preview, root, report)


def _preview(frames_dir, limit=48, width=384):
    """A bounded strip of the output. Never the whole result."""
    from PIL import Image
    import numpy as np

    if not os.path.isdir(frames_dir):
        return torch.zeros(1, 64, 64, 3)
    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if not files:
        return torch.zeros(1, 64, 64, 3)
    step = max(1, len(files) // limit)
    picked = files[::step][:limit]
    out = []
    for name in picked:
        img = Image.open(os.path.join(frames_dir, name)).convert("RGB")
        if img.width > width:
            img = img.resize((width, max(1, int(img.height * width / img.width))))
        out.append(np.asarray(img, dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(out))


class MiniMaxH3LongFormExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3LongFormRef2V]
