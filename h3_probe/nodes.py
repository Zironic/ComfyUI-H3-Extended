"""ComfyUI nodes for the H3 attention probes.

The probes are opt-in and disposable: with no probe node in the graph nothing is
patched, and with `enabled=False` either node is a pass-through.
"""

import logging
import os

import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import capture, moba_capture


def _probe_dir():
    return os.path.join(folder_paths.get_output_directory(), "h3_probe")


class MiniMaxH3AttentionProbe(io.ComfyNode):
    """Attach the original locality/top-k attention probe to an H3 model."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AttentionProbeZi",
            display_name="MiniMax H3 Attention Probe (Zi)",
            category="model/patch/minimax",
            description="Measure H3 attention structure to design a block-sparse mask. Slows sampling on probed steps only; output goes to output/h3_probe/<tag>_<timestamp>/.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.String.Input("run_tag", default="h3", tooltip="Output subdirectory prefix. Use one tag per test-matrix entry."),
                io.String.Input("layers", default="auto", tooltip="DiT layers to probe: 'auto' for early/middle/late, or a comma list like '6,24,44'. Negatives count from the end."),
                io.String.Input("steps", default="auto", tooltip="Denoising steps to probe: 'auto' for early/middle/late, or a comma list like '0,7,15'."),
                io.Int.Input("query_time_positions", default=4, min=1, max=32, tooltip="How many latent frames to place video query blocks at, spread over the clip."),
                io.Int.Input("query_spatial_positions", default=2, min=1, max=16, tooltip="Query blocks per probed frame, spread across the spatial grid."),
                io.Int.Input("kv_block", default=128, min=32, max=1024, step=32, tooltip="KV block size the mask would operate on. Top-k and coverage figures are reported at this granularity."),
                io.Boolean.Input("include_audio_query", default=True, tooltip="Also probe a target-audio query block (audio->video attention drives AV sync)."),
                io.Boolean.Input("include_text_query", default=False),
                io.Boolean.Input("capture_uncond", default=False, tooltip="Also probe the negative pass. Doubles probe cost; the conditional pass is usually what matters."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, run_tag, layers, steps, query_time_positions,
                query_spatial_positions, kv_block, include_audio_query,
                include_text_query, capture_uncond) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        capture.install()
        session = capture.ProbeSession(
            tag=run_tag or "h3",
            layers_spec=layers,
            steps_spec=steps,
            n_time=query_time_positions,
            n_spatial=query_spatial_positions,
            block=kv_block,
            include_audio=include_audio_query,
            include_text=include_text_query,
            capture_uncond=capture_uncond,
            base_dir=_probe_dir(),
        )

        m = model.clone()
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "h3_probe",
            capture.make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3_probe",
            capture.make_wrapper(session), m.model_options, is_model_options=True)
        logging.info("[H3 probe] armed: tag=%s layers=%s steps=%s block=%d",
                     run_tag, layers, steps, kv_block)
        return io.NodeOutput(m)


class MiniMaxH3Moba3DProbe(io.ComfyNode):
    """Simulate the H3 team's publicly described MoBA-style 3D routing.

    Measurement only: non-video context remains dense, target-video keys are
    grouped in 3D blocks and represented by mean-pooled post-RoPE keys. Each
    query token routes independently, and Q/K/V are used to compare the exact
    masked-and-renormalized sparse output with dense attention.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Moba3DProbeZi",
            display_name="MiniMax H3 MoBA 3D Probe (Zi)",
            category="model/patch/minimax",
            description="Probe-only simulation of MiniMax's disclosed H3 sparse-attention idea: video-only 3D candidate blocks, per-query-token mean-pooled routing, dense non-video context, exact sparse-vs-dense output error, and optional sampler-latent convergence measurements. Does not alter inference and is intentionally expensive.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.String.Input("run_tag", default="h3_moba3d", tooltip="Output prefix under output/h3_probe/."),
                io.String.Input("layers", default="auto", tooltip="DiT layers: auto or comma-separated indices."),
                io.String.Input("steps", default="auto", tooltip="Denoising steps: auto or comma-separated indices."),
                io.Int.Input("query_time_positions", default=3, min=1, max=16, tooltip="Video query-frame positions sampled across the clip."),
                io.Int.Input("query_spatial_positions", default=2, min=1, max=8, tooltip="Query regions sampled within each selected latent frame."),
                io.Int.Input("query_block", default=64, min=16, max=512, step=16, tooltip="Number of query tokens evaluated in each sampled region. Every token routes independently; larger values make the probe substantially slower."),
                io.Int.Input("block_t", default=1, min=1, max=16, tooltip="3D candidate block extent in latent time."),
                io.Int.Input("block_h", default=4, min=1, max=32, tooltip="3D candidate block extent in video patch rows."),
                io.Int.Input("block_w", default=4, min=1, max=32, tooltip="3D candidate block extent in video patch columns."),
                io.String.Input("video_budgets", default="10,20,30,40", tooltip="Percent of video blocks retained independently per query token and head, e.g. 10,20,30,40."),
                io.Boolean.Input("include_audio_query", default=True, tooltip="Also test audio queries routing into video blocks."),
                io.Boolean.Input("include_text_query", default=False),
                io.Boolean.Input("capture_uncond", default=False),
                io.Boolean.Input(
                    "capture_latent_dynamics",
                    default=True,
                    tooltip="Measure target-video sampler x/x0 changes between every denoising callback at H3's 1x2x2 patch granularity, plus distance from explicit first/last keyframes. Adds moderate probe overhead but does not alter sampling.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, run_tag, layers, steps, query_time_positions,
                query_spatial_positions, query_block, block_t, block_h, block_w,
                video_budgets, include_audio_query, include_text_query,
                capture_uncond, capture_latent_dynamics=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        # The legacy attention path still needs the H3-only interception. The
        # H3-owned custom forward publishes to the same observer seam directly.
        capture.install()
        session = moba_capture.MobaProbeSession(
            tag=run_tag or "h3_moba3d",
            layers_spec=layers,
            steps_spec=steps,
            n_time=query_time_positions,
            n_spatial=query_spatial_positions,
            query_block=query_block,
            include_audio=include_audio_query,
            include_text=include_text_query,
            capture_uncond=capture_uncond,
            capture_latent_dynamics=capture_latent_dynamics,
            block_t=block_t,
            block_h=block_h,
            block_w=block_w,
            budgets=video_budgets,
            base_dir=_probe_dir(),
        )

        m = model.clone()
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "h3_moba3d_probe",
            moba_capture.make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3_moba3d_probe",
            moba_capture.make_wrapper(session), m.model_options, is_model_options=True)
        logging.info(
            "[H3 MoBA3D probe] armed: tag=%s layers=%s steps=%s block=%dx%dx%d budgets=%s latent_dynamics=%s",
            run_tag, layers, steps, block_t, block_h, block_w, video_budgets,
            capture_latent_dynamics,
        )
        return io.NodeOutput(m)


class MiniMaxH3ProbeExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AttentionProbe, MiniMaxH3Moba3DProbe]
