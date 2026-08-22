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
    """Simulate H3 sparse routing and characterize production-like tile masks.

    Exact sparse-output snapshots remain selective, while the experiment branch
    also records lightweight direct-tile router dynamics across consecutive
    conditional denoising evaluations for HASTE/static-topology analysis.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Moba3DProbeZi",
            display_name="MiniMax H3 MoBA 3D Probe (Zi)",
            category="model/patch/minimax",
            description="Sparse-attention characterization probe. Selected layers/steps run exact sparse-vs-dense output measurements; when attention capture is enabled, a lightweight production-like direct tile router is also tracked across every conditional denoising evaluation to measure Q/K drift, mask reuse, static topology and layer/head budget sensitivity. Does not alter inference.",
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.String.Input("run_tag", default="h3_moba3d", tooltip="Output prefix under output/h3_probe/."),
                io.String.Input("layers", default="auto", tooltip="Layers for expensive exact sparse-output snapshots only: auto or comma-separated indices. Lightweight router dynamics still observe all H3 attention layers."),
                io.String.Input("steps", default="auto", tooltip="Denoising steps for expensive exact sparse-output snapshots only. Lightweight router dynamics still observe every conditional denoising evaluation."),
                io.Int.Input("query_time_positions", default=3, min=1, max=16, tooltip="Video query-frame positions sampled across the clip for exact output-error calibration."),
                io.Int.Input("query_spatial_positions", default=2, min=1, max=8, tooltip="Query regions per sampled frame for exact output-error calibration."),
                io.Int.Input("query_block", default=64, min=16, max=512, step=16, tooltip="Number of query tokens evaluated in each expensive exact snapshot region. Every logical token routes independently; larger values make the exact probe substantially slower."),
                io.Int.Input("block_t", default=1, min=1, max=16, tooltip="Logical 3D candidate block extent in latent time for the original fine-grained router probe."),
                io.Int.Input("block_h", default=4, min=1, max=32, tooltip="Logical 3D candidate block extent in video patch rows."),
                io.Int.Input("block_w", default=4, min=1, max=32, tooltip="Logical 3D candidate block extent in video patch columns."),
                io.String.Input("video_budgets", default="20,30,40,50,60,70", tooltip="Video-KV retention percentages evaluated in exact snapshots. The direct 128x64 calibration uses these same budgets for per-layer/head sensitivity curves."),
                io.Boolean.Input("include_audio_query", default=True, tooltip="Also test audio queries in the legacy logical probe. Head-budget calibration ignores non-video Q records because the production router leaves them dense."),
                io.Boolean.Input("include_text_query", default=False),
                io.Boolean.Input("capture_uncond", default=False, tooltip="Also characterize the unconditional/negative branch. Static-topology analysis is cleanest with the default conditional-only capture."),
                io.Boolean.Input(
                    "capture_latent_dynamics",
                    default=True,
                    tooltip="Measure target-video sampler x/x0 changes between denoising callbacks at H3's 1x2x2 patch granularity, plus distance from explicit first/last keyframes.",
                ),
                io.Boolean.Input(
                    "capture_attention",
                    default=True,
                    tooltip="Enable sparse-attention characterization. On this experiment branch this includes lightweight direct-tile router dynamics on every conditional step plus selective exact sparse-output snapshots.",
                ),
                io.Combo.Input("execution_geometry", options=["logical", "sage_sparse"], default="sage_sparse", tooltip="Use sage_sparse for production-like direct 128x64 calibration and the Q-mask sharing sweep; logical retains only the fine per-token router metrics."),
                io.Int.Input("sage_q_tile", default=128, min=1, max=4096, tooltip="Global packed-sequence Q tile size. SM89 Sparse Sage uses 128."),
                io.Int.Input("sage_kv_tile", default=64, min=1, max=4096, tooltip="Global packed-sequence KV tile size. SM89 Sparse Sage uses 64."),
                io.Boolean.Input("capture_router_dynamics", default=True, tooltip="Track lightweight routes on every conditional layer/step. Disable for bounded exact-snapshot experiments."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, run_tag, layers, steps, query_time_positions,
                query_spatial_positions, query_block, block_t, block_h, block_w,
                video_budgets, include_audio_query, include_text_query,
                capture_uncond, capture_latent_dynamics=True,
                capture_attention=True, execution_geometry="sage_sparse",
                sage_q_tile=128, sage_kv_tile=64,
                capture_router_dynamics=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        if capture_attention:
            # The legacy attention path still needs the H3-only interception.
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
            capture_attention=capture_attention,
            execution_geometry=execution_geometry,
            sage_q_tile=sage_q_tile,
            sage_kv_tile=sage_kv_tile,
            capture_router_dynamics=capture_router_dynamics,
        )

        m = model.clone()
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, "h3_moba3d_probe",
            moba_capture.make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "h3_moba3d_probe",
            moba_capture.make_wrapper(session), m.model_options, is_model_options=True)
        logging.info(
            "[H3 MoBA3D probe] armed: tag=%s exact_layers=%s exact_steps=%s block=%dx%dx%d budgets=%s execution=%s q_tile=%d kv_tile=%d router_dynamics=%s latent_dynamics=%s attention=%s",
            run_tag, layers, steps, block_t, block_h, block_w, video_budgets,
            execution_geometry, sage_q_tile, sage_kv_tile,
            capture_router_dynamics,
            capture_latent_dynamics, capture_attention,
        )
        return io.NodeOutput(m)


class MiniMaxH3ProbeExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AttentionProbe, MiniMaxH3Moba3DProbe]
