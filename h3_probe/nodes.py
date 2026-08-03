"""ComfyUI nodes for the H3 attention probe.

The probe is opt-in and disposable: with no probe node in the graph nothing is
patched, and with `enabled=False` the node is a pass-through.
"""

import logging
import os

import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from . import capture


def _probe_dir():
    return os.path.join(folder_paths.get_output_directory(), "h3_probe")


class MiniMaxH3AttentionProbe(io.ComfyNode):
    """Attach the attention probe to an H3 model.

    Answers, per query block: which text / reference / audio / spatial /
    temporal KV blocks can be dropped while preserving dense-attention output.
    Writes `report.txt`, `summary.json` and `trace.npz` per sampling run.
    """

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


class MiniMaxH3ProbeExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AttentionProbe]
