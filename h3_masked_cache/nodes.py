"""The masked Ref2V node.

Separate from `MiniMaxH3SigmaShiftZi` on purpose: the shift node owns the flow
schedule, the attention backend and the VRAM guard, and this one owns nothing
but masked-Ref2V behaviour. Keeping them apart is what makes an A/B honest -
removing this node from the graph removes the entire feature and nothing else.

Only `measure` is implemented. It installs two wrappers that observe and never
modify, so a `measure` run is a dense run that also writes a report.
"""

import logging
import os

import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .config import IMPLEMENTED_MODES, MODES, MaskedCacheConfig
from .session import MaskedCacheSession
from .wrappers import LOG_PREFIX, make_diffusion_wrapper, make_outer_wrapper

WRAPPER_KEY = "h3_masked_cache"


def _output_dir():
    return os.path.join(folder_paths.get_output_directory(), "h3_masked_cache")


class MiniMaxH3MaskedRef2VCache(io.ComfyNode):
    """Measure whether unchanged target regions can be dropped from H3's blocks.

    Compares H3's predicted clean latent against the source video reference at
    every step and reports how large, how stable and how early-determined the
    edited region is. Does not change the output.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedRef2VCacheZi",
            display_name="MiniMax H3 Masked Ref2V Cache (Zi)",
            category="model/patch/minimax",
            description=(
                "Ref2V edit-mask measurement. Compares the predicted clean latent "
                "against the selected source video reference and writes score maps, "
                "threshold sweeps and mask-stability figures to "
                "output/h3_masked_cache/<run_tag>_<timestamp>/. In 'measure' mode the "
                "model output is untouched - sampling is exactly as dense as without "
                "the node."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "mode", options=list(MODES), default="measure",
                    tooltip=("measure: observe only, output unchanged. "
                             "fixed/dynamic apply the mask to the computation and are "
                             "not implemented yet - selecting one errors rather than "
                             "quietly measuring."),
                ),
                io.Int.Input(
                    "source_video_ref", default=1, min=1, max=16,
                    tooltip=("Which reference video the target is edited from, "
                             "one-based over video references only (reference images "
                             "and standalone audio do not count). Its latent must match "
                             "the target latent exactly in time and size."),
                ),
                io.Float.Input(
                    "score_threshold", default=0.1, min=0.0, max=10.0, step=0.005,
                    tooltip=("Relative per-token difference above which a token counts "
                             "as edited. The report sweeps a range around this value; "
                             "the default is a placeholder until measurement picks one."),
                ),
                io.Float.Input(
                    "score_floor", default=0.001, min=1e-6, max=1.0, step=0.001,
                    tooltip=("Added to the source magnitude before dividing, so flat "
                             "regions do not score high on a tiny absolute error."),
                ),
                io.Combo.Input(
                    "tile_size", options=[1, 2, 4], default=2,
                    tooltip="Token tile the mask is quantized to. Any active token activates its whole tile.",
                ),
                io.Int.Input(
                    "spatial_halo", default=1, min=0, max=16,
                    tooltip="Dilate the mask by this many tiles in each spatial direction.",
                ),
                io.Int.Input(
                    "temporal_halo", default=1, min=0, max=16,
                    tooltip=("Dilate the mask by this many latent frames in each temporal "
                             "direction. Latent frames cover unequal amounts of real time."),
                ),
                io.Int.Input(
                    "warmup_steps", default=2, min=1, max=32,
                    tooltip="Dense steps before a mask may be used. Recorded only, in measure mode.",
                ),
                io.Int.Input(
                    "refresh_interval", default=0, min=0, max=64,
                    tooltip="Distinct sigmas between dense mask refreshes; 0 freezes the mask. Recorded only, in measure mode.",
                ),
                io.Float.Input(
                    "dense_fallback_fraction", default=0.8, min=0.0, max=1.0, step=0.01,
                    tooltip="Above this active fraction a compact pass is not worth its complexity. Recorded only, in measure mode.",
                ),
                io.Boolean.Input(
                    "strict", default=True,
                    tooltip=("Stop the run when the source cannot be resolved or the "
                             "geometry does not match, instead of sampling dense and "
                             "reporting the fallback. A measurement run that quietly "
                             "measured nothing is worse than one that stopped."),
                ),
                io.String.Input("run_tag", default="h3mask",
                                tooltip="Output subdirectory prefix. One tag per test-matrix entry."),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, mode, source_video_ref, score_threshold, score_floor,
                tile_size, spatial_halo, temporal_halo, warmup_steps, refresh_interval,
                dense_fallback_fraction, strict, run_tag) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        if mode not in IMPLEMENTED_MODES:
            raise NotImplementedError(
                "%s mode '%s' is not implemented yet - only %s is. Masked execution "
                "lands after measurement has chosen a threshold; until then this node "
                "will not pretend to apply a mask." % (
                    LOG_PREFIX, mode, ", ".join("'%s'" % m for m in IMPLEMENTED_MODES)))

        config = MaskedCacheConfig(
            mode=mode,
            source_video_ref=int(source_video_ref),
            warmup_steps=int(warmup_steps),
            refresh_interval=int(refresh_interval),
            score_threshold=float(score_threshold),
            score_absolute_floor=float(score_floor),
            tile_h=int(tile_size),
            tile_w=int(tile_size),
            spatial_halo=int(spatial_halo),
            temporal_halo=int(temporal_halo),
            dense_fallback_fraction=float(dense_fallback_fraction),
            strict=bool(strict),
            run_tag=run_tag or "h3mask",
        )

        m = model.clone()
        # the sampling object as configured *at this point in the graph*, so a
        # sigma-shift node upstream is honoured and one downstream is visibly not
        session = MaskedCacheSession(
            config, _output_dir(), model_sampling=m.get_model_object("model_sampling"))

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
            make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
            make_diffusion_wrapper(session), m.model_options, is_model_options=True)

        logging.info("%s armed: mode=%s tag=%s source_video_ref=%d threshold=%.3g "
                     "tile=%dx%d halo=(%d,%d) strict=%s",
                     LOG_PREFIX, mode, config.run_tag, config.source_video_ref,
                     config.score_threshold, config.tile_h, config.tile_w,
                     config.spatial_halo, config.temporal_halo, config.strict)
        return io.NodeOutput(m)


class MiniMaxH3MaskedCacheExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MaskedRef2VCache]
