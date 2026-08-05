"""ComfyUI node for output-neutral H3 Ref2V mask measurement."""

import logging
import os

import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .config import IMPLEMENTED_MODES, MODES, MaskedCacheConfig
from .session import MaskedCacheSession
from .wrappers import (
    LOG_PREFIX,
    make_diffusion_wrapper,
    make_outer_wrapper,
    make_post_cfg_observer,
)

WRAPPER_KEY = "h3_masked_cache"


def _output_dir():
    return os.path.join(folder_paths.get_output_directory(), "h3_masked_cache")


class MiniMaxH3MaskedRef2VCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MaskedRef2VCacheZi",
            display_name="MiniMax H3 Masked Ref2V Cache (Zi)",
            category="model/patch/minimax",
            description=(
                "Output-neutral Ref2V edit-mask measurement. Observes the guided "
                "post-CFG denoised prediction and final sampled latent; writes raw "
                "float32 error/source maps, threshold sweeps and frozen-warmup "
                "coverage to output/h3_masked_cache/<run_tag>_<timestamp>/."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(MODES), default="measure",
                    tooltip="Only measure is implemented; fixed/dynamic error explicitly."),
                io.Int.Input("source_video_ref", default=1, min=1, max=16,
                    tooltip="One-based ordinal over video references only."),
                io.Float.Input("score_threshold", default=0.1, min=0.0, max=10.0, step=0.005,
                    tooltip="Relative token score threshold. The report sweeps 0.01 through 10."),
                io.Float.Input("score_floor", default=0.001, min=1e-6, max=1.0, step=0.001,
                    tooltip="Added to source RMS for the online relative score."),
                io.Combo.Input("tile_size", options=[1, 2, 4], default=2),
                io.Int.Input("spatial_halo", default=1, min=0, max=16),
                io.Int.Input("temporal_halo", default=1, min=0, max=16),
                io.Int.Input("warmup_steps", default=2, min=1, max=32,
                    tooltip="Number of guided predictions whose union becomes the immutable frozen mask."),
                io.Int.Input("refresh_interval", default=0, min=0, max=64,
                    tooltip="Recorded for later policy simulation; 0 means no refresh."),
                io.Float.Input("dense_fallback_fraction", default=0.8, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("strict", default=True,
                    tooltip="Refuse invalid measurement, including EasyCache contamination."),
                io.String.Input("run_tag", default="h3mask"),
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
                "%s mode '%s' is not implemented yet; only %s is available." % (
                    LOG_PREFIX, mode, ", ".join(IMPLEMENTED_MODES)))

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
        session = MaskedCacheSession(
            config, _output_dir(), model_sampling=m.get_model_object("model_sampling"))

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
            make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
            make_diffusion_wrapper(session), m.model_options, is_model_options=True)

        # Comfy invokes these after CFG and all earlier post-CFG hooks.  Preserve
        # existing callbacks and append an observer that returns denoised unchanged.
        post = list(m.model_options.get("sampler_post_cfg_function", []))
        post.append(make_post_cfg_observer(session))
        m.model_options["sampler_post_cfg_function"] = post

        logging.info(
            "%s armed: mode=%s tag=%s source_video_ref=%d threshold=%.3g "
            "tile=%dx%d halo=(%d,%d) warmup=%d strict=%s",
            LOG_PREFIX, mode, config.run_tag, config.source_video_ref,
            config.score_threshold, config.tile_h, config.tile_w,
            config.spatial_halo, config.temporal_halo, config.warmup_steps,
            config.strict)
        return io.NodeOutput(m)


class MiniMaxH3MaskedCacheExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MaskedRef2VCache]
