"""ComfyUI node for H3 block/range cache smoke tests."""
import logging
import os

import comfy.memory_management
import comfy.patcher_extension
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .config import BlockCacheConfig, MODES
from .session import BlockCacheSession, LOG_PREFIX
from .units import parse_unit_spec
from .wrappers import make_block_replacement, make_diffusion_wrapper, make_outer_wrapper

WRAPPER_KEY = "h3_block_cache"


def _output_dir():
    return os.path.join(folder_paths.get_output_directory(), "h3_block_cache")

class MiniMaxH3BlockRangeCache(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3BlockRangeCacheZi",
            display_name="MiniMax H3 Block/Range Cache (Zi)",
            category="model/patch/minimax",
            description=(
                "Experimental H3 block/range residual cache. AIMDO remains mandatory; "
                "the node disables only static all-block vbar prefetch so reused blocks "
                "do not load weights before the cache decision. observe/shadow are "
                "output-neutral; fixed_gpu changes inference."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(MODES), default="observe"),
                io.String.Input("unit_spec", default="25",
                    tooltip="Comma-separated blocks/ranges, e.g. 25 or 20-29,35."),
                io.Int.Input("warmup_steps", default=2, min=0, max=64),
                io.Int.Input("refresh_interval", default=2, min=1, max=64),
                io.Int.Input("max_reuse_span", default=1, min=1, max=16),
                io.Int.Input("force_refresh_last_steps", default=1, min=0, max=16),
                io.Boolean.Input("strict", default=True),
                io.String.Input("run_tag", default="h3block"),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled, mode, unit_spec, warmup_steps,
                refresh_interval, max_reuse_span, force_refresh_last_steps,
                strict, run_tag):
        if not enabled:
            return io.NodeOutput(model)
        if not comfy.memory_management.aimdo_enabled:
            raise RuntimeError(
                f"{LOG_PREFIX}: AIMDO is required for full H3 smoke tests on this machine")
        config = BlockCacheConfig(
            mode=mode,
            unit_spec=unit_spec,
            warmup_steps=int(warmup_steps),
            refresh_interval=int(refresh_interval),
            max_reuse_span=int(max_reuse_span),
            force_refresh_last_steps=int(force_refresh_last_steps),
            strict=bool(strict),
            run_tag=run_tag or "h3block",
        )
        units = parse_unit_spec(config.unit_spec)
        m = model.clone()
        session = BlockCacheSession(config, _output_dir())

        to = m.model_options["transformer_options"] = dict(
            m.model_options.get("transformer_options", {}))
        patches = to["patches_replace"] = dict(to.get("patches_replace", {}))
        dit = patches["dit"] = dict(patches.get("dit", {}))
        for index in range(50):
            key = ("double_block", index)
            if key in dit:
                raise RuntimeError(f"{LOG_PREFIX}: block {index} already has a replacement")
            dit[key] = make_block_replacement(session, index)

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY,
            make_outer_wrapper(session), m.model_options, is_model_options=True)
        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY,
            make_diffusion_wrapper(session), m.model_options, is_model_options=True)

        logging.info(
            "%s armed: mode=%s units=%s AIMDO=required static_vbar_prefetch=disabled",
            LOG_PREFIX, config.mode, ",".join(u.key for u in units))
        return io.NodeOutput(m)

class MiniMaxH3BlockCacheExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3BlockRangeCache]
