"""ComfyUI node that constructs the fixed-policy H3 vector sampler."""

import comfy.samplers
from comfy_api.latest import ComfyExtension, io

from .config import (
    DEFAULT_MAX_EXTRAPOLATION_RATIO,
    DIAGNOSTICS,
    METHODS,
    PROFILES,
    SamplerConfig,
)
from .fingerprint import configuration_fingerprint
from .sampler import sample_vector_accel


class MiniMaxH3VectorAccelSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VectorAccelSamplerZi",
            display_name="MiniMax H3 Vector Accel Sampler (Zi)",
            category="H3-Extender/Experiments",
            inputs=[
                io.Combo.Input("method", options=list(METHODS), default="native"),
                io.Combo.Input(
                    "evaluation_profile",
                    options=list(PROFILES),
                    default="native_20",
                ),
                io.Combo.Input(
                    "diagnostics",
                    options=list(DIAGNOSTICS),
                    default="off",
                ),
                io.Boolean.Input(
                    "fallback_on_guard",
                    default=True,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_extrapolation_ratio",
                    default=DEFAULT_MAX_EXTRAPOLATION_RATIO,
                    min=0.1,
                    max=10.0,
                    step=0.05,
                    round=False,
                    advanced=True,
                ),
            ],
            outputs=[io.Sampler.Output()],
        )

    @classmethod
    def execute(cls, method, evaluation_profile, diagnostics,
                fallback_on_guard=True,
                max_extrapolation_ratio=DEFAULT_MAX_EXTRAPOLATION_RATIO):
        config = SamplerConfig(
            method=method,
            evaluation_profile=evaluation_profile,
            diagnostics=diagnostics,
            fallback_on_guard=fallback_on_guard,
            max_extrapolation_ratio=max_extrapolation_ratio,
        )
        sampler = comfy.samplers.KSAMPLER(
            sample_vector_accel,
            extra_options={"config": config},
        )
        sampler.h3_vector_config = config
        sampler.h3_vector_fingerprint = configuration_fingerprint(config)
        return io.NodeOutput(sampler)


class MiniMaxH3VectorAccelExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3VectorAccelSampler]


async def comfy_entrypoint() -> MiniMaxH3VectorAccelExtension:
    return MiniMaxH3VectorAccelExtension()
