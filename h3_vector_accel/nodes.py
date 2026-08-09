"""ComfyUI node that constructs the H3 vector sampler."""

import comfy.samplers
from comfy_api.latest import ComfyExtension, io

from .config import (
    DEFAULT_MAX_EXTRAPOLATION_RATIO,
    CONDITIONING_MODES,
    DIAGNOSTICS,
    METHODS,
    POLICIES,
    PROFILES,
    QUALITY_PRESETS,
    SamplerConfig,
)
from .fingerprint import configuration_fingerprint
from .repairability import PROFILE_ROOT
from .sampler import sample_vector_accel


def _profile_names():
    return sorted(path.name for path in PROFILE_ROOT.glob("*.json") if path.is_file())


class MiniMaxH3VectorAccelSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        profiles = _profile_names()
        policies = list(POLICIES) if profiles else ["fixed"]
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
                io.Combo.Input("policy", options=policies, default="fixed"),
                io.Combo.Input(
                    "quality_preset",
                    options=list(QUALITY_PRESETS),
                    default="balanced",
                    advanced=True,
                ),
                io.Combo.Input(
                    "repairability_profile",
                    options=profiles or [""],
                    default=profiles[0] if profiles else "",
                    advanced=True,
                ),
                io.Combo.Input(
                    "conditioning_mode",
                    options=list(CONDITIONING_MODES),
                    default="default",
                    advanced=True,
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
                max_extrapolation_ratio=DEFAULT_MAX_EXTRAPOLATION_RATIO,
                policy="fixed", quality_preset="balanced",
                repairability_profile="", conditioning_mode="default"):
        config = SamplerConfig(
            method=method,
            evaluation_profile=evaluation_profile,
            diagnostics=diagnostics,
            fallback_on_guard=fallback_on_guard,
            max_extrapolation_ratio=max_extrapolation_ratio,
            policy=policy,
            quality_preset=quality_preset,
            repairability_profile=repairability_profile or None,
            conditioning_mode=conditioning_mode,
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
