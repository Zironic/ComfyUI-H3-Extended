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
        adaptive_visibility = None if profiles else {"hidden": True}
        return io.Schema(
            node_id="MiniMaxH3VectorAccelSamplerZi",
            display_name="MiniMax H3 Vector Accel Sampler (Zi)",
            category="H3-Extender/Experiments",
            description="Chooses an actual-only core solver or a derivative forecast mode, with a separate evaluation schedule.",
            inputs=[
                io.Combo.Input(
                    "method",
                    options=list(METHODS),
                    default="euler",
                    display_name="solver / forecast mode",
                    tooltip="euler and res_multistep evaluate only the selected schedule's actual points. hold, linear_velocity, and vde forecast between the selected actual anchors.",
                ),
                io.Combo.Input(
                    "evaluation_profile",
                    options=list(PROFILES),
                    default="full_20",
                    display_name="actual-evaluation schedule",
                    tooltip="Select full_20 for every model evaluation or a reduced named schedule such as late_aggressive_13. With res_multistep, late_aggressive_13 is the 13-NFE multistep benchmark.",
                ),
                io.Combo.Input(
                    "diagnostics",
                    options=list(DIAGNOSTICS),
                    default="off",
                    tooltip="summary logs run totals; full also writes per-step and per-anchor diagnostics JSON.",
                ),
                io.Combo.Input(
                    "policy",
                    options=policies,
                    default="fixed",
                    tooltip="Adaptive repair is available only when a compatible measured profile is installed.",
                    extra_dict=adaptive_visibility,
                ),
                io.Combo.Input(
                    "quality_preset",
                    options=list(QUALITY_PRESETS),
                    default="balanced",
                    display_name="adaptive quality tolerance",
                    tooltip="Adaptive policy only. Selects a measured risk tolerance stored in the repairability profile.",
                    extra_dict=adaptive_visibility,
                    advanced=True,
                ),
                io.Combo.Input(
                    "repairability_profile",
                    options=profiles or [""],
                    default=profiles[0] if profiles else "",
                    tooltip="Adaptive policy only. Measured survival profile matched to this model, schedule, and conditioning mode.",
                    extra_dict=adaptive_visibility,
                    advanced=True,
                ),
                io.Combo.Input(
                    "conditioning_mode",
                    options=list(CONDITIONING_MODES),
                    default="default",
                    display_name="adaptive profile conditioning",
                    tooltip="Adaptive profile compatibility label only; this does not alter model conditioning.",
                    extra_dict=adaptive_visibility,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "fallback_on_guard",
                    default=True,
                    tooltip="Run a real H3 evaluation when a requested forecast fails a numerical safety guard.",
                    advanced=True,
                ),
                io.Float.Input(
                    "max_extrapolation_ratio",
                    default=DEFAULT_MAX_EXTRAPOLATION_RATIO,
                    min=0.1,
                    max=10.0,
                    step=0.05,
                    round=False,
                    tooltip="Hard ceiling for forecast derivative RMS versus the last actual derivative. It does not scale forecasts.",
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

    @classmethod
    def validate_inputs(cls, method, evaluation_profile):
        try:
            SamplerConfig(method=method, evaluation_profile=evaluation_profile)
        except ValueError as exc:
            return str(exc)
        return True


class MiniMaxH3VectorAccelExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3VectorAccelSampler]


async def comfy_entrypoint() -> MiniMaxH3VectorAccelExtension:
    return MiniMaxH3VectorAccelExtension()
