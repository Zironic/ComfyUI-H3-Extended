"""ComfyUI nodes for standard and experimental H3 sampling."""

import comfy.samplers
import torch
from comfy_api.latest import ComfyExtension, io

from .config import (
    DEFAULT_MAX_EXTRAPOLATION_RATIO,
    DEFAULT_MAX_ADAPTIVE_STEP_SCALE,
    DEFAULT_EMBEDDED_VIDEO_TOLERANCE,
    DEFAULT_ADAPTIVE_SAFETY_FACTOR,
    DEFAULT_MAX_ADAPTIVE_GROWTH_RATIO,
    CONDITIONING_MODES,
    DIAGNOSTICS,
    EVALUATION_PROFILES,
    METHODS,
    POLICIES,
    QUALITY_PRESETS,
    SamplerConfig,
)
from .fingerprint import configuration_fingerprint
from .repairability import PROFILE_ROOT
from .sampler import sample_vector_accel
from .schedules import CONTINUOUS_SCHEDULE_FAMILIES, continuous_schedule_family


def _profile_names():
    return sorted(path.name for path in PROFILE_ROOT.glob("*.json") if path.is_file())


class MiniMaxH3SamplerScheduler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SamplerSchedulerZi",
            display_name="MiniMax H3 Sampler + Scheduler (Zi)",
            category="H3-Extender/Sampling",
            description="Builds a standard ComfyUI sampler and its sigma schedule in one node.",
            inputs=[
                io.Model.Input("model"),
                io.Combo.Input(
                    "sampler_name",
                    options=comfy.samplers.SAMPLER_NAMES,
                    default="res_multistep",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=[*comfy.samplers.SCHEDULER_NAMES, *CONTINUOUS_SCHEDULE_FAMILIES],
                    default="simple",
                ),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Below 1.0, starts partway through a longer schedule while returning the requested number of steps.",
                ),
            ],
            outputs=[io.Sampler.Output(), io.Sigmas.Output()],
        )

    @classmethod
    def execute(cls, model, sampler_name, scheduler, steps, denoise):
        sampler = comfy.samplers.sampler_object(sampler_name)
        if denoise <= 0.0:
            return io.NodeOutput(sampler, torch.FloatTensor([]))

        total_steps = steps if denoise >= 1.0 else int(steps / denoise)
        model_sampling = model.get_model_object("model_sampling")
        if scheduler in CONTINUOUS_SCHEDULE_FAMILIES:
            source_sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, "simple", 20,
            )
            sigmas, _, _ = continuous_schedule_family(
                source_sigmas, scheduler, total_steps,
            )
            sigmas = sigmas.cpu()
        else:
            sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, total_steps,
            ).cpu()
        return io.NodeOutput(sampler, sigmas[-(steps + 1):])


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
                    options=list(EVALUATION_PROFILES),
                    default="full_20",
                    display_name="actual-evaluation schedule",
                    tooltip="Select full_20, a named reduced schedule, an adaptive_history controller, or adaptive_embedded_res_v1. Adaptive schedules require res_multistep.",
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
                    display_name="forecast max extrapolation ratio",
                    tooltip="Forecast-only safety ceiling; it does not control adaptive RES spacing.",
                    advanced=True,
                ),
                io.Float.Input(
                    "max_adaptive_step_scale",
                    default=DEFAULT_MAX_ADAPTIVE_STEP_SCALE,
                    min=1.0,
                    max=10.0,
                    step=0.1,
                    round=False,
                    display_name="adaptive RES maximum step scale",
                    tooltip="Adaptive RES only. Caps an accepted interval relative to the local 20-step source interval.",
                    advanced=True,
                ),
                io.Float.Input(
                    "embedded_video_tolerance", default=DEFAULT_EMBEDDED_VIDEO_TOLERANCE,
                    min=0.0001, max=10.0, step=0.01, round=False,
                    display_name="embedded RES video defect tolerance",
                    tooltip="adaptive_embedded_res_v1 only. Larger values permit a larger second-order RES correction; this is not a measured visual-quality threshold.",
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_safety_factor", default=DEFAULT_ADAPTIVE_SAFETY_FACTOR,
                    min=0.01, max=1.0, step=0.01, round=False,
                    display_name="embedded RES safety factor",
                    tooltip="adaptive_embedded_res_v1 only. Multiplies the tolerance-selected interval before hard clamps; lower values are more conservative.",
                    advanced=True,
                ),
                io.Float.Input(
                    "max_adaptive_growth_ratio", default=DEFAULT_MAX_ADAPTIVE_GROWTH_RATIO,
                    min=1.0, max=10.0, step=0.1, round=False,
                    display_name="embedded RES maximum interval growth",
                    tooltip="adaptive_embedded_res_v1 only. Caps each interval relative to the preceding accepted interval.",
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
                repairability_profile="", conditioning_mode="default",
                max_adaptive_step_scale=DEFAULT_MAX_ADAPTIVE_STEP_SCALE,
                embedded_video_tolerance=DEFAULT_EMBEDDED_VIDEO_TOLERANCE,
                adaptive_safety_factor=DEFAULT_ADAPTIVE_SAFETY_FACTOR,
                max_adaptive_growth_ratio=DEFAULT_MAX_ADAPTIVE_GROWTH_RATIO):
        config = SamplerConfig(
            method=method,
            evaluation_profile=evaluation_profile,
            diagnostics=diagnostics,
            fallback_on_guard=fallback_on_guard,
            max_extrapolation_ratio=max_extrapolation_ratio,
            max_adaptive_step_scale=max_adaptive_step_scale,
            embedded_video_tolerance=embedded_video_tolerance,
            adaptive_safety_factor=adaptive_safety_factor,
            max_adaptive_growth_ratio=max_adaptive_growth_ratio,
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
        return [MiniMaxH3SamplerScheduler, MiniMaxH3VectorAccelSampler]


async def comfy_entrypoint() -> MiniMaxH3VectorAccelExtension:
    return MiniMaxH3VectorAccelExtension()
