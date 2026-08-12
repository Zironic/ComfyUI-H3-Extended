"""Immutable, order-independent configuration for H3 Sage optimizations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re

PLAN_KEY = "minimax_h3_sage_optimization_plan"
STATUS_KEY = "minimax_h3_sage_optimization_status"
PLAN_VERSION = 7

ATTENTION_AUTO = "auto"
ATTENTION_EXISTING = "existing"
ATTENTION_REQUESTS = (ATTENTION_AUTO, ATTENTION_EXISTING)

FUSED_QKV_AUTO = "auto"
FUSED_QKV_OFF = "off"
FUSED_QKV_REQUIRED = "required"
FUSED_QKV_REQUESTS = (FUSED_QKV_AUTO, FUSED_QKV_OFF, FUSED_QKV_REQUIRED)

MLP_MEMORY_AUTO = "auto"
MLP_MEMORY_EPILOGUE = "epilogue_prototype"
MLP_MEMORY_OFF = "off"

# Internal values used by explicit advanced controls and deprecated workflows.
MLP_MEMORY_LEGACY_BF16 = "legacy_bf16"
MLP_MEMORY_LEGACY_NATIVE = "legacy_native"
MLP_MEMORY_LEGACY_CONVROT_REQUIRED = "legacy_convrot_2slice_required"
MLP_MEMORY_STRICT_AUTO = "strict_auto"
MLP_MEMORY_STRICT_BF16 = "strict_bf16"
MLP_MEMORY_STRICT_NATIVE = "strict_native"

MLP_MEMORY_REQUESTS = (
    MLP_MEMORY_AUTO,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_OFF,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_STRICT_AUTO,
    MLP_MEMORY_STRICT_BF16,
    MLP_MEMORY_STRICT_NATIVE,
)

DENSITY_FIXED = "fixed"
DENSITY_ADAPTIVE_BUDGET = "adaptive_budget"
DENSITY_MODES = (DENSITY_FIXED, DENSITY_ADAPTIVE_BUDGET)

COMPILE_OFF = "off"
COMPILE_INDUCTOR = "inductor"
COMPILE_BACKENDS = (COMPILE_OFF, COMPILE_INDUCTOR)

_RUN_TAG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


@dataclass(frozen=True)
class MemoryRequest:
    """Execution/memory options owned by the Memory Optimizer node."""

    attention: str = ATTENTION_AUTO
    fused_qkv: str = FUSED_QKV_AUTO
    mlp_memory: str = MLP_MEMORY_AUTO
    chunk_rows: int = 2048
    prefer_held_weights: bool = True
    strict: bool = False

    def __post_init__(self):
        if self.attention not in ATTENTION_REQUESTS:
            raise ValueError("unknown H3 Sage attention request %r" % self.attention)
        if self.fused_qkv not in FUSED_QKV_REQUESTS:
            raise ValueError("unknown fused QKV request %r" % self.fused_qkv)
        if self.mlp_memory not in MLP_MEMORY_REQUESTS:
            raise ValueError("unknown MLP memory request %r" % self.mlp_memory)
        if int(self.chunk_rows) <= 0:
            raise ValueError("chunk_rows must be positive")

        # Strictness controls fallback inside the selected production MLP path.
        # It deliberately does not promote QKV auto to the research-only fused
        # QKV implementation. Research kernels require a separate explicit gate.
        if bool(self.strict):
            if self.mlp_memory == MLP_MEMORY_AUTO:
                object.__setattr__(self, "mlp_memory", MLP_MEMORY_STRICT_AUTO)
            elif self.mlp_memory == MLP_MEMORY_LEGACY_BF16:
                object.__setattr__(self, "mlp_memory", MLP_MEMORY_STRICT_BF16)
            elif self.mlp_memory == MLP_MEMORY_LEGACY_NATIVE:
                object.__setattr__(self, "mlp_memory", MLP_MEMORY_STRICT_NATIVE)

    @property
    def signature(self):
        return (
            self.attention,
            self.fused_qkv,
            self.mlp_memory,
            int(self.chunk_rows),
            bool(self.prefer_held_weights),
            bool(self.strict),
        )


@dataclass(frozen=True)
class SparseRequest:
    """Sparse routing, diagnostics, and optional compilation controls."""

    video_budget: float = 0.5
    density_mode: str = DENSITY_FIXED
    min_video_density: float = 0.05
    max_video_density: float = 1.0
    adaptive_temperature: float = 1.0
    adaptive_target_mass: float = 0.80
    strict: bool = True
    write_report: bool = False
    timing: bool = False
    run_tag: str = "sparse50"
    compile_backend: str = COMPILE_OFF

    def __post_init__(self):
        budget = float(self.video_budget)
        if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("video_budget must be finite and in (0, 1]")
        if self.density_mode not in DENSITY_MODES:
            raise ValueError(
                "density_mode %r is unavailable; implemented modes: %s"
                % (self.density_mode, ", ".join(DENSITY_MODES))
            )

        minimum = float(self.min_video_density)
        maximum = float(self.max_video_density)
        if not math.isfinite(minimum) or not 0.0 < minimum <= 1.0:
            raise ValueError("min_video_density must be finite and in (0, 1]")
        if not math.isfinite(maximum) or not 0.0 < maximum <= 1.0:
            raise ValueError("max_video_density must be finite and in (0, 1]")
        if minimum > maximum:
            raise ValueError("min_video_density must not exceed max_video_density")
        if self.density_mode == DENSITY_ADAPTIVE_BUDGET:
            if budget < minimum or budget > maximum:
                raise ValueError(
                    "adaptive video_budget must lie between min_video_density "
                    "and max_video_density"
                )

        temperature = float(self.adaptive_temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                "adaptive_temperature must be finite and greater than zero"
            )
        target_mass = float(self.adaptive_target_mass)
        if not math.isfinite(target_mass) or not 0.0 < target_mass <= 1.0:
            raise ValueError("adaptive_target_mass must be finite and in (0, 1]")

        tag = str(self.run_tag).strip()
        if _RUN_TAG_RE.fullmatch(tag) is None:
            raise ValueError(
                "run_tag must be 1-64 ASCII letters, digits, underscores, or hyphens"
            )
        if self.compile_backend not in COMPILE_BACKENDS:
            raise ValueError("unknown compile backend %r" % self.compile_backend)
        if (
            self.compile_backend == COMPILE_INDUCTOR
            and self.density_mode != DENSITY_FIXED
        ):
            raise ValueError(
                "shared Inductor compilation currently requires fixed density_mode"
            )

    @property
    def reporting_enabled(self):
        return bool(self.write_report or self.timing)

    @property
    def signature(self):
        return (
            float(self.video_budget),
            self.density_mode,
            float(self.min_video_density),
            float(self.max_video_density),
            float(self.adaptive_temperature),
            float(self.adaptive_target_mass),
            bool(self.strict),
            bool(self.write_report),
            bool(self.timing),
            str(self.run_tag),
            self.compile_backend,
        )


@dataclass(frozen=True)
class H3SageOptimizationPlan:
    """Complete composable request carried by one cloned ModelPatcher."""

    version: int = PLAN_VERSION
    memory: MemoryRequest | None = None
    sparse: SparseRequest | None = None

    def __post_init__(self):
        if int(self.version) != PLAN_VERSION:
            raise ValueError(
                "unsupported H3 Sage optimization plan version %r" % self.version
            )

    def with_memory(self, request: MemoryRequest):
        if not isinstance(request, MemoryRequest):
            raise TypeError("request must be MemoryRequest")
        if self.memory is not None and self.memory != request:
            raise ValueError(
                "a different H3 Sage Memory Optimizer is already present; "
                "remove one instead of relying on node order"
            )
        return replace(self, memory=request)

    def with_sparse(self, request: SparseRequest):
        if not isinstance(request, SparseRequest):
            raise TypeError("request must be SparseRequest")
        if self.sparse is not None and self.sparse != request:
            raise ValueError(
                "a different H3 Sparse Sage node is already present; "
                "remove one instead of relying on node order"
            )
        return replace(self, sparse=request)

    @property
    def signature(self):
        return (
            int(self.version),
            None if self.memory is None else self.memory.signature,
            None if self.sparse is None else self.sparse.signature,
        )


def read_plan(model):
    options = getattr(model, "model_options", {}) or {}
    plan = options.get(PLAN_KEY)
    if plan is None:
        return H3SageOptimizationPlan()
    if not isinstance(plan, H3SageOptimizationPlan):
        raise TypeError("%s does not contain an H3SageOptimizationPlan" % PLAN_KEY)
    return plan
