"""Immutable, order-independent configuration for H3 Sage optimizations."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

PLAN_KEY = "minimax_h3_sage_optimization_plan"
STATUS_KEY = "minimax_h3_sage_optimization_status"
PLAN_VERSION = 2

ATTENTION_AUTO = "auto"
ATTENTION_EXISTING = "existing"
ATTENTION_REQUESTS = (ATTENTION_AUTO, ATTENTION_EXISTING)

FUSED_QKV_AUTO = "auto"
FUSED_QKV_OFF = "off"
FUSED_QKV_REQUIRED = "required"
FUSED_QKV_REQUESTS = (FUSED_QKV_AUTO, FUSED_QKV_OFF, FUSED_QKV_REQUIRED)

MLP_MEMORY_AUTO = "auto"
MLP_MEMORY_OFF = "off"
MLP_MEMORY_REQUESTS = (MLP_MEMORY_AUTO, MLP_MEMORY_OFF)

DENSITY_FIXED = "fixed"


@dataclass(frozen=True)
class MemoryRequest:
    """Execution/memory options owned by the Memory Optimizer node."""

    attention: str = ATTENTION_AUTO
    fused_qkv: str = FUSED_QKV_AUTO
    mlp_memory: str = MLP_MEMORY_AUTO
    chunk_rows: int = 2048
    prefer_held_weights: bool = True

    def __post_init__(self):
        if self.attention not in ATTENTION_REQUESTS:
            raise ValueError("unknown H3 Sage attention request %r" % self.attention)
        if self.fused_qkv not in FUSED_QKV_REQUESTS:
            raise ValueError("unknown fused QKV request %r" % self.fused_qkv)
        if self.mlp_memory not in MLP_MEMORY_REQUESTS:
            raise ValueError("unknown MLP memory request %r" % self.mlp_memory)
        if int(self.chunk_rows) <= 0:
            raise ValueError("chunk_rows must be positive")

    @property
    def signature(self):
        return (
            self.attention,
            self.fused_qkv,
            self.mlp_memory,
            int(self.chunk_rows),
            bool(self.prefer_held_weights),
        )


@dataclass(frozen=True)
class SparseRequest:
    """Approximate attention options owned by the Sparse Sage node."""

    video_budget: float = 0.5
    density_mode: str = DENSITY_FIXED

    def __post_init__(self):
        budget = float(self.video_budget)
        if not math.isfinite(budget) or not 0.0 < budget <= 1.0:
            raise ValueError("video_budget must be finite and in (0, 1]")
        if self.density_mode != DENSITY_FIXED:
            raise ValueError(
                "the production Sparse Sage node currently supports fixed density only"
            )

    @property
    def signature(self):
        return (float(self.video_budget), self.density_mode)


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
    """Return the immutable plan already attached to a ModelPatcher."""

    options = getattr(model, "model_options", {}) or {}
    plan = options.get(PLAN_KEY)
    if plan is None:
        return H3SageOptimizationPlan()
    if not isinstance(plan, H3SageOptimizationPlan):
        raise TypeError("%s does not contain an H3SageOptimizationPlan" % PLAN_KEY)
    return plan
