"""Compatibility names for the H3-Optimizations plan contract."""

try:
    from ..h3_optimizations_dependency import dependency_module
except ImportError:
    from h3_optimizations_dependency import dependency_module

_plan = dependency_module("plan")

ATTENTION_AUTO = _plan.ATTENTION_AUTO
ATTENTION_EXISTING = _plan.ATTENTION_EXISTING
ATTENTION_REQUESTS = _plan.ATTENTION_REQUESTS
FUSED_QKV_AUTO = _plan.FUSED_QKV_AUTO
FUSED_QKV_OFF = _plan.FUSED_QKV_OFF
FUSED_QKV_REQUIRED = _plan.FUSED_QKV_REQUIRED
FUSED_QKV_REQUESTS = _plan.FUSED_QKV_REQUESTS
MLP_MEMORY_AUTO = _plan.MLP_MEMORY_AUTO
MLP_MEMORY_EPILOGUE = "epilogue_prototype"
MLP_MEMORY_OFF = _plan.MLP_MEMORY_OFF
MLP_MEMORY_LEGACY_BF16 = _plan.MLP_MEMORY_LEGACY_BF16
MLP_MEMORY_LEGACY_NATIVE = _plan.MLP_MEMORY_LEGACY_NATIVE
MLP_MEMORY_LEGACY_CONVROT_REQUIRED = _plan.MLP_MEMORY_LEGACY_CONVROT_REQUIRED
MLP_MEMORY_REQUESTS = (*_plan.MLP_MEMORY_REQUESTS, MLP_MEMORY_EPILOGUE)
DENSITY_FIXED = _plan.DENSITY_FIXED
DEFAULT_VIDEO_BUDGET = _plan.DEFAULT_VIDEO_BUDGET
DEFAULT_EDGE_STEPS = _plan.DEFAULT_EDGE_STEPS
DEFAULT_EDGE_KV = _plan.DEFAULT_EDGE_KV
PLAN_KEY = _plan.PLAN_KEY
STATUS_KEY = _plan.STATUS_KEY
PLAN_VERSION = _plan.PLAN_VERSION
MemoryRequest = _plan.MemoryRequest
SparseRequest = _plan.SparseRequest
H3SageOptimizationPlan = _plan.H3OptimizationPlan
read_plan = _plan.read_plan

__all__ = [
    "ATTENTION_AUTO",
    "ATTENTION_EXISTING",
    "FUSED_QKV_AUTO",
    "FUSED_QKV_OFF",
    "FUSED_QKV_REQUIRED",
    "MLP_MEMORY_AUTO",
    "MLP_MEMORY_EPILOGUE",
    "MLP_MEMORY_OFF",
    "MLP_MEMORY_LEGACY_BF16",
    "MLP_MEMORY_LEGACY_NATIVE",
    "MLP_MEMORY_LEGACY_CONVROT_REQUIRED",
    "DENSITY_FIXED",
    "DEFAULT_VIDEO_BUDGET",
    "DEFAULT_EDGE_STEPS",
    "DEFAULT_EDGE_KV",
    "PLAN_KEY",
    "STATUS_KEY",
    "PLAN_VERSION",
    "MemoryRequest",
    "SparseRequest",
    "H3SageOptimizationPlan",
    "read_plan",
]
