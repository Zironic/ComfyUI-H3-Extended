"""Incremental mixed-dispatch attention backend for MiniMax H3."""

from .adaptive import (
    AdaptiveDensityError,
    DensityPlan,
    allocate_adaptive_rows,
    resolve_density_plan,
)
from .backend import HybridSparseBackend, PreparedHybrid
from .config import (
    DENSITY_ADAPTIVE_BUDGET,
    DENSITY_FIXED,
    DENSITY_MODES,
    HybridSparseConfig,
    IMPLEMENTED_MODES,
    MODE_SAGE128,
    MODE_SAGE128_FUSED_QKV,
)
from .fused_qkv import (
    FusedQKVError,
    FusedQKVProjector,
    PreparedFusedQKV,
    validate_prepared_fused_qkv,
)
from .router import (
    KV_TILE,
    Q_TILE,
    SparseMaskMetadata,
    SparseRouterError,
    SparseTileGeometry,
    SparseTileRouter,
)
from .sparse_sage import (
    PreparedSparseSage,
    SparseSageKernelSpec,
    SparseSageError,
    SparseSageExecutor,
    load_sparse_sage_spec,
    preflight_sparse_sage,
    resolve_sparse_sage_spec,
)
from .stats import DeferredCudaTiming, HybridStatsCollector, TIMING_STAGES

__all__ = [
    "AdaptiveDensityError",
    "DensityPlan",
    "allocate_adaptive_rows",
    "resolve_density_plan",
    "HybridSparseBackend",
    "PreparedHybrid",
    "HybridSparseConfig",
    "DENSITY_FIXED",
    "DENSITY_ADAPTIVE_BUDGET",
    "DENSITY_MODES",
    "IMPLEMENTED_MODES",
    "MODE_SAGE128",
    "MODE_SAGE128_FUSED_QKV",
    "FusedQKVError",
    "FusedQKVProjector",
    "PreparedFusedQKV",
    "validate_prepared_fused_qkv",
    "KV_TILE",
    "Q_TILE",
    "SparseMaskMetadata",
    "SparseRouterError",
    "SparseTileGeometry",
    "SparseTileRouter",
    "PreparedSparseSage",
    "SparseSageKernelSpec",
    "SparseSageError",
    "SparseSageExecutor",
    "load_sparse_sage_spec",
    "preflight_sparse_sage",
    "resolve_sparse_sage_spec",
    "HybridStatsCollector",
    "DeferredCudaTiming",
    "TIMING_STAGES",
]
