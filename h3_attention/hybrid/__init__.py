"""Incremental mixed-dispatch attention backend for MiniMax H3."""

from .backend import HybridSparseBackend, PreparedHybrid
from .config import HybridSparseConfig, IMPLEMENTED_MODES, MODE_SAGE128
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
    SparseSageAPI,
    SparseSageError,
    SparseSageExecutor,
    load_sparse_sage_api,
    preflight_sparse_sage,
)
from .stats import HybridStatsCollector

__all__ = [
    "HybridSparseBackend",
    "PreparedHybrid",
    "HybridSparseConfig",
    "IMPLEMENTED_MODES",
    "MODE_SAGE128",
    "KV_TILE",
    "Q_TILE",
    "SparseMaskMetadata",
    "SparseRouterError",
    "SparseTileGeometry",
    "SparseTileRouter",
    "PreparedSparseSage",
    "SparseSageAPI",
    "SparseSageError",
    "SparseSageExecutor",
    "load_sparse_sage_api",
    "preflight_sparse_sage",
    "HybridStatsCollector",
]
