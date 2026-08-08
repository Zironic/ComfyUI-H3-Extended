"""H3-owned attention observation and dense/sparse prepared backends."""

from .observer import (
    OBSERVER_KEY,
    OBSERVED_KEY,
    notify_attention,
    observing,
    marked_observed,
)
from . import sage_mem_eff as _sage_mem_eff
from . import sm89_compat as _sm89_compat  # installs hashed torch.ops discovery
from . import v_snapshot_compat as _v_snapshot_compat  # defers stock FP8-V preparation
from .sage_arch import (
    SageSM80MemoryEfficientBackend,
    SageSM86MemoryEfficientBackend,
    SageSM90MemoryEfficientBackend,
    SageSM12xMemoryEfficientBackend,
)
from .sol import (
    DenseBF16SDPABackend,
    SolAttentionBackend,
    SolAttentionConfig,
    SolAttentionError,
    preflight_sol_attention,
)
from .hybrid import (
    HybridSparseBackend,
    HybridSparseConfig,
    HybridStatsCollector,
    SparseSageError,
    preflight_sparse_sage,
)

SM89SageMemoryEfficientBackend = _sage_mem_eff.SM89SageMemoryEfficientBackend

__all__ = [
    "OBSERVER_KEY",
    "OBSERVED_KEY",
    "notify_attention",
    "observing",
    "marked_observed",
    "SageSM80MemoryEfficientBackend",
    "SageSM86MemoryEfficientBackend",
    "SM89SageMemoryEfficientBackend",
    "SageSM90MemoryEfficientBackend",
    "SageSM12xMemoryEfficientBackend",
    "DenseBF16SDPABackend",
    "SolAttentionBackend",
    "SolAttentionConfig",
    "SolAttentionError",
    "preflight_sol_attention",
    "HybridSparseBackend",
    "HybridSparseConfig",
    "HybridStatsCollector",
    "SparseSageError",
    "preflight_sparse_sage",
]
