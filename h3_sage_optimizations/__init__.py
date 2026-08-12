"""Composable dense/fused/MLP and Sparse Sage optimizations for MiniMax H3."""

from .plan import (
    H3SageOptimizationPlan,
    MemoryRequest,
    SparseRequest,
)
from .qkv.formats import (
    H3LinearInventory,
    LinearWeightFormat,
)

__all__ = [
    "H3SageOptimizationPlan",
    "MemoryRequest",
    "SparseRequest",
    "H3LinearInventory",
    "LinearWeightFormat",
]
