"""Unified MiniMax H3 memory-optimization orchestration."""

from .attention import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    ATTENTION_SM89,
    FALLBACK_ALLOW,
    FALLBACK_ERROR,
    AttentionDecision,
    RuntimeEnvironment,
    resolve_attention,
)
from .config import MemoryOptimizerConfig
from .patch import apply

__all__ = [
    "ATTENTION_AUTO",
    "ATTENTION_EXISTING",
    "ATTENTION_SM89",
    "FALLBACK_ALLOW",
    "FALLBACK_ERROR",
    "AttentionDecision",
    "RuntimeEnvironment",
    "MemoryOptimizerConfig",
    "resolve_attention",
    "apply",
]
