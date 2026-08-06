"""Unified MiniMax H3 memory and acceleration orchestration."""

from .attention import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    ATTENTION_SM80,
    ATTENTION_SM86,
    ATTENTION_SM89,
    ATTENTION_SM90,
    ATTENTION_SM12X,
    ATTENTION_SOL,
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
    "ATTENTION_SM80",
    "ATTENTION_SM86",
    "ATTENTION_SM89",
    "ATTENTION_SM90",
    "ATTENTION_SM12X",
    "ATTENTION_SOL",
    "FALLBACK_ALLOW",
    "FALLBACK_ERROR",
    "AttentionDecision",
    "RuntimeEnvironment",
    "MemoryOptimizerConfig",
    "resolve_attention",
    "apply",
]
