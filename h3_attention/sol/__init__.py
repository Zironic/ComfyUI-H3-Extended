"""Optional approximate Sol-Attn backend for MiniMax H3."""

from .backend import (
    DenseBF16SDPABackend,
    SolAttentionBackend,
    SolAttentionError,
    load_sol_attention,
    preflight_sol_attention,
)
from .config import SolAttentionConfig
from . import stats

__all__ = [
    "DenseBF16SDPABackend",
    "SolAttentionBackend",
    "SolAttentionError",
    "SolAttentionConfig",
    "load_sol_attention",
    "preflight_sol_attention",
    "stats",
]
