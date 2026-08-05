"""Backend protocol for H3-owned block attention.

A prepared backend is intentionally split into two phases. ``prepare`` may read
post-RoPE Q/K/V views. The caller then deletes those views, releasing the fused
BF16 QKV projection, before ``execute`` starts the attention kernel.
"""

from typing import Protocol, runtime_checkable, Any


@runtime_checkable
class H3AttentionBackend(Protocol):
    """Two-stage attention backend consumed by ``h3_attention.forward``."""

    name: str

    def prepare(self, q, k, v, *, layer_index: int, transformer_options: dict) -> Any:
        """Return state independent of the original fused QKV storage."""

    def execute(self, prepared):
        """Return HND output ``[batch, heads, sequence, head_dim]``."""
