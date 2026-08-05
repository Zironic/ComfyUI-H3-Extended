"""Immutable configuration for the unified H3 memory optimizer."""

from dataclasses import dataclass
import math

try:
    from ..h3_activation_memory.config import (
        DEFAULT_CHUNK_ROWS,
        IMPLEMENTED_MODES,
        ActivationMemoryConfig,
    )
except ImportError:
    from h3_activation_memory.config import (
        DEFAULT_CHUNK_ROWS,
        IMPLEMENTED_MODES,
        ActivationMemoryConfig,
    )

from .attention import (
    ATTENTION_AUTO,
    ATTENTION_MODES,
    FALLBACK_ALLOW,
    FALLBACK_MODES,
)

ACTIVATION_OFF = "off"
ACTIVATION_MODES = (ACTIVATION_OFF, *sorted(IMPLEMENTED_MODES))


@dataclass(frozen=True)
class MemoryOptimizerConfig:
    attention: str = ATTENTION_AUTO
    attention_fallback: str = FALLBACK_ALLOW
    activation: str = "mlp_chunked_bf16"
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    prefer_held_weights: bool = True
    activation_strict: bool = False
    cuda_async_soft_gc: bool = False
    cuda_async_release_threshold_gib: float = 11.0

    def __post_init__(self):
        if self.attention not in ATTENTION_MODES:
            raise ValueError("unknown attention mode %r" % self.attention)
        if self.attention_fallback not in FALLBACK_MODES:
            raise ValueError("unknown attention fallback %r" % self.attention_fallback)
        if self.activation not in ACTIVATION_MODES:
            raise ValueError("unknown activation mode %r" % self.activation)
        threshold = float(self.cuda_async_release_threshold_gib)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError(
                "cuda_async_release_threshold_gib must be finite and greater than zero"
            )
        if self.activation != ACTIVATION_OFF:
            self.activation_config()

    def activation_config(self):
        if self.activation == ACTIVATION_OFF:
            return None
        return ActivationMemoryConfig(
            mode=self.activation,
            chunk_rows=int(self.chunk_rows),
            strict=bool(self.activation_strict),
            prefer_held_weights=bool(self.prefer_held_weights),
        )
