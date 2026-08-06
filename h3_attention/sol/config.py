"""Configuration for the optional approximate Sol-Attn H3 backend."""

from dataclasses import dataclass
import math

try:
    from ...h3_runtime.layout import SINK_MODES, SINK_PREFIX
except ImportError:
    from h3_runtime.layout import SINK_MODES, SINK_PREFIX


@dataclass(frozen=True)
class SolAttentionConfig:
    tau: float = 1.0
    thresh_type: str = "diag"
    dense_steps: int = 10
    dense_layers: int = 2
    sink_mode: str = SINK_PREFIX
    correctness_gate: bool = True
    strict: bool = False
    kv_splits: int = 1
    gate_heads: int = 0
    density_heads: int = 0
    # Full-reference long-form chunks can place most of the sequence before
    # the target-video tail. Decline to dense prepared attention when the exact
    # sink would consume more than half the packed sequence.
    max_sink_fraction: float = 0.5

    def __post_init__(self):
        if not math.isfinite(float(self.tau)):
            raise ValueError("Sol tau must be finite")
        if self.thresh_type not in ("diag", "exact"):
            raise ValueError("Sol thresh_type must be 'diag' or 'exact'")
        if int(self.dense_steps) < 0 or int(self.dense_layers) < 0:
            raise ValueError("Sol dense_steps and dense_layers must be non-negative")
        if self.sink_mode not in SINK_MODES:
            raise ValueError("unknown Sol sink_mode %r" % self.sink_mode)
        if int(self.kv_splits) not in (1, 2, 4):
            raise ValueError("Sol kv_splits must be 1, 2, or 4")
        if int(self.gate_heads) < 0 or int(self.density_heads) < 0:
            raise ValueError("Sol gate/density head counts must be non-negative")
        if not 0.0 <= float(self.max_sink_fraction) <= 1.0:
            raise ValueError("Sol max_sink_fraction must be in [0, 1]")

    @property
    def signature(self):
        return (
            float(self.tau), self.thresh_type, int(self.dense_steps),
            int(self.dense_layers), self.sink_mode,
            bool(self.correctness_gate), bool(self.strict),
            int(self.kv_splits), int(self.gate_heads),
            int(self.density_heads), float(self.max_sink_fraction),
        )
