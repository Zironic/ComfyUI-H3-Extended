"""Immutable configuration for the unified H3 memory/acceleration optimizer."""

from dataclasses import dataclass
import math

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS, IMPLEMENTED_MODES, ActivationMemoryConfig
    from ..h3_adaln import AdaLNPrecomputeConfig
    from ..h3_block_cache import FirstBlockCacheConfig
    from ..h3_runtime.layout import SINK_PREFIX
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS, IMPLEMENTED_MODES, ActivationMemoryConfig
    from h3_adaln import AdaLNPrecomputeConfig
    from h3_block_cache import FirstBlockCacheConfig
    from h3_runtime.layout import SINK_PREFIX

from .attention import ATTENTION_AUTO, ATTENTION_MODES, FALLBACK_ALLOW, FALLBACK_MODES

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

    sol_tau: float = 1.0
    sol_thresh_type: str = "diag"
    sol_dense_steps: int = 10
    sol_dense_layers: int = 2
    sol_sink_mode: str = SINK_PREFIX
    sol_correctness_gate: bool = True
    sol_gate_heads: int = 4
    sol_density_heads: int = 4
    sol_strict: bool = False
    sol_kv_splits: int = 1
    sol_max_sink_fraction: float = 0.5

    adaln_precompute: str = "off"
    adaln_max_table_gib: float = 2.0
    adaln_strict: bool = False

    block_cache: str = "off"
    block_cache_threshold: float = 0.08
    block_cache_warmup_steps: int = 3
    block_cache_strict: bool = False

    def __post_init__(self):
        if self.attention not in ATTENTION_MODES:
            raise ValueError("unknown attention mode %r" % self.attention)
        if self.attention_fallback not in FALLBACK_MODES:
            raise ValueError("unknown attention fallback %r" % self.attention_fallback)
        if self.activation not in ACTIVATION_MODES:
            raise ValueError("unknown activation mode %r" % self.activation)
        threshold = float(self.cuda_async_release_threshold_gib)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("cuda_async_release_threshold_gib must be finite and greater than zero")
        if self.activation != ACTIVATION_OFF:
            self.activation_config()
        self.sol_config()
        self.adaln_config()
        self.block_cache_config()

    def activation_config(self):
        if self.activation == ACTIVATION_OFF:
            return None
        return ActivationMemoryConfig(
            mode=self.activation,
            chunk_rows=int(self.chunk_rows),
            strict=bool(self.activation_strict),
            prefer_held_weights=bool(self.prefer_held_weights),
        )

    def sol_config(self):
        try:
            from ..h3_attention.sol import SolAttentionConfig
        except ImportError:
            from h3_attention.sol import SolAttentionConfig
        return SolAttentionConfig(
            tau=float(self.sol_tau),
            thresh_type=self.sol_thresh_type,
            dense_steps=int(self.sol_dense_steps),
            dense_layers=int(self.sol_dense_layers),
            sink_mode=self.sol_sink_mode,
            correctness_gate=bool(self.sol_correctness_gate),
            gate_heads=int(self.sol_gate_heads),
            density_heads=int(self.sol_density_heads),
            strict=bool(self.sol_strict),
            kv_splits=int(self.sol_kv_splits),
            max_sink_fraction=float(self.sol_max_sink_fraction),
        )

    def attention_options(self):
        config = self.sol_config()
        return {
            "tau": config.tau,
            "thresh_type": config.thresh_type,
            "dense_steps": config.dense_steps,
            "dense_layers": config.dense_layers,
            "sink_mode": config.sink_mode,
            "correctness_gate": config.correctness_gate,
            "gate_heads": config.gate_heads,
            "density_heads": config.density_heads,
            "strict": config.strict,
            "kv_splits": config.kv_splits,
            "max_sink_fraction": config.max_sink_fraction,
        }

    def adaln_config(self):
        return AdaLNPrecomputeConfig(
            mode=self.adaln_precompute,
            max_table_gib=float(self.adaln_max_table_gib),
            strict=bool(self.adaln_strict),
        )

    def block_cache_config(self):
        return FirstBlockCacheConfig(
            mode=self.block_cache,
            threshold=float(self.block_cache_threshold),
            warmup_steps=int(self.block_cache_warmup_steps),
            strict=bool(self.block_cache_strict),
        )
