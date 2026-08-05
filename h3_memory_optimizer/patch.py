"""Apply efficient attention and activation-memory patches as one transaction."""

from dataclasses import dataclass
import logging

try:
    from ..h3_activation_memory.patch import install as install_activation
    from ..h3_attention.config import configure_backend
except ImportError:
    from h3_activation_memory.patch import install as install_activation
    from h3_attention.config import configure_backend

from .attention import ATTENTION_EXISTING, resolve_attention
from .config import MemoryOptimizerConfig

LOG_PREFIX = "[H3 memory optimizer]"
STATUS_KEY = "minimax_h3_memory_optimizer"


@dataclass(frozen=True)
class ApplyResult:
    attention_requested: str
    attention_selected: str
    attention_reason: str
    attention_blocks: int
    activation_mode: str
    activation_blocks: int
    architecture: str
    device_name: str


def _record_status(model_patcher, result, config, pool_policy=None):
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    status = {
        "attention_requested": result.attention_requested,
        "attention_selected": result.attention_selected,
        "attention_reason": result.attention_reason,
        "attention_blocks": result.attention_blocks,
        "activation_mode": result.activation_mode,
        "activation_blocks": result.activation_blocks,
        "chunk_rows": int(config.chunk_rows),
        "architecture": result.architecture,
        "device_name": result.device_name,
        "cuda_async_soft_gc": bool(config.cuda_async_soft_gc),
        "cuda_async_release_threshold_gib": float(
            config.cuda_async_release_threshold_gib
        ),
    }
    if pool_policy is not None:
        status["cuda_async_pool_policy"] = pool_policy.as_status()
    options[STATUS_KEY] = status


def apply(model_patcher, config=None, decision=None,
          attention_configurer=configure_backend,
          activation_installer=install_activation,
          pool_policy=None):
    """Install both optimizations on one already-cloned ModelPatcher.

    Resolution is performed before patch installation. Unsupported attention
    environments therefore leave the incoming attention backend untouched while
    activation chunking can still be installed. Structural conflicts and core
    drift are not swallowed; those are graph errors rather than capability
    fallbacks.
    """
    config = config or MemoryOptimizerConfig()
    if not isinstance(config, MemoryOptimizerConfig):
        raise TypeError("config must be MemoryOptimizerConfig")
    decision = decision or resolve_attention(
        config.attention,
        config.attention_fallback,
    )

    attention_blocks = 0
    if decision.backend is not None:
        _backend, attention_blocks = attention_configurer(
            model_patcher, decision.backend
        )

    activation_blocks = 0
    activation_config = config.activation_config()
    if activation_config is not None:
        activation_blocks = activation_installer(model_patcher, activation_config)

    result = ApplyResult(
        attention_requested=config.attention,
        attention_selected=decision.selected,
        attention_reason=decision.reason,
        attention_blocks=int(attention_blocks),
        activation_mode=config.activation,
        activation_blocks=int(activation_blocks),
        architecture=decision.environment.architecture,
        device_name=decision.environment.device_name,
    )
    _record_status(
        model_patcher,
        result,
        config,
        pool_policy=pool_policy,
    )

    level = logging.INFO if decision.selected != ATTENTION_EXISTING else logging.WARNING
    logging.log(
        level,
        "%s armed: attention=%s (requested=%s, reason=%s) attention_blocks=%d "
        "activation=%s activation_blocks=%d chunk_rows=%d cuda_async_soft_gc=%s "
        "device=%s arch=%s",
        LOG_PREFIX,
        result.attention_selected,
        result.attention_requested,
        result.attention_reason,
        result.attention_blocks,
        result.activation_mode,
        result.activation_blocks,
        int(config.chunk_rows),
        bool(config.cuda_async_soft_gc),
        result.device_name,
        result.architecture,
    )
    return result
