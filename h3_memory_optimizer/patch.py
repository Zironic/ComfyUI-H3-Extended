"""Apply H3 memory and acceleration patches as one coordinated transaction."""

from dataclasses import dataclass
import logging

try:
    from ..h3_activation_memory.patch import install as install_activation
    from ..h3_adaln.patch import install as install_adaln
    from ..h3_attention.config import configure_backend
    from ..h3_block_cache.patch import install as install_block_cache
    from ..h3_runtime.context import H3RuntimeSession, install_runtime_wrapper
except ImportError:
    from h3_activation_memory.patch import install as install_activation
    from h3_adaln.patch import install as install_adaln
    from h3_attention.config import configure_backend
    from h3_block_cache.patch import install as install_block_cache
    from h3_runtime.context import H3RuntimeSession, install_runtime_wrapper

from .attention import ATTENTION_EXISTING, ATTENTION_SOL, resolve_attention
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
    adaln_mode: str
    adaln_blocks: int
    block_cache_mode: str
    block_cache_blocks: int
    runtime_installed: bool
    architecture: str
    device_name: str


def _component_status(component):
    if component is None:
        return None
    callback = getattr(component, "as_status", None)
    if callable(callback):
        try:
            return callback()
        except Exception as exc:
            return {"status_error": "%s: %s" % (type(exc).__name__, exc)}
    return {"type": type(component).__name__}


def _record_status(
    model_patcher,
    result,
    config,
    *,
    pool_policy=None,
    runtime_session=None,
    adaln_provider=None,
    block_cache=None,
    attention_backend=None,
):
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    status = {
        "attention_requested": result.attention_requested,
        "attention_selected": result.attention_selected,
        "attention_reason": result.attention_reason,
        "attention_blocks": result.attention_blocks,
        "attention_approximate": result.attention_selected == ATTENTION_SOL,
        "activation_mode": result.activation_mode,
        "activation_blocks": result.activation_blocks,
        "chunk_rows": int(config.chunk_rows),
        "adaln_mode": result.adaln_mode,
        "adaln_blocks": result.adaln_blocks,
        "block_cache_mode": result.block_cache_mode,
        "block_cache_blocks": result.block_cache_blocks,
        "block_cache_approximate": result.block_cache_mode != "off",
        "runtime_installed": result.runtime_installed,
        "architecture": result.architecture,
        "device_name": result.device_name,
        "cuda_async_soft_gc": bool(config.cuda_async_soft_gc),
        "cuda_async_release_threshold_gib": float(
            config.cuda_async_release_threshold_gib
        ),
        "sol": _component_status(attention_backend)
        if result.attention_selected == ATTENTION_SOL
        else None,
        "adaln": _component_status(adaln_provider),
        "first_block_cache": _component_status(block_cache),
    }
    if pool_policy is not None:
        status["cuda_async_pool_policy"] = pool_policy.as_status()
    options[STATUS_KEY] = status

    # Keep the live objects available to diagnostics without serializing them
    # into the human-readable status dictionary.
    if runtime_session is not None:
        options["minimax_h3_runtime_session"] = runtime_session
    if adaln_provider is not None:
        options["minimax_h3_adaln_provider"] = adaln_provider
    if block_cache is not None:
        options["minimax_h3_first_block_cache"] = block_cache
    if result.attention_selected == ATTENTION_SOL:
        options["minimax_h3_sol_backend"] = attention_backend


def apply(
    model_patcher,
    config=None,
    decision=None,
    *,
    attention_configurer=configure_backend,
    activation_installer=install_activation,
    adaln_installer=install_adaln,
    block_cache_installer=install_block_cache,
    runtime_installer=install_runtime_wrapper,
    pool_policy=None,
):
    """Install all requested components on one already-cloned ModelPatcher.

    Attention capability resolution happens before this function.  The dense
    default therefore stays unchanged when optional Sol-Attn is unavailable.
    Approximate features are explicit and never enabled by ``attention=auto``.
    """
    config = config or MemoryOptimizerConfig()
    if not isinstance(config, MemoryOptimizerConfig):
        raise TypeError("config must be MemoryOptimizerConfig")
    decision = decision or resolve_attention(
        config.attention,
        config.attention_fallback,
        adapter_options=config.attention_options(),
    )

    # The order is load-bearing: block.forward must see the final attention
    # forward, and FirstBlockCache must wrap the final activation-memory block.
    attention_blocks = 0
    if decision.backend is not None:
        _backend, attention_blocks = attention_configurer(
            model_patcher, decision.backend
        )

    activation_blocks = 0
    activation_config = config.activation_config()
    if activation_config is not None:
        activation_blocks = activation_installer(model_patcher, activation_config)

    adaln_provider, adaln_blocks = adaln_installer(
        model_patcher,
        config.adaln_config(),
    )
    cache_coordinator, cache_blocks = block_cache_installer(
        model_patcher,
        config.block_cache_config(),
    )

    listeners = [
        item for item in (adaln_provider, cache_coordinator)
        if item is not None
    ]
    runtime_needed = bool(
        listeners or decision.selected == ATTENTION_SOL
    )
    runtime_session = None
    if runtime_needed:
        runtime_session = H3RuntimeSession(
            strict_layout=bool(config.sol_strict and decision.selected == ATTENTION_SOL),
            listeners=listeners,
        )
        runtime_installer(model_patcher, runtime_session)

    result = ApplyResult(
        attention_requested=config.attention,
        attention_selected=decision.selected,
        attention_reason=decision.reason,
        attention_blocks=int(attention_blocks),
        activation_mode=config.activation,
        activation_blocks=int(activation_blocks),
        adaln_mode=config.adaln_precompute,
        adaln_blocks=int(adaln_blocks),
        block_cache_mode=config.block_cache,
        block_cache_blocks=int(cache_blocks),
        runtime_installed=runtime_session is not None,
        architecture=decision.environment.architecture,
        device_name=decision.environment.device_name,
    )
    _record_status(
        model_patcher,
        result,
        config,
        pool_policy=pool_policy,
        runtime_session=runtime_session,
        adaln_provider=adaln_provider,
        block_cache=cache_coordinator,
        attention_backend=decision.backend,
    )

    level = logging.INFO if decision.selected != ATTENTION_EXISTING else logging.WARNING
    logging.log(
        level,
        "%s armed: attention=%s requested=%s blocks=%d activation=%s blocks=%d "
        "adaln=%s blocks=%d first_block_cache=%s blocks=%d runtime=%s "
        "device=%s arch=%s",
        LOG_PREFIX,
        result.attention_selected,
        result.attention_requested,
        result.attention_blocks,
        result.activation_mode,
        result.activation_blocks,
        result.adaln_mode,
        result.adaln_blocks,
        result.block_cache_mode,
        result.block_cache_blocks,
        result.runtime_installed,
        result.device_name,
        result.architecture,
    )
    return result
