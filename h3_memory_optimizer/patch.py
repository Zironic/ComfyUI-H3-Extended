"""Apply H3 memory and acceleration patches as one coordinated transaction."""

from dataclasses import dataclass
import logging

try:
    from ..h3_activation_memory.patch import install as install_activation
    from ..h3_adaln.patch import install as install_adaln
    from ..h3_attention.config import configure_backend
    from ..h3_block_cache.patch import install as install_block_cache
    from ..h3_runtime.context import (
        H3RuntimeSession,
        RUNTIME_SESSION_KEY,
        install_runtime_wrapper,
    )
    from ..h3_runtime.compile_compat import configure_shared_block_inductor
except ImportError:
    from h3_activation_memory.patch import install as install_activation
    from h3_adaln.patch import install as install_adaln
    from h3_attention.config import configure_backend
    from h3_block_cache.patch import install as install_block_cache
    from h3_runtime.context import (
        H3RuntimeSession,
        RUNTIME_SESSION_KEY,
        install_runtime_wrapper,
    )
    from h3_runtime.compile_compat import configure_shared_block_inductor

from .attention import ATTENTION_EXISTING, ATTENTION_SOL, resolve_attention
from .config import MemoryOptimizerConfig
from .timing import MemoryOptimizerTimingListener

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


def _backend_flag(backend, name, default=False):
    return bool(getattr(backend, name, default)) if backend is not None else bool(default)


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
    timing_listener=None,
):
    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    status = {
        "attention_requested": result.attention_requested,
        "attention_selected": result.attention_selected,
        "attention_reason": result.attention_reason,
        "attention_blocks": result.attention_blocks,
        "attention_approximate": _backend_flag(attention_backend, "approximate"),
        "attention_backend": _component_status(attention_backend),
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
        "timing": _component_status(timing_listener),
    }
    if pool_policy is not None:
        status["cuda_async_pool_policy"] = pool_policy.as_status()
    options[STATUS_KEY] = status

    # Keep the live objects available to diagnostics without serializing them
    # into the human-readable status dictionary.
    if runtime_session is not None:
        options[RUNTIME_SESSION_KEY] = runtime_session
    if adaln_provider is not None:
        options["minimax_h3_adaln_provider"] = adaln_provider
    if block_cache is not None:
        options["minimax_h3_first_block_cache"] = block_cache
    if result.attention_selected == ATTENTION_SOL:
        options["minimax_h3_sol_backend"] = attention_backend
    if timing_listener is not None:
        options["minimax_h3_memory_optimizer_timing"] = timing_listener


def _reuse_or_install_runtime(model_patcher, listeners, strict_layout, runtime_installer):
    """Return one shared H3 runtime session for every acceleration feature.

    Model-patch nodes may be ordered either before or after the memory optimizer.
    Reusing an already-published session avoids duplicate OUTER_SAMPLE and
    DIFFUSION_MODEL/APPLY_MODEL wrappers and keeps request ids authoritative.
    """
    options = model_patcher.model_options.get("transformer_options", {})
    existing = options.get(RUNTIME_SESSION_KEY)
    if existing is not None:
        if not hasattr(existing, "add_listener"):
            raise TypeError(
                "%s is not an H3 runtime session" % RUNTIME_SESSION_KEY
            )
        if strict_layout:
            existing.strict_layout = True
        for listener in listeners:
            existing.add_listener(listener)
        return existing, False

    session = H3RuntimeSession(
        strict_layout=bool(strict_layout),
        listeners=listeners,
    )
    runtime_installer(model_patcher, session)
    return session, True


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
        if decision.projector is None:
            _backend, attention_blocks = attention_configurer(
                model_patcher, decision.backend
            )
        else:
            _backend, attention_blocks = attention_configurer(
                model_patcher, decision.backend, projector=decision.projector
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
    listeners.extend(
        item for item in getattr(decision.backend, "runtime_listeners", ())
        if item is not None and item not in listeners
    )
    timing_listener = None
    if config.timing:
        timing_listener = MemoryOptimizerTimingListener(
            config.timing_report_directory,
            decision.selected,
            decision.reason,
        )
        listeners.append(timing_listener)

    configure_shared_block_inductor(
        model_patcher,
        backend=decision.backend,
        activation_config=activation_config,
        adaln_provider=adaln_provider,
        block_cache=cache_coordinator,
    )
    runtime_needed = bool(
        listeners or _backend_flag(decision.backend, "requires_runtime_context")
    )
    runtime_session = None
    runtime_created = False
    if runtime_needed:
        runtime_session, runtime_created = _reuse_or_install_runtime(
            model_patcher,
            listeners,
            _backend_flag(decision.backend, "strict_runtime_layout"),
            runtime_installer,
        )

    result = ApplyResult(
        attention_requested=decision.requested,
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
        timing_listener=timing_listener,
    )

    level = logging.INFO if decision.selected != ATTENTION_EXISTING else logging.WARNING
    logging.log(
        level,
        "%s armed: attention=%s requested=%s blocks=%d activation=%s blocks=%d "
        "adaln=%s blocks=%d first_block_cache=%s blocks=%d runtime=%s created=%s "
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
        runtime_created,
        result.device_name,
        result.architecture,
    )
    return result
