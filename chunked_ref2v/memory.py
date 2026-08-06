"""Arm the unified memory optimizer once for a whole harness run.

The harness samples a chunk per arm plus Chunk A, so attention and activation
memory decide whether a suite finishes at all on a 12 GB card. This module is
the harness's single entry point to `h3_memory_optimizer`: the capability
resolver picks an architecture-matched prepared-QKV Sage backend, and the same
patched MODEL is then shared by Chunk A and every Chunk B arm.

Two properties matter more here than in an ordinary graph.

**Arm once, share everywhere.** Resolution and patch installation happen before
Phase A, so an `error` fallback refuses the run in the first second rather than
after Chunk A has been generated. Chunk A and every arm therefore run the *same*
attention backend, so the peak-VRAM and runtime columns are comparable across
arms, and the 50 block forwards are patched once rather than once per experiment.
`patch_target_conditions` clones from the armed model and patches a different
key (`extra_conds`), so the attention patches ride through untouched.

**Say which backend produced the numbers.** A capability fallback is silent by
design - it preserves the incoming attention rather than failing a long run -
but a resource metric attributed to the wrong backend is worse than no metric.
The decision is recorded and surfaced in the report next to the numbers it
explains.
"""

import logging

try:
    from ..h3_memory_optimizer.attention import (
        ATTENTION_AUTO,
        ATTENTION_EXISTING,
        ATTENTION_MODES,
        FALLBACK_ALLOW,
        FALLBACK_MODES,
        resolve_attention,
    )
    from ..h3_memory_optimizer.config import (
        ACTIVATION_MODES,
        ACTIVATION_OFF,
        MemoryOptimizerConfig,
    )
    from ..h3_memory_optimizer.cuda_pool import configure_cuda_async_soft_gc
    from ..h3_memory_optimizer.patch import STATUS_KEY, apply
except ImportError:  # self-tests import the pack as top-level modules
    from h3_memory_optimizer.attention import (
        ATTENTION_AUTO,
        ATTENTION_EXISTING,
        ATTENTION_MODES,
        FALLBACK_ALLOW,
        FALLBACK_MODES,
        resolve_attention,
    )
    from h3_memory_optimizer.config import (
        ACTIVATION_MODES,
        ACTIVATION_OFF,
        MemoryOptimizerConfig,
    )
    from h3_memory_optimizer.cuda_pool import configure_cuda_async_soft_gc
    from h3_memory_optimizer.patch import STATUS_KEY, apply

LOG_PREFIX = "[H3 Extended] harness"

__all__ = [
    "ATTENTION_MODES", "ACTIVATION_MODES", "FALLBACK_MODES",
    "ATTENTION_AUTO", "ATTENTION_EXISTING", "ACTIVATION_OFF", "FALLBACK_ALLOW",
    "STATUS_KEY", "arm", "existing_status", "describe",
]


def existing_status(model):
    """The optimizer status an incoming MODEL already carries, if any."""
    try:
        return model.model_options.get("transformer_options", {}).get(STATUS_KEY)
    except Exception:
        return None


def arm(model, *, attention=ATTENTION_AUTO, attention_fallback=FALLBACK_ALLOW,
        activation="mlp_chunked_bf16", chunk_rows=None,
        cuda_async_soft_gc=False, cuda_async_release_threshold_gib=11.0,
        resolver=resolve_attention, applier=apply,
        pool_configurer=configure_cuda_async_soft_gc, environment=None):
    """Return `(model, status)` with the optimizer armed for the whole run.

    `status` is always a dict describing what happened, including the cases
    where nothing was armed - the report needs to distinguish "ran on optimized
    Sage" from "ran on whatever the graph already had" from "deliberately
    disabled for an A/B".
    """
    if attention not in ATTENTION_MODES:
        raise ValueError("unknown attention mode %r" % attention)
    if attention_fallback not in FALLBACK_MODES:
        raise ValueError("unknown attention fallback %r" % attention_fallback)
    if activation not in ACTIVATION_MODES:
        raise ValueError("unknown activation mode %r" % activation)

    inherited = existing_status(model)
    if inherited is not None:
        # The graph already ran MiniMaxH3MemoryOptimizerZi. Re-applying the same
        # backend is a no-op and a different one raises, so neither is worth
        # risking mid-run: inherit and report what is actually installed.
        status = dict(inherited)
        status["armed_by"] = "incoming model"
        logging.info("%s memory optimizer inherited from the graph: attention=%s "
                     "activation=%s", LOG_PREFIX,
                     status.get("attention_selected"), status.get("activation_mode"))
        return model, status

    if attention == ATTENTION_EXISTING and activation == ACTIVATION_OFF:
        logging.warning("%s memory optimizer disabled (attention=existing, "
                        "activation=off) - resource metrics describe the graph's "
                        "own attention backend", LOG_PREFIX)
        return model, {
            "armed_by": "disabled",
            "attention_requested": attention,
            "attention_selected": ATTENTION_EXISTING,
            "attention_reason": "harness memory optimizer disabled",
            "activation_mode": ACTIVATION_OFF,
        }

    config_kwargs = {
        "attention": attention,
        "attention_fallback": attention_fallback,
        "activation": activation,
        "cuda_async_soft_gc": bool(cuda_async_soft_gc),
        "cuda_async_release_threshold_gib": float(cuda_async_release_threshold_gib),
    }
    if chunk_rows:
        config_kwargs["chunk_rows"] = int(chunk_rows)
    config = MemoryOptimizerConfig(**config_kwargs)

    decision = resolver(config.attention, config.attention_fallback,
                        **({"environment": environment} if environment is not None else {}))
    pool_policy = pool_configurer(
        config.cuda_async_soft_gc,
        config.cuda_async_release_threshold_gib,
        device_index=decision.environment.device_index,
    )

    patched = model.clone()
    result = applier(patched, config=config, decision=decision,
                     pool_policy=pool_policy)

    status = dict(existing_status(patched) or {})
    status["armed_by"] = "harness"
    status.setdefault("attention_requested", result.attention_requested)
    status.setdefault("attention_selected", result.attention_selected)
    status.setdefault("attention_reason", result.attention_reason)

    if status.get("attention_selected") == ATTENTION_EXISTING:
        # Not fatal - all arms still share one backend, so the comparison
        # between them holds. But the absolute VRAM and runtime figures no
        # longer describe the optimized path, and a report that did not say so
        # would invite exactly the wrong conclusion.
        logging.warning(
            "%s memory optimizer fell back to existing attention (%s); arm-to-arm "
            "comparison is still valid, absolute resource figures are not "
            "attributable to optimized Sage", LOG_PREFIX,
            status.get("attention_reason"))
    return patched, status


def describe(status):
    """One line for the log and the report header."""
    if not status:
        return "memory optimizer: not armed"
    selected = status.get("attention_selected")
    parts = ["attention=%s" % selected]
    if status.get("attention_requested") not in (None, selected):
        parts.append("(requested %s)" % status["attention_requested"])
    if status.get("attention_blocks"):
        parts.append("on %d blocks" % status["attention_blocks"])
    parts.append("activation=%s" % status.get("activation_mode"))
    if status.get("activation_blocks"):
        parts.append("on %d blocks" % status["activation_blocks"])
    if status.get("architecture"):
        parts.append("arch=%s" % status["architecture"])
    parts.append("armed by %s" % status.get("armed_by", "?"))
    if selected == ATTENTION_EXISTING and status.get("attention_reason"):
        parts.append("- fallback reason: %s" % status["attention_reason"])
    return "memory optimizer: " + " ".join(parts)


def is_optimized(status):
    """True when an architecture-matched Sage backend is actually installed."""
    return bool(status) and status.get("attention_selected") not in (
        None, ATTENTION_EXISTING)
