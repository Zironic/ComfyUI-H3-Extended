"""Resolve and apply the complete two-node H3 Sage optimization plan."""

from __future__ import annotations

from dataclasses import replace
import logging

from .patch import configure_backend
from .plan import (
    ATTENTION_EXISTING,
    FUSED_QKV_OFF,
    FUSED_QKV_REQUIRED,
    PLAN_KEY,
    STATUS_KEY,
    H3SageOptimizationPlan,
)

LOG_PREFIX = "[H3 Sage optimizations]"
ATTENTION_SPARSE = "hybrid_sparse"


def _imports():
    try:
        from ..h3_memory_optimizer.attention import (
            ATTENTION_EXISTING as CORE_ATTENTION_EXISTING,
            AttentionDecision,
            RuntimeEnvironment,
            resolve_attention,
        )
        from ..h3_memory_optimizer.config import (
            ACTIVATION_OFF,
            MemoryOptimizerConfig,
        )
        from ..h3_memory_optimizer.patch import apply
    except ImportError:
        from h3_memory_optimizer.attention import (
            ATTENTION_EXISTING as CORE_ATTENTION_EXISTING,
            AttentionDecision,
            RuntimeEnvironment,
            resolve_attention,
        )
        from h3_memory_optimizer.config import (
            ACTIVATION_OFF,
            MemoryOptimizerConfig,
        )
        from h3_memory_optimizer.patch import apply
    return (
        CORE_ATTENTION_EXISTING,
        AttentionDecision,
        RuntimeEnvironment,
        resolve_attention,
        ACTIVATION_OFF,
        MemoryOptimizerConfig,
        apply,
    )


def _fused_request(plan):
    return FUSED_QKV_OFF if plan.memory is None else plan.memory.fused_qkv


def _dense_fused_support(decision, environment):
    if decision.backend is None:
        return False, "no prepared dense Sage backend was selected"
    if tuple(environment.capability or ()) != (8, 9):
        return False, "dense fused QKV currently requires SM89"
    if getattr(decision.backend, "name", None) != "sage_mem_eff":
        return False, "the selected dense backend does not consume SM89 carriers"
    try:
        from .dense_fused_qkv import TRITON_AVAILABLE
    except Exception as exc:
        return False, "dense fused QKV import failed: %s: %s" % (
            type(exc).__name__, exc
        )
    if not TRITON_AVAILABLE:
        return False, "Triton is unavailable"
    return True, "SM89 dense Sage per-thread Q/K carrier"


def _resolve_dense(plan, environment, resolve_attention):
    memory = plan.memory
    requested = ATTENTION_EXISTING if memory is None else memory.attention
    decision = resolve_attention(
        requested=requested,
        fallback="allow",
        environment=environment,
    )
    fused = _fused_request(plan)
    if fused == FUSED_QKV_OFF:
        return decision, "off", "fused QKV was disabled"

    supported, reason = _dense_fused_support(decision, environment)
    if not supported:
        if fused == FUSED_QKV_REQUIRED:
            raise RuntimeError("required fused QKV is unavailable: %s" % reason)
        return decision, "standard", reason

    from .dense_backend import ProjectedSM89SageBackend
    from .dense_fused_qkv import DenseFusedQKVProjector

    backend = ProjectedSM89SageBackend(decision.backend)
    projector = DenseFusedQKVProjector()
    return (
        replace(
            decision,
            backend=backend,
            projector=projector,
            reason="%s; fused QKV: %s" % (decision.reason, reason),
        ),
        "dense_per_thread",
        reason,
    )


def _sparse_fused_support(environment, kernel_spec):
    if tuple(environment.capability or ()) != (8, 9):
        return False, "Sparse Sage fused QKV currently requires SM89"
    if (
        tuple(kernel_spec.capability) != (8, 9)
        or int(kernel_spec.q_tile) != 128
        or int(kernel_spec.kv_tile) != 64
    ):
        return False, "the selected Sparse Sage ABI is not SM89 128Q x 64KV"
    try:
        from ..h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
    except ImportError:
        from h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
    if not TRITON_AVAILABLE:
        return False, "Triton is unavailable"
    return True, "SM89 Sparse Sage block Q/K carrier"


def _resolve_sparse(plan, environment, AttentionDecision):
    try:
        from ..h3_attention.hybrid import (
            HybridSparseBackend,
            HybridSparseConfig,
            MODE_SAGE128,
            MODE_SAGE128_FUSED_QKV,
            preflight_sparse_sage,
        )
    except ImportError:
        from h3_attention.hybrid import (
            HybridSparseBackend,
            HybridSparseConfig,
            MODE_SAGE128,
            MODE_SAGE128_FUSED_QKV,
            preflight_sparse_sage,
        )

    kernel_spec = preflight_sparse_sage(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    fused = _fused_request(plan)
    supported, fused_reason = _sparse_fused_support(
        environment, kernel_spec
    )
    use_fused = fused != FUSED_QKV_OFF and supported
    if fused == FUSED_QKV_REQUIRED and not supported:
        raise RuntimeError(
            "required Sparse Sage fused QKV is unavailable: %s" % fused_reason
        )

    sparse = plan.sparse
    config = HybridSparseConfig(
        mode=MODE_SAGE128_FUSED_QKV if use_fused else MODE_SAGE128,
        video_budget=float(sparse.video_budget),
        density_mode=sparse.density_mode,
        strict=True,
        run_tag="production",
        timing=False,
    )
    if use_fused:
        from .sparse_projector import SparseFusedQKVProjector

        projector = SparseFusedQKVProjector()
    else:
        projector = None
    backend = HybridSparseBackend(
        config,
        kernel_spec=kernel_spec,
        projector=projector,
    )
    decision = AttentionDecision(
        requested=ATTENTION_SPARSE,
        selected=ATTENTION_SPARSE,
        backend=backend,
        adapter=ATTENTION_SPARSE,
        reason="explicit fixed-density Sparse Sage attention",
        environment=environment,
        projector=backend.projector,
    )
    if use_fused:
        return decision, "sparse_block", fused_reason
    if fused == FUSED_QKV_OFF:
        return decision, "off", "fused QKV was disabled"
    return decision, "standard", fused_reason


def _optimizer_config(plan, MemoryOptimizerConfig, ACTIVATION_OFF):
    memory = plan.memory
    return MemoryOptimizerConfig(
        attention=(
            ATTENTION_EXISTING if memory is None else memory.attention
        ),
        attention_fallback="allow",
        activation=(
            ACTIVATION_OFF if memory is None else memory.activation
        ),
        chunk_rows=(2048 if memory is None else int(memory.chunk_rows)),
        prefer_held_weights=(
            True if memory is None else bool(memory.prefer_held_weights)
        ),
        activation_strict=False,
        adaln_precompute="off",
        block_cache="off",
        cuda_async_soft_gc=False,
        timing=False,
    )


def _status(plan, decision, qkv_selected, qkv_reason, result):
    environment = decision.environment
    return {
        "plan_version": int(plan.version),
        "plan_signature": plan.signature,
        "attention": {
            "requested": decision.requested,
            "selected": decision.selected,
            "reason": decision.reason,
            "patched_blocks": int(result.attention_blocks),
        },
        "fused_qkv": {
            "requested": _fused_request(plan),
            "selected": qkv_selected,
            "reason": qkv_reason,
            "projector": getattr(decision.projector, "name", None),
        },
        "mlp": None if plan.memory is None else {
            "mode": plan.memory.activation,
            "chunk_rows": int(plan.memory.chunk_rows),
            "prefer_held_weights": bool(plan.memory.prefer_held_weights),
            "patched_blocks": int(result.activation_blocks),
        },
        "sparse": None if plan.sparse is None else {
            "video_budget": float(plan.sparse.video_budget),
            "density_mode": plan.sparse.density_mode,
        },
        "runtime_installed": bool(result.runtime_installed),
        "device": {
            "name": environment.device_name,
            "architecture": environment.architecture,
            "capability": (
                None
                if environment.capability is None
                else [int(value) for value in environment.capability]
            ),
        },
    }


def apply_plan(model, plan: H3SageOptimizationPlan):
    """Clone a model and reconcile the full plan as one transaction."""

    if not isinstance(plan, H3SageOptimizationPlan):
        raise TypeError("plan must be H3SageOptimizationPlan")
    (
        _core_existing,
        AttentionDecision,
        RuntimeEnvironment,
        resolve_attention,
        ACTIVATION_OFF,
        MemoryOptimizerConfig,
        apply,
    ) = _imports()

    environment = RuntimeEnvironment.detect()
    if plan.sparse is not None:
        decision, qkv_selected, qkv_reason = _resolve_sparse(
            plan, environment, AttentionDecision
        )
    else:
        decision, qkv_selected, qkv_reason = _resolve_dense(
            plan, environment, resolve_attention
        )

    config = _optimizer_config(plan, MemoryOptimizerConfig, ACTIVATION_OFF)
    patched = model.clone()
    result = apply(
        patched,
        config=config,
        decision=decision,
        attention_configurer=configure_backend,
        pool_policy=None,
    )
    patched.model_options[PLAN_KEY] = plan
    options = patched.model_options["transformer_options"] = (
        patched.model_options.get("transformer_options", {}).copy()
    )
    options[STATUS_KEY] = _status(
        plan, decision, qkv_selected, qkv_reason, result
    )
    logging.info(
        "%s armed: attention=%s fused_qkv=%s mlp=%s device=%s",
        LOG_PREFIX,
        decision.selected,
        qkv_selected,
        "off" if plan.memory is None else plan.memory.activation,
        environment.device_name,
    )
    return patched
