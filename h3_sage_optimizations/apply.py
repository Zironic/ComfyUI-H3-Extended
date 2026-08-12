"""Resolve and apply the complete two-node H3 Sage optimization plan."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from .dense_resolver import resolve_dense_attention
from .environment import RuntimeEnvironment
from .qkv.formats import inspect_h3_linears
from .model import get_h3_blocks, is_minimax_h3
from .patch import configure_backend
from .plan import (
    ATTENTION_EXISTING,
    COMPILE_INDUCTOR,
    COMPILE_OFF,
    FUSED_QKV_OFF,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    PLAN_KEY,
    STATUS_KEY,
    H3SageOptimizationPlan,
)
from .qkv.providers import (
    MLP_CONVROT_INT8_TWO_SLICE,
    MLP_OFF,
    QKV_DENSE_CONVROT_INT8,
    QKV_SPARSE_CONVROT_INT8,
    resolve_mlp_provider,
    resolve_qkv_provider,
)

LOG_PREFIX = "[H3 Sage optimizations]"
ATTENTION_SPARSE = "sparse_sage"


@dataclass(frozen=True)
class ResolvedAttention:
    requested: str
    selected: str
    backend: object | None
    reason: str
    backend_kind: str
    projector: object | None = None


@dataclass(frozen=True)
class ResolvedCompile:
    requested: str
    state: str
    reason: str

    @property
    def ready(self):
        return self.state == "ready"


def _fused_request(plan):
    return FUSED_QKV_OFF if plan.memory is None else plan.memory.fused_qkv


def _dense_triton_available():
    try:
        from .dense_fused_qkv import TRITON_AVAILABLE
    except Exception:
        return False
    return bool(TRITON_AVAILABLE)


def _sparse_triton_available():
    try:
        from ..h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
    except ImportError:
        try:
            from h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
        except Exception:
            return False
    except Exception:
        return False
    return bool(TRITON_AVAILABLE)


def _sparse_report_root():
    import folder_paths

    return os.path.join(
        folder_paths.get_output_directory(),
        "h3_sparse_sage",
    )


def _resolve_dense(plan, environment, inventory):
    memory = plan.memory
    requested = (
        ATTENTION_EXISTING if memory is None else memory.attention
    )
    dense = resolve_dense_attention(requested, environment)
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind=dense.backend_kind,
        capability=environment.capability,
        triton_available=_dense_triton_available(),
    )
    backend = dense.backend
    projector = None
    if qkv.provider_id == QKV_DENSE_CONVROT_INT8:
        from .dense_backend import ProjectedSM89SageBackend
        from .qkv.projectors import DenseFusedQKVProjector

        backend = ProjectedSM89SageBackend(backend)
        projector = DenseFusedQKVProjector(
            required=_fused_request(plan) == "required"
        )
    return (
        ResolvedAttention(
            requested=dense.requested,
            selected=dense.selected,
            backend=backend,
            reason=dense.reason,
            backend_kind=dense.backend_kind,
            projector=projector,
        ),
        qkv,
    )


def _resolve_sparse(plan, environment, inventory):
    try:
        from ..h3_attention.hybrid import (
            HybridSparseBackend,
            HybridSparseConfig,
            HybridStatsCollector,
            MODE_SAGE128,
            MODE_SAGE128_FUSED_QKV,
            preflight_sparse_sage,
        )
    except ImportError:
        from h3_attention.hybrid import (
            HybridSparseBackend,
            HybridSparseConfig,
            HybridStatsCollector,
            MODE_SAGE128,
            MODE_SAGE128_FUSED_QKV,
            preflight_sparse_sage,
        )

    sparse = plan.sparse
    kernel_spec = preflight_sparse_sage(
        cuda_available=lambda: environment.cuda_available,
        capability_getter=lambda: environment.capability,
    )
    qkv = resolve_qkv_provider(
        inventory,
        request=_fused_request(plan),
        backend_kind="sparse_sage",
        capability=environment.capability,
        triton_available=_sparse_triton_available(),
        sparse_spec=kernel_spec,
    )
    use_fused = qkv.provider_id == QKV_SPARSE_CONVROT_INT8
    config = HybridSparseConfig(
        mode=(
            MODE_SAGE128_FUSED_QKV
            if use_fused
            else MODE_SAGE128
        ),
        video_budget=float(sparse.video_budget),
        density_mode=sparse.density_mode,
        min_video_density=float(sparse.min_video_density),
        max_video_density=float(sparse.max_video_density),
        adaptive_temperature=float(sparse.adaptive_temperature),
        adaptive_target_mass=float(sparse.adaptive_target_mass),
        strict=bool(sparse.strict),
        run_tag=str(sparse.run_tag),
        timing=bool(sparse.timing),
    )
    projector = None
    if use_fused:
        from .qkv.projectors import SparseFusedQKVProjector

        projector = SparseFusedQKVProjector(
            required=_fused_request(plan) == "required"
        )
    collector = (
        HybridStatsCollector(_sparse_report_root(), config.run_tag)
        if sparse.reporting_enabled
        else None
    )
    backend = HybridSparseBackend(
        config,
        kernel_spec=kernel_spec,
        projector=projector,
        collector=collector,
    )
    return (
        ResolvedAttention(
            requested=ATTENTION_SPARSE,
            selected=ATTENTION_SPARSE,
            backend=backend,
            reason=(
                "explicit %s Sparse Sage attention"
                % config.density_mode
            ),
            backend_kind="sparse_sage",
            projector=projector,
        ),
        qkv,
    )


def _resolve_mlp(plan, inventory):
    memory = plan.memory
    request = "off" if memory is None else memory.mlp_memory
    return resolve_mlp_provider(inventory, request=request)


def _install_mlp(model_patcher, memory, resolution):
    if memory is None or resolution.provider_id == MLP_OFF:
        return resolution, 0, None

    try:
        from ..h3_activation_memory.config import ActivationMemoryConfig
        from ..h3_activation_memory.patch import install
    except ImportError:
        from h3_activation_memory.config import ActivationMemoryConfig
        from h3_activation_memory.patch import install

    strict = memory.mlp_memory in (
        MLP_MEMORY_EPILOGUE,
        MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    )
    config = ActivationMemoryConfig(
        mode=resolution.activation_mode,
        chunk_rows=int(memory.chunk_rows),
        strict=bool(strict),
        prefer_held_weights=bool(memory.prefer_held_weights),
    )
    return resolution, int(install(model_patcher, config)), config


def _resolve_compile(plan, qkv, mlp):
    sparse = plan.sparse
    requested = (
        COMPILE_OFF if sparse is None else sparse.compile_backend
    )
    if requested == COMPILE_OFF:
        return ResolvedCompile(
            requested=COMPILE_OFF,
            state="off",
            reason="shared block compilation was disabled",
        )
    if requested != COMPILE_INDUCTOR:
        raise ValueError("unknown compile backend %r" % requested)

    if plan.memory is None:
        return ResolvedCompile(
            requested=requested,
            state="pending",
            reason=(
                "waiting for an H3 Sage Memory Optimizer to provide fused "
                "Sparse QKV and the ConvRot two-slice MLP"
            ),
        )
    if qkv.provider_id != QKV_SPARSE_CONVROT_INT8:
        raise RuntimeError(
            "shared Inductor compilation requires fused Sparse Sage QKV; "
            "resolved provider is %s (%s)"
            % (qkv.provider_id, qkv.reason)
        )
    if mlp.provider_id != MLP_CONVROT_INT8_TWO_SLICE:
        raise RuntimeError(
            "shared Inductor compilation requires the ConvRot two-slice MLP; "
            "resolved provider is %s (%s)"
            % (mlp.provider_id, mlp.reason)
        )
    return ResolvedCompile(
        requested=requested,
        state="ready",
        reason=(
            "fixed Sparse Sage with fused QKV and ConvRot two-slice MLP"
        ),
    )


def _request_compile(model_patcher):
    try:
        from ..h3_runtime.compile_compat import request_shared_block_compile
    except ImportError:
        from h3_runtime.compile_compat import request_shared_block_compile

    request_shared_block_compile(model_patcher)


def _configure_compile(model_patcher, attention, activation_config):
    try:
        from ..h3_runtime.compile_compat import (
            configure_shared_block_inductor,
        )
    except ImportError:
        from h3_runtime.compile_compat import (
            configure_shared_block_inductor,
        )

    return bool(
        configure_shared_block_inductor(
            model_patcher,
            backend=attention.backend,
            activation_config=activation_config,
        )
    )


def _ensure_sparse_runtime(
    model_patcher,
    backend,
    *,
    strict_layout,
):
    try:
        from ..h3_runtime.context import (
            H3RuntimeSession,
            RUNTIME_SESSION_KEY,
            install_runtime_wrapper,
        )
    except ImportError:
        from h3_runtime.context import (
            H3RuntimeSession,
            RUNTIME_SESSION_KEY,
            install_runtime_wrapper,
        )

    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get(
            "transformer_options", {}
        ).copy()
    )
    session = options.get(RUNTIME_SESSION_KEY)
    listeners = tuple(
        listener
        for listener in getattr(
            backend, "runtime_listeners", ()
        )
        if listener is not None
    )
    if session is not None:
        if not hasattr(session, "add_listener"):
            raise TypeError(
                "%s is not an H3 runtime session"
                % RUNTIME_SESSION_KEY
            )
        session.strict_layout = bool(strict_layout)
        if listeners:
            replacement_types = tuple(type(item) for item in listeners)
            session.listeners[:] = [
                item
                for item in session.listeners
                if not isinstance(item, replacement_types)
            ]
            for listener in listeners:
                session.add_listener(listener)
        return session, False

    session = H3RuntimeSession(
        strict_layout=bool(strict_layout),
        listeners=listeners,
    )
    install_runtime_wrapper(model_patcher, session)
    return session, True


def _inventory_status(inventory):
    return {
        "qkv": list(inventory.labels("qkv")),
        "fc1": list(inventory.labels("fc1")),
        "fc2": list(inventory.labels("fc2")),
    }


def _sparse_status(sparse):
    if sparse is None:
        return None
    return {
        "video_budget": float(sparse.video_budget),
        "density_mode": sparse.density_mode,
        "min_video_density": float(sparse.min_video_density),
        "max_video_density": float(sparse.max_video_density),
        "adaptive_temperature": float(sparse.adaptive_temperature),
        "adaptive_target_mass": float(sparse.adaptive_target_mass),
        "strict": bool(sparse.strict),
        "write_report": bool(sparse.write_report),
        "timing": bool(sparse.timing),
        "reporting_enabled": bool(sparse.reporting_enabled),
        "run_tag": str(sparse.run_tag),
        "compile_backend": sparse.compile_backend,
    }


def _status(
    plan,
    environment,
    attention,
    qkv,
    mlp,
    compile_resolution,
    *,
    attention_blocks,
    mlp_blocks,
    runtime_installed,
    compile_configured,
    inventory,
):
    return {
        "plan_version": int(plan.version),
        "plan_signature": plan.signature,
        "attention": {
            "requested": attention.requested,
            "selected": attention.selected,
            "reason": attention.reason,
            "patched_blocks": int(attention_blocks),
        },
        "fused_qkv": {
            "requested": _fused_request(plan),
            "provider": qkv.provider_id,
            "fused": bool(qkv.fused),
            "reason": qkv.reason,
            "projector": getattr(
                attention.projector, "name", None
            ),
        },
        "mlp": {
            "requested": (
                "off"
                if plan.memory is None
                else plan.memory.mlp_memory
            ),
            "provider": mlp.provider_id,
            "activation_mode": mlp.activation_mode,
            "reason": mlp.reason,
            "chunk_rows": (
                None
                if plan.memory is None
                else int(plan.memory.chunk_rows)
            ),
            "patched_blocks": int(mlp_blocks),
        },
        "sparse": _sparse_status(plan.sparse),
        "compile": {
            "requested": compile_resolution.requested,
            "state": (
                "configured"
                if compile_configured
                else compile_resolution.state
            ),
            "reason": compile_resolution.reason,
        },
        "weight_formats": _inventory_status(inventory),
        "runtime_installed": bool(runtime_installed),
        "device": {
            "name": environment.device_name,
            "architecture": environment.architecture,
            "capability": (
                None
                if environment.capability is None
                else [
                    int(value)
                    for value in environment.capability
                ]
            ),
        },
    }


def apply_plan(model, plan: H3SageOptimizationPlan):
    """Apply only compatible H3 features; unknown models are exact no-ops."""

    if not isinstance(plan, H3SageOptimizationPlan):
        raise TypeError("plan must be H3SageOptimizationPlan")
    if not is_minimax_h3(model):
        return model

    blocks = get_h3_blocks(model)
    inventory = inspect_h3_linears(blocks)
    environment = RuntimeEnvironment.detect()

    if plan.sparse is not None:
        attention, qkv = _resolve_sparse(
            plan, environment, inventory
        )
    else:
        attention, qkv = _resolve_dense(
            plan, environment, inventory
        )
    mlp = _resolve_mlp(plan, inventory)
    compile_resolution = _resolve_compile(plan, qkv, mlp)

    patched = model.clone()
    if compile_resolution.ready:
        _request_compile(patched)

    attention_blocks = 0
    if attention.backend is not None:
        _backend, attention_blocks = configure_backend(
            patched,
            attention.backend,
            projector=attention.projector,
        )

    mlp, mlp_blocks, activation_config = _install_mlp(
        patched, plan.memory, mlp
    )
    compile_configured = False
    if compile_resolution.ready:
        compile_configured = _configure_compile(
            patched,
            attention,
            activation_config,
        )

    runtime_installed = False
    if plan.sparse is not None:
        _session, _created = _ensure_sparse_runtime(
            patched,
            attention.backend,
            strict_layout=plan.sparse.strict,
        )
        runtime_installed = True

    patched.model_options[PLAN_KEY] = plan
    options = patched.model_options["transformer_options"] = (
        patched.model_options.get(
            "transformer_options", {}
        ).copy()
    )
    options[STATUS_KEY] = _status(
        plan,
        environment,
        attention,
        qkv,
        mlp,
        compile_resolution,
        attention_blocks=attention_blocks,
        mlp_blocks=mlp_blocks,
        runtime_installed=runtime_installed,
        compile_configured=compile_configured,
        inventory=inventory,
    )
    logging.info(
        "%s armed: attention=%s qkv=%s mlp=%s compile=%s device=%s",
        LOG_PREFIX,
        attention.selected,
        qkv.provider_id,
        mlp.provider_id,
        (
            "configured"
            if compile_configured
            else compile_resolution.state
        ),
        environment.device_name,
    )
    return patched
