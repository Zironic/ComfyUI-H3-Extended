"""Deprecated compatibility adapter for the former combined H3 node."""

import os

import folder_paths

from comfy_api.latest import ComfyExtension, io, ui

try:
    from ..h3_attention.hybrid import (
        DENSITY_ADAPTIVE_BUDGET,
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        preflight_sparse_sage,
    )
    from ..h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
    from ..h3_memory_optimizer.attention import (
        ATTENTION_EXISTING as LEGACY_ATTENTION_EXISTING,
        AttentionDecision,
        RuntimeEnvironment,
    )
    from ..h3_memory_optimizer.config import MemoryOptimizerConfig
    from ..h3_memory_optimizer.patch import apply as apply_legacy
    from ..h3_sage_optimizations.apply import apply_plan
    from ..h3_sage_optimizations.model import get_h3_blocks
    from ..h3_sage_optimizations.plan import (
        ATTENTION_EXISTING,
        FUSED_QKV_AUTO,
        FUSED_QKV_OFF,
        FUSED_QKV_REQUIRED,
        MLP_MEMORY_AUTO,
        MLP_MEMORY_LEGACY_BF16,
        MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
        MLP_MEMORY_LEGACY_NATIVE,
        MLP_MEMORY_OFF,
        MemoryRequest,
        SparseRequest,
        read_plan,
    )
    from ..h3_optimizations_dependency import dependency_module
    from ..h3_sage_optimizations.status import (
        format_disabled_status,
        format_legacy_status,
    )
except ImportError:
    from h3_attention.hybrid import (
        DENSITY_ADAPTIVE_BUDGET,
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        preflight_sparse_sage,
    )
    from h3_attention.hybrid.fused_qkv import TRITON_AVAILABLE
    from h3_memory_optimizer.attention import (
        ATTENTION_EXISTING as LEGACY_ATTENTION_EXISTING,
        AttentionDecision,
        RuntimeEnvironment,
    )
    from h3_memory_optimizer.config import MemoryOptimizerConfig
    from h3_memory_optimizer.patch import apply as apply_legacy
    from h3_sage_optimizations.apply import apply_plan
    from h3_sage_optimizations.model import get_h3_blocks
    from h3_sage_optimizations.plan import (
        ATTENTION_EXISTING,
        FUSED_QKV_AUTO,
        FUSED_QKV_OFF,
        FUSED_QKV_REQUIRED,
        MLP_MEMORY_AUTO,
        MLP_MEMORY_LEGACY_BF16,
        MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
        MLP_MEMORY_LEGACY_NATIVE,
        MLP_MEMORY_OFF,
        MemoryRequest,
        SparseRequest,
        read_plan,
    )
    from h3_optimizations_dependency import dependency_module
    from h3_sage_optimizations.status import (
        format_disabled_status,
        format_legacy_status,
    )

ATTENTION_HYBRID = "hybrid_sparse"
inspect_h3_linears = dependency_module("qkv.formats").inspect_h3_linears
resolve_qkv_provider = dependency_module("qkv.providers").resolve_qkv_provider


def _output_root():
    return os.path.join(folder_paths.get_output_directory(), "h3_hybrid_sparse")

MODE_SAGE128 = "sage128"
MODE_SAGE128_FUSED_QKV = "sage128_fused_qkv"
MODE_AUTO = "auto"
IMPLEMENTED_MODES = (MODE_AUTO, MODE_SAGE128, MODE_SAGE128_FUSED_QKV)

ACTIVATION_OFF = "off"
MODE_BF16 = "mlp_chunked_bf16"
MODE_NATIVE = "mlp_chunked_native"
MODE_CONVROT_2SLICE = "mlp_chunked_convrot_2slice"
ACTIVATION_MODES = (
    ACTIVATION_OFF,
    MODE_BF16,
    MODE_CONVROT_2SLICE,
    MODE_NATIVE,
)

DENSITY_FIXED = "fixed"
DENSITY_ADAPTIVE_BUDGET = "adaptive_budget"
DENSITY_MODES = (DENSITY_FIXED, DENSITY_ADAPTIVE_BUDGET)

DEFAULT_CHUNK_ROWS = 2048
MIN_CHUNK_ROWS = 256


def _memory_request(mode, activation, strict, chunk_rows):
    if mode == MODE_AUTO:
        fused_qkv = FUSED_QKV_AUTO
    elif mode == MODE_SAGE128:
        fused_qkv = FUSED_QKV_OFF
    elif mode == MODE_SAGE128_FUSED_QKV:
        fused_qkv = FUSED_QKV_REQUIRED if strict else FUSED_QKV_AUTO
    else:
        raise ValueError("unknown Hybrid Sparse mode %r" % mode)

    if activation == ACTIVATION_OFF:
        mlp_memory = MLP_MEMORY_OFF
    elif activation == MODE_BF16:
        mlp_memory = MLP_MEMORY_LEGACY_BF16
    elif activation == MODE_NATIVE:
        mlp_memory = MLP_MEMORY_LEGACY_NATIVE
    elif activation == MODE_CONVROT_2SLICE:
        mlp_memory = (
            MLP_MEMORY_LEGACY_CONVROT_REQUIRED
            if strict
            else MLP_MEMORY_AUTO
        )
    else:
        raise ValueError("unknown legacy activation mode %r" % activation)

    return MemoryRequest(
        attention=ATTENTION_EXISTING,
        fused_qkv=fused_qkv,
        mlp_memory=mlp_memory,
        chunk_rows=int(chunk_rows),
        prefer_held_weights=True,
        mlp_strict=bool(strict),
    )


def _resolve_adaptive_mode(model, mode, environment, kernel_spec):
    if mode != MODE_AUTO:
        return mode
    qkv = resolve_qkv_provider(
        inspect_h3_linears(get_h3_blocks(model)),
        request=FUSED_QKV_AUTO,
        backend_kind="sparse_sage",
        triton_available=TRITON_AVAILABLE,
        sparse_spec=kernel_spec,
    )
    return MODE_SAGE128_FUSED_QKV if qkv.fused else MODE_SAGE128


class MiniMaxH3HybridSparseAttention(io.ComfyNode):
    """Translate the former combined node into the two production requests."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridSparseAttentionZi",
            display_name=(
                "MiniMax H3 Hybrid Sparse Attention "
                "(Deprecated Compatibility)"
            ),
            category="model/patch/minimax/compatibility",
            description=(
                "Deprecated compatibility adapter for saved workflows. It "
                "translates the former combined fixed-density Sparse Sage, "
                "fused-QKV, and MLP controls into the new Memory Optimizer and "
                "Sparse Sage requests. Eager adaptive routing remains available "
                "for legacy workflows; shared compilation still requires the "
                "production fixed-density path."
            ),
            search_aliases=[
                "H3 Hybrid Sparse",
                "old H3 sparse",
                "legacy H3 sparse",
                "MiniMax H3 sparse compatibility",
            ],
            is_deprecated=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enable",
                    default=True,
                ),
                io.Combo.Input(
                    "mode",
                    display_name="QKV projection mode",
                    options=list(IMPLEMENTED_MODES),
                    default=MODE_AUTO,
                    tooltip=(
                        "auto prefers fused QKV only when the checkpoint "
                        "format, GPU, Triton, and Sparse Sage ABI are "
                        "compatible; otherwise it uses standard H3 QKV."
                    ),
                ),
                io.Float.Input(
                    "video_budget",
                    display_name="Video KV budget",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Requested pure-video KV tile fraction. The production "
                        "fixed-density route rounds up to a whole tile count."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="Require legacy specialized paths",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "When enabled, an explicitly requested fused-QKV or "
                        "ConvRot two-slice path must pass preflight. When "
                        "disabled, incompatible specialized paths may fall back."
                    ),
                ),
                io.Combo.Input(
                    "activation",
                    display_name="Legacy MLP mode",
                    options=list(ACTIVATION_MODES),
                    default=MODE_NATIVE,
                    advanced=True,
                ),
                io.Int.Input(
                    "chunk_rows",
                    display_name="MLP chunk rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=MIN_CHUNK_ROWS,
                    max=16384,
                    step=256,
                    advanced=True,
                ),
                io.String.Input(
                    "run_tag",
                    display_name="Legacy run tag (ignored)",
                    default="hybrid50",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "timing",
                    display_name="Legacy timing reports (ignored)",
                    default=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "compile_backend",
                    display_name="Legacy compile backend",
                    options=["off", "inductor"],
                    default="off",
                    advanced=True,
                ),
                io.Combo.Input(
                    "density_mode",
                    display_name="Legacy density mode",
                    options=list(DENSITY_MODES),
                    default=DENSITY_FIXED,
                    advanced=True,
                ),
                io.Float.Input(
                    "min_video_density",
                    display_name="Legacy adaptive minimum",
                    default=0.05,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_video_density",
                    display_name="Legacy adaptive maximum",
                    default=0.50,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_temperature",
                    display_name="Legacy adaptive temperature",
                    default=1.0,
                    min=0.05,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_target_mass",
                    display_name="Legacy adaptive target mass",
                    default=0.80,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        mode=MODE_AUTO,
        video_budget=0.5,
        strict=True,
        activation=MODE_NATIVE,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        run_tag="hybrid50",
        timing=True,
        compile_backend="off",
        density_mode=DENSITY_FIXED,
        min_video_density=0.05,
        max_video_density=0.50,
        adaptive_temperature=1.0,
        adaptive_target_mass=0.80,
    ):
        if not enabled:
            return io.NodeOutput(
                model,
                ui=ui.PreviewText(
                    format_disabled_status(
                        "MiniMax H3 Hybrid Sparse compatibility adapter"
                    )
                ),
            )

        if compile_backend not in ("off", "inductor"):
            raise ValueError(
                "unknown compile backend %r" % compile_backend
            )
        if not MIN_CHUNK_ROWS <= int(chunk_rows) <= 16384:
            raise ValueError(
                "chunk_rows must be between %d and 16384, got %r"
                % (MIN_CHUNK_ROWS, chunk_rows)
            )
        if int(chunk_rows) % 256:
            raise ValueError(
                "chunk_rows must be a multiple of 256, got %r"
                % chunk_rows
            )

        if compile_backend == "inductor":
            if density_mode != DENSITY_FIXED:
                raise ValueError(
                    "Inductor shared-block compilation currently requires "
                    "fixed density_mode"
                )
            if mode != MODE_SAGE128_FUSED_QKV:
                raise ValueError(
                    "Inductor requires the sage128_fused_qkv attention mode"
                )
            if activation != MODE_CONVROT_2SLICE:
                raise ValueError(
                    "Inductor requires mlp_chunked_convrot_2slice activation"
                )
            raise ValueError(
                "The deprecated compatibility adapter does not support shared "
                "Inductor compilation. Migrate the workflow to the two "
                "production nodes."
            )

        if density_mode == DENSITY_ADAPTIVE_BUDGET:
            optimizer_config = MemoryOptimizerConfig(
                attention=LEGACY_ATTENTION_EXISTING,
                activation=activation,
                chunk_rows=int(chunk_rows),
                activation_strict=bool(strict),
            )
            environment = RuntimeEnvironment.detect()
            kernel_spec = preflight_sparse_sage(
                cuda_available=lambda: environment.cuda_available,
                capability_getter=lambda: environment.capability,
            )
            resolved_mode = _resolve_adaptive_mode(
                model, mode, environment, kernel_spec,
            )
            hybrid_config = HybridSparseConfig(
                mode=resolved_mode,
                video_budget=float(video_budget),
                density_mode=str(density_mode),
                min_video_density=float(min_video_density),
                max_video_density=float(max_video_density),
                adaptive_temperature=float(adaptive_temperature),
                adaptive_target_mass=float(adaptive_target_mass),
                strict=bool(strict),
                run_tag=run_tag,
                timing=bool(timing),
            )
            collector = HybridStatsCollector(
                _output_root(), hybrid_config.run_tag
            )
            backend = HybridSparseBackend(
                hybrid_config,
                kernel_spec=kernel_spec,
                collector=collector,
            )
            decision = AttentionDecision(
                requested=ATTENTION_HYBRID,
                selected=ATTENTION_HYBRID,
                backend=backend,
                adapter=ATTENTION_HYBRID,
                reason=(
                    "explicit portable Sparse Sage experiment (%s)"
                    % hybrid_config.density_mode
                ),
                environment=environment,
                projector=backend.projector,
            )
            patched = model.clone()
            apply_legacy(
                patched,
                config=optimizer_config,
                decision=decision,
            )
            return io.NodeOutput(patched)

        if density_mode != DENSITY_FIXED:
            raise ValueError(
                "unknown density mode %r" % density_mode
            )

        del min_video_density
        del max_video_density
        del adaptive_temperature
        del adaptive_target_mass

        plan = read_plan(model)
        plan = plan.with_memory(
            _memory_request(
                mode,
                activation,
                bool(strict),
                int(chunk_rows),
            )
        )
        plan = plan.with_sparse(
            SparseRequest(video_budget=float(video_budget))
        )
        patched = apply_plan(model, plan)

        warnings = []
        if timing:
            warnings.append(
                "Legacy timing/report generation is ignored by the production "
                "compatibility path."
            )
        if str(run_tag) != "hybrid50":
            warnings.append(
                "Legacy run_tag=%r is ignored." % str(run_tag)
            )

        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(
                format_legacy_status(patched, warnings=warnings)
            ),
        )


class MiniMaxH3HybridSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridSparseAttention]
