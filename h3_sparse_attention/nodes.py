"""Deprecated compatibility adapter for the former combined H3 node."""

from comfy_api.latest import ComfyExtension, io, ui

try:
    from ..h3_sage_optimizations.apply import apply_plan
    from ..h3_sage_optimizations.plan import (
        ATTENTION_EXISTING,
        COMPILE_INDUCTOR,
        COMPILE_OFF,
        DENSITY_ADAPTIVE_BUDGET,
        DENSITY_FIXED,
        DENSITY_MODES,
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
    from ..h3_sage_optimizations.status import (
        format_disabled_status,
        format_legacy_status,
    )
except ImportError:
    from h3_sage_optimizations.apply import apply_plan
    from h3_sage_optimizations.plan import (
        ATTENTION_EXISTING,
        COMPILE_INDUCTOR,
        COMPILE_OFF,
        DENSITY_ADAPTIVE_BUDGET,
        DENSITY_FIXED,
        DENSITY_MODES,
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
    from h3_sage_optimizations.status import (
        format_disabled_status,
        format_legacy_status,
    )

MODE_SAGE128 = "sage128"
MODE_SAGE128_FUSED_QKV = "sage128_fused_qkv"
IMPLEMENTED_MODES = (MODE_SAGE128, MODE_SAGE128_FUSED_QKV)

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

DEFAULT_CHUNK_ROWS = 2048
MIN_CHUNK_ROWS = 256


def _memory_request(mode, activation, strict, chunk_rows):
    if mode == MODE_SAGE128:
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
    )


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
                "translates every meaningful former Sparse Sage, fused-QKV, "
                "MLP, adaptive-routing, reporting, and shared-compilation "
                "control into the new composable requests."
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
                    display_name="Legacy attention mode",
                    options=list(IMPLEMENTED_MODES),
                    default=MODE_SAGE128,
                ),
                io.Float.Input(
                    "video_budget",
                    display_name="Video KV budget",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Requested pure-video KV tile fraction. Fixed and "
                        "adaptive routing use the same quantized global budget."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="Require legacy specialized paths",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Require explicitly selected fused-QKV and ConvRot "
                        "two-slice paths and strict packed-layout validation."
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
                    display_name="Report run tag",
                    default="hybrid50",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "timing",
                    display_name="Include deferred CUDA timing",
                    default=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "compile_backend",
                    display_name="Shared block compilation",
                    options=[COMPILE_OFF, COMPILE_INDUCTOR],
                    default=COMPILE_OFF,
                    advanced=True,
                ),
                io.Combo.Input(
                    "density_mode",
                    display_name="Routing policy",
                    options=list(DENSITY_MODES),
                    default=DENSITY_FIXED,
                    advanced=True,
                ),
                io.Float.Input(
                    "min_video_density",
                    display_name="Minimum video density",
                    default=0.05,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_video_density",
                    display_name="Maximum video density",
                    default=0.50,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_temperature",
                    display_name="Adaptive temperature",
                    default=1.0,
                    min=0.05,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Float.Input(
                    "adaptive_target_mass",
                    display_name="Adaptive target mass",
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
        mode=MODE_SAGE128,
        video_budget=0.5,
        strict=True,
        activation=MODE_NATIVE,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        run_tag="hybrid50",
        timing=True,
        compile_backend=COMPILE_OFF,
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

        if compile_backend not in (COMPILE_OFF, COMPILE_INDUCTOR):
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

        # Preserve the former node's early validation order so invalid compile
        # requests fail before any CUDA or Sparse Sage dependency preflight.
        if compile_backend == COMPILE_INDUCTOR:
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
            SparseRequest(
                video_budget=float(video_budget),
                density_mode=str(density_mode),
                min_video_density=float(min_video_density),
                max_video_density=float(max_video_density),
                adaptive_temperature=float(adaptive_temperature),
                adaptive_target_mass=float(adaptive_target_mass),
                strict=bool(strict),
                # The former combined node always wrote structural reports;
                # timing only controlled inclusion of deferred CUDA events.
                write_report=True,
                timing=bool(timing),
                run_tag=str(run_tag),
                compile_backend=str(compile_backend),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_legacy_status(patched)),
        )


class MiniMaxH3HybridSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridSparseAttention]
