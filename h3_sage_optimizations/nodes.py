"""Two composable public nodes for H3 Sage execution."""

from comfy_api.latest import ComfyExtension, io, ui

from .apply import apply_plan
from .plan import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    COMPILE_BACKENDS,
    COMPILE_OFF,
    DENSITY_FIXED,
    DENSITY_MODES,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_LEGACY_BF16,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
    MLP_MEMORY_LEGACY_NATIVE,
    MLP_MEMORY_OFF,
    MemoryRequest,
    SparseRequest,
    read_plan,
)
from .status import (
    format_disabled_status,
    format_memory_status,
    format_sparse_status,
)

DEFAULT_CHUNK_ROWS = 2048

MLP_EXECUTION_AUTO = "auto"
MLP_EXECUTION_BF16 = "chunked_bf16"
MLP_EXECUTION_NATIVE = "chunked_native"
MLP_EXECUTION_CONVROT = "convrot_two_slice"
MLP_EXECUTION_OPTIONS = (
    MLP_EXECUTION_AUTO,
    MLP_EXECUTION_BF16,
    MLP_EXECUTION_NATIVE,
    MLP_EXECUTION_CONVROT,
)


def _resolve_mlp_request(mlp_memory, mlp_execution):
    """Translate the optional exact-provider override into one plan request."""

    if mlp_execution == MLP_EXECUTION_AUTO:
        return mlp_memory
    if mlp_execution == MLP_EXECUTION_BF16:
        return MLP_MEMORY_LEGACY_BF16
    if mlp_execution == MLP_EXECUTION_NATIVE:
        return MLP_MEMORY_LEGACY_NATIVE
    if mlp_execution == MLP_EXECUTION_CONVROT:
        return MLP_MEMORY_LEGACY_CONVROT_REQUIRED
    raise ValueError("unknown MLP execution override %r" % mlp_execution)


class MiniMaxH3SageMemoryOptimizer(io.ComfyNode):
    """Format-aware fused QKV and MLP memory execution controls."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SageMemoryOptimizerZi",
            display_name="MiniMax H3 Sage Memory Optimizer (Zi)",
            category="model/patch/minimax",
            description=(
                "H3-only memory and execution optimizations. Unknown models "
                "pass through unchanged. Auto preserves each H3 checkpoint's "
                "QKV and MLP weight layouts, selecting a specialized fused or "
                "tiled provider only when that exact format is supported."
            ),
            search_aliases=[
                "H3 VRAM",
                "H3 memory",
                "H3 fused QKV",
                "H3 chunked MLP",
                "MiniMax memory optimizer",
                "Sage optimizer",
            ],
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enable",
                    default=True,
                    tooltip=(
                        "When disabled, this node applies no new request and "
                        "leaves all upstream model patches unchanged."
                    ),
                ),
                io.Combo.Input(
                    "attention",
                    display_name="Dense attention when Sparse is absent",
                    options=[ATTENTION_AUTO, ATTENTION_EXISTING],
                    default=ATTENTION_AUTO,
                    advanced=True,
                    tooltip=(
                        "auto selects prepared dense Sage unless Sparse Sage "
                        "is also present. existing preserves incoming dense "
                        "attention. This setting has no effect while the Sparse "
                        "Sage node owns attention execution."
                    ),
                ),
                io.Combo.Input(
                    "fused_qkv",
                    display_name="QKV projection optimization",
                    options=[FUSED_QKV_AUTO, FUSED_QKV_OFF],
                    default=FUSED_QKV_AUTO,
                    tooltip=(
                        "auto uses a fused provider only when the actual H3 "
                        "QKV weight format and resolved attention backend are "
                        "compatible; otherwise it safely uses standard H3 QKV."
                    ),
                ),
                io.Combo.Input(
                    "mlp_memory",
                    display_name="MLP memory optimization",
                    options=[
                        MLP_MEMORY_AUTO,
                        MLP_MEMORY_EPILOGUE,
                        MLP_MEMORY_OFF,
                    ],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        "auto uses the established ConvRot two-slice path when "
                        "compatible and otherwise generic token chunking. "
                        "epilogue_prototype tests fused fc1+SwiGLU and "
                        "fc2+gated-residual kernels on compatible ConvRot-256 "
                        "TensorWise INT8 MLP weights."
                    ),
                ),
                io.Combo.Input(
                    "mlp_execution",
                    display_name="Explicit MLP execution override",
                    options=list(MLP_EXECUTION_OPTIONS),
                    default=MLP_EXECUTION_AUTO,
                    advanced=True,
                    tooltip=(
                        "auto follows MLP memory optimization. Explicit values "
                        "restore the former BF16, native, or required ConvRot "
                        "two-slice execution modes and take precedence over the "
                        "ordinary MLP selector."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="Error instead of specialized fallback",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "When enabled, fused_qkv=auto becomes a required fused "
                        "request, and runtime MLP acquisition/provider failures "
                        "raise instead of falling back. QKV optimization set to "
                        "off remains off."
                    ),
                ),
                io.Int.Input(
                    "chunk_rows",
                    display_name="MLP chunk rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=256,
                    max=65_536,
                    step=256,
                    advanced=True,
                    tooltip=(
                        "Maximum token rows processed by one MLP chunk. Larger "
                        "values may be faster but use more activation memory."
                    ),
                ),
                io.Boolean.Input(
                    "prefer_held_weights",
                    display_name="Hold weights across chunks",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Try to acquire fc1 and fc2 once for all chunks. The "
                        "implementation validates reusable cast-buffer safety "
                        "and falls back when holding both weights is unsafe."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        attention=ATTENTION_AUTO,
        fused_qkv=FUSED_QKV_AUTO,
        mlp_memory=MLP_MEMORY_AUTO,
        mlp_execution=MLP_EXECUTION_AUTO,
        strict=False,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        prefer_held_weights=True,
    ):
        if not enabled:
            return io.NodeOutput(
                model,
                ui=ui.PreviewText(
                    format_disabled_status("MiniMax H3 Sage Memory Optimizer")
                ),
            )
        plan = read_plan(model).with_memory(
            MemoryRequest(
                attention=attention,
                fused_qkv=fused_qkv,
                mlp_memory=_resolve_mlp_request(
                    mlp_memory, mlp_execution
                ),
                chunk_rows=int(chunk_rows),
                prefer_held_weights=bool(prefer_held_weights),
                strict=bool(strict),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )


class MiniMaxH3SparseSageAttention(io.ComfyNode):
    """Sparse Sage routing with optional adaptive, reporting, and compile controls."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SparseSageAttentionZi",
            display_name="MiniMax H3 Sparse Sage Attention (Zi)",
            category="model/patch/minimax",
            description=(
                "H3-only Sparse Sage attention. Unknown models pass through "
                "unchanged. Non-video context and mixed boundary tiles remain "
                "dense; Video KV budget controls retained pure target-video "
                "KV tiles. Advanced controls preserve the meaningful routing, "
                "validation, diagnostics, and compilation behavior of the "
                "former combined node."
            ),
            search_aliases=[
                "H3 sparse",
                "H3 sparse attention",
                "MiniMax sparse",
                "Sparse Sage",
                "Sparge",
                "H3 acceleration",
                "H3 adaptive attention",
            ],
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enable",
                    default=True,
                    tooltip=(
                        "When disabled, this node applies no new sparse request "
                        "and leaves all upstream model patches unchanged."
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
                        "Requested fraction of pure target-video KV tiles "
                        "retained per head and query tile. The request is rounded "
                        "up to a whole KV-tile count, so effective density depends "
                        "on video geometry. Non-video context and mixed boundary "
                        "tiles remain dense. 1.0 preserves the complete pure-video "
                        "route while still using the Sparse Sage execution path."
                    ),
                ),
                io.Combo.Input(
                    "density_mode",
                    display_name="Routing policy",
                    options=list(DENSITY_MODES),
                    default=DENSITY_FIXED,
                    advanced=True,
                    tooltip=(
                        "fixed gives every pure-video head/query row the same "
                        "quantized K. adaptive_budget preserves the same global "
                        "block count but redistributes K between rows according "
                        "to pooled Q/K attention estimates."
                    ),
                ),
                io.Float.Input(
                    "min_video_density",
                    display_name="Minimum video density",
                    default=0.05,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Adaptive lower rail for each pure-video head/query row. "
                        "It must not exceed Video KV budget."
                    ),
                ),
                io.Float.Input(
                    "max_video_density",
                    display_name="Maximum video density",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Adaptive upper rail for each pure-video head/query row. "
                        "It must be at least Video KV budget. A maximum equal to "
                        "the budget leaves no room to move blocks above target."
                    ),
                ),
                io.Float.Input(
                    "adaptive_temperature",
                    display_name="Adaptive temperature",
                    default=1.0,
                    min=0.05,
                    max=20.0,
                    step=0.05,
                    advanced=True,
                    tooltip=(
                        "Temperature applied to pooled Q/K scores before "
                        "estimating each row's block demand."
                    ),
                ),
                io.Float.Input(
                    "adaptive_target_mass",
                    display_name="Adaptive target mass",
                    default=0.80,
                    min=0.05,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip=(
                        "Coarse cumulative video-attention mass used to estimate "
                        "each row's unconstrained K before exact budget balancing."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="Strict packed-layout validation",
                    default=True,
                    advanced=True,
                    tooltip=(
                        "Require a valid H3 packed-token layout for Sparse Sage. "
                        "Disable only for diagnostics; the sparse backend still "
                        "cannot route an unknown layout safely."
                    ),
                ),
                io.Boolean.Input(
                    "write_report",
                    display_name="Write sparse report",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Write per-request JSON and text reports containing "
                        "effective density, route distributions, and selected "
                        "providers. Enabling CUDA timing also enables reports."
                    ),
                ),
                io.Boolean.Input(
                    "timing",
                    display_name="Include deferred CUDA timing",
                    default=False,
                    advanced=True,
                    tooltip=(
                        "Record request-scoped deferred CUDA events in the sparse "
                        "report. Stage timings overlap and are not additive."
                    ),
                ),
                io.String.Input(
                    "run_tag",
                    display_name="Report run tag",
                    default="sparse50",
                    advanced=True,
                    tooltip=(
                        "Prefix used for report directories. Use 1-64 ASCII "
                        "letters, digits, underscores, or hyphens."
                    ),
                ),
                io.Combo.Input(
                    "compile_backend",
                    display_name="Shared block compilation",
                    options=list(COMPILE_BACKENDS),
                    default=COMPILE_OFF,
                    advanced=True,
                    tooltip=(
                        "inductor compiles one shared CUDA tensor program for all "
                        "50 H3 blocks. It requires fixed routing plus an upstream "
                        "Memory Optimizer that resolves fused Sparse QKV and the "
                        "ConvRot two-slice MLP. When this Sparse node appears "
                        "first, the request remains pending until that Memory "
                        "Optimizer is applied. Do not combine with TorchCompileModel."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        video_budget=0.5,
        density_mode=DENSITY_FIXED,
        min_video_density=0.05,
        max_video_density=1.0,
        adaptive_temperature=1.0,
        adaptive_target_mass=0.80,
        strict=True,
        write_report=False,
        timing=False,
        run_tag="sparse50",
        compile_backend=COMPILE_OFF,
    ):
        if not enabled:
            return io.NodeOutput(
                model,
                ui=ui.PreviewText(
                    format_disabled_status(
                        "MiniMax H3 Sparse Sage Attention"
                    )
                ),
            )
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                density_mode=str(density_mode),
                min_video_density=float(min_video_density),
                max_video_density=float(max_video_density),
                adaptive_temperature=float(adaptive_temperature),
                adaptive_target_mass=float(adaptive_target_mass),
                strict=bool(strict),
                write_report=bool(write_report),
                timing=bool(timing),
                run_tag=str(run_tag),
                compile_backend=str(compile_backend),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )


class MiniMaxH3SageOptimizationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SageMemoryOptimizer,
            MiniMaxH3SparseSageAttention,
        ]
