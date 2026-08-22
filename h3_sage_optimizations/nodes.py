"""Two composable public nodes for H3 Sage execution."""

from comfy_api.latest import ComfyExtension, io, ui

from .apply import apply_plan
from .plan import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    DEFAULT_EDGE_KV,
    DEFAULT_EDGE_STEPS,
    DEFAULT_VIDEO_BUDGET,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_EPILOGUE,
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
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
                mlp_memory=(
                    MLP_MEMORY_LEGACY_CONVROT_REQUIRED
                    if mlp_memory == MLP_MEMORY_EPILOGUE
                    else mlp_memory
                ),
                chunk_rows=int(chunk_rows),
                prefer_held_weights=bool(prefer_held_weights),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_memory_status(patched)),
        )


class MiniMaxH3SparseSageAttention(io.ComfyNode):
    """Approximate fixed-density Sparse Sage routing only."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SparseSageAttentionZi",
            display_name="MiniMax H3 Sparse Sage Attention (Zi)",
            category="model/patch/minimax",
            description=(
                "H3-only fixed-density Sparse Sage attention. Unknown models "
                "pass through unchanged. Non-video context and mixed boundary "
                "tiles remain dense; Video KV budget controls retained pure "
                "target-video KV tiles."
            ),
            search_aliases=[
                "H3 sparse",
                "H3 sparse attention",
                "MiniMax sparse",
                "Sparse Sage",
                "Sparge",
                "H3 acceleration",
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
                io.Boolean.Input(
                    "denser_early_late_steps",
                    display_name="Denser Early/Late steps",
                    default=False,
                    tooltip=(
                        "Add 30 percentage points to the Video KV budget for "
                        "the first 2 and last 2 sampling steps, capped at 100%."
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
        denser_early_late_steps=False,
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
                denser_early_late_steps=bool(denser_early_late_steps),
            )
        )
        patched = apply_plan(model, plan)
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(format_sparse_status(patched)),
        )


class MiniMaxH3SparseSageAttentionAdvanced(io.ComfyNode):
    """Fixed-density sparse attention with explicit edge-step budgets."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SparseSageAttentionAdvancedZi",
            display_name="MiniMax H3 Sparse Sage Attention Advanced (Zi)",
            category="model/patch/minimax",
            description=(
                "H3 fixed-density sparse attention with independent early, "
                "middle, and late target-video KV budgets."
            ),
            search_aliases=[
                "H3 sparse advanced",
                "H3 sparse schedule",
                "Sparse Sage advanced",
                "H3 early late KV",
            ],
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", display_name="Enable", default=True),
                io.Float.Input(
                    "video_budget",
                    display_name="Video KV budget",
                    default=DEFAULT_VIDEO_BUDGET,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "early_steps",
                    display_name="Early steps",
                    default=DEFAULT_EDGE_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                ),
                io.Float.Input(
                    "early_kv",
                    display_name="Early KV",
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                ),
                io.Int.Input(
                    "late_steps",
                    display_name="Late steps",
                    default=DEFAULT_EDGE_STEPS,
                    min=0,
                    max=1000,
                    step=1,
                ),
                io.Float.Input(
                    "late_kv",
                    display_name="Late KV",
                    default=DEFAULT_EDGE_KV,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        video_budget=DEFAULT_VIDEO_BUDGET,
        early_steps=DEFAULT_EDGE_STEPS,
        early_kv=DEFAULT_EDGE_KV,
        late_steps=DEFAULT_EDGE_STEPS,
        late_kv=DEFAULT_EDGE_KV,
    ):
        if not enabled:
            return io.NodeOutput(
                model,
                ui=ui.PreviewText(
                    format_disabled_status(
                        "MiniMax H3 Sparse Sage Attention Advanced"
                    )
                ),
            )
        plan = read_plan(model).with_sparse(
            SparseRequest(
                video_budget=float(video_budget),
                early_steps=int(early_steps),
                early_kv=float(early_kv),
                late_steps=int(late_steps),
                late_kv=float(late_kv),
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
            MiniMaxH3SparseSageAttentionAdvanced,
        ]
