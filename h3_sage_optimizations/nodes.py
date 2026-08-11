"""Two composable public nodes for H3 Sage execution."""

from comfy_api.latest import ComfyExtension, io

from .apply import apply_plan
from .plan import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MLP_MEMORY_AUTO,
    MLP_MEMORY_OFF,
    MemoryRequest,
    SparseRequest,
    read_plan,
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
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "attention",
                    options=[ATTENTION_AUTO, ATTENTION_EXISTING],
                    default=ATTENTION_AUTO,
                    tooltip=(
                        "auto selects prepared dense Sage unless Sparse Sage "
                        "is also present. existing preserves incoming dense "
                        "attention."
                    ),
                ),
                io.Combo.Input(
                    "fused_qkv",
                    options=[FUSED_QKV_AUTO, FUSED_QKV_OFF],
                    default=FUSED_QKV_AUTO,
                    tooltip=(
                        "auto uses a fused provider only when the actual H3 "
                        "QKV weight format and resolved attention backend are "
                        "compatible; otherwise it uses standard H3 QKV."
                    ),
                ),
                io.Combo.Input(
                    "mlp_memory",
                    options=[MLP_MEMORY_AUTO, MLP_MEMORY_OFF],
                    default=MLP_MEMORY_AUTO,
                    tooltip=(
                        "auto uses ConvRot feature tiling when compatible and "
                        "otherwise performs generic token chunking through "
                        "the model's existing Comfy quantized linear format."
                    ),
                ),
                io.Int.Input(
                    "chunk_rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=256,
                    max=65_536,
                    step=256,
                ),
                io.Boolean.Input(
                    "prefer_held_weights", default=True
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
            return io.NodeOutput(model)
        plan = read_plan(model).with_memory(
            MemoryRequest(
                attention=attention,
                fused_qkv=fused_qkv,
                mlp_memory=mlp_memory,
                chunk_rows=int(chunk_rows),
                prefer_held_weights=bool(
                    prefer_held_weights
                ),
            )
        )
        return io.NodeOutput(apply_plan(model, plan))


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
                "tiles remain dense; video_budget controls retained pure "
                "target-video KV tiles."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input(
                    "video_budget",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Fraction of pure target-video KV tiles retained per "
                        "head/query tile. 1.0 preserves the full video route."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model, enabled=True, video_budget=0.5
    ):
        if not enabled:
            return io.NodeOutput(model)
        plan = read_plan(model).with_sparse(
            SparseRequest(video_budget=float(video_budget))
        )
        return io.NodeOutput(apply_plan(model, plan))


class MiniMaxH3SageOptimizationsExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            MiniMaxH3SageMemoryOptimizer,
            MiniMaxH3SparseSageAttention,
        ]
