"""Two composable public nodes for H3 Sage execution."""

from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS, DEFAULT_MODE
    from ..h3_memory_optimizer.config import ACTIVATION_MODES
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS, DEFAULT_MODE
    from h3_memory_optimizer.config import ACTIVATION_MODES

from .apply import apply_plan
from .plan import (
    ATTENTION_AUTO,
    ATTENTION_EXISTING,
    FUSED_QKV_AUTO,
    FUSED_QKV_OFF,
    MemoryRequest,
    SparseRequest,
    read_plan,
)


class MiniMaxH3SageMemoryOptimizer(io.ComfyNode):
    """Lossless-oriented dense/fused QKV and MLP execution controls."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SageMemoryOptimizerZi",
            display_name="MiniMax H3 Sage Memory Optimizer (Zi)",
            category="model/patch/minimax",
            description=(
                "H3 execution and memory optimizations without sparse routing. "
                "Fused QKV is selected only when the resolved dense or Sparse "
                "Sage backend supports its native projected carrier; MLP "
                "chunking/tiling remains independently configurable."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "attention",
                    options=[ATTENTION_AUTO, ATTENTION_EXISTING],
                    default=ATTENTION_AUTO,
                    tooltip=(
                        "auto selects prepared dense Sage unless a Sparse Sage "
                        "node is also present. existing preserves incoming dense "
                        "attention when no Sparse Sage node is present."
                    ),
                ),
                io.Combo.Input(
                    "fused_qkv",
                    options=[FUSED_QKV_AUTO, FUSED_QKV_OFF],
                    default=FUSED_QKV_OFF,
                    tooltip=(
                        "auto emits the projected-QKV format requested by the "
                        "resolved attention backend. It is opt-in while dense "
                        "fused-QKV CUDA parity is being validated; unsupported "
                        "combinations use the standard H3 projection."
                    ),
                ),
                io.Combo.Input(
                    "mlp_mode",
                    options=list(ACTIVATION_MODES),
                    default=DEFAULT_MODE,
                ),
                io.Int.Input(
                    "chunk_rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=256,
                    max=65_536,
                    step=256,
                ),
                io.Boolean.Input("prefer_held_weights", default=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        attention=ATTENTION_AUTO,
        fused_qkv=FUSED_QKV_OFF,
        mlp_mode=DEFAULT_MODE,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        prefer_held_weights=True,
    ):
        if not enabled:
            return io.NodeOutput(model)
        plan = read_plan(model).with_memory(
            MemoryRequest(
                attention=attention,
                fused_qkv=fused_qkv,
                activation=mlp_mode,
                chunk_rows=int(chunk_rows),
                prefer_held_weights=bool(prefer_held_weights),
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
                "Approximate H3 acceleration through fixed-density Sparse Sage. "
                "All non-video context and mixed boundary tiles remain dense; "
                "video_budget controls retained pure target-video KV tiles."
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
    def execute(cls, model, enabled=True, video_budget=0.5):
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
