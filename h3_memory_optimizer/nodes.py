"""One user-facing model patch for H3 attention and activation memory."""

from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS

from .attention import ATTENTION_MODES, FALLBACK_MODES, resolve_attention
from .config import ACTIVATION_MODES, MemoryOptimizerConfig
from .patch import apply


class MiniMaxH3MemoryOptimizer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MemoryOptimizerZi",
            display_name="MiniMax H3 Memory Optimizer (Zi)",
            category="model/patch/minimax",
            description=(
                "Unified H3 memory patch. On a supported SM89 setup, auto uses "
                "the prepared-QKV efficient Sage path; otherwise it preserves "
                "the model's existing attention backend. BF16 MLP chunking is "
                "device-independent and remains available on fallback systems."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="False returns the input MODEL unchanged.",
                ),
                io.Combo.Input(
                    "attention",
                    options=list(ATTENTION_MODES),
                    default="auto",
                    tooltip=(
                        "auto selects a fully preflighted optimized adapter. "
                        "efficient_sage_sm89 requests the current RTX 40-series "
                        "adapter. existing preserves whatever attention backend "
                        "the incoming MODEL already uses."
                    ),
                ),
                io.Combo.Input(
                    "attention_fallback",
                    options=list(FALLBACK_MODES),
                    default="allow",
                    tooltip=(
                        "allow preserves existing attention when no optimized "
                        "adapter is supported. error raises during node execution "
                        "instead. Runtime CUDA kernel failures never fall back."
                    ),
                ),
                io.Combo.Input(
                    "activation",
                    options=list(ACTIVATION_MODES),
                    default="mlp_chunked_bf16",
                    tooltip=(
                        "off disables MLP chunking. mlp_chunked_bf16 is the "
                        "portable conservative path. mlp_chunked_native requests "
                        "Comfy's fused native SwiGLU path where available."
                    ),
                ),
                io.Int.Input(
                    "chunk_rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=256,
                    max=65_536,
                    step=256,
                    tooltip="Maximum packed-token rows per MLP slab.",
                ),
                io.Boolean.Input(
                    "prefer_held_weights",
                    default=True,
                    tooltip=(
                        "Acquire MLP weights once per block when safe. Async "
                        "single-buffer layouts fall back to ordinary module calls."
                    ),
                ),
                io.Boolean.Input(
                    "activation_strict",
                    default=False,
                    tooltip=(
                        "Raise on activation-runtime capability issues. False is "
                        "the portable default and permits same-graph fallbacks."
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
        attention="auto",
        attention_fallback="allow",
        activation="mlp_chunked_bf16",
        chunk_rows=DEFAULT_CHUNK_ROWS,
        prefer_held_weights=True,
        activation_strict=False,
    ):
        if not enabled:
            return io.NodeOutput(model)

        config = MemoryOptimizerConfig(
            attention=attention,
            attention_fallback=attention_fallback,
            activation=activation,
            chunk_rows=int(chunk_rows),
            prefer_held_weights=bool(prefer_held_weights),
            activation_strict=bool(activation_strict),
        )

        decision = resolve_attention(
            config.attention,
            config.attention_fallback,
        )
        patched = model.clone()
        apply(patched, config=config, decision=decision)
        return io.NodeOutput(patched)


class MiniMaxH3MemoryOptimizerExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MemoryOptimizer]
