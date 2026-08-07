"""User-facing H3 memory and experimental acceleration patches."""

from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS
    from ..h3_adaln.config import MODES as ADALN_MODES
    from ..h3_block_cache.config import MODES as BLOCK_CACHE_MODES
    from ..h3_runtime.layout import SINK_MODES
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS
    from h3_adaln.config import MODES as ADALN_MODES
    from h3_block_cache.config import MODES as BLOCK_CACHE_MODES
    from h3_runtime.layout import SINK_MODES

from .attention import ATTENTION_SOL, FALLBACK_MODES, resolve_attention
from .config import ACTIVATION_MODES, MemoryOptimizerConfig
from .cuda_pool import configure_cuda_async_soft_gc
from .patch import apply


class MiniMaxH3MemoryOptimizer(io.ComfyNode):
    """Stable, lossless H3 memory-efficiency path.

    This node intentionally exposes only the prepared Sage attention path and
    activation-memory controls. Experimental Sol-Attn, AdaLN precompute,
    FirstBlockCache, and CUDA pool tuning live in MiniMaxH3SolEngine.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MemoryOptimizerZi",
            display_name="MiniMax H3 Memory Optimizer (Zi)",
            category="model/patch/minimax",
            description=(
                "Stable H3 memory-efficiency patch. Auto attention selects the "
                "prepared architecture-specific Sage backend; activation chunking "
                "reduces H3 MLP peak memory without enabling approximate features."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "attention",
                    options=["auto", "existing"],
                    default="auto",
                    tooltip=(
                        "auto selects the prepared Sage backend for the detected GPU; "
                        "existing leaves incoming attention unchanged."
                    ),
                ),
                io.Combo.Input(
                    "activation",
                    options=list(ACTIVATION_MODES),
                    default="mlp_chunked_bf16",
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
        attention="auto",
        activation="mlp_chunked_bf16",
        chunk_rows=DEFAULT_CHUNK_ROWS,
        prefer_held_weights=True,
    ):
        if not enabled:
            return io.NodeOutput(model)

        config = MemoryOptimizerConfig(
            attention=attention,
            attention_fallback="allow",
            activation=activation,
            chunk_rows=int(chunk_rows),
            prefer_held_weights=bool(prefer_held_weights),
            activation_strict=False,
            # Experimental features are deliberately pinned off here.
            adaln_precompute="off",
            block_cache="off",
            cuda_async_soft_gc=False,
        )
        decision = resolve_attention(
            config.attention,
            config.attention_fallback,
            adapter_options=config.attention_options(),
        )
        patched = model.clone()
        apply(patched, config=config, decision=decision, pool_policy=None)
        return io.NodeOutput(patched)


class MiniMaxH3SolEngine(io.ComfyNode):
    """Experimental H3 acceleration controls kept separate from stable memory code."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SolEngineZi",
            display_name="MiniMax H3 Sol Engine (Zi)",
            category="model/patch/minimax",
            description=(
                "Experimental H3 acceleration patch. Sol-Attn and FirstBlockCache "
                "can be approximate; AdaLN precompute and CUDA pool tuning are also "
                "kept here so the stable Memory Optimizer remains small."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input(
                    "attention",
                    options=[ATTENTION_SOL, "existing"],
                    default=ATTENTION_SOL,
                    tooltip=(
                        "sol_attn enables the experimental sparse/approximate backend; "
                        "existing leaves the incoming attention backend unchanged."
                    ),
                ),
                io.Combo.Input(
                    "attention_fallback",
                    options=list(FALLBACK_MODES),
                    default="allow",
                ),
                io.Float.Input("sol_tau", default=1.0, min=-100.0, max=100.0, step=0.05),
                io.Combo.Input("sol_thresh_type", options=["diag", "exact"], default="diag"),
                io.Int.Input("sol_dense_steps", default=10, min=0, max=256),
                io.Int.Input("sol_dense_layers", default=2, min=0, max=50),
                io.Combo.Input("sol_sink_mode", options=list(SINK_MODES), default="prefix"),
                io.Boolean.Input("sol_correctness_gate", default=True),
                io.Int.Input(
                    "sol_gate_heads",
                    default=4,
                    min=0,
                    max=56,
                    tooltip="Heads checked against dense SDPA; 0 checks all heads.",
                ),
                io.Int.Input(
                    "sol_density_heads",
                    default=4,
                    min=0,
                    max=56,
                    tooltip="Heads sampled for route-density diagnostics; 0 uses all.",
                ),
                io.Combo.Input("sol_kv_splits", options=[1, 2, 4], default=1),
                io.Float.Input(
                    "sol_max_sink_fraction",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Decline to dense prepared attention when the exact KV sink "
                        "would exceed this fraction."
                    ),
                ),
                io.Boolean.Input("sol_strict", default=False),
                io.Combo.Input("adaln_precompute", options=list(ADALN_MODES), default="off"),
                io.Float.Input("adaln_max_table_gib", default=2.0, min=0.05, max=32.0, step=0.05),
                io.Boolean.Input("adaln_strict", default=False),
                io.Combo.Input("block_cache", options=list(BLOCK_CACHE_MODES), default="off"),
                io.Float.Input("block_cache_threshold", default=0.08, min=0.0, max=1.0, step=0.005),
                io.Int.Input("block_cache_warmup_steps", default=3, min=0, max=256),
                io.Boolean.Input("block_cache_strict", default=False),
                io.Boolean.Input("cuda_async_soft_gc", default=False),
                io.Float.Input(
                    "cuda_async_release_threshold_gib",
                    default=11.0,
                    min=0.25,
                    max=256.0,
                    step=0.25,
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        enabled=True,
        attention=ATTENTION_SOL,
        attention_fallback="allow",
        sol_tau=1.0,
        sol_thresh_type="diag",
        sol_dense_steps=10,
        sol_dense_layers=2,
        sol_sink_mode="prefix",
        sol_correctness_gate=True,
        sol_gate_heads=4,
        sol_density_heads=4,
        sol_kv_splits=1,
        sol_max_sink_fraction=0.5,
        sol_strict=False,
        adaln_precompute="off",
        adaln_max_table_gib=2.0,
        adaln_strict=False,
        block_cache="off",
        block_cache_threshold=0.08,
        block_cache_warmup_steps=3,
        block_cache_strict=False,
        cuda_async_soft_gc=False,
        cuda_async_release_threshold_gib=11.0,
    ):
        if not enabled:
            return io.NodeOutput(model)

        # activation="off" is load-bearing: this node must compose after the
        # stable Memory Optimizer without trying to own the H3 block forwards.
        config = MemoryOptimizerConfig(
            attention=attention,
            attention_fallback=attention_fallback,
            activation="off",
            sol_tau=float(sol_tau),
            sol_thresh_type=sol_thresh_type,
            sol_dense_steps=int(sol_dense_steps),
            sol_dense_layers=int(sol_dense_layers),
            sol_sink_mode=sol_sink_mode,
            sol_correctness_gate=bool(sol_correctness_gate),
            sol_gate_heads=int(sol_gate_heads),
            sol_density_heads=int(sol_density_heads),
            sol_kv_splits=int(sol_kv_splits),
            sol_max_sink_fraction=float(sol_max_sink_fraction),
            sol_strict=bool(sol_strict),
            adaln_precompute=adaln_precompute,
            adaln_max_table_gib=float(adaln_max_table_gib),
            adaln_strict=bool(adaln_strict),
            block_cache=block_cache,
            block_cache_threshold=float(block_cache_threshold),
            block_cache_warmup_steps=int(block_cache_warmup_steps),
            block_cache_strict=bool(block_cache_strict),
            cuda_async_soft_gc=bool(cuda_async_soft_gc),
            cuda_async_release_threshold_gib=float(cuda_async_release_threshold_gib),
        )
        decision = resolve_attention(
            config.attention,
            config.attention_fallback,
            adapter_options=config.attention_options(),
        )
        pool_policy = configure_cuda_async_soft_gc(
            config.cuda_async_soft_gc,
            config.cuda_async_release_threshold_gib,
            device_index=decision.environment.device_index,
        )
        patched = model.clone()
        apply(patched, config=config, decision=decision, pool_policy=pool_policy)
        return io.NodeOutput(patched)


class MiniMaxH3MemoryOptimizerExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MemoryOptimizer, MiniMaxH3SolEngine]
