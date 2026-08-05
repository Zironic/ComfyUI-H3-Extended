"""One user-facing model patch for H3 attention and activation memory."""

from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS

from .attention import ATTENTION_MODES, FALLBACK_MODES, resolve_attention
from .config import ACTIVATION_MODES, MemoryOptimizerConfig
from .cuda_pool import configure_cuda_async_soft_gc
from .patch import apply


class MiniMaxH3MemoryOptimizer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3MemoryOptimizerZi",
            display_name="MiniMax H3 Memory Optimizer (Zi)",
            category="model/patch/minimax",
            description=(
                "Unified H3 memory patch. Auto selects a preflighted prepared-QKV "
                "Sage path on SageAttention-supported NVIDIA SM80, SM86, SM89, "
                "SM90, SM120, or SM121 devices. Unsupported architectures or "
                "incomplete Sage builds preserve the incoming attention backend. "
                "BF16 MLP chunking remains available on fallback systems. An "
                "optional cudaMallocAsync soft-GC policy can discourage the CUDA "
                "pool from retaining an obsolete high-water footprint without "
                "setting a hard allocation limit."
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
                        "auto selects the adapter matching the active GPU. "
                        "Architecture-specific entries force that family and "
                        "then follow attention_fallback. existing preserves "
                        "whatever attention backend the incoming MODEL uses."
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
                io.Boolean.Input(
                    "cuda_async_soft_gc",
                    default=False,
                    tooltip=(
                        "When PyTorch uses cudaMallocAsync, replace its unlimited "
                        "default-pool retention policy with the soft threshold below. "
                        "CUDA may still grow above the threshold when live demand "
                        "requires it. This does not set a hard VRAM limit, empty the "
                        "cache, trim the pool, or intentionally cause an early OOM."
                    ),
                ),
                io.Float.Input(
                    "cuda_async_release_threshold_gib",
                    default=11.0,
                    min=0.25,
                    max=256.0,
                    step=0.25,
                    tooltip=(
                        "Soft amount of backing memory the cudaMallocAsync default "
                        "pool should retain before trying to return excess memory "
                        "on a later CUDA synchronization. Ignored when soft GC is "
                        "off or the allocator backend is not cudaMallocAsync."
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
        cuda_async_soft_gc=False,
        cuda_async_release_threshold_gib=11.0,
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
            cuda_async_soft_gc=bool(cuda_async_soft_gc),
            cuda_async_release_threshold_gib=float(
                cuda_async_release_threshold_gib
            ),
        )

        decision = resolve_attention(
            config.attention,
            config.attention_fallback,
        )
        pool_policy = configure_cuda_async_soft_gc(
            config.cuda_async_soft_gc,
            config.cuda_async_release_threshold_gib,
            device_index=decision.environment.device_index,
        )
        patched = model.clone()
        apply(
            patched,
            config=config,
            decision=decision,
            pool_policy=pool_policy,
        )
        return io.NodeOutput(patched)


class MiniMaxH3MemoryOptimizerExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3MemoryOptimizer]
