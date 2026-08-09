"""Experimental ComfyUI node for H3 hybrid sparse attention."""

import os

import folder_paths
from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import DEFAULT_CHUNK_ROWS, DEFAULT_MODE
    from ..h3_attention.hybrid import (
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        IMPLEMENTED_MODES,
        preflight_sparse_sage,
    )
    from ..h3_memory_optimizer.attention import (
        ATTENTION_EXISTING,
        AttentionDecision,
        RuntimeEnvironment,
    )
    from ..h3_memory_optimizer.config import ACTIVATION_MODES, MemoryOptimizerConfig
    from ..h3_memory_optimizer.patch import apply
    from ..h3_runtime.compile_compat import request_shared_block_compile
except ImportError:
    from h3_activation_memory.config import DEFAULT_CHUNK_ROWS, DEFAULT_MODE
    from h3_attention.hybrid import (
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        IMPLEMENTED_MODES,
        preflight_sparse_sage,
    )
    from h3_memory_optimizer.attention import (
        ATTENTION_EXISTING,
        AttentionDecision,
        RuntimeEnvironment,
    )
    from h3_memory_optimizer.config import ACTIVATION_MODES, MemoryOptimizerConfig
    from h3_memory_optimizer.patch import apply
    from h3_runtime.compile_compat import request_shared_block_compile

ATTENTION_HYBRID = "hybrid_sparse"


def _output_root():
    return os.path.join(folder_paths.get_output_directory(), "h3_hybrid_sparse")


class MiniMaxH3HybridSparseAttention(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridSparseAttentionZi",
            display_name="MiniMax H3 Hybrid Sparse Attention (Zi)",
            category="model/patch/minimax",
            description=(
                "Phase A experimental SM89 path: route H3 directly at Sparse "
                "Sage's 128Q x 64KV geometry, retaining the requested fraction "
                "of pure target-video KV tiles. Flex, Sol head dispatch, and "
                "automatic planning are not enabled yet."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(IMPLEMENTED_MODES), default="sage128"),
                io.Float.Input(
                    "video_budget", default=0.5, min=0.01, max=1.0, step=0.01,
                ),
                io.Boolean.Input("strict", default=True),
                io.Combo.Input(
                    "activation", options=list(ACTIVATION_MODES),
                    default=DEFAULT_MODE,
                ),
                io.Int.Input(
                    "chunk_rows", default=DEFAULT_CHUNK_ROWS,
                    min=128, max=16384, step=256,
                ),
                io.String.Input("run_tag", default="hybrid50"),
                io.Boolean.Input("timing", default=True),
                io.Combo.Input(
                    "compile_backend", options=["off", "inductor"], default="off",
                    tooltip=(
                        "Compile one shared tensor program for all 50 main H3 blocks. "
                        "Do not combine this with TorchCompileModel."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled=True, mode="sage128", video_budget=0.5,
                strict=True, activation=DEFAULT_MODE,
                chunk_rows=DEFAULT_CHUNK_ROWS, run_tag="hybrid50", timing=True,
                compile_backend="off") -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)

        hybrid_config = HybridSparseConfig(
            mode=mode,
            video_budget=float(video_budget),
            strict=bool(strict),
            run_tag=run_tag,
            timing=bool(timing),
        )
        environment = RuntimeEnvironment.detect()
        api = preflight_sparse_sage(
            cuda_available=lambda: environment.cuda_available,
            capability_getter=lambda: environment.capability,
        )
        collector = HybridStatsCollector(_output_root(), hybrid_config.run_tag)
        backend = HybridSparseBackend(
            hybrid_config,
            api=api,
            collector=collector,
        )
        decision = AttentionDecision(
            requested=ATTENTION_HYBRID,
            selected=ATTENTION_HYBRID,
            backend=backend,
            adapter=ATTENTION_HYBRID,
            reason="explicit Phase A direct 128Q x 64KV Sparse Sage experiment",
            environment=environment,
            projector=backend.projector,
        )
        optimizer_config = MemoryOptimizerConfig(
            attention=ATTENTION_EXISTING,
            activation=activation,
            chunk_rows=int(chunk_rows),
            activation_strict=bool(strict),
        )
        patched = model.clone()
        if compile_backend == "inductor":
            request_shared_block_compile(patched)
        apply(patched, config=optimizer_config, decision=decision)
        return io.NodeOutput(patched)


class MiniMaxH3HybridSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridSparseAttention]
