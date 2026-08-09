"""Experimental ComfyUI node for H3 hybrid sparse attention."""

import os

import folder_paths
from comfy_api.latest import ComfyExtension, io

try:
    from ..h3_activation_memory.config import (
        DEFAULT_CHUNK_ROWS,
        DEFAULT_MODE,
        MIN_CHUNK_ROWS,
        MODE_CONVROT_2SLICE,
    )
    from ..h3_attention.hybrid import (
        DENSITY_FIXED,
        DENSITY_MODES,
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        IMPLEMENTED_MODES,
        MODE_SAGE128_FUSED_QKV,
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
    from h3_activation_memory.config import (
        DEFAULT_CHUNK_ROWS,
        DEFAULT_MODE,
        MIN_CHUNK_ROWS,
        MODE_CONVROT_2SLICE,
    )
    from h3_attention.hybrid import (
        DENSITY_FIXED,
        DENSITY_MODES,
        HybridSparseBackend,
        HybridSparseConfig,
        HybridStatsCollector,
        IMPLEMENTED_MODES,
        MODE_SAGE128_FUSED_QKV,
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
                "Experimental SM89 path: route H3 directly at Sparse Sage's "
                "128Q x 64KV geometry. Fixed density retains the same fraction "
                "for every pure target-video row. Adaptive budget density keeps "
                "the same aggregate block budget while moving blocks toward "
                "head/query rows with larger omitted coarse attention mass."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(IMPLEMENTED_MODES), default="sage128"),
                io.Float.Input(
                    "video_budget", default=0.5, min=0.01, max=1.0, step=0.01,
                    tooltip=(
                        "Fixed: fraction retained by every pure-video row. "
                        "Adaptive budget: target mean fraction across all "
                        "pure-video head/query rows."
                    ),
                ),
                io.Boolean.Input("strict", default=True),
                io.Combo.Input(
                    "activation", options=list(ACTIVATION_MODES),
                    default=DEFAULT_MODE,
                ),
                io.Int.Input(
                    "chunk_rows", default=DEFAULT_CHUNK_ROWS,
                    min=MIN_CHUNK_ROWS, max=16384, step=256,
                ),
                io.String.Input("run_tag", default="hybrid50"),
                io.Boolean.Input("timing", default=True),
                io.Combo.Input(
                    "compile_backend", options=["off", "inductor"], default="off",
                    tooltip=(
                        "Compile one shared tensor program for all 50 main H3 blocks. "
                        "The current shared program supports fixed density only. "
                        "Do not combine this with TorchCompileModel."
                    ),
                ),
                # Append adaptive controls after every established widget so old
                # serialized workflow widget positions retain their meaning.
                io.Combo.Input(
                    "density_mode", options=list(DENSITY_MODES), default=DENSITY_FIXED,
                    tooltip=(
                        "adaptive_budget preserves the fixed route's total block "
                        "count but redistributes blocks between head/query rows."
                    ),
                ),
                io.Float.Input(
                    "min_video_density", default=0.05, min=0.01, max=1.0, step=0.01,
                    tooltip="Minimum pure-video KV density per adaptive row.",
                ),
                io.Float.Input(
                    "max_video_density", default=0.50, min=0.01, max=1.0, step=0.01,
                    tooltip="Maximum pure-video KV density per adaptive row.",
                ),
                io.Float.Input(
                    "adaptive_temperature", default=1.0, min=0.05, max=20.0, step=0.05,
                    tooltip=(
                        "Temperature applied to pooled QK scores before adaptive "
                        "cumulative-mass estimation."
                    ),
                ),
                io.Float.Input(
                    "adaptive_target_mass", default=0.80, min=0.05, max=1.0, step=0.01,
                    tooltip=(
                        "Coarse video-attention mass used to estimate each row's "
                        "unconstrained block demand before exact budget balancing."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled=True, mode="sage128", video_budget=0.5,
                strict=True, activation=DEFAULT_MODE,
                chunk_rows=DEFAULT_CHUNK_ROWS, run_tag="hybrid50", timing=True,
                compile_backend="off", density_mode=DENSITY_FIXED,
                min_video_density=0.05, max_video_density=0.50,
                adaptive_temperature=1.0, adaptive_target_mass=0.80) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)
        if compile_backend not in ("off", "inductor"):
            raise ValueError("unknown compile backend %r" % compile_backend)

        hybrid_config = HybridSparseConfig(
            mode=mode,
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
        optimizer_config = MemoryOptimizerConfig(
            attention=ATTENTION_EXISTING,
            activation=activation,
            chunk_rows=int(chunk_rows),
            activation_strict=bool(strict),
        )
        if compile_backend == "inductor":
            if hybrid_config.density_mode != DENSITY_FIXED:
                raise ValueError(
                    "Inductor shared-block compilation currently requires fixed "
                    "density_mode; adaptive_budget is available in eager mode"
                )
            if hybrid_config.mode != MODE_SAGE128_FUSED_QKV:
                raise ValueError(
                    "Inductor requires the sage128_fused_qkv attention mode"
                )
            if optimizer_config.activation != MODE_CONVROT_2SLICE:
                raise ValueError(
                    "Inductor requires mlp_chunked_convrot_2slice activation"
                )
        collector = HybridStatsCollector(_output_root(), hybrid_config.run_tag)
        environment = RuntimeEnvironment.detect()
        api = preflight_sparse_sage(
            cuda_available=lambda: environment.cuda_available,
            capability_getter=lambda: environment.capability,
        )
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
            reason=(
                "explicit direct 128Q x 64KV Sparse Sage experiment (%s)"
                % hybrid_config.density_mode
            ),
            environment=environment,
            projector=backend.projector,
        )
        patched = model.clone()
        if compile_backend == "inductor":
            request_shared_block_compile(patched)
        apply(patched, config=optimizer_config, decision=decision)
        return io.NodeOutput(patched)


class MiniMaxH3HybridSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridSparseAttention]

