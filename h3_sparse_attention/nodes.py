"""Experimental combined H3 Sparse Sage, fused-QKV, and MLP node.

This node intentionally remains the full experimental surface in H3-Extended.
The two production nodes use the smaller format-aware plan implementation; this
node keeps adaptive routing, timing reports, run tags, and shared Inductor
compilation for development workflows.
"""

import os

import folder_paths
from comfy_api.latest import ComfyExtension, io, ui

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
    from ..h3_memory_optimizer.config import (
        ACTIVATION_MODES,
        MemoryOptimizerConfig,
    )
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
    from h3_memory_optimizer.config import (
        ACTIVATION_MODES,
        MemoryOptimizerConfig,
    )
    from h3_memory_optimizer.patch import apply
    from h3_runtime.compile_compat import request_shared_block_compile

ATTENTION_HYBRID = "hybrid_sparse"
PRODUCTION_PLAN_KEY = "minimax_h3_sage_optimization_plan"


def _output_root():
    return os.path.join(
        folder_paths.get_output_directory(),
        "h3_hybrid_sparse",
    )


def _reject_production_plan(model):
    """Keep the monolithic experiment separate from the production patches."""

    options = getattr(model, "model_options", {}) or {}
    if options.get(PRODUCTION_PLAN_KEY) is not None:
        raise ValueError(
            "MiniMax H3 Hybrid Sparse Attention is the monolithic experimental "
            "node and cannot be combined with MiniMax H3 Sage Memory Optimizer "
            "or MiniMax H3 Sparse Sage Attention on the same model branch. Use "
            "either this experimental node or the two production nodes."
        )


class MiniMaxH3HybridSparseAttention(io.ComfyNode):
    """Full experimental H3 sparse-attention and memory-optimization surface."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3HybridSparseAttentionZi",
            display_name=(
                "MiniMax H3 Hybrid Sparse Attention (Experimental)"
            ),
            category="model/patch/minimax/experiments",
            description=(
                "Full experimental H3 Sparse Sage node. It retains fixed and "
                "adaptive routing, optional fused QKV, chunked/tiled MLP modes, "
                "timing reports, run tags, and shared Inductor compilation. "
                "Do not combine it with the two production H3 Sage nodes."
            ),
            search_aliases=[
                "H3 Hybrid Sparse",
                "H3 adaptive sparse",
                "H3 sparse experiment",
                "H3 compiled sparse",
                "MiniMax experimental sparse",
            ],
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    display_name="Enable",
                    default=True,
                    tooltip=(
                        "When disabled, this node is an exact pass-through. It "
                        "does not remove patches already applied upstream."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    display_name="Attention mode",
                    options=list(IMPLEMENTED_MODES),
                    default="sage128",
                ),
                io.Float.Input(
                    "video_budget",
                    display_name="Video KV budget",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Fixed: fraction retained by every pure-video row. "
                        "Adaptive budget: target mean fraction across all "
                        "pure-video head/query rows."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    display_name="Strict experimental paths",
                    default=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "activation",
                    display_name="MLP execution",
                    options=list(ACTIVATION_MODES),
                    default=DEFAULT_MODE,
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
                    display_name="Write timing reports",
                    default=True,
                    advanced=True,
                ),
                io.Combo.Input(
                    "compile_backend",
                    display_name="Shared block compilation",
                    options=["off", "inductor"],
                    default="off",
                    advanced=True,
                    tooltip=(
                        "Compile one shared tensor program for all 50 main H3 "
                        "blocks. The current program requires fixed density, "
                        "fused QKV, and ConvRot two-slice MLP. Do not combine "
                        "with TorchCompileModel."
                    ),
                ),
                io.Combo.Input(
                    "density_mode",
                    display_name="Density allocation",
                    options=list(DENSITY_MODES),
                    default=DENSITY_FIXED,
                    advanced=True,
                    tooltip=(
                        "adaptive_budget preserves the fixed route's aggregate "
                        "block count while redistributing blocks between "
                        "head/query rows."
                    ),
                ),
                io.Float.Input(
                    "min_video_density",
                    display_name="Adaptive minimum density",
                    default=0.05,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                ),
                io.Float.Input(
                    "max_video_density",
                    display_name="Adaptive maximum density",
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
                    tooltip=(
                        "Temperature applied to pooled QK scores before "
                        "cumulative-mass demand estimation."
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
                        "Coarse video-attention mass used to estimate each "
                        "row's unconstrained demand before exact budget "
                        "balancing."
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
        mode="sage128",
        video_budget=0.5,
        strict=True,
        activation=DEFAULT_MODE,
        chunk_rows=DEFAULT_CHUNK_ROWS,
        run_tag="hybrid50",
        timing=True,
        compile_backend="off",
        density_mode=DENSITY_FIXED,
        min_video_density=0.05,
        max_video_density=0.50,
        adaptive_temperature=1.0,
        adaptive_target_mass=0.80,
    ) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)
        _reject_production_plan(model)
        if compile_backend not in ("off", "inductor"):
            raise ValueError(
                "unknown compile backend %r" % compile_backend
            )

        hybrid_config = HybridSparseConfig(
            mode=mode,
            video_budget=float(video_budget),
            density_mode=str(density_mode),
            min_video_density=float(min_video_density),
            max_video_density=float(max_video_density),
            adaptive_temperature=float(adaptive_temperature),
            adaptive_target_mass=float(adaptive_target_mass),
            strict=bool(strict),
            run_tag=str(run_tag),
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
                    "Inductor shared-block compilation currently requires "
                    "fixed density_mode; adaptive_budget remains available "
                    "in eager mode"
                )
            if hybrid_config.mode != MODE_SAGE128_FUSED_QKV:
                raise ValueError(
                    "Inductor requires the sage128_fused_qkv attention mode"
                )
            if optimizer_config.activation != MODE_CONVROT_2SLICE:
                raise ValueError(
                    "Inductor requires mlp_chunked_convrot_2slice activation"
                )

        collector = HybridStatsCollector(
            _output_root(),
            hybrid_config.run_tag,
        )
        environment = RuntimeEnvironment.detect()
        kernel_spec = preflight_sparse_sage(
            cuda_available=lambda: environment.cuda_available,
            capability_getter=lambda: environment.capability,
        )
        backend = HybridSparseBackend(
            hybrid_config,
            kernel_spec=kernel_spec,
            collector=collector,
        )
        decision = AttentionDecision(
            requested=ATTENTION_HYBRID,
            selected=ATTENTION_HYBRID,
            backend=backend,
            adapter=ATTENTION_HYBRID,
            reason=(
                "explicit portable Sparse Sage experiment (%s)"
                % hybrid_config.density_mode
            ),
            environment=environment,
            projector=backend.projector,
        )

        patched = model.clone()
        if compile_backend == "inductor":
            request_shared_block_compile(patched)
        apply(
            patched,
            config=optimizer_config,
            decision=decision,
        )

        status = (
            "Experimental Hybrid Sparse armed: mode=%s, density=%s, "
            "video budget=%.1f%%, MLP=%s, compile=%s"
            % (
                hybrid_config.mode,
                hybrid_config.density_mode,
                hybrid_config.video_budget * 100.0,
                optimizer_config.activation,
                compile_backend,
            )
        )
        if hybrid_config.timing:
            status += "\nReports: %s (run tag %s)" % (
                _output_root(),
                hybrid_config.run_tag,
            )
        return io.NodeOutput(
            patched,
            ui=ui.PreviewText(status),
        )


class MiniMaxH3HybridSparseAttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3HybridSparseAttention]
