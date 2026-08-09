from comfy_api.latest import ComfyExtension, io

from .config import (
    H3ChipmunkConfig,
    MODES,
    SCOPES,
    CACHE_LOCATIONS,
    DENSITY_PROFILES,
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MEASURE_LAYER_STRIDE,
    DEFAULT_GPU_STAGING_BUDGET_GB,
)
from .patch import install


class MiniMaxH3ChipmunkMLP(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ChipmunkMLPZi",
            display_name="MiniMax H3 Chipmunk MLP (Zi)",
            category="model/patch/minimax",
            description=(
                "Experimental H3 SwiGLU MLP delta acceleration. MLP math stays on CUDA; "
                "persistent cache state is JIT-offloaded to pinned host RAM through "
                "non-blocking CUDA streams with a bounded VRAM staging pool."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(MODES), default="measure"),
                io.Float.Input("top_fraction", default=0.25, min=0.05, max=1.0, step=0.05),
                io.Int.Input("refresh_every", default=6, min=1, max=50, step=1),
                io.Int.Input("first_dense_steps", default=2, min=0, max=20, step=1),
                io.Int.Input("last_dense_steps", default=2, min=0, max=20, step=1),
                io.Int.Input("first_dense_layers", default=0, min=0, max=50, step=1),
                io.Int.Input("layer_start", default=0, min=0, max=49, step=1),
                io.Int.Input("layer_stop", default=50, min=1, max=50, step=1),
                io.Int.Input("chunk_rows", default=DEFAULT_CHUNK_ROWS, min=128, max=4096, step=128),
                io.Int.Input("token_group_rows", default=128, min=32, max=1024, step=32),
                io.Combo.Input("scope", options=list(SCOPES), default="target_video"),
                io.Combo.Input(
                    "cache_location",
                    options=list(CACHE_LOCATIONS),
                    default="async_pinned",
                    tooltip=(
                        "Persistent state is pinned host backing. H2D/D2H use dedicated CUDA "
                        "streams and events; the model thread never waits on a CPU/device sync."
                    ),
                ),
                io.Float.Input(
                    "cache_budget_gb",
                    default=DEFAULT_GPU_STAGING_BUDGET_GB,
                    min=0.25,
                    max=8.0,
                    step=0.25,
                    tooltip=(
                        "Hard VRAM cap for Chipmunk staging buffers only. Persistent cache is "
                        "offloaded; 1 GiB is intentionally enough for the normal two-slot pipeline."
                    ),
                ),
                io.Float.Input("random_groups", default=0.0, min=0.0, max=0.25, step=0.01),
                io.Boolean.Input("strict", default=True),
                io.Boolean.Input(
                    "save_report",
                    default=False,
                    tooltip="Writes host-native path/count metadata only; never reads CUDA tensors.",
                ),
                io.String.Input("run_tag", default="chipmunk"),
                io.Int.Input(
                    "measure_layer_stride",
                    default=DEFAULT_MEASURE_LAYER_STRIDE,
                    min=1,
                    max=50,
                    step=1,
                    tooltip=(
                        "Retained for saved-workflow compatibility. Production measure is a "
                        "dense no-sync dry-run and does not collect CUDA-valued diagnostics."
                    ),
                ),
                # New fields are appended so old saved widget positions do not shift.
                io.Combo.Input(
                    "density_profile",
                    options=list(DENSITY_PROFILES),
                    default="depth_safe_v1",
                    tooltip=(
                        "depth_safe_v1 keeps blocks 0-10 dense, then uses 40%/50%/60% "
                        "requested density across progressively later depth bands. uniform uses top_fraction."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model, enabled=True, mode="measure", top_fraction=0.25,
        refresh_every=6, first_dense_steps=2, last_dense_steps=2,
        first_dense_layers=0, layer_start=0, layer_stop=50,
        chunk_rows=DEFAULT_CHUNK_ROWS, token_group_rows=128, scope="target_video",
        cache_location="async_pinned", cache_budget_gb=DEFAULT_GPU_STAGING_BUDGET_GB,
        random_groups=0.0, strict=True, save_report=False, run_tag="chipmunk",
        measure_layer_stride=DEFAULT_MEASURE_LAYER_STRIDE,
        density_profile="depth_safe_v1",
    ):
        if not enabled:
            return io.NodeOutput(model)
        config = H3ChipmunkConfig(
            mode=mode,
            top_fraction=float(top_fraction),
            refresh_every=int(refresh_every),
            first_dense_steps=int(first_dense_steps),
            last_dense_steps=int(last_dense_steps),
            first_dense_layers=int(first_dense_layers),
            layer_start=int(layer_start),
            layer_stop=int(layer_stop),
            chunk_rows=int(chunk_rows),
            token_group_rows=int(token_group_rows),
            feature_group=256,
            scope=scope,
            cache_location=cache_location,
            cache_budget_gb=float(cache_budget_gb),
            random_groups=float(random_groups),
            measure_layer_stride=int(measure_layer_stride),
            strict=bool(strict),
            save_report=bool(save_report),
            run_tag=str(run_tag),
            density_profile=str(density_profile),
        )
        patched = model.clone()
        install(patched, config)
        return io.NodeOutput(patched)


class MiniMaxH3ChipmunkExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ChipmunkMLP]
