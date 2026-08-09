from comfy_api.latest import ComfyExtension, io

from .config import (
    H3ChipmunkConfig,
    MODES,
    SCOPES,
    CACHE_LOCATIONS,
    DEFAULT_CHUNK_ROWS,
    DEFAULT_MEASURE_LAYER_STRIDE,
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
                "Experimental training-free H3 SwiGLU MLP delta acceleration. "
                "measure is output-exact and records lightweight ConvRot group dynamics; "
                "reference_delta enables approximate cached sparse-delta execution. "
                "Use after the H3 attention patch and instead of Activation Memory/shared block compile."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Combo.Input("mode", options=list(MODES), default="measure"),
                io.Float.Input("top_fraction", default=0.25, min=0.05, max=1.0, step=0.05),
                io.Int.Input("refresh_every", default=6, min=1, max=50, step=1),
                io.Int.Input("first_dense_steps", default=2, min=0, max=20, step=1),
                io.Int.Input("last_dense_steps", default=2, min=0, max=20, step=1),
                io.Int.Input("first_dense_layers", default=2, min=0, max=50, step=1),
                io.Int.Input("layer_start", default=0, min=0, max=49, step=1),
                io.Int.Input("layer_stop", default=50, min=1, max=50, step=1),
                io.Int.Input("chunk_rows", default=DEFAULT_CHUNK_ROWS, min=128, max=4096, step=128),
                io.Int.Input("token_group_rows", default=128, min=32, max=1024, step=32),
                io.Combo.Input("scope", options=list(SCOPES), default="target_video"),
                io.Combo.Input("cache_location", options=list(CACHE_LOCATIONS), default="cpu"),
                io.Float.Input("cache_budget_gb", default=24.0, min=1.0, max=512.0, step=1.0),
                io.Float.Input("random_groups", default=0.0, min=0.0, max=0.25, step=0.01),
                io.Boolean.Input("strict", default=True),
                io.Boolean.Input("save_report", default=True),
                io.String.Input("run_tag", default="chipmunk"),
                # Append new widgets after all original fields so saved Comfy
                # workflows retain their positional widget values.
                io.Int.Input(
                    "measure_layer_stride",
                    default=DEFAULT_MEASURE_LAYER_STRIDE,
                    min=1,
                    max=50,
                    step=1,
                    tooltip=(
                        "measure mode observes every Nth main block plus the last block. "
                        "5 is the low-overhead default; 1 measures all 50 blocks."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls, model, enabled=True, mode="measure", top_fraction=0.25,
        refresh_every=6, first_dense_steps=2, last_dense_steps=2,
        first_dense_layers=2, layer_start=0, layer_stop=50,
        chunk_rows=DEFAULT_CHUNK_ROWS, token_group_rows=128, scope="target_video",
        cache_location="cpu", cache_budget_gb=24.0, random_groups=0.0,
        strict=True, save_report=True, run_tag="chipmunk",
        measure_layer_stride=DEFAULT_MEASURE_LAYER_STRIDE,
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
        )
        patched = model.clone()
        install(patched, config)
        return io.NodeOutput(patched)


class MiniMaxH3ChipmunkExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ChipmunkMLP]
