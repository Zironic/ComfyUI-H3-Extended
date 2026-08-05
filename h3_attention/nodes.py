"""ComfyUI node exposing the opt-in efficient H3 Sage backend."""

from comfy_api.latest import ComfyExtension, io

from .config import configure


class MiniMaxH3EfficientSagePatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3EfficientSagePatchZi",
            display_name="MiniMax H3 Efficient Sage Patch (Zi)",
            category="model/patch/minimax",
            description=(
                "Experimental SM89-only H3 SageAttention path. Quantizes Q/K with "
                "local int64-safe Triton kernels and releases the fused BF16 QKV "
                "projection before the attention kernel. Place after the H3 Sigma "
                "Shift node. Fails explicitly on unsupported GPUs or Sage versions."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(model)
        patched = model.clone()
        configure(patched)
        return io.NodeOutput(patched)


class MiniMaxH3AttentionExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3EfficientSagePatch]
