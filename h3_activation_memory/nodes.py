"""ComfyUI node for experimental H3 activation-memory execution."""

import logging

from comfy_api.latest import ComfyExtension, io

from .config import (
    DEFAULT_CHUNK_ROWS,
    IMPLEMENTED_MODES,
    ActivationMemoryConfig,
)
from .patch import install

LOG_PREFIX = "[H3 activation memory]"


class MiniMaxH3ActivationMemory(io.ComfyNode):
    """Bound H3's MLP activations by evaluating token slabs sequentially."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ActivationMemoryZi",
            display_name="MiniMax H3 Activation Memory (Zi)",
            category="model/patch/minimax",
            description=(
                "Experimental exact-graph activation-lifetime optimization. "
                "Keeps H3's persistent residual stream in BF16 but evaluates "
                "the tokenwise MLP in bounded slabs, avoiding the full "
                "[sequence, 28672] BF16 expansion. This is separate from the "
                "attention backend and composes with the H3 attention patch."
            ),
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input(
                    "enabled",
                    default=True,
                    tooltip="False returns the input MODEL unchanged.",
                ),
                io.Combo.Input(
                    "mode",
                    options=sorted(IMPLEMENTED_MODES),
                    default="mlp_chunked_bf16",
                    tooltip=(
                        "mlp_chunked_bf16 materializes a bounded BF16 SwiGLU "
                        "slab before fc2. mlp_chunked_native uses Comfy's fused "
                        "TensorWise-INT8 SwiGLU path when available."
                    ),
                ),
                io.Int.Input(
                    "chunk_rows",
                    default=DEFAULT_CHUNK_ROWS,
                    min=256,
                    max=65_536,
                    step=256,
                    tooltip=(
                        "Maximum packed-token rows per MLP slab. 4096 is the "
                        "initial conservative value; larger slabs use more "
                        "VRAM but usually launch more efficient GEMMs."
                    ),
                ),
                io.Boolean.Input(
                    "prefer_held_weights",
                    default=True,
                    tooltip=(
                        "Acquire fc1/fc2 once per block when Comfy's async cast "
                        "buffers are distinct. Unsafe single-buffer cases fall "
                        "back to ordinary per-slab module calls."
                    ),
                ),
                io.Boolean.Input(
                    "strict",
                    default=True,
                    tooltip=(
                        "Raise on unsupported weight acquisition, core drift, "
                        "or torch.compile instead of silently running a "
                        "different path. Safe same-graph module fallback for an "
                        "async cast-buffer conflict remains allowed."
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
        mode="mlp_chunked_bf16",
        chunk_rows=DEFAULT_CHUNK_ROWS,
        prefer_held_weights=True,
        strict=True,
    ):
        if not enabled:
            return io.NodeOutput(model)

        config = ActivationMemoryConfig(
            mode=mode,
            chunk_rows=int(chunk_rows),
            strict=bool(strict),
            prefer_held_weights=bool(prefer_held_weights),
        )
        patched = model.clone()
        count = install(patched, config)
        logging.info(
            "%s armed: blocks=%d mode=%s chunk_rows=%d strict=%s "
            "prefer_held_weights=%s",
            LOG_PREFIX,
            count,
            config.mode,
            config.chunk_rows,
            config.strict,
            config.prefer_held_weights,
        )
        return io.NodeOutput(patched)


class MiniMaxH3ActivationMemoryExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3ActivationMemory]
