"""ComfyUI-H3-Extended: forked MiniMax H3 nodes and experimental tooling."""

from comfy_api.latest import ComfyExtension

from .chunked_ref2v.nodes import MiniMaxH3HarnessExtension
from .h3_activation_memory.nodes import MiniMaxH3ActivationMemoryExtension
from .h3_attention.nodes import MiniMaxH3AttentionExtension
from .h3_masked_cache.nodes import MiniMaxH3MaskedCacheExtension
from .h3_probe.nodes import MiniMaxH3ProbeExtension
from .nodes_minimax_h3 import MiniMaxH3Extension


class H3ExtendedExtension(ComfyExtension):
    async def get_node_list(self):
        nodes = []
        for ext in (
            MiniMaxH3Extension(),
            MiniMaxH3ProbeExtension(),
            MiniMaxH3HarnessExtension(),
            MiniMaxH3MaskedCacheExtension(),
            MiniMaxH3AttentionExtension(),
            MiniMaxH3ActivationMemoryExtension(),
        ):
            nodes.extend(await ext.get_node_list())
        return nodes


async def comfy_entrypoint() -> H3ExtendedExtension:
    return H3ExtendedExtension()
