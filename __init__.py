"""ComfyUI-H3-Extended: MiniMax H3 nodes forked from core, plus the attention probe."""

from comfy_api.latest import ComfyExtension

from .h3_probe.nodes import MiniMaxH3ProbeExtension
from .nodes_minimax_h3 import MiniMaxH3Extension


class H3ExtendedExtension(ComfyExtension):
    async def get_node_list(self):
        nodes = []
        for ext in (MiniMaxH3Extension(), MiniMaxH3ProbeExtension()):
            nodes.extend(await ext.get_node_list())
        return nodes


async def comfy_entrypoint() -> H3ExtendedExtension:
    return H3ExtendedExtension()
