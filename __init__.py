"""ComfyUI-H3-Extended: forked MiniMax H3 nodes and experimental tooling."""

from dataclasses import is_dataclass, replace

from comfy_api.latest import ComfyExtension

from . import nodes_minimax_h3
from .chunked_ref2v.nodes import MiniMaxH3HarnessExtension
from .cond_cache_diagnostics import encode as encode_conditioning_diagnostic
from .h3_activation_memory.nodes import MiniMaxH3ActivationMemoryExtension
from .h3_attention.nodes import MiniMaxH3AttentionExtension
from .h3_masked_cache.nodes import MiniMaxH3MaskedCacheExtension
from .h3_memory_optimizer.nodes import MiniMaxH3MemoryOptimizerExtension
from .h3_probe.nodes import MiniMaxH3ProbeExtension

# The conditioning node module binds cond_cache.encode at import time. Replace
# that module-level seam with a transparent diagnostic wrapper; the wrapper
# delegates to the original cache without changing key or storage behaviour.
nodes_minimax_h3.encode_conditioning = encode_conditioning_diagnostic
MiniMaxH3Extension = nodes_minimax_h3.MiniMaxH3Extension


NODE_CATEGORIES = {
    "EmptyMiniMaxH3LatentAVZi": "H3-Extender/Generation",
    "MiniMaxH3ImageToVideoZi": "H3-Extender/Generation",
    "MiniMaxH3ReferenceToVideoZi": "H3-Extender/Generation",
    "MiniMaxH3SigmaShiftZi": "H3-Extender/Model Patches",
    "MiniMaxH3EfficientSagePatchZi": "H3-Extender/Model Patches",
    "MiniMaxH3ActivationMemoryZi": "H3-Extender/Model Patches",
    "MiniMaxH3MaskedRef2VCacheZi": "H3-Extender/Model Patches",
    "MiniMaxH3MemoryOptimizerZi": "H3-Extender/Model Patches",
    "MiniMaxH3AttentionProbeZi": "H3-Extender/Diagnostics",
    "MiniMaxH3Ref2VExperimentHarnessZi": "H3-Extender/Experiments",
}


def _replace_schema_category(schema, category):
    """Return the schema with a new category across supported schema models."""
    if is_dataclass(schema):
        return replace(schema, category=category)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"category": category})
    schema.category = category
    return schema


def _categorized_node(node):
    """Wrap a node class so its existing schema is exposed under our menu."""
    schema = node.define_schema()
    category = NODE_CATEGORIES.get(schema.node_id)
    if category is None:
        return node

    class CategorizedNode(node):
        @classmethod
        def define_schema(cls):
            return _replace_schema_category(
                super().define_schema(),
                category,
            )

    CategorizedNode.__name__ = node.__name__
    CategorizedNode.__qualname__ = node.__qualname__
    CategorizedNode.__module__ = node.__module__
    return CategorizedNode


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
            MiniMaxH3MemoryOptimizerExtension(),
        ):
            nodes.extend(_categorized_node(node) for node in await ext.get_node_list())
        return nodes


async def comfy_entrypoint() -> H3ExtendedExtension:
    return H3ExtendedExtension()
