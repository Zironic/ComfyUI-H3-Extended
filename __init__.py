"""ComfyUI-H3-Extended: forked MiniMax H3 nodes and experimental tooling."""

from dataclasses import is_dataclass, replace

from comfy_api.latest import ComfyExtension

from . import nodes_minimax_h3
from .chunked_ref2v.longform.preview_nodes import (
    MiniMaxH3LongFormPreviewExtension,
)
from .chunked_ref2v.longform.reference_preview_nodes import (
    MiniMaxH3LongFormReferencePreviewExtension,
)
from .chunked_ref2v.longform.video_output_contract import adapt_longform_node
from .chunked_ref2v.longform.chunk_prompt_timeline import (
    MiniMaxH3ChunkPromptTimelineExtension,
)
from .chunked_ref2v.longform import (
    completed_preview_runtime,
    v2v_audio_runtime,
)
from .chunked_ref2v.nodes import MiniMaxH3HarnessExtension
from . import cond_cache_diagnostics
from .h3_activation_memory.nodes import MiniMaxH3ActivationMemoryExtension
from .h3_attention.nodes import MiniMaxH3AttentionExtension
from .h3_masked_cache.nodes import MiniMaxH3MaskedCacheExtension
from .h3_memory_optimizer.nodes import MiniMaxH3MemoryOptimizerExtension
from .h3_probe.nodes import MiniMaxH3ProbeExtension

# Comfy serves JavaScript from this directory. The long-form nodes use it for
# two independent live preview panes because the standard progress protocol
# carries only one replaceable preview image.
WEB_DIRECTORY = "./web"

# The preview module patches the original Ref2V sampling seam first. Install the
# audiovisual runtime afterwards so it retains those preview callbacks while
# replacing the duplicated video-only sampler with synchronized video+audio
# carry, source-audio conditioning, and dual-track output.
v2v_audio_runtime.install()

# Keep finalized MP4 segments as the primary completed-output preview, but make
# that channel failure-safe: GIF fallback first, explicit browser error second.
completed_preview_runtime.install()

# Install the diagnostic wrapper at `cond_cache.encode` itself. Rebinding only
# `nodes_minimax_h3.encode_conditioning` covered the (Zi) nodes but silently
# missed the harness, which resolves the function inside a function body — so
# the diagnostics never once ran on the path that most needed them.
encode_conditioning_diagnostic = cond_cache_diagnostics.install()
# ...and the conditioning nodes bound the name at import, before the patch.
nodes_minimax_h3.encode_conditioning = encode_conditioning_diagnostic
MiniMaxH3Extension = nodes_minimax_h3.MiniMaxH3Extension


NODE_CATEGORIES = {
    "EmptyMiniMaxH3LatentAVZi": "H3-Extender/Generation",
    "MiniMaxH3ImageToVideoZi": "H3-Extender/Generation",
    "MiniMaxH3ReferenceToVideoZi": "H3-Extender/Generation",
    "MiniMaxH3LongFormReferenceVideoZi": "H3-Extender/Generation",
    "MiniMaxH3ChunkPromptTimelineZi": "H3-Extender/Generation",
    "MiniMaxH3SigmaShiftZi": "H3-Extender/Model Patches",
    "MiniMaxH3EfficientSagePatchZi": "H3-Extender/Model Patches",
    "MiniMaxH3ActivationMemoryZi": "H3-Extender/Model Patches",
    "MiniMaxH3MaskedRef2VCacheZi": "H3-Extender/Model Patches",
    "MiniMaxH3MemoryOptimizerZi": "H3-Extender/Model Patches",
    "MiniMaxH3AttentionProbeZi": "H3-Extender/Diagnostics",
    "MiniMaxH3Ref2VExperimentHarnessZi": "H3-Extender/Experiments",
    "MiniMaxH3LongFormRef2VZi": "H3-Extender/Experiments",
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
    """Wrap a node class with the H3 category and native video contract."""

    node = adapt_longform_node(node)
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
            MiniMaxH3ChunkPromptTimelineExtension(),
            MiniMaxH3LongFormPreviewExtension(),
            MiniMaxH3LongFormReferencePreviewExtension(),
            MiniMaxH3MaskedCacheExtension(),
            MiniMaxH3AttentionExtension(),
            MiniMaxH3ActivationMemoryExtension(),
            MiniMaxH3MemoryOptimizerExtension(),
        ):
            nodes.extend(
                _categorized_node(node)
                for node in await ext.get_node_list()
            )
        return nodes


async def comfy_entrypoint() -> H3ExtendedExtension:
    return H3ExtendedExtension()
