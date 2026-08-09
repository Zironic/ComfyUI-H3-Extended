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
from .chunked_ref2v.longform.av_continuation_nodes import (
    MiniMaxH3LongFormAVContinuationExtension,
)
from .chunked_ref2v.longform.chunk_prompt_timeline import (
    MiniMaxH3ChunkPromptTimelineExtension,
)
from .chunked_ref2v.longform.nplusone_chunk_prompt_timeline import (
    MiniMaxH3NPlusOneChunkPromptTimelineExtension,
)
from .chunked_ref2v.longform import (
    aligned_source_runtime,
    completed_preview_runtime,
    opening_picture_runtime,
    v2v_audio_runtime,
)
from .chunked_ref2v.nodes import MiniMaxH3HarnessExtension
from .chunked_ref2v.bridge.nodes import MiniMaxH3BridgeExtension
from . import cond_cache_diagnostics
from . import taeh3_latent_preview
from .diagnostics.nodes import MiniMaxH3DiagnosticsExtension
from .h3_activation_memory.nodes import MiniMaxH3ActivationMemoryExtension
from .h3_attention.nodes import MiniMaxH3AttentionExtension
from .h3_masked_cache.nodes import MiniMaxH3MaskedCacheExtension
from .h3_memory_optimizer.nodes import MiniMaxH3MemoryOptimizerExtension
from .h3_probe.nodes import MiniMaxH3ProbeExtension
from .h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttentionExtension
from .h3_vector_accel.nodes import MiniMaxH3VectorAccelExtension

# Comfy serves JavaScript from this directory. The long-form nodes use it for
# two independent live preview panes because the standard progress protocol
# carries only one replaceable preview image.
WEB_DIRECTORY = "./web"

# The preview module patches the original Ref2V sampling seam first. Install the
# audiovisual runtime afterwards so it retains those preview callbacks while
# replacing the duplicated video-only sampler with synchronized video+audio
# carry, source-audio conditioning, and dual-track output.
v2v_audio_runtime.install()

# Optional same-time source AV conditioning for the native N+1 continuation
# node. This is appended after all existing widgets so saved workflows keep
# their positional values. The source reference can remain at source/native
# resolution while the generated target uses an independent larger canvas.
aligned_source_runtime.install()

# Keep finalized MP4 segments as the primary completed-output preview, but make
# that channel failure-safe: GIF fallback first, explicit browser error second.
completed_preview_runtime.install()

# Give every other way of sampling H3 the same fast decoder the long-form nodes
# use. Core cannot find taeh3 by itself and would mis-build it if it did, so the
# patch lives here rather than in the auto-updating checkout.
taeh3_latent_preview.install()

# Optional long-form reference continuity: retain the decoded frame that maps
# to the next chunk's frame zero and inject it as the next <Picture N> reference
# for a just-in-time Qwen conditioning pass. This installs after the prompt,
# audio-reference, and preview patches so it composes with all three.
opening_picture_runtime.install()

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
    "MiniMaxH3LongFormAVContinuationZi": "H3-Extender/Generation",
    "MiniMaxH3ChunkPromptTimelineZi": "H3-Extender/Generation",
    "MiniMaxH3NPlusOneChunkPromptTimelineZi": "H3-Extender/Generation",
    "MiniMaxH3SigmaShiftZi": "H3-Extender/Model Patches",
    "MiniMaxH3EfficientSagePatchZi": "H3-Extender/Model Patches",
    "MiniMaxH3ActivationMemoryZi": "H3-Extender/Model Patches",
    "MiniMaxH3MaskedRef2VCacheZi": "H3-Extender/Model Patches",
    "MiniMaxH3MemoryOptimizerZi": "H3-Extender/Model Patches",
    "MiniMaxH3SolEngineZi": "H3-Extender/Experiments",
    "MiniMaxH3HybridSparseAttentionZi": "H3-Extender/Experiments",
    "MiniMaxH3AttentionProbeZi": "H3-Extender/Diagnostics",
    "MiniMaxH3Ref2VExperimentHarnessZi": "H3-Extender/Experiments",
    "MiniMaxH3LongFormRef2VZi": "H3-Extender/Experiments",
    "MiniMaxH3BridgeExperimentZi": "H3-Extender/Experiments",
    "MiniMaxH3VectorAccelSamplerZi": "H3-Extender/Experiments",
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
    """Wrap a node class with the H3 category."""

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
            MiniMaxH3BridgeExtension(),
            MiniMaxH3ChunkPromptTimelineExtension(),
            MiniMaxH3NPlusOneChunkPromptTimelineExtension(),
            MiniMaxH3LongFormPreviewExtension(),
            MiniMaxH3LongFormReferencePreviewExtension(),
            MiniMaxH3LongFormAVContinuationExtension(),
            MiniMaxH3MaskedCacheExtension(),
            MiniMaxH3AttentionExtension(),
            MiniMaxH3ActivationMemoryExtension(),
            MiniMaxH3MemoryOptimizerExtension(),
            MiniMaxH3HybridSparseAttentionExtension(),
            MiniMaxH3VectorAccelExtension(),
            MiniMaxH3DiagnosticsExtension(),
        ):
            nodes.extend(
                _categorized_node(node)
                for node in await ext.get_node_list()
            )
        return nodes


async def comfy_entrypoint() -> H3ExtendedExtension:
    return H3ExtendedExtension()
