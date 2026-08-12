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
from .h3_chipmunk.nodes import MiniMaxH3ChipmunkExtension
from .h3_masked_cache.nodes import MiniMaxH3MaskedCacheExtension
from .h3_memory_optimizer.nodes import MiniMaxH3MemoryOptimizerExtension
from .h3_probe.nodes import MiniMaxH3ProbeExtension
from .h3_sage_optimizations.nodes import MiniMaxH3SageOptimizationsExtension
from .h3_sparse_attention.nodes import MiniMaxH3HybridSparseAttentionExtension
from .h3_vector_accel.nodes import MiniMaxH3VectorAccelExtension

WEB_DIRECTORY = "./web"

v2v_audio_runtime.install()
aligned_source_runtime.install()
completed_preview_runtime.install()
taeh3_latent_preview.install()
opening_picture_runtime.install()

encode_conditioning_diagnostic = cond_cache_diagnostics.install()
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
    "MiniMaxH3ChipmunkMLPZi": "H3-Extender/Experiments",
    "MiniMaxH3MaskedRef2VCacheZi": "H3-Extender/Model Patches",
    "MiniMaxH3MemoryOptimizerZi": "H3-Extender/Model Patches",
    "MiniMaxH3SageMemoryOptimizerZi": "H3-Extender/Model Patches",
    "MiniMaxH3SparseSageAttentionZi": "H3-Extender/Model Patches",
    "MiniMaxH3SolEngineZi": "H3-Extender/Experiments",
    "MiniMaxH3HybridSparseAttentionZi": "H3-Extender/Compatibility",
    "MiniMaxH3AttentionProbeZi": "H3-Extender/Diagnostics",
    "MiniMaxH3Ref2VExperimentHarnessZi": "H3-Extender/Experiments",
    "MiniMaxH3LongFormRef2VZi": "H3-Extender/Experiments",
    "MiniMaxH3BridgeExperimentZi": "H3-Extender/Experiments",
    "MiniMaxH3VectorAccelSamplerZi": "H3-Extender/Experiments",
}


def _replace_schema_category(schema, category):
    if is_dataclass(schema):
        return replace(schema, category=category)
    if hasattr(schema, "model_copy"):
        return schema.model_copy(update={"category": category})
    schema.category = category
    return schema


def _categorized_node(node):
    schema = node.define_schema()
    category = NODE_CATEGORIES.get(schema.node_id)
    if category is None:
        return node

    class CategorizedNode(node):
        @classmethod
        def define_schema(cls):
            return _replace_schema_category(
                super().define_schema(), category
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
            MiniMaxH3ChipmunkExtension(),
            MiniMaxH3MemoryOptimizerExtension(),
            MiniMaxH3SageOptimizationsExtension(),
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
