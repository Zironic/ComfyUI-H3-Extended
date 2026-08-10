"""Sparse-Sage projected-QKV adapter selected outside the sparse backend."""

try:
    from ..h3_attention.hybrid.fused_qkv import FusedQKVProjector
except ImportError:
    from h3_attention.hybrid.fused_qkv import FusedQKVProjector

SPARSE_QK_FORMAT = "sparge_block_int8"


class SparseFusedQKVProjector(FusedQKVProjector):
    """Label the established sparse projector as one negotiated Q/K format."""

    qk_format = SPARSE_QK_FORMAT

    @property
    def installation_signature(self):
        return (super().installation_signature, self.qk_format)
