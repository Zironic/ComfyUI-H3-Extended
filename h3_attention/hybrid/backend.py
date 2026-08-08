"""Prepared H3 hybrid backend, implemented through Phase A Sparse Sage."""

from dataclasses import dataclass

from .config import HybridSparseConfig
from .router import SparseRouterError, SparseTileRouter
from .sparse_sage import SparseSageError, SparseSageExecutor, load_sparse_sage_api

try:
    from ...h3_runtime.context import get_runtime_snapshot
except ImportError:
    from h3_runtime.context import get_runtime_snapshot


@dataclass
class PreparedHybrid:
    sparse: object


class HybridSparseBackend:
    name = "hybrid_sparse"
    requires_registered_sage = True
    requires_runtime_context = True
    approximate = True

    def __init__(self, config=None, *, api=None, router=None, collector=None,
                 allow_cpu_for_tests=False):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError("config must be HybridSparseConfig")
        self.router = router if router is not None else SparseTileRouter()
        self.collector = collector
        self.runtime_listeners = () if collector is None else (collector,)
        self.strict_runtime_layout = bool(self.config.strict)
        self.executor = SparseSageExecutor(
            api if api is not None else load_sparse_sage_api(),
            allow_cpu_for_tests=allow_cpu_for_tests,
        )

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            raise SparseSageError("Hybrid Sparse Attention requires an H3 runtime snapshot")
        if not snapshot.valid_layout:
            raise SparseSageError(
                "Hybrid Sparse Attention requires a valid packed layout: %s"
                % (snapshot.error or "layout unavailable")
            )
        if int(snapshot.layout.seq_len) != q.shape[-2]:
            raise SparseSageError(
                "runtime layout sequence %d does not match attention sequence %d"
                % (snapshot.layout.seq_len, q.shape[-2])
            )

        # These are the Phase-A owned copies. Routing and Sparse Sage share them;
        # the H3 forward can release the fused source allocation afterwards.
        q_owned = q.contiguous()
        k_owned = k.contiguous()
        try:
            mask_id, mask_metadata = self.router.build_mask(
                q_owned,
                k_owned,
                snapshot.layout,
                self.config.video_budget,
            )
        except SparseRouterError as exc:
            raise SparseSageError("hybrid routing failed: %s" % exc) from exc

        metadata = mask_metadata.as_dict()
        metadata.update({
            "request_id": int(snapshot.request_id),
            "step": int(snapshot.step_index),
            "total_steps": int(snapshot.total_steps),
            "branch": [int(x) for x in snapshot.branch],
            "layer": int(layer_index),
            "dense_sage_heads": 0,
            "sparse_sage_heads": int(q.shape[1]),
            "sol_heads": 0,
            "flex_fallback_tiles": 0,
            "total_128q_video_tiles": (
                int(mask_metadata.pure_video_q_tiles) * int(q.shape[1])
            ),
        })
        sparse = self.executor.prepare(
            q_owned,
            k_owned,
            v,
            mask_id,
            layer_index=layer_index,
            metadata=metadata,
        )
        return PreparedHybrid(sparse=sparse)

    def execute(self, prepared):
        output = self.executor.execute(prepared.sparse)
        if self.collector is not None:
            self.collector.record(prepared.sparse.metadata)
        return output

    def as_status(self):
        return {
            "phase": "A",
            "mode": self.config.mode,
            "video_budget": float(self.config.video_budget),
            "sparge_attention": self.executor.api.version,
            "approximate": True,
        }
