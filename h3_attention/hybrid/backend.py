"""Prepared H3 hybrid backend, implemented through Phase A Sparse Sage."""

from dataclasses import dataclass

import torch

from .config import HybridSparseConfig, MODE_SAGE128_FUSED_QKV
from .fused_qkv import FusedQKVProjector
from .router import SparseRouterError, SparseTileRouter
from .sparse_sage import SparseSageError, SparseSageExecutor, load_sparse_sage_api
from .stats import DeferredCudaTiming

try:
    from ...h3_runtime.context import get_runtime_snapshot
except ImportError:
    from h3_runtime.context import get_runtime_snapshot


@dataclass
class PreparedHybrid:
    sparse: object
    total_timing: object = None


class HybridSparseBackend:
    name = "hybrid_sparse"
    requires_registered_sage = True
    requires_runtime_context = True
    approximate = True

    def __init__(self, config=None, *, api=None, router=None, collector=None,
                 allow_cpu_for_tests=False, event_factory=None, timing_timer=None,
                 qk_quantizer=None, v_preparer=None, low_level_selector=None,
                 projector=None):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError("config must be HybridSparseConfig")
        self.router = router if router is not None else SparseTileRouter()
        self.collector = collector
        self.runtime_listeners = () if collector is None else (collector,)
        self.timing = DeferredCudaTiming(
            self.config.timing,
            event_factory=event_factory,
            timer=timing_timer,
        )
        if collector is not None:
            attach = getattr(collector, "attach_timing", None)
            if attach is not None:
                attach(self.timing)
        self.strict_runtime_layout = bool(self.config.strict)
        self.projector = projector
        if self.config.mode == MODE_SAGE128_FUSED_QKV and self.projector is None:
            self.projector = FusedQKVProjector()
        self.executor = SparseSageExecutor(
            api if api is not None else load_sparse_sage_api(),
            allow_cpu_for_tests=allow_cpu_for_tests,
            qk_quantizer=qk_quantizer,
            v_preparer=v_preparer,
            low_level_selector=low_level_selector,
        )

    @staticmethod
    def _callable_signature(value):
        if value is None:
            return None
        function = getattr(value, "__func__", value)
        return (
            getattr(function, "__module__", type(function).__module__),
            getattr(function, "__qualname__", type(function).__qualname__),
            id(function),
        )

    @property
    def installation_signature(self):
        collector = self.collector
        projector = self.projector
        api = self.executor.api
        return (
            self.name,
            self.config.signature,
            (type(self.router).__module__, type(self.router).__qualname__),
            None if collector is None else (
                type(collector).__module__,
                type(collector).__qualname__,
                str(collector.output_root),
                str(collector.run_tag),
            ),
            getattr(api, "signature", (
                str(getattr(api, "version", "unknown")),
                id(getattr(api, "low_level_f16", None)),
                id(getattr(api, "low_level_f32", None)),
                id(getattr(api, "v_fused", None)),
            )),
            self._callable_signature(self.executor.qk_quantizer),
            self._callable_signature(self.executor.v_preparer),
            self._callable_signature(self.executor.low_level_selector),
            None if projector is None else (
                type(projector).__module__,
                type(projector).__qualname__,
                getattr(projector, "name", None),
                self._callable_signature(getattr(projector, "tensor_core", None)),
            ),
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

        if self.collector is None:
            self.timing.begin_request(
                snapshot.request_id,
                cuda=getattr(snapshot.device, "type", None) == "cuda",
            )
        total_timing = self.timing.begin("total_hybrid_attention")
        router_timing = self.timing.begin("direct_lut_construction")
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                self.config.video_budget,
            )
        except SparseRouterError as exc:
            self.timing.end(total_timing)
            raise SparseSageError("hybrid routing failed: %s" % exc) from exc
        except Exception:
            self.timing.end(total_timing)
            raise
        finally:
            self.timing.end(router_timing)

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
        try:
            sparse = self.executor.prepare(
                q,
                k,
                v,
                lut,
                valid_block_num,
                layer_index=layer_index,
                metadata=metadata,
                timing=self.timing,
            )
        except Exception:
            self.timing.end(total_timing)
            raise
        return PreparedHybrid(sparse=sparse, total_timing=total_timing)

    def prepare_projected(self, projected, *, layer_index, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            raise SparseSageError("Hybrid Sparse Attention requires an H3 runtime snapshot")
        if not snapshot.valid_layout:
            raise SparseSageError(
                "Hybrid Sparse Attention requires a valid packed layout: %s"
                % (snapshot.error or "layout unavailable")
            )
        if int(snapshot.layout.seq_len) != int(projected.sequence):
            raise SparseSageError(
                "runtime layout sequence %d does not match fused QKV sequence %d"
                % (snapshot.layout.seq_len, projected.sequence)
            )

        if self.collector is None:
            self.timing.begin_request(
                snapshot.request_id,
                cuda=getattr(snapshot.device, "type", None) == "cuda",
            )
        total_timing = self.timing.begin("total_hybrid_attention")
        router_timing = self.timing.begin("direct_lut_construction")
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut_from_summaries(
                projected.q_summary,
                projected.k_summary,
                snapshot.layout,
                self.config.video_budget,
            )
        except SparseRouterError as exc:
            self.timing.end(total_timing)
            raise SparseSageError("hybrid routing failed: %s" % exc) from exc
        except Exception:
            self.timing.end(total_timing)
            raise
        finally:
            self.timing.end(router_timing)

        heads = int(projected.heads)
        metadata = mask_metadata.as_dict()
        metadata.update({
            "request_id": int(snapshot.request_id),
            "step": int(snapshot.step_index),
            "total_steps": int(snapshot.total_steps),
            "branch": [int(x) for x in snapshot.branch],
            "layer": int(layer_index),
            "dense_sage_heads": 0,
            "sparse_sage_heads": heads,
            "sol_heads": 0,
            "flex_fallback_tiles": 0,
            "total_128q_video_tiles": (
                int(mask_metadata.pure_video_q_tiles) * heads
            ),
        })
        try:
            sparse = self.executor.prepare_projected(
                projected,
                lut,
                valid_block_num,
                metadata=metadata,
                timing=self.timing,
            )
        except Exception:
            self.timing.end(total_timing)
            raise
        return PreparedHybrid(sparse=sparse, total_timing=total_timing)

    def execute(self, prepared):
        try:
            output = self.executor.execute(prepared.sparse)
            if self.collector is not None and not torch.compiler.is_compiling():
                self.collector.record(prepared.sparse.metadata)
            return output
        finally:
            self.timing.end(prepared.total_timing)

    def as_status(self):
        return {
            "phase": "A",
            "mode": self.config.mode,
            "video_budget": float(self.config.video_budget),
            "sparge_attention": self.executor.api.version,
            "approximate": True,
            "timing": bool(self.config.timing),
            "fused_qkv": self.projector is not None,
            "smooth_k": False if self.projector is not None else True,
        }
