"""Prepared H3 hybrid backend, implemented through portable Sparse Sage."""

from dataclasses import dataclass

import torch

from .config import (
    HybridSparseConfig,
    MODE_SAGE128_FUSED_QKV,
    resolve_video_budget,
)
from .fused_qkv import FusedQKVProjector
from .router import SparseRouterError, SparseTileRouter
from .sparse_sage import SparseSageError, SparseSageExecutor, load_sparse_sage_spec
from .stats import DeferredCudaTiming, ROUTE_HISTOGRAM_KEY, build_route_histogram

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

    def __init__(self, config=None, *, kernel_spec=None, router=None, collector=None,
                 allow_cpu_for_tests=False, event_factory=None, timing_timer=None,
                 qk_quantizer=None, v_preparer=None, low_level_selector=None,
                 projector=None):
        self.config = config or HybridSparseConfig()
        if not isinstance(self.config, HybridSparseConfig):
            raise TypeError("config must be HybridSparseConfig")
        kernel_spec = kernel_spec if kernel_spec is not None else load_sparse_sage_spec()
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
            kernel_spec,
            allow_cpu_for_tests=allow_cpu_for_tests,
            qk_quantizer=qk_quantizer,
            v_preparer=v_preparer,
            low_level_selector=low_level_selector,
        )
        self.router = (
            router if router is not None else SparseTileRouter(
                self.config, spec=self.executor.spec
            )
        )
        if (self.router.q_tile, self.router.kv_tile) != (
                self.executor.spec.q_tile, self.executor.spec.kv_tile):
            raise SparseSageError(
                "Sparse Sage router geometry %dQ x %dKV does not match %s's %dQ x %dKV ABI"
                % (self.router.q_tile, self.router.kv_tile,
                   self.executor.spec.architecture, self.executor.spec.q_tile,
                   self.executor.spec.kv_tile)
            )
        if self.config.mode == MODE_SAGE128_FUSED_QKV and (
                self.executor.spec.capability != (8, 9)
                or self.executor.spec.q_tile != 128
                or self.executor.spec.kv_tile != 64):
            raise SparseSageError(
                "sage128_fused_qkv requires SM89 and the 128Q x 64KV Sparse Sage ABI"
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
        spec = self.executor.spec
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
            spec.signature,
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

    @staticmethod
    def _attach_route_telemetry(metadata, valid_block_num, mask_metadata):
        histogram = build_route_histogram(valid_block_num, mask_metadata)
        if histogram is not None:
            metadata[ROUTE_HISTOGRAM_KEY] = histogram

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
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut(
                q,
                k,
                snapshot.layout,
                video_budget,
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
            "total_q_video_tiles": (
                int(mask_metadata.pure_video_q_tiles) * int(q.shape[1])
            ),
        })
        self._attach_route_telemetry(metadata, valid_block_num, mask_metadata)
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
        video_budget = resolve_video_budget(
            self.config,
            snapshot.step_index,
            snapshot.total_steps,
        )
        try:
            lut, valid_block_num, mask_metadata = self.router.build_lut_from_summaries(
                projected.q_summary,
                projected.k_summary,
                snapshot.layout,
                video_budget,
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
            "total_q_video_tiles": (
                int(mask_metadata.pure_video_q_tiles) * heads
            ),
        })
        self._attach_route_telemetry(metadata, valid_block_num, mask_metadata)
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
            "denser_early_late_steps": bool(
                self.config.denser_early_late_steps
            ),
            "density_mode": self.config.density_mode,
            "min_video_density": float(self.config.min_video_density),
            "max_video_density": float(self.config.max_video_density),
            "adaptive_temperature": float(self.config.adaptive_temperature),
            "adaptive_target_mass": float(self.config.adaptive_target_mass),
            "sparge_attention": self.executor.spec.version,
            "sparse_architecture": self.executor.spec.architecture,
            "sparse_q_tile": self.executor.spec.q_tile,
            "sparse_kv_tile": self.executor.spec.kv_tile,
            "sparse_v_format": self.executor.spec.v_format,
            "sparse_v_quant_bound": self.executor.spec.v_quant_bound,
            "sparse_extension_layout": self.executor.spec.extension_layout,
            "approximate": True,
            "timing": bool(self.config.timing),
            "fused_qkv": self.projector is not None,
            "smooth_k": False if self.projector is not None else True,
        }
