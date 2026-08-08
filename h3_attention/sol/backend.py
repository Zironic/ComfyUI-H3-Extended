"""Optional Sol-Attn backend for MiniMax H3.

Sol-Attn is an approximate sibling of the prepared dense Sage backends.  Dense
warmup steps/layers delegate to the architecture-selected prepared Sage backend;
sparse calls copy post-RoPE Q/K/V into the contiguous BF16 BTHD layout required
by the released Sol kernel, release the fused source allocation, and then route.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import logging
import math
import os
import sys

import torch
import torch.nn.functional as F

from .config import SolAttentionConfig
from .policy import decline_reason, exact_sink
from . import stats

try:
    from ...h3_runtime.context import get_runtime_snapshot
    from ...h3_runtime.metrics import tensor_error_metrics
except ImportError:
    from h3_runtime.context import get_runtime_snapshot
    from h3_runtime.metrics import tensor_error_metrics

LOG_PREFIX = "[H3 Sol-Attn]"


class SolAttentionError(RuntimeError):
    pass


def load_sol_attention():
    root = os.environ.get("H3_SOL_ATTN_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    errors = []
    for module_name in ("sol_attn", "techniques.sparse_backends.sol_attn"):
        try:
            module = importlib.import_module(module_name)
            fn = getattr(module, "sol_attn", None)
            if callable(fn):
                return fn, module
            errors.append("%s has no sol_attn callable" % module_name)
        except Exception as exc:
            errors.append("%s: %s: %s" % (module_name, type(exc).__name__, exc))
    raise SolAttentionError(
        "optional Sol-Attn package is unavailable; install the released "
        "NVlabs/Sana sol-engine sparse backend or set H3_SOL_ATTN_ROOT. "
        + "; ".join(errors)
    )


def preflight_sol_attention(environment=None, *, run_probe=True, kv_splits=1):
    if not torch.cuda.is_available():
        raise SolAttentionError("Sol-Attn requires CUDA")
    if environment is not None and environment.capability is not None:
        capability = tuple(environment.capability)
    else:
        capability = tuple(torch.cuda.get_device_capability())
    if capability[0] < 8:
        raise SolAttentionError("Sol-Attn requires SM80 or newer; got SM%d%d" % capability)
    fn, module = load_sol_attention()
    probe = None
    if run_probe:
        device_index = (
            int(environment.device_index)
            if environment is not None and environment.device_index is not None
            else torch.cuda.current_device()
        )
        device = torch.device("cuda", device_index)
        try:
            generator = torch.Generator(device=device).manual_seed(7391)
            tensors = [
                torch.randn(1, 128, 1, 128, device=device, dtype=torch.bfloat16, generator=generator)
                for _ in range(3)
            ]
            out = fn(
                *tensors,
                tau=-1000.0,
                thresh_type="diag",
                kv_splits=int(kv_splits),
            )
            torch.cuda.synchronize(device)
            if tuple(out.shape) != tuple(tensors[0].shape):
                raise RuntimeError(
                    "probe returned %s, expected %s"
                    % (tuple(out.shape), tuple(tensors[0].shape))
                )
            if not bool(torch.isfinite(out).all().item()):
                raise RuntimeError("probe returned non-finite output")
            probe = {"shape": tuple(out.shape), "dtype": str(out.dtype)}
            del out, tensors
        except Exception as exc:
            raise SolAttentionError(
                "Sol-Attn compile/run preflight failed on SM%d%d: %s: %s"
                % (capability[0], capability[1], type(exc).__name__, exc)
            ) from exc
    return {
        "callable": fn,
        "module": getattr(module, "__name__", type(module).__name__),
        "capability": capability,
        "version": getattr(module, "__version__", None),
        "probe": probe,
        "kv_splits": int(kv_splits),
    }


def _dense_bthd(q, k, v):
    # BTHD -> BHSD for SDPA -> BTHD
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)




@dataclass
class PreparedDenseBF16:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


class DenseBF16SDPABackend:
    """Portable consuming dense fallback for Sol warmup steps/layers."""

    name = "dense_sdpa_bf16"
    requires_registered_sage = False

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        # HND -> independent BTHD, allowing the fused source allocation to die.
        return PreparedDenseBF16(
            q=q.transpose(1, 2).contiguous(),
            k=k.transpose(1, 2).contiguous(),
            v=v.transpose(1, 2).contiguous(),
        )

    def execute(self, prepared):
        return _dense_bthd(prepared.q, prepared.k, prepared.v).transpose(1, 2)

    def as_status(self):
        return {"name": self.name}


def _load_preprocess_module():
    try:
        return importlib.import_module("sol_attn.preprocess")
    except ImportError:
        return importlib.import_module(
            "techniques.sparse_backends.sol_attn.preprocess"
        )


def _clear_preprocess_autotune_cache():
    """Clear T-only Triton autotune choices after subset-head diagnostics.

    The released preprocess reduction kernels autotune on token count alone.
    Running a correctness/density diagnostic on fewer heads must not leave a
    launch configuration chosen for that smaller grid cached for production.
    """
    try:
        module = _load_preprocess_module()
    except Exception:
        return
    for name in (
        "_reduce_kc_kernel",
        "_reduce_vc_kernel",
        "_reduce_kc_stats_kernel",
        "_diag_threshold_kernel",
        "_pool_query_kernel",
        "_exact_fused_threshold_kernel",
    ):
        kernel = getattr(module, name, None)
        cache = getattr(kernel, "cache", None)
        if hasattr(cache, "clear"):
            cache.clear()


@dataclass
class PreparedSol:
    mode: str
    layer_index: int
    snapshot: object
    decline: str | None = None
    delegated: object | None = None
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    sink_start: int = 0
    sink_tokens: int = 0


class SolAttentionBackend:
    name = "sol_attn"
    # The token refiner can keep the incoming Comfy attention path.
    requires_registered_sage = False
    requires_runtime_context = True
    approximate = True

    def __init__(self, dense_backend, config=None, sol_callable=None):
        if dense_backend is None:
            raise TypeError("SolAttentionBackend requires a prepared dense fallback backend")
        self.dense_backend = dense_backend
        self.config = config or SolAttentionConfig()
        if not isinstance(self.config, SolAttentionConfig):
            raise TypeError("config must be SolAttentionConfig")
        self.strict_runtime_layout = bool(self.config.strict)
        if sol_callable is None:
            self.sol_attn, module = load_sol_attention()
            self.sol_module = getattr(module, "__name__", type(module).__name__)
        else:
            self.sol_attn = sol_callable
            self.sol_module = "injected"
        self._gated_shapes = set()
        self._density_shapes = set()
        self._request_id = None
        self._request_sparse_start = 0
        self._last_request_sparse = None
        stats.increment("configured")
        logging.info(
            "%s configured: dense=%s tau=%s threshold=%s dense_steps=%d "
            "dense_layers=%d sink=%s module=%s",
            LOG_PREFIX,
            getattr(dense_backend, "name", type(dense_backend).__name__),
            self.config.tau,
            self.config.thresh_type,
            self.config.dense_steps,
            self.config.dense_layers,
            self.config.sink_mode,
            self.sol_module,
        )

    def _close_request(self, snapshot):
        request_id = getattr(snapshot, "request_id", None)
        if request_id == self._request_id:
            return
        if self._request_id is not None:
            current = stats.get()["sparse_calls"]
            self._last_request_sparse = current - self._request_sparse_start
            self._request_sparse_start = current
            stats.set_last("previous_request_sparse_calls", self._last_request_sparse)
            if (
                self.config.strict
                and getattr(snapshot, "step_index", -1) >= self.config.dense_steps
                and self._last_request_sparse == 0
            ):
                raise SolAttentionError(
                    "previous request reached sparse-eligible steps without one Sol-Attn call"
                )
        self._request_id = request_id

    def prepare(self, q, k, v, *, layer_index, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        self._close_request(snapshot)
        reason = decline_reason(self.config, snapshot, layer_index, q)
        if reason is not None:
            stats.decline(reason)
            stats.increment("dense_calls")
            delegated = self.dense_backend.prepare(
                q,
                k,
                v,
                layer_index=layer_index,
                transformer_options=transformer_options,
            )
            return PreparedSol(
                mode="dense",
                layer_index=int(layer_index),
                snapshot=snapshot,
                decline=reason,
                delegated=delegated,
            )

        # HND [B,H,T,D] -> contiguous BTHD [B,T,H,D].  These independent
        # buffers permit h3_attention.forward to release the fused QKV source.
        qb = q.transpose(1, 2).contiguous()
        kb = k.transpose(1, 2).contiguous()
        vb = v.transpose(1, 2).contiguous()
        prepared_bytes = sum(x.numel() * x.element_size() for x in (qb, kb, vb))
        stats.increment("prepared_bf16_bytes", prepared_bytes)
        start, tokens = exact_sink(self.config, snapshot)
        return PreparedSol(
            mode="sparse",
            layer_index=int(layer_index),
            snapshot=snapshot,
            q=qb,
            k=kb,
            v=vb,
            sink_start=int(start),
            sink_tokens=int(tokens),
        )

    def _gate_limits(self, tokens):
        return {
            "max_abs": 0.15 if int(tokens) >= 32768 else 0.08,
            "mean_abs": 0.002,
            "relative_l2": 0.005,
        }

    def _run_gate(self, prepared):
        key = (tuple(prepared.q.shape), str(prepared.q.dtype), prepared.q.device.index)
        if not self.config.correctness_gate or key in self._gated_shapes:
            return
        heads = int(self.config.gate_heads) or int(prepared.q.shape[2])
        heads = min(heads, int(prepared.q.shape[2]))
        q = prepared.q[:, :, :heads].contiguous()
        k = prepared.k[:, :, :heads].contiguous()
        v = prepared.v[:, :, :heads].contiguous()
        got = self.sol_attn(
            q,
            k,
            v,
            tau=-1000.0,
            thresh_type=self.config.thresh_type,
            kv_splits=int(self.config.kv_splits),
        )
        want = _dense_bthd(q, k, v)
        torch.cuda.synchronize(q.device)
        metrics = tensor_error_metrics(got, want)
        if heads != int(prepared.q.shape[2]):
            _clear_preprocess_autotune_cache()
        limits = self._gate_limits(q.shape[1])
        passed = (
            metrics.get("shape_match", False)
            and metrics.get("nonfinite_count", 1) == 0
            and metrics["max_abs"] <= limits["max_abs"]
            and metrics["mean_abs"] <= limits["mean_abs"]
            and metrics["relative_l2"] <= limits["relative_l2"]
        )
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor(
                    [1 if passed else 0], device=q.device, dtype=torch.int32
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                passed = bool(flag.item())
        except Exception:
            if self.config.strict:
                raise
            logging.warning(
                "%s could not coordinate correctness gate across ranks",
                LOG_PREFIX,
                exc_info=True,
            )
        stats.set_last("gate", {"passed": passed, "metrics": metrics, "limits": limits})
        if not passed:
            stats.increment("gate_failures")
            raise SolAttentionError(
                "Sol-Attn real-QKV correctness gate failed: metrics=%s limits=%s"
                % (metrics, limits)
            )
        stats.increment("gate_passes")
        self._gated_shapes.add(key)
        logging.info("%s correctness gate PASS: %s", LOG_PREFIX, metrics)

    def _estimate_density(self, prepared):
        shape_key = tuple(prepared.q.shape)
        if shape_key in self._density_shapes:
            return
        self._density_shapes.add(shape_key)
        try:
            preprocess = _load_preprocess_module()
            block_size = int(getattr(preprocess, "BLOCK_SIZE", 64))
            prepare = preprocess.prepare
            heads = int(self.config.density_heads) or int(prepared.q.shape[2])
            heads = min(heads, int(prepared.q.shape[2]))
            q = prepared.q[:, :, :heads].contiguous()
            k = prepared.k[:, :, :heads].contiguous()
            v = prepared.v[:, :, :heads].contiguous()
            scale = q.shape[-1] ** -0.5
            kc, _vc, threshold = prepare(
                q,
                k,
                v,
                scale=scale,
                tau=float(self.config.tau),
                thresh_type=self.config.thresh_type,
            )
            tokens = int(q.shape[1])
            blocks = math.ceil(tokens / block_size)
            padded = F.pad(q, (0, 0, 0, 0, 0, blocks * block_size - tokens))
            counts = torch.full((blocks,), float(block_size), device=q.device, dtype=torch.float32)
            counts[-1] = tokens - (blocks - 1) * block_size
            q_bar = padded.view(q.shape[0], blocks, block_size, heads, q.shape[3]).float().sum(2)
            q_bar /= counts.view(1, blocks, 1, 1)
            scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float()).mul_(
                scale * math.log2(math.e)
            )
            routed = scores > threshold[:, :, None, :]
            threshold_density = float(routed.float().mean().item())
            ids = torch.arange(blocks, device=q.device)
            routed |= ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]
            sink_blocks = 0
            if prepared.sink_tokens:
                first = prepared.sink_start // block_size
                last = math.ceil((prepared.sink_start + prepared.sink_tokens) / block_size)
                routed[:, :, first:last, :] = True
                sink_blocks = last - first
            value = {
                "blocks": blocks,
                "sink_blocks": sink_blocks,
                "threshold_density": threshold_density,
                "effective_density": float(routed.float().mean().item()),
            }
            stats.set_last("route_density", value)
            if heads != int(prepared.q.shape[2]):
                _clear_preprocess_autotune_cache()
            logging.info("%s route density: %s", LOG_PREFIX, value)
        except Exception as exc:
            _clear_preprocess_autotune_cache()
            stats.set_last("route_density_error", "%s: %s" % (type(exc).__name__, exc))
            logging.warning("%s route-density diagnostic unavailable: %s", LOG_PREFIX, exc)

    def as_status(self):
        value = stats.get()
        value.update(
            {
                "name": self.name,
                "module": self.sol_module,
                "dense_backend": getattr(
                    self.dense_backend, "name", type(self.dense_backend).__name__
                ),
                "config": {
                    "tau": self.config.tau,
                    "thresh_type": self.config.thresh_type,
                    "dense_steps": self.config.dense_steps,
                    "dense_layers": self.config.dense_layers,
                    "sink_mode": self.config.sink_mode,
                    "correctness_gate": self.config.correctness_gate,
                    "gate_heads": self.config.gate_heads,
                    "density_heads": self.config.density_heads,
                },
            }
        )
        return value

    def execute(self, prepared):
        if prepared.mode == "dense":
            return self.dense_backend.execute(prepared.delegated)
        try:
            self._run_gate(prepared)
            self._estimate_density(prepared)
            out = self.sol_attn(
                prepared.q,
                prepared.k,
                prepared.v,
                tau=float(self.config.tau),
                thresh_type=self.config.thresh_type,
                kv_splits=int(self.config.kv_splits),
                sink_start=int(prepared.sink_start),
                sink_tokens=int(prepared.sink_tokens),
            )
            if prepared.sink_tokens:
                lo = int(prepared.sink_start)
                hi = lo + int(prepared.sink_tokens)
                out[:, lo:hi] = _dense_bthd(
                    prepared.q[:, lo:hi],
                    prepared.k,
                    prepared.v,
                )
            if out.shape != prepared.q.shape:
                raise SolAttentionError(
                    "Sol-Attn returned %s, expected %s" % (tuple(out.shape), tuple(prepared.q.shape))
                )
            if not torch.isfinite(out).all():
                raise SolAttentionError("Sol-Attn returned non-finite output")
            stats.increment("sparse_calls")
            # BTHD -> HND expected by h3_attention.forward.
            return out.transpose(1, 2)
        except Exception as exc:
            stats.increment("kernel_errors")
            if isinstance(exc, SolAttentionError):
                raise
            raise SolAttentionError(
                "Sol-Attn runtime failed at layer=%d request=%s step=%s shape=%s: %s: %s"
                % (
                    prepared.layer_index,
                    getattr(prepared.snapshot, "request_id", None),
                    getattr(prepared.snapshot, "step_index", None),
                    tuple(prepared.q.shape),
                    type(exc).__name__,
                    exc,
                )
            ) from exc
