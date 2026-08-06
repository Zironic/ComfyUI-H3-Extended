"""Signal-driven cache for the H3 block stack.

The decision follows FirstBlockCache exactly: compare the residual produced by
block 0 in the current step with the previous block-0 residual.  If the ratio is
below threshold, add the cached total residual of blocks 1..N once and treat the
remaining block calls as identities.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
import threading
import time

import torch

from .config import FirstBlockCacheConfig

try:
    from ..h3_runtime.context import get_runtime_snapshot
except ImportError:
    from h3_runtime.context import get_runtime_snapshot

LOG_PREFIX = "[H3 FirstBlockCache]"


@dataclass
class BranchState:
    request_id: int = -1
    shape: tuple | None = None
    dtype: torch.dtype | None = None
    device: torch.device | None = None
    prev_head_residual: torch.Tensor | None = None
    current_head_output: torch.Tensor | None = None
    cached_tail: torch.Tensor | None = None
    skip: bool = False
    step_index: int = -1
    last_diff: float | None = None

    def release(self):
        self.prev_head_residual = None
        self.current_head_output = None
        self.cached_tail = None
        self.skip = False
        self.last_diff = None


class FirstBlockCacheCoordinator:
    def __init__(self, config=None):
        self.config = config or FirstBlockCacheConfig()
        self.states: dict[tuple[int, ...], BranchState] = {}
        self.stats = {
            "head_calls": 0,
            "computed_tails": 0,
            "skipped_tails": 0,
            "cache_bytes": 0,
            "peak_cache_bytes": 0,
            "declines": {},
            "diffs": [],
        }
        self._lock = threading.RLock()
        self._records = []
        self._request_id = -1
        self._request_started = None
        self._sequence_length = None
        self._min_free_bytes = None

    def _sample_free_bytes(self):
        """Physical free VRAM, which PyTorch's allocator counters do not show."""
        if not torch.cuda.is_available():
            return
        try:
            free = int(torch.cuda.mem_get_info()[0])
        except Exception:
            return
        if self._min_free_bytes is None or free < self._min_free_bytes:
            self._min_free_bytes = free

    def _record(self, snapshot, diff, decision, reason=None):
        if self._sequence_length is None and snapshot.layout is not None:
            self._sequence_length = int(snapshot.layout.seq_len)
        row = {
            "step": int(snapshot.step_index),
            "branch": list(snapshot.branch),
            "sigma": float(snapshot.sigma),
            "diff": None if diff is None else float(diff),
            "decision": decision,
        }
        if reason is not None:
            row["reason"] = reason
        self._records.append(row)

    def on_request_reset(self, request_id):
        with self._lock:
            for state in self.states.values():
                state.release()
            self.states.clear()
            self._records = []
            self._request_id = int(request_id)
            self._sequence_length = None
            self._min_free_bytes = None
            self._request_started = time.perf_counter()
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass

    def _decline(self, reason):
        declines = self.stats["declines"]
        declines[reason] = declines.get(reason, 0) + 1

    def _state(self, snapshot, tensor):
        branch = snapshot.branch
        state = self.states.get(branch)
        signature = (tuple(tensor.shape), tensor.dtype, tensor.device)
        if state is None:
            state = BranchState()
            self.states[branch] = state
        if (
            state.request_id != snapshot.request_id
            or state.shape != signature[0]
            or state.dtype != signature[1]
            or state.device != signature[2]
        ):
            state.release()
            state.request_id = snapshot.request_id
            state.shape, state.dtype, state.device = signature
        return state

    def _global_residual_diff(self, output, original_input, previous, chunk_rows=1024):
        # Bound temporary arithmetic to row slabs.  The two accumulated sums have
        # the same denominator, so their ratio equals the absmean ratio used by
        # FirstBlockCache without materializing another full residual tensor.
        values = torch.zeros(2, device=output.device, dtype=torch.float64)
        rows = int(output.shape[0])
        for start in range(0, rows, max(1, int(chunk_rows))):
            stop = min(rows, start + max(1, int(chunk_rows)))
            current = output[start:stop].detach() - original_input[start:stop]
            values[0].add_((current - previous[start:stop]).abs().sum(dtype=torch.float64))
            values[1].add_(previous[start:stop].abs().sum(dtype=torch.float64))
        if self.config.collective:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(values, op=dist.ReduceOp.SUM)
            except Exception:
                if self.config.strict:
                    raise
                logging.warning(
                    "%s collective reduction unavailable",
                    LOG_PREFIX,
                    exc_info=True,
                )
        return float((values[0] / values[1].clamp_min(1e-8)).item())

    def after_head(self, original_input, output, transformer_options):
        """Record block-0 residual and decide whether the tail is needed.

        ``original_input`` is an independent snapshot made before block 0 because
        Comfy's H3 block updates its residual stream in place.  On a computed
        step that allocation is reused as ``current_head_output`` for the final
        tail-residual calculation instead of allocating a second snapshot.
        """
        snapshot = get_runtime_snapshot(transformer_options)
        self.stats["head_calls"] += 1
        if snapshot is None or snapshot.step_index < 0:
            self._decline("runtime step unavailable")
            return False
        if "easycache" in transformer_options:
            reason = "EasyCache already owns whole-step reuse"
            if self.config.strict:
                raise RuntimeError(reason)
            self._decline(reason)
            return False

        state = self._state(snapshot, output)
        state.step_index = snapshot.step_index
        self._sample_free_bytes()

        if state.prev_head_residual is None:
            state.prev_head_residual = torch.empty_like(output)
            torch.sub(output.detach(), original_input, out=state.prev_head_residual)
            state.current_head_output = original_input
            state.current_head_output.copy_(output.detach())
            state.skip = False
            reason = (
                "warmup_step"
                if snapshot.step_index < self.config.warmup_steps
                else "cache_not_ready"
            )
            self._decline(reason)
            self._record(snapshot, None, "compute", reason)
            return False

        diff = self._global_residual_diff(
            output.detach(), original_input, state.prev_head_residual
        )
        state.last_diff = diff
        torch.sub(output.detach(), original_input, out=state.prev_head_residual)

        if snapshot.step_index < self.config.warmup_steps:
            state.current_head_output = original_input
            state.current_head_output.copy_(output.detach())
            state.skip = False
            self._decline("warmup_step")
            self._record(snapshot, diff, "compute", "warmup_step")
            return False
        if state.cached_tail is None:
            state.current_head_output = original_input
            state.current_head_output.copy_(output.detach())
            state.skip = False
            self._decline("cache_not_ready")
            self._record(snapshot, diff, "compute", "cache_not_ready")
            return False

        state.skip = diff <= float(self.config.threshold)
        self.stats["diffs"].append(diff)
        self._record(
            snapshot,
            diff,
            "skip" if state.skip else "compute",
            None if state.skip else "above_threshold",
        )
        if state.skip:
            state.current_head_output = None
        else:
            state.current_head_output = original_input
            state.current_head_output.copy_(output.detach())
        return state.skip

    def should_skip(self, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return False
        state = self.states.get(snapshot.branch)
        return bool(
            state is not None
            and state.request_id == snapshot.request_id
            and state.step_index == snapshot.step_index
            and state.skip
            and state.cached_tail is not None
        )

    def apply_cached_tail(self, x, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        state = self.states[snapshot.branch]
        x.add_(state.cached_tail)
        return x

    def finish_compute(self, output, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return
        state = self.states.get(snapshot.branch)
        if state is None or state.current_head_output is None:
            return
        if state.cached_tail is None:
            state.cached_tail = torch.empty_like(output)
        torch.sub(output.detach(), state.current_head_output, out=state.cached_tail)
        state.current_head_output = None
        state.skip = False
        self.stats["computed_tails"] += 1
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (state.prev_head_residual, state.cached_tail)
            if tensor is not None
        )
        self.stats["cache_bytes"] = cache_bytes
        self.stats["peak_cache_bytes"] = max(
            self.stats["peak_cache_bytes"], cache_bytes
        )

    def finish_skip(self, transformer_options):
        snapshot = get_runtime_snapshot(transformer_options)
        if snapshot is None:
            return
        state = self.states.get(snapshot.branch)
        if state is None:
            return
        state.current_head_output = None
        state.skip = False
        self.stats["skipped_tails"] += 1

    def report(self, seconds=None):
        """Per-request decision trace: scalar diffs only, never residual tensors."""
        head_calls = int(self.stats["head_calls"])
        skipped = int(self.stats["skipped_tails"])
        computed = int(self.stats["computed_tails"])
        decided = skipped + computed
        peak_allocated = None
        if torch.cuda.is_available():
            try:
                peak_allocated = int(torch.cuda.max_memory_allocated())
            except Exception:
                pass
        return {
            "request_id": self._request_id,
            "mode": self.config.mode,
            "threshold": float(self.config.threshold),
            "warmup_steps": int(self.config.warmup_steps),
            "sequence_length": self._sequence_length,
            "sampler_seconds": None if seconds is None else float(seconds),
            "peak_allocated_gib": (
                None if peak_allocated is None else peak_allocated / (1024 ** 3)
            ),
            "minimum_physical_free_mib": (
                None
                if self._min_free_bytes is None
                else self._min_free_bytes / (1024 ** 2)
            ),
            "head_calls": head_calls,
            "computed_tails": computed,
            "skipped_tails": skipped,
            "skip_fraction": (skipped / decided) if decided else 0.0,
            "cache_bytes": int(self.stats["cache_bytes"]),
            "peak_cache_bytes": int(self.stats["peak_cache_bytes"]),
            "declines": dict(self.stats["declines"]),
            "steps": list(self._records),
        }

    def on_request_end(self, request_id, seconds=None):
        directory = str(self.config.report_directory or "").strip()
        if not directory:
            return None
        payload = self.report(seconds)
        try:
            os.makedirs(directory, exist_ok=True)
            path = os.path.join(
                directory,
                "fbc_request_%d_t%s.json"
                % (int(request_id), ("%.4f" % self.config.threshold).replace(".", "p")),
            )
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except Exception:
            if self.config.strict:
                raise
            logging.warning("%s could not write report", LOG_PREFIX, exc_info=True)
            return None
        logging.info(
            "%s wrote %s (skip_fraction=%.3f)",
            LOG_PREFIX,
            path,
            payload["skip_fraction"],
        )
        return path

    def as_status(self):
        return {
            **self.stats,
            "mode": self.config.mode,
            "threshold": self.config.threshold,
            "warmup_steps": self.config.warmup_steps,
            "branches": len(self.states),
            "signal": "block0_residual",
        }
