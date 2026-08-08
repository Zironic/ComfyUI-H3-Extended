"""Lightweight request-scoped statistics for hybrid sparse attention."""

from datetime import datetime
import logging
import threading

import torch

from .report import validate_run_tag, write_request

LOG_PREFIX = "[H3 hybrid sparse]"


TIMING_STAGES = (
    "direct_lut_construction",
    "v_fp8_preparation",
    "q_k_int8_quantization",
    "sparse_sage_low_level_kernel",
    "total_hybrid_attention",
)


class DeferredCudaTiming:
    """Request-scoped CUDA event pairs resolved by one final synchronization."""

    def __init__(self, enabled=False, *, event_factory=None, timer=None):
        self.enabled = bool(enabled)
        self._injected = event_factory is not None or timer is not None
        self._event_factory = event_factory
        self._timer = timer
        self._request_id = None
        self._samples = {stage: [] for stage in TIMING_STAGES}
        self._active = []
        self._last_event = None
        self._resolved = None

    @property
    def active(self):
        return self.enabled and (self._injected or self._request_cuda)

    @property
    def _request_cuda(self):
        return bool(getattr(self, "_cuda", False))

    def begin_request(self, request_id, *, cuda=False):
        request_id = int(request_id)
        if self._request_id != request_id:
            self._request_id = request_id
            self._samples = {stage: [] for stage in TIMING_STAGES}
            self._active = []
            self._last_event = None
            self._resolved = None
        self._cuda = bool(cuda)

    def on_request_reset(self, request_id):
        self.begin_request(request_id)

    def _new_event(self):
        factory = self._event_factory
        if factory is None and self._timer is not None:
            if callable(self._timer):
                factory = self._timer
            else:
                factory = getattr(self._timer, "event", None)
                if factory is None:
                    factory = getattr(self._timer, "create_event", None)
        if factory is None:
            factory = lambda **kwargs: torch.cuda.Event(**kwargs)
        try:
            return factory(enable_timing=True)
        except TypeError:
            return factory()

    def begin(self, stage):
        if stage not in TIMING_STAGES:
            raise ValueError("unknown hybrid timing stage: %s" % stage)
        if not self.active:
            return None
        start = self._new_event()
        end = self._new_event()
        start.record()
        token = (stage, start, end)
        self._active.append(token)
        return token

    def end(self, token):
        if token is None or token not in self._active:
            return
        stage, _start, end = token
        end.record()
        self._active.remove(token)
        self._samples[stage].append(token)
        self._last_event = end

    def _elapsed_ms(self, start, end):
        if self._timer is not None:
            elapsed = getattr(self._timer, "elapsed_ms", None)
            if elapsed is not None:
                return float(elapsed(start, end))
        return float(start.elapsed_time(end))

    def resolve(self, request_wall_seconds=None):
        if not self.active or self._resolved is not None:
            return self._resolved or self.summary(request_wall_seconds)
        if self._last_event is not None:
            synchronize = getattr(self._timer, "synchronize", None) if self._timer else None
            if synchronize is not None:
                try:
                    synchronize(self._last_event)
                except TypeError:
                    synchronize()
            else:
                self._last_event.synchronize()
        stages = {}
        total_ms = 0.0
        for stage in TIMING_STAGES:
            values = [self._elapsed_ms(start, end) for _, start, end in self._samples[stage]]
            stage_sum = sum(values)
            stages[stage] = {
                "count": len(values),
                "sum_ms": stage_sum,
                "mean_ms": stage_sum / len(values) if values else 0.0,
            }
            if stage == "total_hybrid_attention":
                total_ms = stage_sum
        self._resolved = self.summary(request_wall_seconds, stages=stages,
                                      measured_ms=total_ms)
        return self._resolved

    def summary(self, request_wall_seconds=None, *, stages=None, measured_ms=0.0):
        stages = stages or {
            stage: {"count": 0, "sum_ms": 0.0, "mean_ms": 0.0}
            for stage in TIMING_STAGES
        }
        wall = None if request_wall_seconds is None else float(request_wall_seconds)
        cuda_seconds = float(measured_ms) / 1000.0
        ratio = None if wall is None or wall <= 0.0 else cuda_seconds / wall
        return {
            "enabled": bool(self.enabled),
            "call_count": int(stages["total_hybrid_attention"]["count"]),
            "stages": stages,
            "total_measured_attention_cuda_seconds": cuda_seconds,
            "request_wall_seconds": wall,
            "attention_cuda_to_request_wall_ratio": ratio,
            "ratio_caveat": (
                "CUDA event time is asynchronous and overlaps request wall time; "
                "the ratio is indicative, not an exact decomposition."
            ),
        }

    def on_request_end(self, request_id, seconds=None):
        self.begin_request(request_id, cuda=self._request_cuda)
        return self.resolve(seconds)


class HybridStatsCollector:
    def __init__(self, output_root, run_tag):
        self.output_root = str(output_root)
        self.run_tag = validate_run_tag(run_tag)
        self._lock = threading.RLock()
        self._request_id = -1
        self._timestamp = None
        self._records = []
        self.last_report_directory = None
        self._timing = None

    def attach_timing(self, timing):
        self._timing = timing

    def on_request_reset(self, request_id):
        with self._lock:
            self._request_id = int(request_id)
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._records = []
            if self._timing is not None:
                self._timing.on_request_reset(self._request_id)

    def record(self, metadata):
        with self._lock:
            self._records.append(dict(metadata))

    def on_request_end(self, request_id, seconds=None):
        with self._lock:
            timestamp = self._timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            records = list(self._records)
            timing = self._timing.resolve(seconds) if self._timing is not None else None
        directory = write_request(
            self.output_root,
            self.run_tag,
            timestamp,
            request_id,
            records,
            seconds,
            timing=timing,
        )
        self.last_report_directory = directory
        logging.info("%s wrote %d records to %s", LOG_PREFIX, len(records), directory)
        return directory
