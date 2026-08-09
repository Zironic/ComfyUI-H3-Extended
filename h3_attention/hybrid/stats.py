"""Lightweight request-scoped statistics for hybrid sparse attention."""

from datetime import datetime
import logging
import math
import threading

import torch

from .report import validate_run_tag, write_request

try:
    from ...h3_runtime.timing import publish_timing
except ImportError:
    from h3_runtime.timing import publish_timing

LOG_PREFIX = "[H3 hybrid sparse]"
ROUTE_HISTOGRAM_KEY = "_adaptive_route_histogram"


TIMING_STAGES = (
    "total_dit_block",
    "adaln_proj",
    "norm1_modulation",
    "qkv_proj",
    "qk_rmsnorm_rope",
    "fused_qkv_projection",
    "direct_lut_construction",
    "v_fp8_preparation",
    "q_k_int8_quantization",
    "sparse_sage_low_level_kernel",
    "total_hybrid_attention",
    "out_proj",
    "attention_residual_gate",
    "norm2_modulation",
    "mlp_fc1",
    "mlp_swiglu_fc2",
    "final_mlp_gate",
    "model_forward",
)


_ROUTE_PERCENTILES = (
    ("p05", 0.05),
    ("p25", 0.25),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p95", 0.95),
)


def build_route_histogram(valid_block_num, mask_metadata):
    """Build exact per-row adaptive-K telemetry without synchronizing CUDA."""
    if getattr(mask_metadata, "density_mode", None) != "adaptive_budget":
        return None
    pure_q = int(mask_metadata.pure_video_q_tiles)
    pure_kv = int(mask_metadata.pure_video_kv_tiles)
    if pure_q <= 0 or pure_kv <= 0:
        return None
    q_start = int(mask_metadata.q_tiles) - pure_q
    non_video_kv = int(mask_metadata.kv_tiles) - pure_kv
    counts = valid_block_num[..., q_start:].to(torch.int64) - non_video_kv
    return torch.bincount(
        counts.reshape(-1), minlength=pure_kv + 1
    ).detach()


def _nearest_rank(cdf, row_count, fraction):
    threshold = max(1, int(math.ceil(float(fraction) * int(row_count))))
    value = torch.tensor(threshold, dtype=cdf.dtype)
    return int(torch.searchsorted(cdf, value, right=False).item())


def _histogram_stats(histogram, *, pure_kv, target, minimum, maximum):
    histogram = histogram.to(dtype=torch.int64, device="cpu")
    row_count = int(histogram.sum().item())
    if row_count <= 0:
        return {
            "row_count": 0,
            "adaptive_reallocation_observed": False,
        }
    positions_i64 = torch.arange(histogram.numel(), dtype=torch.int64)
    positions = positions_i64.to(torch.float64)
    weights = histogram.to(torch.float64)
    total = float((weights * positions).sum().item())
    total_sq = float((weights * positions.square()).sum().item())
    mean = total / row_count
    variance = max(0.0, total_sq / row_count - mean * mean)
    nonzero = torch.nonzero(histogram > 0, as_tuple=False).flatten()
    minimum_actual = int(nonzero[0].item())
    maximum_actual = int(nonzero[-1].item())
    cdf = torch.cumsum(histogram, dim=0)
    percentiles = {
        name: _nearest_rank(cdf, row_count, fraction)
        for name, fraction in _ROUTE_PERCENTILES
    }
    below = int(histogram[:target].sum().item()) if target > 0 else 0
    equal = int(histogram[target].item()) if target < histogram.numel() else 0
    above = int(histogram[target + 1:].sum().item()) if target + 1 < histogram.numel() else 0
    changed = below + above
    at_minimum = int(histogram[minimum].item()) if minimum < histogram.numel() else 0
    at_maximum = int(histogram[maximum].item()) if maximum < histogram.numel() else 0
    unique = int((histogram > 0).sum().item())

    result = {
        "row_count": row_count,
        "target_video_kv_tiles": int(target),
        "minimum_rail_video_kv_tiles": int(minimum),
        "maximum_rail_video_kv_tiles": int(maximum),
        "mean_video_kv_tiles": mean,
        "std_video_kv_tiles": math.sqrt(variance),
        "min_video_kv_tiles": minimum_actual,
        "max_video_kv_tiles": maximum_actual,
        "unique_video_kv_tile_counts": unique,
        "rows_below_target": below,
        "rows_equal_target": equal,
        "rows_above_target": above,
        "rows_changed_from_target": changed,
        "rows_changed_from_target_rate": changed / row_count,
        "rows_at_minimum_rail": at_minimum,
        "rows_at_minimum_rail_rate": at_minimum / row_count,
        "rows_at_maximum_rail": at_maximum,
        "rows_at_maximum_rail_rate": at_maximum / row_count,
        "adaptive_reallocation_observed": bool(changed and unique > 1),
        "target_video_tile_density": float(target) / pure_kv,
        "mean_video_tile_density": mean / pure_kv,
        "std_video_tile_density": math.sqrt(variance) / pure_kv,
        "min_video_tile_density": float(minimum_actual) / pure_kv,
        "max_video_tile_density": float(maximum_actual) / pure_kv,
    }
    for name, value in percentiles.items():
        result["%s_video_kv_tiles" % name] = int(value)
        result["%s_video_tile_density" % name] = float(value) / pure_kv
    return result


def _record_route_fields(record, stats):
    fields = (
        "mean_video_kv_tiles",
        "std_video_kv_tiles",
        "min_video_kv_tiles",
        "p05_video_kv_tiles",
        "p25_video_kv_tiles",
        "p50_video_kv_tiles",
        "p75_video_kv_tiles",
        "p95_video_kv_tiles",
        "max_video_kv_tiles",
        "min_video_tile_density",
        "p05_video_tile_density",
        "p25_video_tile_density",
        "p50_video_tile_density",
        "p75_video_tile_density",
        "p95_video_tile_density",
        "max_video_tile_density",
        "std_video_tile_density",
        "unique_video_kv_tile_counts",
        "rows_below_target",
        "rows_equal_target",
        "rows_above_target",
        "rows_changed_from_target",
        "rows_changed_from_target_rate",
        "rows_at_minimum_rail",
        "rows_at_minimum_rail_rate",
        "rows_at_maximum_rail",
        "rows_at_maximum_rail_rate",
        "adaptive_reallocation_observed",
    )
    for key in fields:
        if key in stats:
            record["actual_row_%s" % key] = stats[key]


def resolve_route_telemetry(records):
    """Resolve deferred route histograms with one host transfer per request."""
    resolved = [dict(record) for record in records]
    entries = []
    for index, record in enumerate(resolved):
        histogram = record.get(ROUTE_HISTOGRAM_KEY)
        if torch.is_tensor(histogram):
            entries.append((index, histogram))
    if not entries:
        for record in resolved:
            record.pop(ROUTE_HISTOGRAM_KEY, None)
        return resolved, None

    first = resolved[entries[0][0]]
    pure_kv = int(first["pure_video_kv_tiles"])
    target = int(first["retained_video_kv_tiles"])
    minimum = int(first["configured_minimum_video_kv_tiles"])
    maximum = int(first["configured_maximum_video_kv_tiles"])
    compatible = all(
        int(resolved[index]["pure_video_kv_tiles"]) == pure_kv
        and int(resolved[index]["retained_video_kv_tiles"]) == target
        and int(resolved[index]["configured_minimum_video_kv_tiles"]) == minimum
        and int(resolved[index]["configured_maximum_video_kv_tiles"]) == maximum
        and int(histogram.numel()) == pure_kv + 1
        for index, histogram in entries
    )
    if not compatible:
        for record in resolved:
            record.pop(ROUTE_HISTOGRAM_KEY, None)
        return resolved, {
            "observed": True,
            "density_mode": "adaptive_budget",
            "mixed_geometry": True,
            "records": len(entries),
            "error": "adaptive route telemetry used multiple incompatible geometries",
        }

    # Histograms are tiny compared with Q/K/V. Stack them on-device and perform
    # one request-end device-to-host transfer. When CUDA timing is enabled, the
    # timing resolver has already synchronized the final event before this runs.
    stacked = torch.stack(
        [histogram.detach() for _index, histogram in entries], dim=0
    ).to(device="cpu")
    record_to_hist = {}
    for offset, (index, _histogram) in enumerate(entries):
        record_to_hist[index] = stacked[offset]
        record = resolved[index]
        stats = _histogram_stats(
            stacked[offset], pure_kv=pure_kv, target=target,
            minimum=minimum, maximum=maximum,
        )
        _record_route_fields(record, stats)
        record.pop(ROUTE_HISTOGRAM_KEY, None)
    for record in resolved:
        record.pop(ROUTE_HISTOGRAM_KEY, None)

    def aggregate(indices):
        histogram = torch.stack(
            [record_to_hist[index] for index in indices], dim=0
        ).sum(dim=0)
        return _histogram_stats(
            histogram, pure_kv=pure_kv, target=target,
            minimum=minimum, maximum=maximum,
        )

    all_indices = [index for index, _histogram in entries]
    summary = aggregate(all_indices)
    summary.update({
        "observed": True,
        "density_mode": "adaptive_budget",
        "mixed_geometry": False,
        "records": len(entries),
        "pure_video_kv_tiles": pure_kv,
    })

    by_step = {}
    by_layer = {}
    for index, _histogram in entries:
        record = resolved[index]
        by_step.setdefault(int(record.get("step", -1)), []).append(index)
        by_layer.setdefault(int(record.get("layer", -1)), []).append(index)
    summary["per_step"] = [
        dict(aggregate(by_step[step]), step_index=int(step))
        for step in sorted(by_step) if step >= 0
    ]
    summary["per_layer"] = [
        dict(aggregate(by_layer[layer]), layer=int(layer))
        for layer in sorted(by_layer) if layer >= 0
    ]
    return resolved, summary


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
        self._step_index = -1
        self._branch = (0,)

    @property
    def active(self):
        return self.enabled and (self._injected or self._request_cuda)

    @property
    def _request_cuda(self):
        return bool(getattr(self, "_cuda", False))

    def begin_request(self, request_id, *, cuda=False, snapshot=None):
        request_id = int(request_id)
        if self._request_id != request_id:
            self._request_id = request_id
            self._samples = {stage: [] for stage in TIMING_STAGES}
            self._active = []
            self._last_event = None
            self._resolved = None
            self._step_index = -1
            self._branch = (0,)
        self._cuda = bool(cuda)
        if snapshot is not None:
            self.set_context(snapshot)

    def set_context(self, snapshot):
        """Set the authoritative request context used by subsequent events."""
        self._step_index = int(getattr(snapshot, "step_index", -1))
        branch = getattr(snapshot, "branch", (0,))
        self._branch = tuple(int(value) for value in branch) or (0,)

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
        if torch.compiler.is_compiling():
            return None
        if not self.active:
            return None
        start = self._new_event()
        end = self._new_event()
        start.record()
        token = (stage, start, end, self._step_index, self._branch)
        self._active.append(token)
        return token

    def end(self, token):
        if torch.compiler.is_compiling():
            return
        if token is None or token not in self._active:
            return
        stage, _start, end, _step_index, _branch = token
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
        total_block_ms = 0.0
        model_forward_ms = 0.0
        for stage in TIMING_STAGES:
            values = [self._elapsed_ms(start, end) for _, start, end, _, _ in self._samples[stage]]
            stage_sum = sum(values)
            stages[stage] = {
                "count": len(values),
                "sum_ms": stage_sum,
                "mean_ms": stage_sum / len(values) if values else 0.0,
            }
            if stage == "total_hybrid_attention":
                total_ms = stage_sum
            elif stage == "total_dit_block":
                total_block_ms = stage_sum
            elif stage == "model_forward":
                model_forward_ms = stage_sum
        per_step = self._per_step_summary()
        self._resolved = self.summary(request_wall_seconds, stages=stages,
                                      measured_ms=total_ms,
                                      measured_block_ms=total_block_ms,
                                      model_forward_ms=model_forward_ms,
                                      per_step=per_step)
        return self._resolved

    @staticmethod
    def _stage_stats(samples, elapsed):
        values = [elapsed(start, end) for _, start, end, _, _ in samples]
        total = sum(values)
        return {
            "count": len(values),
            "sum_ms": total,
            "mean_ms": total / len(values) if values else 0.0,
        }

    def _timing_bucket(self, samples):
        if isinstance(samples, dict):
            grouped = samples
        else:
            grouped = {stage: [] for stage in TIMING_STAGES}
            for sample in samples:
                grouped[sample[0]].append(sample)
        stages = {
            stage: self._stage_stats(grouped.get(stage, ()), self._elapsed_ms)
            for stage in TIMING_STAGES
        }
        attention_ms = stages["total_hybrid_attention"]["sum_ms"]
        block_ms = stages["total_dit_block"]["sum_ms"]
        model_ms = stages["model_forward"]["sum_ms"]
        return {
            "stages": stages,
            "total_measured_attention_cuda_seconds": attention_ms / 1000.0,
            "total_measured_dit_block_cuda_seconds": block_ms / 1000.0,
            "total_model_forward_cuda_seconds": model_ms / 1000.0,
        }

    def _per_step_summary(self):
        grouped = {}
        for stage_samples in self._samples.values():
            for _stage, start, end, step_index, branch in stage_samples:
                if step_index < 0:
                    continue
                grouped.setdefault((step_index, branch), []).append(
                    (_stage, start, end, step_index, branch)
                )
        by_step = {}
        for (step_index, branch), samples in grouped.items():
            by_step.setdefault(step_index, {})[branch] = samples
        result = []
        for step_index in sorted(by_step):
            branches = []
            all_samples = []
            for branch in sorted(by_step[step_index]):
                bucket = self._timing_bucket(by_step[step_index][branch])
                bucket.update({
                    "step_index": int(step_index),
                    "ordinal": int(step_index) + 1,
                    "branch": list(branch),
                })
                branches.append(bucket)
                all_samples.extend(by_step[step_index][branch])
            rolled = self._timing_bucket(all_samples)
            rolled.update({
                "step_index": int(step_index),
                "ordinal": int(step_index) + 1,
                "branches": branches,
            })
            # Keep the stage data at the step level while retaining exact CFG
            # branch identity in the list entries above.
            result.append(rolled)
        return result

    def summary(self, request_wall_seconds=None, *, stages=None, measured_ms=0.0,
                measured_block_ms=0.0, model_forward_ms=0.0, per_step=None):
        stages = stages or {
            stage: {"count": 0, "sum_ms": 0.0, "mean_ms": 0.0}
            for stage in TIMING_STAGES
        }
        wall = None if request_wall_seconds is None else float(request_wall_seconds)
        cuda_seconds = float(measured_ms) / 1000.0
        ratio = None if wall is None or wall <= 0.0 else cuda_seconds / wall
        block_seconds = float(measured_block_ms) / 1000.0
        block_ratio = None if wall is None or wall <= 0.0 else block_seconds / wall
        model_forward_seconds = float(model_forward_ms) / 1000.0
        model_forward_ratio = None if wall is None or wall <= 0.0 else model_forward_seconds / wall
        return {
            "enabled": bool(self.enabled),
            "call_count": int(stages["total_hybrid_attention"]["count"]),
            "stages": stages,
            "per_step": list(per_step or []),
            "total_measured_attention_cuda_seconds": cuda_seconds,
            "request_wall_seconds": wall,
            "attention_cuda_to_request_wall_ratio": ratio,
            "total_measured_dit_block_cuda_seconds": block_seconds,
            "dit_block_cuda_to_request_wall_ratio": block_ratio,
            "model_forward_call_count": int(stages["model_forward"]["count"]),
            "total_model_forward_cuda_seconds": model_forward_seconds,
            "model_forward_cuda_to_request_wall_ratio": model_forward_ratio,
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

    def before_forward(self, snapshot, transformer_options, payload):
        """Start and publish one timer before the first DiT block executes."""
        with self._lock:
            if self._timing is None:
                return
            device = getattr(snapshot, "device", None)
            cuda = getattr(device, "type", str(device).split(":", 1)[0]) == "cuda"
            self._timing.begin_request(snapshot.request_id, cuda=cuda, snapshot=snapshot)
            publish_timing(transformer_options, self._timing)

    def record(self, metadata):
        with self._lock:
            self._records.append(dict(metadata))

    def on_request_end(self, request_id, seconds=None):
        with self._lock:
            timestamp = self._timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            records = list(self._records)
            timing = self._timing.resolve(seconds) if self._timing is not None else None
            records, route_summary = resolve_route_telemetry(records)
        directory = write_request(
            self.output_root,
            self.run_tag,
            timestamp,
            request_id,
            records,
            seconds,
            timing=timing,
            route_summary=route_summary,
        )
        self.last_report_directory = directory
        logging.info("%s wrote %d records to %s", LOG_PREFIX, len(records), directory)
        return directory
