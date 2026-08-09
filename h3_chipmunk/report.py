from __future__ import annotations

import json
import os
import time


SHADOW_METRICS = (
    "raw_relative_l2",
    "gated_relative_l2",
    "block_relative_l2",
    "raw_cosine",
    "error_rms",
    "dense_rms",
    "max_abs_error",
)


def _metric_stats(rows, key):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    p95 = ordered[min(n - 1, max(0, int(round(0.95 * (n - 1)))))]
    return {
        "count": n,
        "mean": sum(values) / n,
        "median": ordered[n // 2],
        "p95": p95,
        "max": ordered[-1],
    }


def _shadow_bucket(rows):
    return {
        key: stats
        for key in SHADOW_METRICS
        if (stats := _metric_stats(rows, key)) is not None
    }


def _group_shadow(rows, field):
    grouped = {}
    for row in rows:
        value = row.get(field)
        if value is None:
            continue
        grouped.setdefault(int(value), []).append(row)
    return {
        str(key): _shadow_bucket(grouped[key])
        for key in sorted(grouped)
    }


class H3ChipmunkReportListener:
    def __init__(self, session, config):
        self.session = session
        self.config = config
        self.started = time.time()

    def on_request_reset(self, request_id):
        # H3ChipmunkSession resets lazily from the published RuntimeSnapshot so
        # it receives the matching layout signature as well as request id.
        pass

    def on_request_end(self, request_id, seconds=None):
        if not self.session.records:
            self.session.caches.clear()
            return
        if not self.config.save_report:
            self.session.caches.clear()
            self.session.records.clear()
            return
        try:
            import folder_paths
            root = folder_paths.get_output_directory()
        except Exception:
            root = os.path.abspath("output")
        directory = os.path.join(root, "h3_chipmunk")
        os.makedirs(directory, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        base = f"{self.config.run_tag}_{request_id}_{stamp}"

        try:
            # Any CUDA scalar diagnostics are copied here, after model execution,
            # rather than synchronizing inside every MLP chunk.
            records = self.session.materialize_records()
            records_path = os.path.join(directory, base + ".jsonl")
            with open(records_path, "w", encoding="utf-8") as handle:
                for row in records:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

            measured = [
                row for row in records
                if row.get("path") == "measure_selector" and row.get("cross_step", False)
            ]
            capture_keys = ("0.10", "0.20", "0.25", "0.30", "0.40", "0.50")
            mean_capture = {}
            for key in capture_keys:
                values = [
                    float(row.get("energy_capture", {}).get(key))
                    for row in measured
                    if row.get("energy_capture", {}).get(key) is not None
                ]
                if values:
                    mean_capture[key] = sum(values) / len(values)

            shadow_rows = [row for row in records if row.get("path") == "shadow_delta"]
            shadow_refresh = [row for row in records if row.get("path") == "shadow_refresh"]
            shadow_summary = None
            if shadow_rows or shadow_refresh:
                by_fraction = {}
                for row in shadow_rows:
                    fraction = f"{float(row.get('active_fraction', 0.0)):.2f}"
                    by_fraction.setdefault(fraction, []).append(row)
                shadow_summary = {
                    "profile": [list(item) for item in self.config.shadow_profile],
                    "layer_stride": int(self.config.shadow_layer_stride),
                    "sample_rows": int(self.config.shadow_sample_rows),
                    "refresh_every": int(self.config.refresh_every),
                    "refresh_records": len(shadow_refresh),
                    "delta_records": len(shadow_rows),
                    "overall": _shadow_bucket(shadow_rows),
                    "by_layer": _group_shadow(shadow_rows, "layer"),
                    "by_step": _group_shadow(shadow_rows, "step"),
                    "by_fraction": {
                        key: _shadow_bucket(by_fraction[key])
                        for key in sorted(by_fraction)
                    },
                }

            summary = {
                "request_id": int(request_id),
                "seconds": seconds,
                "config": list(self.config.signature),
                "records": len(records),
                "dense_refresh": sum(r.get("path") == "dense_refresh" for r in records),
                "sparse_delta": sum(r.get("path") == "sparse_delta" for r in records),
                "measure": sum(r.get("path") == "measure_selector" for r in records),
                "measure_cross_step": len(measured),
                "measure_mean_energy_capture": mean_capture,
                "shadow_validate": shadow_summary,
                "fallback": sum("fallback" in str(r.get("path", "")) for r in records),
                "effective_chunk_rows": int(self.config.effective_chunk_rows),
                "measure_layer_stride": int(self.config.measure_layer_stride),
            }
            with open(os.path.join(directory, base + ".summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        finally:
            # Diagnostic selector/shadow state lives on GPU during a request to
            # avoid PCIe stalls. Drop it after report materialization so it does
            # not pin VRAM between generations.
            self.session.caches.clear()
            self.session.records.clear()
