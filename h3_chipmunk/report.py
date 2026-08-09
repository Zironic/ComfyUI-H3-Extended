from __future__ import annotations

import json
import os
import time


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
                "fallback": sum("fallback" in str(r.get("path", "")) for r in records),
                "effective_chunk_rows": int(self.config.effective_chunk_rows),
                "measure_layer_stride": int(self.config.measure_layer_stride),
            }
            with open(os.path.join(directory, base + ".summary.json"), "w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        finally:
            # Measurement selector summaries intentionally live on GPU during a
            # request to avoid PCIe stalls. Drop them immediately after report
            # materialization so they do not pin VRAM between generations.
            self.session.caches.clear()
            self.session.records.clear()
