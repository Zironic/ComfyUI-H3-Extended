from __future__ import annotations

import json
import os
import time


class H3ChipmunkReportListener:
    """Optional host-metadata report writer.

    The production node never transfers CUDA diagnostics to the host. Records
    therefore contain only Python metadata that was already available without a
    device synchronization (step/layer/chunk/path/static active fraction).
    """

    def __init__(self, session, config):
        self.session = session
        self.config = config
        self.started = time.time()

    def on_request_reset(self, request_id):
        pass

    def on_request_end(self, request_id, seconds=None):
        try:
            if not self.config.save_report or not self.session.records:
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
            records = list(self.session.records)

            with open(
                os.path.join(directory, base + ".jsonl"),
                "w",
                encoding="utf-8",
            ) as handle:
                for row in records:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")

            summary = {
                "request_id": int(request_id),
                "seconds": seconds,
                "config": list(self.config.signature),
                "records": len(records),
                "dense_refresh": sum(
                    row.get("path") == "dense_refresh" for row in records
                ),
                "sparse_delta": sum(
                    row.get("path") == "sparse_delta" for row in records
                ),
                "fallback": sum(
                    "fallback" in str(row.get("path", "")) for row in records
                ),
                "gpu_only": True,
                "cuda_metrics_materialized": False,
            }
            with open(
                os.path.join(directory, base + ".summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        finally:
            # Release GPU cache references after the request. This is ordinary
            # Python reference cleanup and performs no CUDA-to-host transfer.
            self.session.caches.clear()
            self.session.records.clear()
