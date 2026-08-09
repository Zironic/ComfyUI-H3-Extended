from __future__ import annotations

import json
import os
import time


class H3ChipmunkReportListener:
    """Optional host-metadata report writer.

    Records contain only Python metadata already known without reading CUDA
    tensors. Persistent cache data stays in pinned host backing and is never
    serialized by this reporter.
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
                "dense_refresh_async": sum(
                    row.get("path") == "dense_refresh_async" for row in records
                ),
                "sparse_delta_async": sum(
                    row.get("path") == "sparse_delta_async" for row in records
                ),
                "dma_miss": sum(
                    row.get("path") in ("dense_dma_miss", "dense_cache_not_ready")
                    for row in records
                ),
                "fallback": sum(
                    "fallback" in str(row.get("path", "")) for row in records
                ),
                "compute_device": "cuda",
                "cache_backing": "pinned_host_async",
                "cuda_metrics_materialized": False,
                "gpu_staging_budget_gb": float(self.config.cache_budget_gb),
                "density_profile": self.config.density_profile,
                "profile": [list(item) for item in self.config.profile],
            }
            with open(
                os.path.join(directory, base + ".summary.json"),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        finally:
            # Release only per-request staging leases/validity. Allocated pinned
            # host buffers remain warm and reusable for the next generation.
            self.session.finish_request()
