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
        records_path = os.path.join(directory, base + ".jsonl")
        with open(records_path, "w", encoding="utf-8") as handle:
            for row in self.session.records:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary = {
            "request_id": int(request_id),
            "seconds": seconds,
            "config": list(self.config.signature),
            "records": len(self.session.records),
            "dense_refresh": sum(r.get("path") == "dense_refresh" for r in self.session.records),
            "sparse_delta": sum(r.get("path") == "sparse_delta" for r in self.session.records),
            "measure": sum(r.get("path") == "measure" for r in self.session.records),
            "fallback": sum("fallback" in str(r.get("path", "")) for r in self.session.records),
        }
        with open(os.path.join(directory, base + ".summary.json"), "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
