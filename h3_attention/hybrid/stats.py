"""Lightweight request-scoped statistics for hybrid sparse attention."""

from datetime import datetime
import logging
import threading

from .report import validate_run_tag, write_request

LOG_PREFIX = "[H3 hybrid sparse]"


class HybridStatsCollector:
    def __init__(self, output_root, run_tag):
        self.output_root = str(output_root)
        self.run_tag = validate_run_tag(run_tag)
        self._lock = threading.RLock()
        self._request_id = -1
        self._timestamp = None
        self._records = []
        self.last_report_directory = None

    def on_request_reset(self, request_id):
        with self._lock:
            self._request_id = int(request_id)
            self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self._records = []

    def record(self, metadata):
        with self._lock:
            self._records.append(dict(metadata))

    def on_request_end(self, request_id, seconds=None):
        with self._lock:
            timestamp = self._timestamp or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            records = list(self._records)
        directory = write_request(
            self.output_root,
            self.run_tag,
            timestamp,
            request_id,
            records,
            seconds,
        )
        self.last_report_directory = directory
        logging.info("%s wrote %d records to %s", LOG_PREFIX, len(records), directory)
        return directory
