"""CPU-only tests for the latest-only TAEH3 preview worker."""

from __future__ import annotations

import os
import sys
import threading
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from chunked_ref2v.longform.taeh3_preview_worker import AsyncTAEH3PreviewWorker


class FakeStream:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


class FakeEvent:
    def __init__(self, log):
        self.log = log

    def record(self, stream):
        self.log.append("record")

    def wait(self, stream):
        self.log.append("wait")


class FakePreviewer:
    device = "cpu"

    def __init__(self, log, *, block=False, fail=False):
        self.log = log
        self.block = block
        self.fail = fail
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = []

    def frames(self, snapshot, *, limit=0):
        self.log.append("decode")
        self.calls.append((int(snapshot.shape[2]), int(limit)))
        self.started.set()
        if self.block:
            self.release.wait(2)
        if self.fail:
            raise RuntimeError("decoder failed")
        return torch.zeros(1, 2, 2, 3, dtype=torch.uint8)


class FakeSnapshot:
    shape = (1, 24, 3, 2, 2)

    def record_stream(self, stream):
        return None


def wait_until(predicate):
    deadline = time.time() + 2
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class WorkerTests(unittest.TestCase):
    def make_worker(self, previewer, published, errors, log):
        stream = FakeStream()
        return AsyncTAEH3PreviewWorker(
            previewer,
            lambda job, frames: published.append(job.sequence),
            errors.append,
            stream_factory=lambda device: stream,
            event_factory=lambda: FakeEvent(log),
            current_stream_factory=lambda device: stream,
            stream_context_factory=lambda value: _NullContext(),
        )

    def submit(self, worker, sequence=0, limit=0):
        worker.submit_snapshot(
            FakeSnapshot(),
            worker.record_producer_event("cpu"),
            chunk_index=0,
            step=sequence,
            total_steps=3,
            limit=limit,
        )

    def test_event_wait_precedes_decode(self):
        log, published, errors = [], [], []
        worker = self.make_worker(FakePreviewer(log), published, errors, log)
        self.submit(worker, limit=4)
        wait_until(lambda: published)
        worker.close()
        self.assertLess(log.index("record"), log.index("wait"))
        self.assertLess(log.index("wait"), log.index("decode"))
        self.assertFalse(errors)

    def test_pending_queue_replaces_historical_job(self):
        log, published, errors = [], [], []
        previewer = FakePreviewer(log, block=True)
        worker = self.make_worker(previewer, published, errors, log)
        self.submit(worker, 1)
        self.assertTrue(previewer.started.wait(1))
        self.submit(worker, 2)
        self.submit(worker, 3)
        previewer.release.set()
        wait_until(lambda: len(published) == 1)
        worker.close()
        self.assertEqual(published, [3])
        self.assertEqual([call[1] for call in previewer.calls], [0, 0])

    def test_newer_active_result_is_suppressed(self):
        log, published, errors = [], [], []
        previewer = FakePreviewer(log, block=True)
        worker = self.make_worker(previewer, published, errors, log)
        self.submit(worker, 1)
        self.assertTrue(previewer.started.wait(1))
        self.submit(worker, 2)
        previewer.release.set()
        wait_until(lambda: published)
        worker.close()
        self.assertEqual(published, [2])

    def test_failure_stops_acceptance_and_close_is_idempotent(self):
        log, published, errors = [], [], []
        worker = self.make_worker(
            FakePreviewer(log, fail=True), published, errors, log
        )
        self.submit(worker)
        wait_until(lambda: errors)
        self.assertFalse(worker.accepting)
        self.assertFalse(worker.submit_snapshot(
            FakeSnapshot(),
            worker.record_producer_event("cpu"),
            chunk_index=0,
            step=2,
            total_steps=3,
            limit=0,
        ))
        worker.close()
        worker.close()
        self.assertEqual(len(errors), 1)

    def test_close_drains_only_the_active_and_latest_jobs(self):
        log, published, errors = [], [], []
        previewer = FakePreviewer(log, block=True)
        worker = self.make_worker(previewer, published, errors, log)
        self.submit(worker, 1)
        self.assertTrue(previewer.started.wait(1))
        self.submit(worker, 2)
        self.submit(worker, 3)

        closed = threading.Event()
        closer = threading.Thread(target=lambda: (worker.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(0.05))
        previewer.release.set()
        closer.join(2)

        self.assertTrue(closed.is_set())
        self.assertEqual(published, [3])
        self.assertEqual(len(previewer.calls), 2)
        self.assertFalse(errors)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    unittest.main()
