"""Latest-only asynchronous CUDA worker for long-form TAEH3 previews."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import torch


LOG = "[H3 Extended] taeh3 preview worker"


@dataclass
class _PreviewJob:
    sequence: int
    snapshot: torch.Tensor
    producer_event: object
    chunk_index: int
    step: int
    total_steps: int
    limit: int


class AsyncTAEH3PreviewWorker:
    """Decode current-chunk TAEH3 previews on a dedicated CUDA stream.

    The sampler only clones a bounded latent snapshot and records an event.
    All decoding and host-side publication happens on this thread.  The queue
    intentionally keeps at most one pending job: a preview is disposable, and
    retaining history only increases latency and memory pressure.
    """

    def __init__(
        self,
        previewer,
        publish_result,
        on_error,
        *,
        stream_factory=None,
        event_factory=None,
        current_stream_factory=None,
        stream_context_factory=None,
    ):
        self.previewer = previewer
        self.publish_result = publish_result
        self.on_error = on_error
        self._stream_factory = stream_factory or self._default_stream
        self._event_factory = event_factory or self._default_event
        self._current_stream_factory = (
            current_stream_factory or self._default_current_stream
        )
        self._stream_context_factory = (
            stream_context_factory or torch.cuda.stream
        )
        device = torch.device(previewer.device)
        try:
            self.stream = self._stream_factory(device)
        except TypeError:
            self.stream = self._stream_factory()
        self._condition = threading.Condition()
        self._pending = None
        self._active = None
        self._accepting = True
        self._stopping = False
        self._failed = False
        self._error_announced = False
        self._sequence = 0
        self._thread = threading.Thread(
            target=self._run,
            name="h3-taeh3-preview",
            daemon=False,
        )
        self._thread.start()

    @staticmethod
    def _default_stream(device):
        return torch.cuda.Stream(device=device)

    @staticmethod
    def _default_event():
        return torch.cuda.Event()

    @staticmethod
    def _default_current_stream(device):
        return torch.cuda.current_stream(device=device)

    @property
    def failed(self):
        with self._condition:
            return self._failed

    @property
    def accepting(self):
        with self._condition:
            return self._accepting

    def record_producer_event(self, device):
        """Record the producer event after the caller has cloned its snapshot."""
        try:
            event = self._event_factory()
        except TypeError:
            event = self._event_factory(device)
        try:
            stream = self._current_stream_factory(device)
        except TypeError:
            stream = self._current_stream_factory()
        event.record(stream)
        return event

    def submit_snapshot(
        self,
        snapshot,
        producer_event,
        *,
        chunk_index,
        step,
        total_steps,
        limit,
    ):
        """Submit a cloned latent and its producer event.

        Returns ``False`` after failure/close; this is a normal race during
        shutdown and does not raise into the sampler.
        """
        with self._condition:
            if not self._accepting:
                return False
            self._sequence += 1
            job = _PreviewJob(
                sequence=self._sequence,
                snapshot=snapshot,
                producer_event=producer_event,
                chunk_index=int(chunk_index),
                step=int(step),
                total_steps=int(total_steps),
                limit=int(limit),
            )
            # Releasing the old pending job here is the latest-only queue.
            self._pending = job
            self._condition.notify()
            return True

    submit = submit_snapshot

    def _run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return
                self._active = self._pending
                self._pending = None
                job = self._active
            try:
                self._execute(job)
            except Exception as exc:
                self._fail(exc)
                return
            finally:
                with self._condition:
                    self._active = None

    def _execute(self, job):
        with self._stream_context_factory(self.stream):
            job.producer_event.wait(self.stream)
            job.snapshot.record_stream(self.stream)
            frames = self.previewer.frames(job.snapshot, limit=job.limit)

        # A newer submission makes this active result disposable.  The check
        # is intentionally short; publication itself may do resize/ffmpeg/UI
        # work without holding the submission lock.
        with self._condition:
            if self._sequence > job.sequence:
                return
        self.publish_result(job, frames)

    def _fail(self, exc):
        with self._condition:
            if self._failed:
                return
            self._failed = True
            self._accepting = False
            self._pending = None
            self._stopping = True
            if self._error_announced:
                return
            self._error_announced = True
        try:
            self.on_error(exc)
        except Exception:
            logging.exception("%s error handler failed", LOG)
        with self._condition:
            self._condition.notify_all()

    def close(self):
        """Stop accepting work, drain queued jobs, and join the worker."""
        with self._condition:
            self._accepting = False
            self._stopping = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()
        synchronize = getattr(self.stream, "synchronize", None)
        if callable(synchronize):
            synchronize()


__all__ = ["AsyncTAEH3PreviewWorker"]
