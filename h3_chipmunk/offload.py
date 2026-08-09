from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
from typing import Iterable

import torch


H3_HIDDEN = 5376
H3_FFN = 14336


@dataclass(frozen=True)
class CacheSpec:
    rows: int
    selected_features: int
    selector_rows: int
    selected_groups: int
    hidden: int = H3_HIDDEN
    ffn: int = H3_FFN


@dataclass
class HostRecord:
    spec: CacheSpec
    allocating: bool = True
    allocated: bool = False
    error: str | None = None
    activation: torch.Tensor | None = None
    output: torch.Tensor | None = None
    selector_summary: torch.Tensor | None = None
    selected_groups: torch.Tensor | None = None
    ready_event: torch.cuda.Event | None = None


@dataclass
class StageSlot:
    activation: torch.Tensor
    output: torch.Tensor
    selector_summary: torch.Tensor
    selected_groups: torch.Tensor
    in_use: bool = False
    ready_event: torch.cuda.Event | None = None


@dataclass
class StageLease:
    slot: StageSlot
    spec: CacheSpec
    load_event: torch.cuda.Event | None = None

    @property
    def activation(self):
        return self.slot.activation[: self.spec.rows, : self.spec.selected_features]

    @property
    def output(self):
        return self.slot.output[: self.spec.rows, : self.spec.hidden]

    @property
    def selector_summary(self):
        return self.slot.selector_summary[
            : self.spec.selector_rows, : self.spec.ffn
        ]

    @property
    def selected_groups(self):
        return self.slot.selected_groups[
            : self.spec.selector_rows, : self.spec.selected_groups
        ]


@dataclass
class Prefetch:
    lease: StageLease
    fields: tuple[str, ...]


class AsyncPinnedOffload:
    """Pinned-host persistent cache with bounded CUDA staging.

    Persistent per-layer/chunk state is never resident in VRAM between uses.
    H2D and D2H copies run on dedicated CUDA streams. The Python/model thread
    never calls Event.synchronize(), Stream.synchronize(), Tensor.cpu(), item(),
    or any equivalent CUDA->host materialization. Consumers express ordering with
    CUDA events; if PCIe cannot keep up, the GPU stream waits for DMA rather than
    blocking the CPU to poll device state.

    Pinned host buffers are allocated by a background thread. A cache entry that
    has not finished allocating simply remains dense for that evaluation instead
    of stalling model execution.
    """

    def __init__(self, config):
        self.config = config
        self._lock = threading.RLock()
        self._hosts: dict[tuple, HostRecord] = {}
        self._alloc_queue: queue.Queue = queue.Queue()
        self._allocator = threading.Thread(
            target=self._allocator_loop,
            name="h3-chipmunk-pinned-allocator",
            daemon=True,
        )
        self._allocator.start()

        self._device: torch.device | None = None
        self._h2d_stream: torch.cuda.Stream | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        self._slots: list[StageSlot] = []
        self._slot_cursor = 0

    def _allocator_loop(self):
        while True:
            item = self._alloc_queue.get()
            if item is None:
                return
            key, spec = item
            try:
                activation = torch.empty(
                    (spec.rows, spec.selected_features),
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                output = torch.empty(
                    (spec.rows, spec.hidden),
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                selector_summary = torch.empty(
                    (spec.selector_rows, spec.ffn),
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=True,
                )
                selected_groups = torch.empty(
                    (spec.selector_rows, spec.selected_groups),
                    dtype=torch.int32,
                    device="cpu",
                    pin_memory=True,
                )
                with self._lock:
                    record = self._hosts.get(key)
                    if record is None or record.spec != spec:
                        continue
                    record.activation = activation
                    record.output = output
                    record.selector_summary = selector_summary
                    record.selected_groups = selected_groups
                    record.allocated = True
                    record.allocating = False
            except Exception as exc:
                with self._lock:
                    record = self._hosts.get(key)
                    if record is not None and record.spec == spec:
                        record.error = f"{type(exc).__name__}: {exc}"
                        record.allocating = False
                        record.allocated = False
            finally:
                self._alloc_queue.task_done()

    def request_host(self, key: tuple, spec: CacheSpec) -> HostRecord:
        with self._lock:
            record = self._hosts.get(key)
            if record is not None:
                if record.spec != spec:
                    raise RuntimeError(
                        "Chipmunk host-cache shape changed for an existing key: "
                        f"{record.spec} -> {spec}"
                    )
                return record
            record = HostRecord(spec=spec)
            self._hosts[key] = record
            self._alloc_queue.put((key, spec))
            return record

    @staticmethod
    def _tensor_bytes(shape: Iterable[int], dtype: torch.dtype) -> int:
        elements = 1
        for value in shape:
            elements *= int(value)
        return elements * torch.empty((), dtype=dtype).element_size()

    def _ensure_device(self, device: torch.device):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("Chipmunk async cache requires a CUDA device")
        if self._device is not None:
            if device != self._device:
                raise RuntimeError(
                    f"one Chipmunk cache session cannot span CUDA devices: "
                    f"{self._device} vs {device}"
                )
            return

        max_selected = int(self.config.max_selected_features(H3_FFN))
        max_selector_rows = (
            int(self.config.chunk_rows) + int(self.config.token_group_rows) - 1
        ) // int(self.config.token_group_rows)
        max_groups = max_selected // int(self.config.feature_group)
        slots = int(self.config.staging_slots)

        bytes_per_slot = (
            self._tensor_bytes(
                (int(self.config.chunk_rows), max_selected), torch.bfloat16
            )
            + self._tensor_bytes(
                (int(self.config.chunk_rows), H3_HIDDEN), torch.bfloat16
            )
            + self._tensor_bytes(
                (max_selector_rows, H3_FFN), torch.bfloat16
            )
            + self._tensor_bytes(
                (max_selector_rows, max_groups), torch.int32
            )
        )
        staging_bytes = bytes_per_slot * slots
        budget_bytes = int(float(self.config.cache_budget_gb) * (1024 ** 3))
        if staging_bytes > budget_bytes:
            raise RuntimeError(
                "Chipmunk CUDA staging requires %.3f GiB but cache_budget_gb is %.3f; "
                "reduce chunk_rows/staging_slots or raise the bounded staging budget"
                % (
                    staging_bytes / (1024 ** 3),
                    float(self.config.cache_budget_gb),
                )
            )

        self._device = device
        self._h2d_stream = torch.cuda.Stream(device=device)
        self._d2h_stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device):
            self._slots = [
                StageSlot(
                    activation=torch.empty(
                        (int(self.config.chunk_rows), max_selected),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    output=torch.empty(
                        (int(self.config.chunk_rows), H3_HIDDEN),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    selector_summary=torch.empty(
                        (max_selector_rows, H3_FFN),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    selected_groups=torch.empty(
                        (max_selector_rows, max_groups),
                        dtype=torch.int32,
                        device=device,
                    ),
                )
                for _ in range(slots)
            ]

    def _acquire_slot(self, device: torch.device) -> StageSlot | None:
        self._ensure_device(device)
        count = len(self._slots)
        for offset in range(count):
            index = (self._slot_cursor + offset) % count
            slot = self._slots[index]
            if slot.in_use:
                continue
            slot.in_use = True
            self._slot_cursor = (index + 1) % count
            return slot
        return None

    @staticmethod
    def _copy_if_different(dst: torch.Tensor, src: torch.Tensor):
        if dst.data_ptr() != src.data_ptr():
            dst.copy_(src, non_blocking=True)

    def release_lease(self, lease: StageLease | None):
        if lease is None:
            return
        if lease.load_event is not None:
            lease.slot.ready_event = lease.load_event
        lease.slot.in_use = False

    def prefetch(
        self,
        cache,
        key: tuple,
        spec: CacheSpec,
        fields: tuple[str, ...],
        device: torch.device,
    ) -> bool:
        if not fields or not getattr(cache, "valid", False):
            self.request_host(key, spec)
            return False
        if getattr(cache, "prefetch", None) is not None:
            return True

        record = self.request_host(key, spec)
        if record.error is not None:
            raise RuntimeError(f"Chipmunk pinned allocation failed: {record.error}")
        if not record.allocated:
            return False

        slot = self._acquire_slot(device)
        if slot is None:
            return False
        lease = StageLease(slot=slot, spec=spec)
        stream = self._h2d_stream
        assert stream is not None
        if slot.ready_event is not None:
            stream.wait_event(slot.ready_event)
        if record.ready_event is not None:
            stream.wait_event(record.ready_event)

        with torch.cuda.stream(stream):
            for field in fields:
                getattr(lease, field).copy_(getattr(record, field), non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)
        lease.load_event = event
        cache.prefetch = Prefetch(lease=lease, fields=tuple(fields))
        return True

    def load(
        self,
        cache,
        key: tuple,
        spec: CacheSpec,
        fields: tuple[str, ...],
        device: torch.device,
    ) -> StageLease | None:
        prefetched = getattr(cache, "prefetch", None)
        if prefetched is None or tuple(prefetched.fields) != tuple(fields):
            self.release_prefetch(cache)
            if not self.prefetch(cache, key, spec, fields, device):
                return None
            prefetched = cache.prefetch
        lease = prefetched.lease
        if lease.load_event is not None:
            torch.cuda.current_stream(device).wait_event(lease.load_event)
        cache.prefetch = None
        return lease

    def release_prefetch(self, cache):
        prefetched = getattr(cache, "prefetch", None)
        if prefetched is None:
            return
        self.release_lease(prefetched.lease)
        cache.prefetch = None

    def store(
        self,
        cache,
        key: tuple,
        spec: CacheSpec,
        *,
        activation: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        selector_summary: torch.Tensor | None = None,
        selected_groups: torch.Tensor | None = None,
        lease: StageLease | None = None,
    ) -> bool:
        record = self.request_host(key, spec)
        if record.error is not None:
            raise RuntimeError(f"Chipmunk pinned allocation failed: {record.error}")
        if not record.allocated:
            self.release_lease(lease)
            return False

        device = (
            activation.device
            if activation is not None
            else output.device
            if output is not None
            else selector_summary.device
            if selector_summary is not None
            else selected_groups.device
            if selected_groups is not None
            else self._device
        )
        if device is None:
            raise RuntimeError("Chipmunk store needs at least one CUDA tensor")
        current = torch.cuda.current_stream(device)

        if lease is None:
            slot = self._acquire_slot(device)
            if slot is None:
                return False
            lease = StageLease(slot=slot, spec=spec)
            if slot.ready_event is not None:
                current.wait_event(slot.ready_event)

        fields = []
        if activation is not None:
            self._copy_if_different(lease.activation, activation)
            fields.append("activation")
        if output is not None:
            self._copy_if_different(lease.output, output)
            fields.append("output")
        if selector_summary is not None:
            self._copy_if_different(lease.selector_summary, selector_summary)
            fields.append("selector_summary")
        if selected_groups is not None:
            self._copy_if_different(lease.selected_groups, selected_groups)
            fields.append("selected_groups")

        stream = self._d2h_stream
        assert stream is not None
        if record.ready_event is not None:
            stream.wait_event(record.ready_event)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for field in fields:
                getattr(record, field).copy_(getattr(lease, field), non_blocking=True)
            event = torch.cuda.Event()
            event.record(stream)

        record.ready_event = event
        lease.slot.ready_event = event
        lease.slot.in_use = False
        cache.prefetch = None
        cache.valid = True
        return True

    @property
    def staging_bytes(self) -> int:
        total = 0
        for slot in self._slots:
            total += sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    slot.activation,
                    slot.output,
                    slot.selector_summary,
                    slot.selected_groups,
                )
            )
        return int(total)
