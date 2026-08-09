from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Any

import torch

from .offload import AsyncPinnedOffload


@dataclass
class ChunkCache:
    # Persistent tensor state lives in AsyncPinnedOffload host backing. The
    # per-request cache stores only validity/scheduling metadata and optional
    # staging leases.
    valid: bool = False
    prefetch: Any | None = None
    refresh_step: int = -1
    update_step: int = -1
    dense_calls: int = 0
    sparse_calls: int = 0
    fallback_calls: int = 0

    def clear(self):
        self.valid = False
        self.prefetch = None
        self.refresh_step = -1
        self.update_step = -1


@dataclass
class H3ChipmunkSession:
    config: object
    request_id: int = -1
    layout_signature: tuple | None = None
    caches: Dict[Tuple[tuple, int, int], ChunkCache] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    offload: AsyncPinnedOffload = field(init=False)

    def __post_init__(self):
        self.offload = AsyncPinnedOffload(self.config)

    def _release_request_state(self):
        for cache in self.caches.values():
            self.offload.release_prefetch(cache)
        self.caches.clear()

    def reset(self, request_id: int, layout_signature=None):
        # Drop request validity but keep AsyncPinnedOffload's allocated host
        # buffers. Subsequent generations with the same geometry reuse the pinned
        # allocation without paying the initialization cost again.
        self._release_request_state()
        self.request_id = int(request_id)
        self.layout_signature = layout_signature
        self.records.clear()

    def finish_request(self):
        self._release_request_state()
        self.records.clear()

    def ensure_request(self, snapshot):
        request_id = int(getattr(snapshot, "request_id", -1))
        layout_signature = getattr(snapshot, "layout_signature", None)
        if request_id != self.request_id or layout_signature != self.layout_signature:
            self.reset(request_id, layout_signature)

    def cache(self, branch, layer_index: int, chunk_index: int) -> ChunkCache:
        key = (tuple(int(x) for x in branch), int(layer_index), int(chunk_index))
        value = self.caches.get(key)
        if value is None:
            value = ChunkCache()
            self.caches[key] = value
        return value

    @staticmethod
    def host_key(branch, layer_index: int, chunk_index: int, spec) -> tuple:
        # Shape is part of the persistent key so a changed sequence/chunk layout
        # cannot accidentally reuse incompatible backing storage.
        return (
            tuple(int(x) for x in branch),
            int(layer_index),
            int(chunk_index),
            int(spec.rows),
            int(spec.selected_features),
            int(spec.selector_rows),
            int(spec.selected_groups),
        )

    @staticmethod
    def _assert_host_metadata(value):
        if torch.is_tensor(value):
            raise RuntimeError(
                "Chipmunk report records may not contain tensors; production nodes "
                "must never materialize CUDA diagnostics on the host"
            )
        if isinstance(value, dict):
            for item in value.values():
                H3ChipmunkSession._assert_host_metadata(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                H3ChipmunkSession._assert_host_metadata(item)

    def record(self, **values):
        for value in values.values():
            self._assert_host_metadata(value)
        self.records.append(dict(values))

    def invalidate_branch(self, branch):
        branch = tuple(int(x) for x in branch)
        for key, value in tuple(self.caches.items()):
            if key[0] == branch:
                self.offload.release_prefetch(value)
                value.clear()
