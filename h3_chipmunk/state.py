from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch


@dataclass
class ChunkCache:
    activation: torch.Tensor | None = None
    output: torch.Tensor | None = None
    selector_summary: torch.Tensor | None = None
    selected_groups: torch.Tensor | None = None
    selected_counts: torch.Tensor | None = None
    refresh_step: int = -1
    update_step: int = -1
    dense_calls: int = 0
    sparse_calls: int = 0
    fallback_calls: int = 0

    def clear(self):
        self.activation = None
        self.output = None
        self.selector_summary = None
        self.selected_groups = None
        self.selected_counts = None
        self.refresh_step = -1
        self.update_step = -1


@dataclass
class H3ChipmunkSession:
    request_id: int = -1
    layout_signature: tuple | None = None
    caches: Dict[Tuple[tuple, int, int], ChunkCache] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def reset(self, request_id: int, layout_signature=None):
        self.request_id = int(request_id)
        self.layout_signature = layout_signature
        self.caches.clear()
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
                value.clear()
