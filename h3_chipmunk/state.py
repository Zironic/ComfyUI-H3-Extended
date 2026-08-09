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

    def record(self, **values):
        # Measurement scalars may remain as CUDA tensors here. Converting them
        # with .item() in the block loop would synchronize the whole device for
        # every chunk. materialize_records() batches those transfers at request
        # end instead.
        self.records.append(dict(values))

    def materialize_records(self):
        """Return JSON-safe records with one batched scalar copy per device."""
        scalar_tensors = []

        def encode(value):
            if torch.is_tensor(value):
                tensor = value.detach()
                if tensor.numel() == 1:
                    slot = len(scalar_tensors)
                    scalar_tensors.append(tensor.reshape(()))
                    return ("__h3_chipmunk_scalar__", slot)
                return tensor.to("cpu").tolist()
            if isinstance(value, dict):
                return {key: encode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [encode(item) for item in value]
            if isinstance(value, tuple):
                return tuple(encode(item) for item in value)
            return value

        encoded = [encode(row) for row in self.records]
        resolved = [None] * len(scalar_tensors)
        by_device = {}
        for index, tensor in enumerate(scalar_tensors):
            by_device.setdefault(str(tensor.device), []).append((index, tensor))
        for items in by_device.values():
            # One stack/copy per device replaces thousands of per-chunk .item()
            # synchronizations. Values are diagnostic only, so float32 is enough.
            packed = torch.stack([tensor.float() for _, tensor in items]).to("cpu")
            values = packed.tolist()
            for (index, _tensor), value in zip(items, values):
                resolved[index] = float(value)

        def decode(value):
            if (
                isinstance(value, tuple)
                and len(value) == 2
                and value[0] == "__h3_chipmunk_scalar__"
            ):
                return resolved[int(value[1])]
            if isinstance(value, dict):
                return {key: decode(item) for key, item in value.items()}
            if isinstance(value, list):
                return [decode(item) for item in value]
            if isinstance(value, tuple):
                return tuple(decode(item) for item in value)
            return value

        return [decode(row) for row in encoded]

    def invalidate_branch(self, branch):
        branch = tuple(int(x) for x in branch)
        for key, value in tuple(self.caches.items()):
            if key[0] == branch:
                value.clear()
