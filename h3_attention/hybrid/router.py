"""Direct Sparse-Sage tile routing for MiniMax H3 hybrid attention."""

from dataclasses import dataclass
import math

import torch

Q_TILE = 128
KV_TILE = 64


class SparseRouterError(RuntimeError):
    pass


@dataclass(frozen=True)
class SparseTileGeometry:
    signature: tuple
    sequence: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_start: int
    pure_video_kv_start: int

    @property
    def pure_video_q_tiles(self):
        return self.q_tiles - self.pure_video_q_start

    @property
    def pure_video_kv_tiles(self):
        return self.kv_tiles - self.pure_video_kv_start


@dataclass(frozen=True)
class SparseMaskMetadata:
    requested_video_budget: float
    actual_video_tile_density: float
    full_mask_density: float
    dense_q_tiles: int
    sparse_q_tiles: int
    q_tiles: int
    kv_tiles: int
    pure_video_q_tiles: int
    pure_video_kv_tiles: int
    retained_video_kv_tiles: int

    def as_dict(self):
        return {
            "requested_video_budget": self.requested_video_budget,
            "actual_video_tile_density": self.actual_video_tile_density,
            "full_mask_density": self.full_mask_density,
            "dense_q_tiles": self.dense_q_tiles,
            "sparse_q_tiles": self.sparse_q_tiles,
            "q_tiles": self.q_tiles,
            "kv_tiles": self.kv_tiles,
            "pure_video_q_tiles": self.pure_video_q_tiles,
            "pure_video_kv_tiles": self.pure_video_kv_tiles,
            "retained_video_kv_tiles": self.retained_video_kv_tiles,
        }


class SparseTileRouter:
    """Build one per-head route for each global 128-token query tile."""

    def __init__(self):
        self._geometry_cache = {}

    @staticmethod
    def _layout_signature(layout):
        return (
            int(layout.seq_len),
            tuple(int(x) for x in layout.video_range),
            tuple((int(a), int(b), str(kind)) for a, b, kind in layout.segments),
            tuple(int(x) for x in layout.video_shape),
            int(layout.audio_t),
        )

    def geometry(self, layout):
        signature = self._layout_signature(layout)
        cached = self._geometry_cache.get(signature)
        if cached is not None:
            return cached

        sequence = int(layout.seq_len)
        video_start, video_stop = (int(x) for x in layout.video_range)
        if sequence <= 0:
            raise SparseRouterError("packed sequence is empty")
        if not (0 <= video_start < video_stop == sequence):
            raise SparseRouterError(
                "Sparse Sage requires H3 target video to be the final packed segment; "
                "got video=%s sequence=%d" % (tuple(layout.video_range), sequence)
            )

        q_tiles = (sequence + Q_TILE - 1) // Q_TILE
        kv_tiles = (sequence + KV_TILE - 1) // KV_TILE
        pure_q_start = (video_start + Q_TILE - 1) // Q_TILE
        pure_kv_start = (video_start + KV_TILE - 1) // KV_TILE
        geometry = SparseTileGeometry(
            signature=signature,
            sequence=sequence,
            q_tiles=q_tiles,
            kv_tiles=kv_tiles,
            pure_video_q_start=pure_q_start,
            pure_video_kv_start=pure_kv_start,
        )
        if not geometry.pure_video_q_tiles or not geometry.pure_video_kv_tiles:
            raise SparseRouterError(
                "packed layout has no pure-video Sparse Sage tiles: video=%s sequence=%d"
                % (tuple(layout.video_range), sequence)
            )
        self._geometry_cache[signature] = geometry
        return geometry

    @staticmethod
    def _mean_pool(x, block):
        sequence = x.shape[-2]
        full = sequence // block
        remainder = sequence % block
        pieces = []
        if full:
            pieces.append(
                x[..., :full * block, :]
                .reshape(*x.shape[:-2], full, block, x.shape[-1])
                .mean(dim=-2)
            )
        if remainder:
            pieces.append(x[..., full * block:, :].mean(dim=-2, keepdim=True))
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)

    @staticmethod
    def _metadata(geometry, video_budget, retained):
        pure_q = geometry.pure_video_q_tiles
        pure_kv = geometry.pure_video_kv_tiles
        sparse_q = pure_q if retained < pure_kv else 0
        dense_q = geometry.q_tiles - sparse_q
        non_video_kv = geometry.kv_tiles - pure_kv
        true_blocks = (
            (geometry.q_tiles - pure_q) * geometry.kv_tiles
            + pure_q * (non_video_kv + retained)
        )
        return SparseMaskMetadata(
            requested_video_budget=float(video_budget),
            actual_video_tile_density=float(retained) / pure_kv,
            full_mask_density=float(true_blocks) / (geometry.q_tiles * geometry.kv_tiles),
            dense_q_tiles=dense_q,
            sparse_q_tiles=sparse_q,
            q_tiles=geometry.q_tiles,
            kv_tiles=geometry.kv_tiles,
            pure_video_q_tiles=pure_q,
            pure_video_kv_tiles=pure_kv,
            retained_video_kv_tiles=retained,
        )

    def build_mask(self, q, k, layout, video_budget):
        if q.ndim != 4 or k.ndim != 4:
            raise SparseRouterError("tile router expects HND rank-4 Q/K")
        if q.shape != k.shape:
            raise SparseRouterError(
                "tile router requires equal self-attention Q/K shapes; got %s %s"
                % (tuple(q.shape), tuple(k.shape))
            )
        if not 0.0 < float(video_budget) <= 1.0:
            raise SparseRouterError("video_budget must be in (0, 1]")

        geometry = self.geometry(layout)
        if q.shape[-2] != geometry.sequence:
            raise SparseRouterError(
                "runtime layout sequence %d does not match Q/K sequence %d"
                % (geometry.sequence, q.shape[-2])
            )

        batch, heads = q.shape[:2]
        mask = torch.ones(
            (batch, heads, geometry.q_tiles, geometry.kv_tiles),
            dtype=torch.bool,
            device=q.device,
        )
        pure_kv = geometry.pure_video_kv_tiles
        retained = min(pure_kv, math.ceil(float(video_budget) * pure_kv))
        metadata = self._metadata(geometry, video_budget, retained)
        if float(video_budget) == 1.0:
            return mask, metadata

        q_means = self._mean_pool(q, Q_TILE)
        k_means = self._mean_pool(k, KV_TILE)
        scores = torch.matmul(
            q_means[..., geometry.pure_video_q_start:, :],
            k_means[..., geometry.pure_video_kv_start:, :].transpose(-1, -2),
        )
        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.scatter_(-1, torch.topk(scores, retained, dim=-1).indices, True)
        mask[
            ...,
            geometry.pure_video_q_start:,
            geometry.pure_video_kv_start:,
        ] = selected
        return mask, metadata
