"""CPU tests for architecture-native Sparse Sage routing."""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def layout(sequence=384, video_start=128):
    return SimpleNamespace(
        seq_len=sequence,
        video_range=(video_start, sequence),
        segments=[
            (0, video_start - 32, "text"),
            (video_start - 32, video_start, "audio"),
            (video_start, sequence, "video"),
        ],
        video_shape=(1, 1, sequence - video_start),
        audio_t=16,
    )


def routed_inputs():
    q = torch.zeros((1, 2, 384, 2), dtype=torch.float32)
    k = torch.zeros_like(q)
    q[0, 0, 128:, 0] = 1
    q[0, 1, 128:, 1] = 1
    head0 = ((4, 0), (3, 0), (2, 0), (1, 0))
    head1 = ((0, 1), (0, 2), (0, 3), (0, 4))
    for index, (a, b) in enumerate(head0, start=2):
        k[0, 0, index * 64:(index + 1) * 64] = torch.tensor((a, b))
    for index, (a, b) in enumerate(head1, start=2):
        k[0, 1, index * 64:(index + 1) * 64] = torch.tensor((a, b))
    return q, k


def decode(lut, valid):
    mask = torch.zeros(lut.shape, dtype=torch.bool)
    for index in range(valid.shape[-1]):
        count = int(valid[..., index].max().item())
        if count:
            delta = lut[..., index, :count]
            mask[..., index, :] = torch.nn.functional.one_hot(
                torch.cumsum(delta, dim=-1).long(), num_classes=lut.shape[-1]
            ).any(dim=-2)
    return mask


def test_exact_direct_routing():
    print("direct per-head routing")
    router = SparseTileRouter()
    q, k = routed_inputs()
    lut, valid, metadata = router.build_lut(q, k, layout(), 0.5)
    mask = decode(lut, valid)
    check(mask.shape == (1, 2, 3, 6), "mask uses global 128Q x 64KV geometry")
    check(mask[..., :2].all(), "all non-video KV tiles stay dense")
    check(mask[:, :, 0].all(), "non-video Q tile stays dense")
    check(set(torch.where(mask[0, 0, 1, 2:])[0].tolist()) == {0, 1},
          "head zero retains its top two video KV tiles")
    check(set(torch.where(mask[0, 1, 1, 2:])[0].tolist()) == {2, 3},
          "head one retains a different top-two route")
    check(metadata.retained_video_kv_tiles == 2,
          "50% retains exactly two of four pure-video KV tiles")
    check(metadata.actual_video_tile_density == 0.5,
          "reported video density is exactly 50%")
    check(abs(metadata.full_mask_density - 14 / 18) < 1e-12,
          "packed density includes dense context and non-video Q rows")


def test_mixed_boundary_and_partial_tiles():
    print("mixed and partial tiles")
    router = SparseTileRouter()
    q = torch.randn((1, 1, 350, 4))
    k = torch.randn_like(q)
    mixed_layout = layout(sequence=350, video_start=96)
    lut, valid, metadata = router.build_lut(q, k, mixed_layout, 0.5)
    mask = decode(lut, valid)
    check(mask[:, :, 0].all(), "128Q tile crossing the video boundary stays dense")
    check(mask[..., 1].all(), "64KV tile crossing the video boundary stays dense")
    check(metadata.q_tiles == 3 and metadata.kv_tiles == 6,
          "partial final Q and KV tiles are represented")
    check(metadata.pure_video_q_tiles == 2 and metadata.pure_video_kv_tiles == 4,
          "partial final tiles count as pure video when all real rows are video")


def test_full_budget_skips_scoring():
    print("100% fast path")

    class NoPoolingRouter(SparseTileRouter):
        @staticmethod
        def _mean_pool(x, block):
            raise AssertionError("100% budget must not pool Q/K")

    q = torch.randn((1, 2, 384, 8))
    lut, valid, metadata = NoPoolingRouter().build_lut(q, q, layout(), 1.0)
    mask = decode(lut, valid)
    check(mask.all(), "100% budget produces an all-one block mask")
    check(metadata.full_mask_density == 1.0 and metadata.sparse_q_tiles == 0,
          "100% metadata reports a fully dense executable mask")


def test_geometry_cache():
    print("static geometry cache")
    router = SparseTileRouter()
    first = router.geometry(layout())
    second = router.geometry(layout())
    check(first is second, "identical layout signatures reuse static geometry only")


def main():
    test_exact_direct_routing()
    test_mixed_boundary_and_partial_tiles()
    test_full_budget_skips_scoring()
    test_geometry_cache()
    print("\nall hybrid router tests passed")


if __name__ == "__main__":
    main()
