"""Opt-in CUDA parity test for variable Sparse-Sage row lengths.

Set ``H3_RUN_ADAPTIVE_SPARSE_SAGE_CUDA_TESTS=1`` on an authorized SM89 machine
with the compiled ``spas_sage_attn`` package installed.
"""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_attention.hybrid.config import (  # noqa: E402
    DENSITY_ADAPTIVE_BUDGET,
    HybridSparseConfig,
)
from h3_attention.hybrid.router import SparseTileRouter  # noqa: E402
from h3_attention.hybrid.sparse_sage import (  # noqa: E402
    SparseSageExecutor,
    preflight_sparse_sage,
)


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def layout():
    return SimpleNamespace(
        seq_len=384,
        video_range=(128, 384),
        segments=[(0, 96, "text"), (96, 128, "audio"), (128, 384, "video")],
        video_shape=(1, 1, 256),
        audio_t=16,
    )


def controlled_qk(device):
    q_summary = torch.zeros((1, 2, 3, 128), dtype=torch.float16, device=device)
    k_summary = torch.zeros((1, 2, 6, 128), dtype=torch.float16, device=device)
    q_summary[0, 0, 1, 0] = 1
    q_summary[0, 0, 2, 1] = 1
    q_summary[0, 1, 1, 0] = 1
    q_summary[0, 1, 2, 1] = 1
    k_summary[0, 0, 2:, :2] = torch.tensor(
        ((10.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        dtype=torch.float16,
        device=device,
    )
    k_summary[0, 1, 2:, :2] = torch.tensor(
        ((4.0, 4.0), (3.0, 3.0), (2.0, 2.0), (1.0, 1.0)),
        dtype=torch.float16,
        device=device,
    )
    q = q_summary.repeat_interleave(128, dim=-2)[..., :384, :].contiguous()
    k = k_summary.repeat_interleave(64, dim=-2)[..., :384, :].contiguous()
    return q_summary, k_summary, q, k


def decode_block_mask(lut, valid):
    indices = torch.cumsum(lut, dim=-1).long()
    rank = torch.arange(lut.shape[-1], device=lut.device)
    active = rank.view(1, 1, 1, -1) < valid[..., None]
    one_hot = F.one_hot(indices.clamp(0, lut.shape[-1] - 1), lut.shape[-1]).bool()
    return (one_hot & active[..., None]).any(dim=-2)


def dense_reference(q, k, v, lut, valid):
    sequence = q.shape[-2]
    block_mask = decode_block_mask(lut, valid)
    token_mask = block_mask.repeat_interleave(128, dim=-2).repeat_interleave(64, dim=-1)
    token_mask = token_mask[..., :sequence, :sequence]
    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * (128 ** -0.5)
    scores.masked_fill_(~token_mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def main():
    if os.environ.get("H3_RUN_ADAPTIVE_SPARSE_SAGE_CUDA_TESTS") != "1":
        print(
            "adaptive variable-row CUDA parity: SKIP "
            "(set H3_RUN_ADAPTIVE_SPARSE_SAGE_CUDA_TESTS=1 after authorization)"
        )
        return

    api = preflight_sparse_sage()
    device = torch.device("cuda")
    config = HybridSparseConfig(
        video_budget=0.5,
        density_mode=DENSITY_ADAPTIVE_BUDGET,
        min_video_density=0.25,
        max_video_density=1.0,
        adaptive_temperature=1.0,
        adaptive_target_mass=0.80,
    )
    router = SparseTileRouter(config)
    q_summary, k_summary, q, k = controlled_qk(device)
    generator = torch.Generator(device=device).manual_seed(8127)
    v = torch.randn(q.shape, dtype=torch.float16, device=device, generator=generator)
    lut, valid, metadata = router.build_lut_from_summaries(
        q_summary, k_summary, layout(), config.video_budget
    )
    counts = valid[..., 1:] - 2
    check(counts.unique().numel() > 1,
          "one Sparse-Sage launch contains genuinely different row lengths")
    check(int(counts.sum()) == metadata.retained_video_kv_tiles * counts.numel(),
          "variable rows preserve the exact aggregate block budget")

    executor = SparseSageExecutor(api)
    prepared = executor.prepare(
        q,
        k,
        v,
        lut,
        valid,
        layer_index=0,
        metadata=metadata.as_dict(),
    )
    reference = dense_reference(q, k, v, lut, valid)
    output = executor.execute(prepared)
    relative_rmse = (
        (output.float() - reference.float()).square().mean().sqrt()
        / reference.float().square().mean().sqrt().clamp_min(1e-8)
    ).item()
    check(relative_rmse < 0.08,
          "variable-row Sparse-Sage matches explicit masked attention")
    print("relative RMSE: %.6f" % relative_rmse)
    print("\nadaptive variable-row CUDA parity passed")


if __name__ == "__main__":
    main()
