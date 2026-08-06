"""CPU stand-in tests for the optional H3 Sol-Attn backend."""

import os
import sys
from types import SimpleNamespace

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_attention.sol import SolAttentionBackend, SolAttentionConfig  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def dense_bthd(q, k, v):
    return F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)


class DenseBackend:
    name = "fake_dense"

    def __init__(self):
        self.prepares = 0

    def prepare(self, q, k, v, **kwargs):
        self.prepares += 1
        return (q.clone(), k.clone(), v.clone())

    def execute(self, prepared):
        q, k, v = prepared
        return dense_bthd(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)


def fake_sol(q, k, v, **kwargs):
    return dense_bthd(q, k, v)


def runtime(step):
    layout = SimpleNamespace(
        seq_len=16,
        text_range=(0, 4),
        audio_range=(4, 6),
        video_range=(6, 16),
        reference_ranges=[],
    )
    return RuntimeSnapshot(
        request_id=0,
        step_index=step,
        total_steps=20,
        sigma=0.5,
        branch=(0,),
        layout=layout,
        layout_signature=(16,),
        compute_dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )


def qkv():
    generator = torch.Generator().manual_seed(4)
    return tuple(
        torch.randn(1, 2, 16, 128, generator=generator, dtype=torch.bfloat16)
        for _ in range(3)
    )


def main():
    dense = DenseBackend()
    config = SolAttentionConfig(
        dense_steps=10,
        dense_layers=2,
        correctness_gate=True,
        gate_heads=1,
        density_heads=1,
        strict=True,
    )
    backend = SolAttentionBackend(dense, config=config, sol_callable=fake_sol)

    q, k, v = qkv()
    prepared = backend.prepare(
        q, k, v,
        layer_index=10,
        transformer_options={RUNTIME_KEY: runtime(0)},
    )
    check(prepared.mode == "dense" and dense.prepares == 1, "warmup step delegates to dense backend")
    dense_out = backend.execute(prepared)
    check(dense_out.shape == q.shape, "dense delegate returns HND output")

    q, k, v = qkv()
    prepared = backend.prepare(
        q, k, v,
        layer_index=10,
        transformer_options={RUNTIME_KEY: runtime(10)},
    )
    check(prepared.mode == "sparse", "eligible call prepares Sol BTHD buffers")
    check(prepared.q.is_contiguous() and prepared.q.shape == (1, 16, 2, 128), "Sol preparation is contiguous BTHD")
    check((prepared.sink_start, prepared.sink_tokens) == (0, 6), "prefix sink protects text and audio")

    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *args, **kwargs: None
    try:
        out = backend.execute(prepared)
    finally:
        torch.cuda.synchronize = original_sync
    check(out.shape == q.shape, "Sol output is converted back to HND")
    expected = dense_bthd(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
    check(torch.equal(out, expected), "exact fake Sol path and prefix recompute preserve output")
    status = backend.as_status()
    check(status["sparse_calls"] == 1 and status["gate_passes"] == 1, "backend reports sparse and gate activity")
    print("\nall Sol-Attn backend tests passed")


if __name__ == "__main__":
    with torch.no_grad():
        main()
