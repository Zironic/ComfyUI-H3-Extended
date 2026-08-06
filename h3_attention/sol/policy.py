"""MiniMax H3 sparse-routing policy derived from Sol-Engine's released line."""

from __future__ import annotations

try:
    from ...h3_runtime.layout import sink_fraction, sink_range
except ImportError:
    from h3_runtime.layout import sink_fraction, sink_range


def decline_reason(config, snapshot, layer_index: int, q) -> str | None:
    if snapshot is None:
        return "no runtime context"
    if not snapshot.valid_layout:
        return "packed layout unavailable%s" % (
            "" if snapshot.error is None else ": %s" % snapshot.error
        )
    if snapshot.step_index < 0:
        return "sampler step index unavailable"
    if q.ndim != 4:
        return "expected HND rank-4 Q/K/V"
    if q.shape[0] != 1:
        return "Sol H3 path requires attention batch 1"
    import torch
    if q.dtype != torch.bfloat16:
        return "Sol kernel requires bfloat16, got %s" % q.dtype
    if q.shape[-1] != 128:
        return "Sol kernel requires head_dim 128, got %d" % q.shape[-1]
    if int(q.shape[2]) != int(snapshot.layout.seq_len):
        return "layout length %d != attention rows %d" % (
            snapshot.layout.seq_len,
            q.shape[2],
        )
    if snapshot.step_index < int(config.dense_steps):
        return "warmup_step"
    if int(layer_index) < int(config.dense_layers):
        return "dense_layer"
    fraction = sink_fraction(snapshot.layout, config.sink_mode)
    if fraction > float(config.max_sink_fraction):
        return "sink fraction %.3f exceeds configured maximum %.3f" % (
            fraction,
            config.max_sink_fraction,
        )
    return None


def exact_sink(config, snapshot) -> tuple[int, int]:
    if snapshot is None or not snapshot.valid_layout:
        return 0, 0
    return sink_range(snapshot.layout, config.sink_mode)
