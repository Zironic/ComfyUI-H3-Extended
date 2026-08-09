from __future__ import annotations

import math
import torch

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout

from ..h3_activation_memory.linear import (
    ConvRotTwoSliceMLP,
    acquire_linear,
    _convrot_parts,
)
from .selector import logical_swiglu, select_top_groups


class ChipmunkExecutionError(RuntimeError):
    pass


def _segment_kind(snapshot, start: int, stop: int):
    layout = getattr(snapshot, "layout", None)
    segments = getattr(layout, "segments", ()) if layout is not None else ()
    for a, b, kind in segments:
        if int(a) <= start and stop <= int(b):
            return str(kind)
    return None


def _must_dense(config, snapshot, layer_index: int):
    step = int(getattr(snapshot, "step_index", -1))
    total = int(getattr(snapshot, "total_steps", 0))
    if step < 0:
        return True
    if layer_index < int(config.first_dense_layers):
        return True
    if step < int(config.first_dense_steps):
        return True
    if total and step >= max(0, total - int(config.last_dense_steps)):
        return True
    return step % int(config.refresh_every) == 0


class ConvRotGroupWeights:
    """Acquire one H3 MLP's ConvRot carriers for a block invocation."""

    def __init__(self, mlp, sample):
        self.mlp = mlp
        self.sample = sample
        self.fc1 = None
        self.fc2 = None
        self.fc1_q = self.fc1_s = self.fc2_q = self.fc2_s = None

    def __enter__(self):
        self.fc1 = acquire_linear(self.mlp.fc1, self.sample)
        self.fc2 = acquire_linear(self.mlp.fc2, self.sample)
        if self.fc1.bias is not None or self.fc2.bias is not None:
            raise ChipmunkExecutionError("H3 Chipmunk requires bias-free fc1/fc2")
        self.fc1_q, self.fc1_s = _convrot_parts(self.fc1.weight, "chipmunk.fc1")
        self.fc2_q, self.fc2_s = _convrot_parts(self.fc2.weight, "chipmunk.fc2")
        if self.fc1_q.shape[0] != 2 * self.fc2_q.shape[1]:
            raise ChipmunkExecutionError("fc1/fc2 are not a SwiGLU pair")
        return self

    @property
    def ffn(self):
        return int(self.fc2_q.shape[1])

    def full_fc1(self, x):
        return comfy.quant_ops.ck.int8_linear(
            x, self.fc1_q, self.fc1_s, None, x.dtype,
            convrot=True, convrot_groupsize=256,
        )

    def selected_activation(self, x, logical_indices):
        logical_indices = logical_indices.long()
        rows = torch.cat((logical_indices, logical_indices + self.ffn), dim=0)
        q = self.fc1_q.index_select(0, rows).contiguous()
        scale = self.fc1_s if self.fc1_s.numel() == 1 else self.fc1_s.index_select(0, rows).contiguous()
        expanded = comfy.quant_ops.ck.int8_linear(
            x, q, scale, None, x.dtype,
            convrot=True, convrot_groupsize=256,
        )
        return logical_swiglu(expanded)

    def selected_fc2(self, activation, logical_indices):
        logical_indices = logical_indices.long()
        q = self.fc2_q.index_select(1, logical_indices).contiguous()
        return comfy.quant_ops.ck.int8_linear(
            activation, q, self.fc2_s, None, activation.dtype,
            convrot=True, convrot_groupsize=256,
        )

    def release(self):
        if self.fc2 is not None:
            self.fc2.release()
            self.fc2 = None
        if self.fc1 is not None:
            self.fc1.release()
            self.fc1 = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


def _logical_indices(groups: torch.Tensor, count: int, feature_group: int):
    groups = groups[:count].long()
    offsets = torch.arange(feature_group, device=groups.device, dtype=torch.long)
    return (groups[:, None] * feature_group + offsets[None]).reshape(-1)


def _selector(weights: ConvRotGroupWeights, h: torch.Tensor, cache, config):
    """Chipmunk-style cheap selector using fc1 of token-group means."""
    tg = int(config.token_group_rows)
    means = []
    for a in range(0, h.shape[0], tg):
        means.append(h[a:min(h.shape[0], a + tg)].mean(dim=0))
    means = torch.stack(means, dim=0)
    mid = weights.full_fc1(means)
    act = logical_swiglu(mid)
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    grouped = act.float().reshape(act.shape[0], act.shape[1] // fg, fg).mean(dim=-1)
    if cache.selector_summary is None:
        scores = grouped.abs()
    else:
        scores = (grouped - cache.selector_summary.float()).abs()
    indices, counts = select_top_groups(
        scores, config.top_fraction, config.random_groups
    )
    cache.selector_summary = grouped.to(torch.bfloat16)
    cache.selected_groups = indices
    cache.selected_counts = counts
    return indices, counts, scores


def _selected_activation_all_groups(weights, h, indices, counts, config):
    """Packed cache: list represented as dense [rows, kept_features] tensor.

    Counts are currently uniform because top-k returns a fixed keep count.
    """
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    pieces = []
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], int(counts[gi].item()), fg)
        pieces.append(weights.selected_activation(h[a:b], logical))
    return torch.cat(pieces, dim=0)


def _delta_update(weights, h, cache, config):
    indices, counts = cache.selected_groups, cache.selected_counts
    if indices is None or counts is None or cache.activation is None or cache.output is None:
        raise ChipmunkExecutionError("sparse update requested without a dense cache")
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    out = cache.output
    activation_offset = 0
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], int(counts[gi].item()), fg)
        current = weights.selected_activation(h[a:b], logical)
        old = cache.activation[a:b]
        # Two selected ConvRot fc2 calls deliberately preserve the quantized
        # contribution semantics better than quantizing a BF16 delta once.
        old_part = weights.selected_fc2(old, logical)
        new_part = weights.selected_fc2(current, logical)
        out[a:b].sub_(old_part).add_(new_part)
        cache.activation[a:b].copy_(current)
        activation_offset += current.shape[0]
    cache.output = out
    return out


def run_chipmunk_chunk(
    *, block, h, layer_index: int, chunk_index: int, chunk_start: int,
    chunk_stop: int, snapshot, session, config, dense_runner,
):
    """Return raw MLP output before H3's current-step residual gate."""
    session.ensure_request(snapshot)
    branch = tuple(getattr(snapshot, "branch", (0,)))
    cache = session.cache(branch, layer_index, chunk_index)
    kind = _segment_kind(snapshot, chunk_start, chunk_stop)
    eligible = (
        config.layer_start <= layer_index < config.layer_stop
        and (config.scope == "all_dynamic" or kind == "video")
    )
    step = int(getattr(snapshot, "step_index", -1))

    if not eligible or _must_dense(config, snapshot, layer_index):
        out = dense_runner(h)
        cache.dense_calls += 1
        if eligible:
            try:
                with ConvRotGroupWeights(block.mlp, h[:1]) as weights:
                    indices, counts, scores = _selector(weights, h, cache, config)
                    cache.activation = _selected_activation_all_groups(
                        weights, h, indices, counts, config
                    ).contiguous()
                    cache.output = out.detach().clone()
                    cache.refresh_step = step
                    cache.update_step = step
                    session.record(
                        step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                        path="dense_refresh", active_fraction=float(indices.shape[1] * config.feature_group / weights.ffn),
                        selector_mean=float(scores.mean().item()),
                    )
            except Exception as exc:
                cache.clear()
                cache.fallback_calls += 1
                session.record(step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                               path="dense_refresh_no_cache", reason=f"{type(exc).__name__}: {exc}")
                if config.strict and config.mode == "reference_delta":
                    raise
        return out, "chipmunk_dense"

    if config.mode == "measure":
        out = dense_runner(h)
        cache.dense_calls += 1
        try:
            with ConvRotGroupWeights(block.mlp, h[:1]) as weights:
                indices, counts, scores = _selector(weights, h, cache, config)
                current = _selected_activation_all_groups(weights, h, indices, counts, config)
                if cache.activation is not None and cache.activation.shape == current.shape:
                    delta_rms = float((current.float() - cache.activation.float()).square().mean().sqrt().item())
                else:
                    delta_rms = float("nan")
                cache.activation = current.detach().clone()
                cache.output = out.detach().clone()
                session.record(step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                               path="measure", active_fraction=float(indices.shape[1] * config.feature_group / weights.ffn),
                               selector_mean=float(scores.mean().item()), selected_delta_rms=delta_rms)
        except Exception as exc:
            cache.fallback_calls += 1
            session.record(step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                           path="measure_fallback", reason=f"{type(exc).__name__}: {exc}")
            if config.strict:
                raise
        return out, "chipmunk_measure"

    try:
        with ConvRotGroupWeights(block.mlp, h[:1]) as weights:
            out = _delta_update(weights, h, cache, config)
        cache.sparse_calls += 1
        cache.update_step = step
        session.record(step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                       path="sparse_delta", active_fraction=float(cache.selected_groups.shape[1] * config.feature_group / weights.ffn))
        return out, "chipmunk_sparse_delta"
    except Exception as exc:
        cache.clear()
        cache.fallback_calls += 1
        if config.strict:
            raise
        out = dense_runner(h)
        session.record(step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                       path="dense_fallback", reason=f"{type(exc).__name__}: {exc}")
        return out, "chipmunk_dense_fallback"
