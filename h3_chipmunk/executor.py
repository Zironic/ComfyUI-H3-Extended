from __future__ import annotations

import torch

import comfy.quant_ops

from ..h3_activation_memory.linear import acquire_linear, _convrot_parts
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
    if step < 0 or layer_index < int(config.first_dense_layers):
        return True
    if step < int(config.first_dense_steps):
        return True
    if total and step >= max(0, total - int(config.last_dense_steps)):
        return True
    return step % int(config.refresh_every) == 0


class ConvRotGroupWeights:
    """Acquire H3 ConvRot INT8 MLP carriers for one block invocation."""

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
        scale = (
            self.fc1_s
            if self.fc1_s.numel() == 1
            else self.fc1_s.index_select(0, rows).contiguous()
        )
        expanded = comfy.quant_ops.ck.int8_linear(
            x, q, scale, None, x.dtype,
            convrot=True, convrot_groupsize=256,
        )
        return logical_swiglu(expanded)

    def selected_fc2(self, activation, logical_indices):
        q = self.fc2_q.index_select(1, logical_indices.long()).contiguous()
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


def _selector(weights, h, cache, config):
    tg = int(config.token_group_rows)
    means = torch.stack([
        h[a:min(h.shape[0], a + tg)].mean(dim=0)
        for a in range(0, h.shape[0], tg)
    ], dim=0)
    act = logical_swiglu(weights.full_fc1(means))
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    grouped = act.float().reshape(act.shape[0], act.shape[1] // fg, fg).mean(dim=-1)
    scores = grouped.abs() if cache.selector_summary is None else (
        grouped - cache.selector_summary.float()
    ).abs()
    indices, counts = select_top_groups(scores, config.top_fraction, config.random_groups)
    cache.selector_summary = grouped.to(torch.bfloat16)
    cache.selected_groups = indices
    cache.selected_counts = counts
    return indices, counts, scores


def _selected_activation_all_groups(weights, h, indices, counts, config):
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
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], int(counts[gi].item()), fg)
        current = weights.selected_activation(h[a:b], logical)
        old = cache.activation[a:b]
        old_part = weights.selected_fc2(old, logical)
        new_part = weights.selected_fc2(current, logical)
        out[a:b].sub_(old_part).add_(new_part)
        cache.activation[a:b].copy_(current)
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
                    active = float(indices.shape[1] * config.feature_group / weights.ffn)
                session.record(
                    step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                    path="dense_refresh", active_fraction=active,
                    selector_mean=float(scores.mean().item()),
                )
            except Exception as exc:
                cache.clear()
                cache.fallback_calls += 1
                session.record(
                    step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                    path="dense_refresh_no_cache", reason=f"{type(exc).__name__}: {exc}",
                )
                if config.strict and config.mode == "reference_delta":
                    raise
        return out, "chipmunk_dense"

    if config.mode == "measure":
        out = dense_runner(h)
        cache.dense_calls += 1
        try:
            with ConvRotGroupWeights(block.mlp, h[:1]) as weights:
                previous_indices = None if cache.selected_groups is None else cache.selected_groups.clone()
                previous_counts = None if cache.selected_counts is None else cache.selected_counts.clone()
                previous_activation = cache.activation
                delta_rms = float("nan")
                if previous_indices is not None and previous_activation is not None:
                    current_previous = _selected_activation_all_groups(
                        weights, h, previous_indices, previous_counts, config
                    )
                    if current_previous.shape == previous_activation.shape:
                        delta_rms = float(
                            (current_previous.float() - previous_activation.float())
                            .square().mean().sqrt().item()
                        )
                indices, counts, scores = _selector(weights, h, cache, config)
                cache.activation = _selected_activation_all_groups(
                    weights, h, indices, counts, config
                ).detach().clone()
                cache.output = out.detach().clone()
                active = float(indices.shape[1] * config.feature_group / weights.ffn)
            session.record(
                step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                path="measure", active_fraction=active,
                selector_mean=float(scores.mean().item()), selected_delta_rms=delta_rms,
            )
        except Exception as exc:
            cache.fallback_calls += 1
            session.record(
                step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                path="measure_fallback", reason=f"{type(exc).__name__}: {exc}",
            )
            if config.strict:
                raise
        return out, "chipmunk_measure"

    try:
        with ConvRotGroupWeights(block.mlp, h[:1]) as weights:
            out = _delta_update(weights, h, cache, config)
            active = float(
                cache.selected_groups.shape[1] * config.feature_group / weights.ffn
            )
        cache.sparse_calls += 1
        cache.update_step = step
        session.record(
            step=step, layer=layer_index, chunk=chunk_index, kind=kind,
            path="sparse_delta", active_fraction=active,
        )
        return out, "chipmunk_sparse_delta"
    except Exception as exc:
        cache.clear()
        cache.fallback_calls += 1
        if config.strict:
            raise
        out = dense_runner(h)
        session.record(
            step=step, layer=layer_index, chunk=chunk_index, kind=kind,
            path="dense_fallback", reason=f"{type(exc).__name__}: {exc}",
        )
        return out, "chipmunk_dense_fallback"
