from __future__ import annotations

import math
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


def _cache_on_cpu(config):
    return config.mode == "measure" or config.cache_location == "cpu"


def _store_tensor(value, config):
    value = value.detach()
    if _cache_on_cpu(config):
        return value.to("cpu", non_blocking=False)
    return value.clone()


def _load_tensor(value, device):
    if value is None:
        return None
    return value if value.device == device else value.to(device, non_blocking=False)


def _dynamic_rows(snapshot, scope):
    layout = getattr(snapshot, "layout", None)
    segments = getattr(layout, "segments", ()) if layout is not None else ()
    kinds = {"video"} if scope == "target_video" else {"audio", "video"}
    return sum(int(b) - int(a) for a, b, kind in segments if str(kind) in kinds)


def estimated_cache_bytes(snapshot, config, hidden: int, ffn: int):
    rows = _dynamic_rows(snapshot, config.scope)
    layer_start = max(int(config.layer_start), int(config.first_dense_layers))
    layers = max(0, int(config.layer_stop) - layer_start)
    groups = int(ffn) // int(config.feature_group)
    keep = max(1, min(groups, int(math.ceil(groups * float(config.top_fraction)))))
    if float(config.random_groups) > 0 and keep < groups:
        keep += min(groups - keep, max(1, int(math.ceil(groups * float(config.random_groups)))))
    selected = keep * int(config.feature_group)
    return int(rows) * int(layers) * (int(hidden) + selected) * 2


def _check_gpu_budget(snapshot, config, hidden, ffn):
    if config.mode != "reference_delta" or config.cache_location != "gpu":
        return
    need = estimated_cache_bytes(snapshot, config, hidden, ffn)
    limit = int(float(config.cache_budget_gb) * (1024 ** 3))
    if need > limit:
        raise ChipmunkExecutionError(
            "estimated GPU Chipmunk cache %.2f GiB exceeds cache_budget_gb %.2f; "
            "use cache_location=cpu, reduce layer range/top_fraction, or raise the budget"
            % (need / (1024 ** 3), float(config.cache_budget_gb))
        )


class ConvRotFC1:
    """Short-lived fc1 lease. Never overlaps an fc2 lease."""

    def __init__(self, mlp, sample):
        self.mlp = mlp
        self.sample = sample
        self.lease = None
        self.q = self.scale = None

    def __enter__(self):
        self.lease = acquire_linear(self.mlp.fc1, self.sample)
        if self.lease.bias is not None:
            raise ChipmunkExecutionError("H3 Chipmunk requires bias-free fc1")
        self.q, self.scale = _convrot_parts(self.lease.weight, "chipmunk.fc1")
        if self.q.shape[0] % 2:
            raise ChipmunkExecutionError("fc1 output must split into a SwiGLU pair")
        if self.ffn % 256 or self.hidden % 256:
            raise ChipmunkExecutionError("H3 Chipmunk requires ConvRot-256 alignment")
        return self

    @property
    def ffn(self):
        return int(self.q.shape[0]) // 2

    @property
    def hidden(self):
        return int(self.q.shape[1])

    def full(self, x):
        return comfy.quant_ops.ck.int8_linear(
            x, self.q, self.scale, None, x.dtype,
            convrot=True, convrot_groupsize=256,
        )

    def selected_activation(self, x, logical_indices):
        logical_indices = logical_indices.long()
        rows = torch.cat((logical_indices, logical_indices + self.ffn), dim=0)
        q = self.q.index_select(0, rows).contiguous()
        scale = (
            self.scale
            if self.scale.numel() == 1
            else self.scale.index_select(0, rows).contiguous()
        )
        expanded = comfy.quant_ops.ck.int8_linear(
            x, q, scale, None, x.dtype,
            convrot=True, convrot_groupsize=256,
        )
        return logical_swiglu(expanded)

    def __exit__(self, exc_type, exc, tb):
        if self.lease is not None:
            self.lease.release()
            self.lease = None
        self.q = self.scale = None
        return False


class ConvRotFC2:
    """Short-lived fc2 lease. Acquired only after fc1 has been released."""

    def __init__(self, mlp, sample):
        self.mlp = mlp
        self.sample = sample
        self.lease = None
        self.q = self.scale = None

    def __enter__(self):
        self.lease = acquire_linear(self.mlp.fc2, self.sample)
        if self.lease.bias is not None:
            raise ChipmunkExecutionError("H3 Chipmunk requires bias-free fc2")
        self.q, self.scale = _convrot_parts(self.lease.weight, "chipmunk.fc2")
        if self.ffn % 256:
            raise ChipmunkExecutionError("fc2 input must be ConvRot-256 aligned")
        return self

    @property
    def ffn(self):
        return int(self.q.shape[1])

    @property
    def hidden(self):
        return int(self.q.shape[0])

    def selected(self, activation, logical_indices):
        q = self.q.index_select(1, logical_indices.long()).contiguous()
        return comfy.quant_ops.ck.int8_linear(
            activation, q, self.scale, None, activation.dtype,
            convrot=True, convrot_groupsize=256,
        )

    def __exit__(self, exc_type, exc, tb):
        if self.lease is not None:
            self.lease.release()
            self.lease = None
        self.q = self.scale = None
        return False


def _logical_indices(groups: torch.Tensor, count: int, feature_group: int):
    groups = groups[:count].long()
    offsets = torch.arange(feature_group, device=groups.device, dtype=torch.long)
    return (groups[:, None] * feature_group + offsets[None]).reshape(-1)


def _selector(fc1, h, cache, config):
    """Select ConvRot groups from block-mean SwiGLU activation deltas."""
    tg = int(config.token_group_rows)
    means = torch.stack([
        h[a:min(h.shape[0], a + tg)].mean(dim=0)
        for a in range(0, h.shape[0], tg)
    ], dim=0)
    act = logical_swiglu(fc1.full(means)).float()
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    previous = _load_tensor(cache.selector_summary, act.device)
    delta = act if previous is None else act - previous.float()
    scores = delta.reshape(delta.shape[0], delta.shape[1] // fg, fg).square().mean(dim=-1).sqrt()
    indices, counts = select_top_groups(scores, config.top_fraction, config.random_groups)
    # This summary is touched only on refresh/measurement calls. Keep the full
    # per-feature block-mean signal on CPU so signed values cannot cancel inside
    # a 256-neuron group before the RMS delta is measured.
    cache.selector_summary = act.to(torch.bfloat16).cpu()
    cache.selected_groups = _store_tensor(indices, config)
    cache.selected_counts = _store_tensor(counts, config)
    return indices, counts, scores


def _selected_activation_all_groups(fc1, h, indices, counts, config):
    indices = _load_tensor(indices, h.device)
    counts = _load_tensor(counts, h.device)
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    pieces = []
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], int(counts[gi].item()), fg)
        pieces.append(fc1.selected_activation(h[a:b], logical))
    return torch.cat(pieces, dim=0)


def _delta_update(block, h, cache, config):
    if (
        cache.selected_groups is None or cache.selected_counts is None
        or cache.activation is None or cache.output is None
    ):
        raise ChipmunkExecutionError("sparse update requested without a dense cache")
    indices = _load_tensor(cache.selected_groups, h.device)
    counts = _load_tensor(cache.selected_counts, h.device)
    old_activation = _load_tensor(cache.activation, h.device)
    out = _load_tensor(cache.output, h.device)

    with ConvRotFC1(block.mlp, h[:1]) as fc1:
        current_activation = _selected_activation_all_groups(fc1, h, indices, counts, config)

    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    with ConvRotFC2(block.mlp, current_activation[:1]) as fc2:
        for gi, a in enumerate(range(0, h.shape[0], tg)):
            b = min(h.shape[0], a + tg)
            logical = _logical_indices(indices[gi], int(counts[gi].item()), fg)
            old_part = fc2.selected(old_activation[a:b], logical)
            new_part = fc2.selected(current_activation[a:b], logical)
            out[a:b].sub_(old_part).add_(new_part)

    cache.activation = _store_tensor(current_activation, config)
    cache.output = _store_tensor(out, config)
    return out


def _refresh_cache(block, h, out, cache, config, snapshot):
    with ConvRotFC1(block.mlp, h[:1]) as fc1:
        _check_gpu_budget(snapshot, config, fc1.hidden, fc1.ffn)
        indices, counts, scores = _selector(fc1, h, cache, config)
        selected = _selected_activation_all_groups(fc1, h, indices, counts, config)
        active = float(indices.shape[1] * config.feature_group / fc1.ffn)
    cache.activation = _store_tensor(selected, config)
    cache.output = _store_tensor(out, config) if config.mode == "reference_delta" else None
    return active, scores


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
        max(config.layer_start, config.first_dense_layers) <= layer_index < config.layer_stop
        and (
            (config.scope == "all_dynamic" and kind in ("audio", "video"))
            or (config.scope == "target_video" and kind == "video")
        )
    )
    step = int(getattr(snapshot, "step_index", -1))

    if not eligible or _must_dense(config, snapshot, layer_index):
        out = dense_runner(h)
        cache.dense_calls += 1
        if eligible:
            try:
                active, scores = _refresh_cache(block, h, out, cache, config, snapshot)
                cache.refresh_step = step
                cache.update_step = step
                session.record(
                    step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                    path="dense_refresh", active_fraction=active,
                    selector_mean=float(scores.mean().item()),
                    cache_location=config.cache_location,
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
            previous_indices = cache.selected_groups
            previous_counts = cache.selected_counts
            previous_activation = cache.activation
            delta_rms = float("nan")
            with ConvRotFC1(block.mlp, h[:1]) as fc1:
                if previous_indices is not None and previous_activation is not None:
                    current_previous = _selected_activation_all_groups(
                        fc1, h, previous_indices, previous_counts, config
                    )
                    previous_gpu = _load_tensor(previous_activation, current_previous.device)
                    if current_previous.shape == previous_gpu.shape:
                        delta_rms = float(
                            (current_previous.float() - previous_gpu.float())
                            .square().mean().sqrt().item()
                        )
                indices, counts, scores = _selector(fc1, h, cache, config)
                selected = _selected_activation_all_groups(fc1, h, indices, counts, config)
                active = float(indices.shape[1] * config.feature_group / fc1.ffn)
            cache.activation = _store_tensor(selected, config)
            cache.output = None
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
        hidden = int(block.mlp.fc2.out_features)
        ffn = int(block.mlp.fc2.in_features)
        _check_gpu_budget(snapshot, config, hidden, ffn)
        out = _delta_update(block, h, cache, config)
        active = float(cache.selected_groups.shape[1] * config.feature_group / ffn)
        cache.sparse_calls += 1
        cache.update_step = step
        session.record(
            step=step, layer=layer_index, chunk=chunk_index, kind=kind,
            path="sparse_delta", active_fraction=active,
            cache_location=config.cache_location,
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
