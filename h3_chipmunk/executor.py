from __future__ import annotations

import math
import torch

import comfy.quant_ops

from ..h3_activation_memory.linear import acquire_linear, _convrot_parts
from .selector import logical_swiglu, select_top_groups, energy_capture_by_fraction


class ChipmunkExecutionError(RuntimeError):
    pass


def _segment_kind(snapshot, start: int, stop: int):
    layout = getattr(snapshot, "layout", None)
    segments = getattr(layout, "segments", ()) if layout is not None else ()
    for a, b, kind in segments:
        if int(a) <= start and stop <= int(b):
            return str(kind)
    return None


def _kind_eligible(config, kind):
    return (
        (config.scope == "all_dynamic" and kind in ("audio", "video"))
        or (config.scope == "target_video" and kind == "video")
    )


def _measure_layer_enabled(config, layer_index: int):
    if not (int(config.layer_start) <= layer_index < int(config.layer_stop)):
        return False
    stride = int(config.measure_layer_stride)
    return (
        layer_index == int(config.layer_stop) - 1
        or (layer_index - int(config.layer_start)) % stride == 0
    )


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


def _shadow_must_refresh(config, snapshot, cache):
    step = int(getattr(snapshot, "step_index", -1))
    total = int(getattr(snapshot, "total_steps", 0))
    if cache.output is None or cache.activation is None or cache.selected_groups is None:
        return True
    if step < 0 or step < int(config.first_dense_steps):
        return True
    if total and step >= max(0, total - int(config.last_dense_steps)):
        return True
    return step % int(config.refresh_every) == 0


def _cache_on_cpu(config):
    return config.cache_location == "cpu"


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


def _token_means(h, token_group_rows: int):
    """Compute block means with one reduction for all complete token groups."""
    rows = int(h.shape[0])
    tg = int(token_group_rows)
    full_rows = rows // tg * tg
    pieces = []
    if full_rows:
        pieces.append(h[:full_rows].reshape(-1, tg, h.shape[-1]).mean(dim=1))
    if full_rows < rows:
        pieces.append(h[full_rows:].mean(dim=0, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _selector_from_activation(
    act,
    cache,
    config,
    *,
    store_selection=True,
    top_fraction=None,
):
    act = act.float()
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    previous = _load_tensor(cache.selector_summary, act.device)
    had_previous = previous is not None
    delta = act if previous is None else act - previous.float()
    scores = delta.reshape(delta.shape[0], delta.shape[1] // fg, fg).square().mean(dim=-1).sqrt()
    fraction = float(config.top_fraction if top_fraction is None else top_fraction)
    indices, counts = select_top_groups(scores, fraction, config.random_groups)

    if config.mode in ("measure", "shadow_validate"):
        # Diagnostic summaries stay on GPU. In shadow_validate the selected row
        # window is small enough that its recurrent state can also remain on GPU,
        # avoiding the synchronous CPU-cache path entirely.
        cache.selector_summary = act.to(torch.bfloat16).detach()
        if config.mode == "shadow_validate" and store_selection:
            cache.selected_groups = indices.detach().clone()
            cache.selected_counts = counts.detach().clone()
        else:
            cache.selected_groups = None
            cache.selected_counts = None
    else:
        cache.selector_summary = _store_tensor(act.to(torch.bfloat16), config)
        if store_selection:
            cache.selected_groups = _store_tensor(indices, config)
            cache.selected_counts = _store_tensor(counts, config)
    return indices, counts, scores, had_previous


def _selector(fc1, h, cache, config, *, store_selection=True):
    act = logical_swiglu(fc1.full(_token_means(h, config.token_group_rows)))
    return _selector_from_activation(act, cache, config, store_selection=store_selection)


def _selected_activation_all_groups(fc1, h, indices, counts, config):
    indices = _load_tensor(indices, h.device)
    counts = _load_tensor(counts, h.device)
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    count = int(indices.shape[1])
    pieces = []
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], count, fg)
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
    count = int(indices.shape[1])
    with ConvRotFC2(block.mlp, current_activation[:1]) as fc2:
        for gi, a in enumerate(range(0, h.shape[0], tg)):
            b = min(h.shape[0], a + tg)
            logical = _logical_indices(indices[gi], count, fg)
            old_part = fc2.selected(old_activation[a:b], logical)
            new_part = fc2.selected(current_activation[a:b], logical)
            out[a:b].sub_(old_part).add_(new_part)

    cache.activation = _store_tensor(current_activation, config)
    cache.output = _store_tensor(out, config)
    return out


def _refresh_cache(block, h, out, cache, config, snapshot):
    with ConvRotFC1(block.mlp, h[:1]) as fc1:
        _check_gpu_budget(snapshot, config, fc1.hidden, fc1.ffn)
        indices, counts, scores, _had_previous = _selector(fc1, h, cache, config)
        selected = _selected_activation_all_groups(fc1, h, indices, counts, config)
        active = float(indices.shape[1] * config.feature_group / fc1.ffn)
    cache.activation = _store_tensor(selected, config)
    cache.output = _store_tensor(out, config)
    return active, scores


def _shadow_sample_bounds(rows: int, config):
    rows = int(rows)
    sample_rows = min(rows, int(config.shadow_sample_rows))
    if sample_rows <= 0:
        return 0, 0
    alignment = min(int(config.token_group_rows), sample_rows)
    start = max(0, (rows - sample_rows) // 2)
    start = start // alignment * alignment
    stop = min(rows, start + sample_rows)
    return int(start), int(stop)


def _relative_l2(error, reference):
    error_norm = torch.linalg.vector_norm(error.float())
    reference_norm = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    return error_norm / reference_norm


def _cosine_similarity(a, b):
    a32 = a.float().reshape(-1)
    b32 = b.float().reshape(-1)
    denom = (
        torch.linalg.vector_norm(a32) * torch.linalg.vector_norm(b32)
    ).clamp_min(1e-12)
    return torch.dot(a32, b32) / denom


def _shadow_metrics(approx, dense, residual_base, gate):
    diff = approx.float() - dense.float()
    gate32 = gate.float()
    dense_gated = dense.float() * gate32
    approx_gated = approx.float() * gate32
    gated_diff = approx_gated - dense_gated
    dense_block = residual_base.float() + dense_gated
    approx_block = residual_base.float() + approx_gated
    return {
        "raw_relative_l2": _relative_l2(diff, dense),
        "gated_relative_l2": _relative_l2(gated_diff, dense_gated),
        "block_relative_l2": _relative_l2(approx_block - dense_block, dense_block),
        "raw_cosine": _cosine_similarity(approx, dense),
        "error_rms": diff.square().mean().sqrt(),
        "dense_rms": dense.float().square().mean().sqrt(),
        "max_abs_error": diff.abs().amax(),
    }


def _shadow_refresh(
    *, h, dense_out, cache, config, layer_index, measure_activation_runner,
):
    start, stop = _shadow_sample_bounds(h.shape[0], config)
    if stop <= start:
        raise ChipmunkExecutionError("shadow validation selected an empty row window")
    sample_h = h[start:stop]
    logical_full = measure_activation_runner(sample_h)
    mean_logical = measure_activation_runner(sample_h.mean(dim=0, keepdim=True))
    fraction = config.shadow_fraction_for_layer(layer_index)
    indices, counts, scores, had_previous = _selector_from_activation(
        mean_logical,
        cache,
        config,
        top_fraction=fraction,
    )
    logical = _logical_indices(indices[0], int(indices.shape[1]), config.feature_group)
    selected = logical_full.index_select(-1, logical).contiguous()
    cache.activation = selected.detach().clone()
    cache.output = dense_out[start:stop].detach().clone()
    return start, stop, fraction, scores, had_previous


def _shadow_delta(
    *, block, h, dense_out, residual_base, gate, cache, config,
    layer_index, measure_activation_runner,
):
    start, stop = _shadow_sample_bounds(h.shape[0], config)
    sample_h = h[start:stop]
    indices = cache.selected_groups
    if indices is None or cache.activation is None or cache.output is None:
        raise ChipmunkExecutionError("shadow delta requested without a refresh cache")
    logical = _logical_indices(indices[0], int(indices.shape[1]), config.feature_group)
    current_full = measure_activation_runner(sample_h)
    current_selected = current_full.index_select(-1, logical).contiguous()
    old_selected = cache.activation
    approx = cache.output

    # Use the same selected-column ConvRot fc2 operation as reference_delta so
    # shadow error measures the actual approximation semantics we intend to run.
    with ConvRotFC2(block.mlp, current_selected[:1]) as fc2:
        old_part = fc2.selected(old_selected, logical)
        new_part = fc2.selected(current_selected, logical)
    approx.sub_(old_part).add_(new_part)

    dense_sample = dense_out[start:stop]
    base_sample = residual_base[start:stop]
    metrics = _shadow_metrics(approx, dense_sample, base_sample, gate)
    cache.activation = current_selected.detach().clone()
    cache.output = approx.detach()
    return start, stop, config.shadow_fraction_for_layer(layer_index), metrics


def run_chipmunk_chunk(
    *, block, h, layer_index: int, chunk_index: int, chunk_start: int,
    chunk_stop: int, snapshot, session, config, dense_runner,
    measure_activation_runner=None, residual_base=None, mlp_gate=None,
):
    """Return raw MLP output before H3's current-step residual gate."""
    session.ensure_request(snapshot)
    branch = tuple(getattr(snapshot, "branch", (0,)))
    kind = _segment_kind(snapshot, chunk_start, chunk_stop)
    step = int(getattr(snapshot, "step_index", -1))

    # Measurement is intentionally a lightweight observer. It always returns
    # the exact dense MLP result, observes only a representative layer subset,
    # keeps only token-mean feature summaries on GPU, and never computes or
    # transfers per-token selected activations.
    if config.mode == "measure":
        out = dense_runner(h)
        if not (
            _kind_eligible(config, kind)
            and _measure_layer_enabled(config, layer_index)
        ):
            return out, "chipmunk_measure_unobserved"
        if measure_activation_runner is None:
            raise ChipmunkExecutionError("measure mode requires held-fc1 activation runner")
        cache = session.cache(branch, layer_index, chunk_index)
        cache.dense_calls += 1
        try:
            means = _token_means(h, config.token_group_rows)
            logical_activation = measure_activation_runner(means)
            indices, _counts, scores, had_previous = _selector_from_activation(
                logical_activation,
                cache,
                config,
                store_selection=False,
            )
            ffn = int(logical_activation.shape[-1])
            active = float(indices.shape[1] * config.feature_group / ffn)
            captures = energy_capture_by_fraction(scores)
            cache.activation = None
            cache.output = None
            session.record(
                step=step,
                layer=layer_index,
                chunk=chunk_index,
                kind=kind,
                path="measure_selector",
                cross_step=bool(had_previous),
                active_fraction=active,
                selector_mean=scores.mean(),
                selector_rms=scores.square().mean().sqrt(),
                energy_capture=captures,
                effective_chunk_rows=int(config.effective_chunk_rows),
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

    # Shadow validation computes the exact dense result first and always returns
    # it. Only sampled video rows/layers receive a parallel recurrent delta path.
    if config.mode == "shadow_validate":
        out = dense_runner(h)
        if not (
            _kind_eligible(config, kind)
            and config.shadow_layer_enabled(layer_index)
        ):
            return out, "chipmunk_shadow_unobserved"
        if measure_activation_runner is None or residual_base is None or mlp_gate is None:
            raise ChipmunkExecutionError(
                "shadow_validate requires held-fc1 runner, residual base, and current MLP gate"
            )
        cache = session.cache(branch, layer_index, chunk_index)
        cache.dense_calls += 1
        try:
            if _shadow_must_refresh(config, snapshot, cache):
                start, stop, fraction, scores, had_previous = _shadow_refresh(
                    h=h,
                    dense_out=out,
                    cache=cache,
                    config=config,
                    layer_index=layer_index,
                    measure_activation_runner=measure_activation_runner,
                )
                cache.refresh_step = step
                cache.update_step = step
                session.record(
                    step=step,
                    layer=layer_index,
                    chunk=chunk_index,
                    kind=kind,
                    path="shadow_refresh",
                    cross_step=bool(had_previous),
                    active_fraction=float(fraction),
                    sample_start=int(start),
                    sample_rows=int(stop - start),
                    selector_mean=scores.mean(),
                    selector_rms=scores.square().mean().sqrt(),
                )
            else:
                start, stop, fraction, metrics = _shadow_delta(
                    block=block,
                    h=h,
                    dense_out=out,
                    residual_base=residual_base,
                    gate=mlp_gate,
                    cache=cache,
                    config=config,
                    layer_index=layer_index,
                    measure_activation_runner=measure_activation_runner,
                )
                cache.sparse_calls += 1
                cache.update_step = step
                session.record(
                    step=step,
                    layer=layer_index,
                    chunk=chunk_index,
                    kind=kind,
                    path="shadow_delta",
                    active_fraction=float(fraction),
                    refresh_age=int(step - cache.refresh_step),
                    sample_start=int(start),
                    sample_rows=int(stop - start),
                    **metrics,
                )
        except Exception as exc:
            cache.clear()
            cache.fallback_calls += 1
            session.record(
                step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                path="shadow_fallback", reason=f"{type(exc).__name__}: {exc}",
            )
            if config.strict:
                raise
        return out, "chipmunk_shadow_validate"

    cache = session.cache(branch, layer_index, chunk_index)
    eligible = (
        max(config.layer_start, config.first_dense_layers) <= layer_index < config.layer_stop
        and _kind_eligible(config, kind)
    )

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
                    selector_mean=scores.mean(),
                    cache_location=config.cache_location,
                )
            except Exception as exc:
                cache.clear()
                cache.fallback_calls += 1
                session.record(
                    step=step, layer=layer_index, chunk=chunk_index, kind=kind,
                    path="dense_refresh_no_cache", reason=f"{type(exc).__name__}: {exc}",
                )
                if config.strict:
                    raise
        return out, "chipmunk_dense"

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
