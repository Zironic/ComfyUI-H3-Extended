from __future__ import annotations

import math
import torch

from .selector import select_top_groups


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


def _store_gpu(value):
    value = value.detach()
    if value.device.type != "cuda":
        raise ChipmunkExecutionError(
            "production Chipmunk cache must remain on CUDA; refusing host-backed state"
        )
    return value.clone()


def _load_gpu(value, device):
    if value is None:
        return None
    if value.device != device or value.device.type != "cuda":
        raise ChipmunkExecutionError(
            "production Chipmunk state moved off the active CUDA device"
        )
    return value


def _dynamic_rows(snapshot, scope):
    layout = getattr(snapshot, "layout", None)
    segments = getattr(layout, "segments", ()) if layout is not None else ()
    kinds = {"video"} if scope == "target_video" else {"audio", "video"}
    return sum(int(b) - int(a) for a, b, kind in segments if str(kind) in kinds)


def _selected_feature_count(config, ffn: int):
    """Static selected width for the balanced two-tile selector."""
    fg = int(config.feature_group)
    groups = int(ffn) // fg
    if groups % 2:
        raise ChipmunkExecutionError("H3 FFN group count must split into two ConvRot tiles")
    half_groups = groups // 2
    keep = max(
        1,
        min(
            half_groups,
            int(math.ceil(half_groups * float(config.top_fraction))),
        ),
    )
    if float(config.random_groups) > 0.0 and keep < half_groups:
        keep += min(
            half_groups - keep,
            max(1, int(math.ceil(half_groups * float(config.random_groups)))),
        )
    return int(2 * keep * fg)


def estimated_cache_bytes(snapshot, config, hidden: int, ffn: int):
    rows = _dynamic_rows(snapshot, config.scope)
    layer_start = max(int(config.layer_start), int(config.first_dense_layers))
    layers = max(0, int(config.layer_stop) - layer_start)
    selected = _selected_feature_count(config, ffn)
    return int(rows) * int(layers) * (int(hidden) + int(selected)) * 2


def _check_gpu_budget(snapshot, config, hidden, ffn):
    if config.mode != "reference_delta":
        return
    if config.cache_location != "gpu":
        raise ChipmunkExecutionError(
            "reference_delta requires cache_location=gpu; synchronous CPU cache is disabled"
        )
    need = estimated_cache_bytes(snapshot, config, hidden, ffn)
    limit = int(float(config.cache_budget_gb) * (1024 ** 3))
    if need > limit:
        raise ChipmunkExecutionError(
            "estimated GPU Chipmunk cache %.2f GiB exceeds cache_budget_gb %.2f; "
            "reduce layer range/top_fraction or raise the budget"
            % (need / (1024 ** 3), float(config.cache_budget_gb))
        )


def _logical_indices(groups: torch.Tensor, feature_group: int):
    groups = groups.long()
    offsets = torch.arange(feature_group, device=groups.device, dtype=torch.long)
    return (groups[:, None] * feature_group + offsets[None]).reshape(-1)


def _token_means(h, token_group_rows: int):
    rows = int(h.shape[0])
    tg = int(token_group_rows)
    full_rows = rows // tg * tg
    pieces = []
    if full_rows:
        pieces.append(h[:full_rows].reshape(-1, tg, h.shape[-1]).mean(dim=1))
    if full_rows < rows:
        pieces.append(h[full_rows:].mean(dim=0, keepdim=True))
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)


def _balanced_selector(full_activation, cache, config):
    """Select a fixed number of groups from each prepacked ConvRot half.

    Equal per-half counts make the selected fc1/fc2 tensors rectangular without
    host reads or CUDA-value-dependent Python branches. At top_fraction=0.30,
    H3's 56 groups become 9+9 selected groups (32.14% actual density).
    """
    act = full_activation.float()
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    previous = _load_gpu(cache.selector_summary, act.device)
    delta = act if previous is None else act - previous.float()
    scores = (
        delta.reshape(delta.shape[0], delta.shape[1] // fg, fg)
        .square()
        .mean(dim=-1)
        .sqrt()
    )
    groups = int(scores.shape[1])
    if groups % 2:
        raise ChipmunkExecutionError("selector group count must split into two tiles")
    half = groups // 2
    first, _ = select_top_groups(
        scores[:, :half],
        config.top_fraction,
        config.random_groups,
    )
    second, _ = select_top_groups(
        scores[:, half:],
        config.top_fraction,
        config.random_groups,
    )
    second = second + half
    indices = torch.cat((first, second), dim=1).contiguous()
    cache.selector_summary = _store_gpu(act.to(torch.bfloat16))
    cache.selected_groups = _store_gpu(indices)
    cache.selected_counts = None
    return indices


def _selected_activation_all_groups(
    h,
    indices,
    config,
    selected_activation_runner,
):
    indices = _load_gpu(indices, h.device)
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    pieces = []
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], fg)
        pieces.append(selected_activation_runner(h[a:b], logical))
    return torch.cat(pieces, dim=0)


def _delta_update(
    h,
    cache,
    config,
    selected_activation_runner,
    selected_fc2_runner,
):
    if (
        cache.selected_groups is None
        or cache.activation is None
        or cache.output is None
    ):
        raise ChipmunkExecutionError("sparse update requested without a dense cache")

    indices = _load_gpu(cache.selected_groups, h.device)
    old_activation = _load_gpu(cache.activation, h.device)
    out = _load_gpu(cache.output, h.device)
    current_activation = _selected_activation_all_groups(
        h,
        indices,
        config,
        selected_activation_runner,
    )

    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], fg)
        old_part = selected_fc2_runner(old_activation[a:b], logical)
        new_part = selected_fc2_runner(current_activation[a:b], logical)
        out[a:b].sub_(old_part).add_(new_part)

    cache.activation = _store_gpu(current_activation)
    cache.output = out.detach()
    return out


def _refresh_cache(
    h,
    out,
    cache,
    config,
    snapshot,
    hidden,
    ffn,
    full_activation_runner,
    selected_activation_runner,
):
    _check_gpu_budget(snapshot, config, hidden, ffn)
    means = _token_means(h, config.token_group_rows)
    full_activation = full_activation_runner(means)
    indices = _balanced_selector(full_activation, cache, config)
    selected = _selected_activation_all_groups(
        h,
        indices,
        config,
        selected_activation_runner,
    )
    cache.activation = _store_gpu(selected)
    cache.output = _store_gpu(out)
    return float(selected.shape[-1] / int(ffn))


def run_chipmunk_chunk(
    *,
    block,
    h,
    layer_index: int,
    chunk_index: int,
    chunk_start: int,
    chunk_stop: int,
    snapshot,
    session,
    config,
    dense_runner,
    full_activation_runner=None,
    selected_activation_runner=None,
    selected_fc2_runner=None,
    **_unused,
):
    """Return raw MLP output before H3's current-step residual gate.

    Production invariant: this function never materializes CUDA data on the CPU
    and never reacquires MLP weights for the sparse path.
    """
    session.ensure_request(snapshot)
    branch = tuple(getattr(snapshot, "branch", (0,)))
    kind = _segment_kind(snapshot, chunk_start, chunk_stop)
    step = int(getattr(snapshot, "step_index", -1))

    if config.mode == "measure":
        return dense_runner(h), "chipmunk_measure_dense"

    cache = session.cache(branch, layer_index, chunk_index)
    eligible = (
        max(int(config.layer_start), int(config.first_dense_layers))
        <= layer_index
        < int(config.layer_stop)
        and _kind_eligible(config, kind)
    )

    if eligible and (
        full_activation_runner is None
        or selected_activation_runner is None
        or selected_fc2_runner is None
    ):
        raise ChipmunkExecutionError(
            "reference_delta requires held ConvRot fc1/fc2 runners"
        )

    hidden = int(block.mlp.fc2.out_features)
    ffn = int(block.mlp.fc2.in_features)

    if not eligible or _must_dense(config, snapshot, layer_index):
        out = dense_runner(h)
        cache.dense_calls += 1
        if eligible:
            try:
                active = _refresh_cache(
                    h,
                    out,
                    cache,
                    config,
                    snapshot,
                    hidden,
                    ffn,
                    full_activation_runner,
                    selected_activation_runner,
                )
                cache.refresh_step = step
                cache.update_step = step
                session.record(
                    step=step,
                    layer=layer_index,
                    chunk=chunk_index,
                    kind=kind,
                    path="dense_refresh",
                    active_fraction=active,
                )
            except Exception as exc:
                cache.clear()
                cache.fallback_calls += 1
                session.record(
                    step=step,
                    layer=layer_index,
                    chunk=chunk_index,
                    kind=kind,
                    path="dense_refresh_no_cache",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if config.strict:
                    raise
        return out, "chipmunk_dense"

    try:
        out = _delta_update(
            h,
            cache,
            config,
            selected_activation_runner,
            selected_fc2_runner,
        )
        active = float(cache.activation.shape[-1] / ffn)
        cache.sparse_calls += 1
        cache.update_step = step
        session.record(
            step=step,
            layer=layer_index,
            chunk=chunk_index,
            kind=kind,
            path="sparse_delta",
            active_fraction=active,
        )
        return out, "chipmunk_sparse_delta"
    except Exception as exc:
        cache.clear()
        cache.fallback_calls += 1
        if config.strict:
            raise
        out = dense_runner(h)
        session.record(
            step=step,
            layer=layer_index,
            chunk=chunk_index,
            kind=kind,
            path="dense_fallback",
            reason=f"{type(exc).__name__}: {exc}",
        )
        return out, "chipmunk_dense_fallback"
