from __future__ import annotations

import torch

from .offload import CacheSpec
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
    if step < 0:
        return True
    if step < int(config.first_dense_steps):
        return True
    if total and step >= max(0, total - int(config.last_dense_steps)):
        return True
    return step % int(config.refresh_every) == 0


def _should_build_cache(config, snapshot):
    step = int(getattr(snapshot, "step_index", -1))
    total = int(getattr(snapshot, "total_steps", 0))
    if step < 0:
        return False
    if step < max(0, int(config.first_dense_steps) - 1):
        return False
    if total and step >= max(0, total - int(config.last_dense_steps)):
        return False
    return True


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


def _cache_spec(config, layer_index: int, rows: int, hidden: int, ffn: int):
    selected = int(config.selected_features_for_layer(layer_index, ffn))
    selector_rows = (
        int(rows) + int(config.token_group_rows) - 1
    ) // int(config.token_group_rows)
    return CacheSpec(
        rows=int(rows),
        selected_features=selected,
        selector_rows=selector_rows,
        selected_groups=selected // int(config.feature_group),
        hidden=int(hidden),
        ffn=int(ffn),
    )


def _balanced_selector(
    full_activation,
    previous_summary,
    config,
    layer_index: int,
):
    act = full_activation.float()
    fg = int(config.feature_group)
    if act.shape[-1] % fg:
        raise ChipmunkExecutionError("H3 FFN is not feature-group aligned")
    delta = act if previous_summary is None else act - previous_summary.float()
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
    fraction = float(config.fraction_for_layer(layer_index))
    first, _ = select_top_groups(
        scores[:, :half], fraction, config.random_groups
    )
    second, _ = select_top_groups(
        scores[:, half:], fraction, config.random_groups
    )
    return torch.cat((first, second + half), dim=1).contiguous()


def _selected_activation_all_groups(
    h,
    indices,
    config,
    selected_activation_runner,
):
    tg = int(config.token_group_rows)
    fg = int(config.feature_group)
    pieces = []
    for gi, a in enumerate(range(0, h.shape[0], tg)):
        b = min(h.shape[0], a + tg)
        logical = _logical_indices(indices[gi], fg)
        pieces.append(selected_activation_runner(h[a:b], logical))
    return torch.cat(pieces, dim=0)


def _refresh_cache(
    *,
    h,
    out,
    cache,
    key,
    spec,
    session,
    config,
    layer_index,
    full_activation_runner,
    selected_activation_runner,
):
    previous = None
    lease = None
    if cache.valid:
        lease = session.offload.load(
            cache,
            key,
            spec,
            ("selector_summary",),
            h.device,
        )
        if lease is None:
            return False
        previous = lease.selector_summary

    try:
        full_activation = full_activation_runner(
            _token_means(h, config.token_group_rows)
        )
        indices = _balanced_selector(
            full_activation,
            previous,
            config,
            layer_index,
        )
        selected = _selected_activation_all_groups(
            h,
            indices,
            config,
            selected_activation_runner,
        )
        stored = session.offload.store(
            cache,
            key,
            spec,
            activation=selected,
            output=out,
            selector_summary=full_activation.to(torch.bfloat16),
            selected_groups=indices,
            lease=lease,
        )
        if not stored:
            cache.clear()
            return False
        return True
    except Exception:
        session.offload.release_lease(lease)
        raise


def _delta_update(
    *,
    h,
    cache,
    key,
    spec,
    session,
    config,
    selected_activation_runner,
    selected_fc2_runner,
):
    lease = session.offload.load(
        cache,
        key,
        spec,
        ("activation", "output", "selected_groups"),
        h.device,
    )
    if lease is None:
        return None

    try:
        indices = lease.selected_groups
        old_activation = lease.activation
        out = lease.output
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

        # The staging slot is released as soon as its D2H write is queued. The
        # block residual gate executes afterwards on the main stream, so return a
        # normal CUDA tensor rather than a slot view that could be recycled by H2D.
        result = out.clone()
        stored = session.offload.store(
            cache,
            key,
            spec,
            activation=current_activation,
            output=out,
            lease=lease,
        )
        if not stored:
            cache.clear()
        return result
    except Exception:
        session.offload.release_lease(lease)
        raise


def _entry_context(
    *,
    block,
    layer_index,
    chunk_index,
    chunk_start,
    chunk_stop,
    snapshot,
    session,
    config,
):
    session.ensure_request(snapshot)
    branch = tuple(getattr(snapshot, "branch", (0,)))
    kind = _segment_kind(snapshot, chunk_start, chunk_stop)
    eligible = config.layer_eligible(layer_index) and _kind_eligible(config, kind)
    hidden = int(block.mlp.fc2.out_features)
    ffn = int(block.mlp.fc2.in_features)
    spec = _cache_spec(
        config,
        layer_index,
        int(chunk_stop) - int(chunk_start),
        hidden,
        ffn,
    )
    cache = session.cache(branch, layer_index, chunk_index)
    key = session.host_key(branch, layer_index, chunk_index, spec)
    return branch, kind, eligible, cache, key, spec, hidden, ffn


def prefetch_chipmunk_chunk(
    *,
    block,
    layer_index: int,
    chunk_index: int,
    chunk_start: int,
    chunk_stop: int,
    snapshot,
    session,
    config,
    device,
):
    if config.mode != "reference_delta":
        return False
    (
        _branch,
        _kind,
        eligible,
        cache,
        key,
        spec,
        _hidden,
        _ffn,
    ) = _entry_context(
        block=block,
        layer_index=layer_index,
        chunk_index=chunk_index,
        chunk_start=chunk_start,
        chunk_stop=chunk_stop,
        snapshot=snapshot,
        session=session,
        config=config,
    )
    if not eligible:
        return False

    session.offload.request_host(key, spec)
    if not cache.valid:
        return False

    fields = (
        ("selector_summary",)
        if _must_dense(config, snapshot, layer_index)
        else ("activation", "output", "selected_groups")
    )
    return session.offload.prefetch(cache, key, spec, fields, device)


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

    All MLP math runs on CUDA. Persistent state is pinned host backing accessed
    only through non-blocking DMA and CUDA event dependencies.
    """
    if config.mode == "measure":
        return dense_runner(h), "chipmunk_measure_dense"

    (
        _branch,
        kind,
        eligible,
        cache,
        key,
        spec,
        _hidden,
        ffn,
    ) = _entry_context(
        block=block,
        layer_index=layer_index,
        chunk_index=chunk_index,
        chunk_start=chunk_start,
        chunk_stop=chunk_stop,
        snapshot=snapshot,
        session=session,
        config=config,
    )
    step = int(getattr(snapshot, "step_index", -1))

    if eligible and (
        full_activation_runner is None
        or selected_activation_runner is None
        or selected_fc2_runner is None
    ):
        raise ChipmunkExecutionError(
            "reference_delta requires held ConvRot fc1/fc2 runners"
        )

    if not eligible:
        return dense_runner(h), "chipmunk_dense_ineligible"

    session.offload.request_host(key, spec)

    dense_required = _must_dense(config, snapshot, layer_index) or not cache.valid
    if dense_required:
        out = dense_runner(h)
        cache.dense_calls += 1
        if _should_build_cache(config, snapshot):
            try:
                stored = _refresh_cache(
                    h=h,
                    out=out,
                    cache=cache,
                    key=key,
                    spec=spec,
                    session=session,
                    config=config,
                    layer_index=layer_index,
                    full_activation_runner=full_activation_runner,
                    selected_activation_runner=selected_activation_runner,
                )
                if stored:
                    cache.refresh_step = step
                    cache.update_step = step
                    session.record(
                        step=step,
                        layer=layer_index,
                        chunk=chunk_index,
                        kind=kind,
                        path="dense_refresh_async",
                        active_fraction=float(spec.selected_features / ffn),
                    )
                else:
                    session.record(
                        step=step,
                        layer=layer_index,
                        chunk=chunk_index,
                        kind=kind,
                        path="dense_cache_not_ready",
                    )
            except Exception as exc:
                session.offload.release_prefetch(cache)
                cache.clear()
                cache.fallback_calls += 1
                if config.strict:
                    raise
                session.record(
                    step=step,
                    layer=layer_index,
                    chunk=chunk_index,
                    kind=kind,
                    path="dense_refresh_fallback",
                    reason=f"{type(exc).__name__}: {exc}",
                )
        return out, "chipmunk_dense"

    try:
        out = _delta_update(
            h=h,
            cache=cache,
            key=key,
            spec=spec,
            session=session,
            config=config,
            selected_activation_runner=selected_activation_runner,
            selected_fc2_runner=selected_fc2_runner,
        )
        if out is None:
            cache.clear()
            dense = dense_runner(h)
            stored = _refresh_cache(
                h=h,
                out=dense,
                cache=cache,
                key=key,
                spec=spec,
                session=session,
                config=config,
                layer_index=layer_index,
                full_activation_runner=full_activation_runner,
                selected_activation_runner=selected_activation_runner,
            )
            if stored:
                cache.refresh_step = step
                cache.update_step = step
            session.record(
                step=step,
                layer=layer_index,
                chunk=chunk_index,
                kind=kind,
                path="dense_dma_miss",
            )
            return dense, "chipmunk_dense_dma_miss"

        cache.sparse_calls += 1
        cache.update_step = step
        session.record(
            step=step,
            layer=layer_index,
            chunk=chunk_index,
            kind=kind,
            path="sparse_delta_async",
            active_fraction=float(spec.selected_features / ffn),
        )
        return out, "chipmunk_sparse_delta"
    except Exception as exc:
        session.offload.release_prefetch(cache)
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
