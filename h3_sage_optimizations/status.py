"""Compact UI summaries for resolved H3 Sage optimization plans."""

from __future__ import annotations

from .plan import (
    DENSITY_ADAPTIVE_BUDGET,
    PLAN_KEY,
    STATUS_KEY,
)


def _model_options(model):
    return getattr(model, "model_options", {}) or {}


def _status(model):
    transformer_options = (
        _model_options(model).get("transformer_options", {}) or {}
    )
    value = transformer_options.get(STATUS_KEY)
    return value if isinstance(value, dict) else None


def _plan(model):
    return _model_options(model).get(PLAN_KEY)


def _provider_text(section, *, fallback_label):
    provider = section.get("provider") or fallback_label
    reason = str(section.get("reason") or "").strip()
    return provider if not reason else "%s — %s" % (provider, reason)


def format_disabled_status(node_name):
    return (
        "%s is disabled. No new optimization request was applied; "
        "upstream model patches are left unchanged." % node_name
    )


def format_memory_status(model):
    status = _status(model)
    if status is None:
        return "Skipped: input model is not MiniMax H3."

    attention = status.get("attention", {})
    qkv = status.get("fused_qkv", {})
    mlp = status.get("mlp", {})
    compile_status = status.get("compile", {})

    lines = [
        "Attention: %s" % (attention.get("selected") or "preserve incoming"),
        "QKV: %s" % _provider_text(qkv, fallback_label="standard_h3_qkv"),
        "MLP: %s" % _provider_text(mlp, fallback_label="off"),
    ]
    chunk_rows = mlp.get("chunk_rows")
    if chunk_rows is not None and mlp.get("provider") != "off":
        lines[-1] += " (%d-row chunks)" % int(chunk_rows)
    if compile_status.get("state") not in (None, "off"):
        lines.append(
            "Shared compilation: %s — %s"
            % (
                compile_status.get("state"),
                compile_status.get("reason") or "",
            )
        )
    return "\n".join(lines)


def _adaptive_warning(sparse_status):
    if sparse_status.get("density_mode") != DENSITY_ADAPTIVE_BUDGET:
        return None
    budget = float(sparse_status.get("video_budget", 0.0))
    minimum = float(sparse_status.get("min_video_density", 0.0))
    maximum = float(sparse_status.get("max_video_density", 0.0))
    if abs(minimum - budget) < 1e-9 and abs(maximum - budget) < 1e-9:
        return (
            "Adaptive redistribution is disabled by equal minimum, budget, "
            "and maximum densities."
        )
    if abs(maximum - budget) < 1e-9:
        return (
            "Maximum density equals the target budget, so no row can receive "
            "more than the fixed-route K."
        )
    if abs(minimum - budget) < 1e-9:
        return (
            "Minimum density equals the target budget, so no row can give up "
            "blocks below the fixed-route K."
        )
    return None


def format_sparse_status(model):
    status = _status(model)
    if status is None:
        return "Skipped: input model is not MiniMax H3."

    qkv = status.get("fused_qkv", {})
    mlp = status.get("mlp", {})
    sparse_status = status.get("sparse") or {}
    compile_status = status.get("compile", {})
    budget = sparse_status.get("video_budget")
    if budget is None:
        plan = _plan(model)
        sparse_request = getattr(plan, "sparse", None)
        budget = getattr(sparse_request, "video_budget", 0.0)
    budget = float(budget)
    density_mode = sparse_status.get("density_mode") or "fixed"

    lines = [
        "Sparse Sage routing: %s; requested video KV budget: %.1f%%"
        % (density_mode, budget * 100.0),
        "QKV: %s" % _provider_text(qkv, fallback_label="standard_h3_qkv"),
    ]
    if density_mode == DENSITY_ADAPTIVE_BUDGET:
        lines.append(
            "Adaptive rails: %.1f%%–%.1f%%; temperature %.3g; target mass %.1f%%"
            % (
                float(sparse_status.get("min_video_density", 0.0)) * 100.0,
                float(sparse_status.get("max_video_density", 1.0)) * 100.0,
                float(sparse_status.get("adaptive_temperature", 1.0)),
                float(sparse_status.get("adaptive_target_mass", 0.8)) * 100.0,
            )
        )
        warning = _adaptive_warning(sparse_status)
        if warning is not None:
            lines.append("Adaptive warning: %s" % warning)
    else:
        lines.append(
            "Effective density is rounded up to a whole KV-tile count at "
            "runtime; non-video context and mixed boundary tiles remain dense."
        )

    if sparse_status.get("reporting_enabled"):
        timing = " with deferred CUDA timing" if sparse_status.get("timing") else ""
        lines.append(
            "Reports enabled%s; run tag: %s"
            % (timing, sparse_status.get("run_tag") or "sparse")
        )
    compile_state = compile_status.get("state")
    if compile_state not in (None, "off"):
        lines.append(
            "Shared compilation: %s — %s"
            % (compile_state, compile_status.get("reason") or "")
        )
    if mlp.get("provider") not in (None, "off"):
        lines.append(
            "Upstream MLP optimization: %s"
            % _provider_text(mlp, fallback_label="off")
        )
    return "\n".join(lines)


def format_legacy_status(model, warnings=()):
    details = format_sparse_status(model)
    lines = [
        "Deprecated compatibility node: replace this with MiniMax H3 Sage "
        "Memory Optimizer plus MiniMax H3 Sparse Sage Attention.",
    ]
    lines.extend(str(item) for item in warnings if str(item).strip())
    lines.append(details)
    return "\n".join(lines)
