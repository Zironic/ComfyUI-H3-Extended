"""Compact UI summaries for resolved H3 Sage optimization plans."""

from __future__ import annotations

from .plan import PLAN_KEY, STATUS_KEY


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

    lines = [
        "Attention: %s" % (attention.get("selected") or "preserve incoming"),
        "QKV: %s" % _provider_text(qkv, fallback_label="standard_h3_qkv"),
        "MLP: %s" % _provider_text(mlp, fallback_label="off"),
    ]
    chunk_rows = mlp.get("chunk_rows")
    if chunk_rows is not None and mlp.get("provider") != "off":
        lines[-1] += " (%d-row chunks)" % int(chunk_rows)
    return "\n".join(lines)


def format_sparse_status(model):
    status = _status(model)
    if status is None:
        return "Skipped: input model is not MiniMax H3."

    qkv = status.get("fused_qkv", {})
    mlp = status.get("mlp", {})
    sparse_status = status.get("sparse") or {}
    plan = _plan(model)
    sparse_request = getattr(plan, "sparse", None)
    budget = sparse_status.get("video_budget")
    if budget is None:
        budget = getattr(sparse_request, "video_budget", 0.0)
    budget = float(budget)
    denser_early_late_steps = sparse_status.get(
        "denser_early_late_steps",
        getattr(sparse_request, "denser_early_late_steps", False),
    )

    lines = [
        "Sparse Sage requested video KV budget: %.1f%%" % (budget * 100.0),
        "QKV: %s" % _provider_text(qkv, fallback_label="standard_h3_qkv"),
        (
            "Effective density is rounded up to a whole KV-tile count at "
            "runtime; non-video context and mixed boundary tiles remain dense."
        ),
    ]
    if denser_early_late_steps:
        lines.insert(
            1,
            "First 2 and last 2 steps add 30 percentage points, capped at 100%.",
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
