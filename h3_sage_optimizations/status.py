"""H3-Extended status text around dependency-owned production status."""

try:
    from ..h3_optimizations_dependency import dependency_module
except ImportError:
    from h3_optimizations_dependency import dependency_module

_status = dependency_module("status")
format_memory_status = _status.format_memory_status
format_sparse_status = _status.format_sparse_status


def format_disabled_status(node_name):
    return (
        "%s is disabled. No new optimization request was applied; "
        "upstream model patches are left unchanged." % node_name
    )


def format_legacy_status(model, warnings=()):
    lines = [
        "Deprecated compatibility node: replace this with MiniMax H3 Sage "
        "Memory Optimizer plus MiniMax H3 Sparse Sage Attention.",
    ]
    lines.extend(str(item) for item in warnings if str(item).strip())
    lines.append(format_sparse_status(model))
    return "\n".join(lines)


__all__ = [
    "format_disabled_status",
    "format_memory_status",
    "format_sparse_status",
    "format_legacy_status",
]
