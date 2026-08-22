"""Compatibility import for the H3-Optimizations apply contract."""

try:
    from ..h3_optimizations_dependency import dependency_module
except ImportError:
    from h3_optimizations_dependency import dependency_module

apply_plan = dependency_module("apply").apply_plan

__all__ = ["apply_plan"]
