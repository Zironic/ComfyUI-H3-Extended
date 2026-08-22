"""Compatibility imports for H3-Optimizations model discovery."""

try:
    from ..h3_optimizations_dependency import dependency_module
except ImportError:
    from h3_optimizations_dependency import dependency_module

_model = dependency_module("model")
get_minimax_h3_model = _model.get_minimax_h3_model
is_minimax_h3 = _model.is_minimax_h3
get_h3_blocks = _model.get_h3_blocks

__all__ = ["get_minimax_h3_model", "is_minimax_h3", "get_h3_blocks"]
