"""Configure H3-specific prepared-QKV attention backends."""

import logging

from comfy.ldm.modules.attention import get_attention_function

from .patch import install
from .sage_mem_eff import SM89SageMemoryEfficientBackend

BACKEND_NAME = "sage_mem_eff"


def _pin_token_refiner_to_sage(transformer_options):
    sage = get_attention_function("sage", default=None)
    if sage is None:
        raise RuntimeError(
            "prepared H3 attention requires the registered 'sage' backend for "
            "the H3 token refiner"
        )
    sage_impl = getattr(sage, "__wrapped__", sage)

    def override(_original, *args, **kwargs):
        return sage_impl(*args, **kwargs)

    transformer_options["optimized_attention_override"] = override


def configure_backend(model_patcher, backend):
    """Install one already-preflighted prepared-QKV backend.

    Keeping construction separate from installation lets a capability resolver
    fall back before any model options or object patches are mutated.
    """
    if backend is None:
        raise TypeError("backend must not be None")
    backend_name = getattr(backend, "name", type(backend).__name__)
    transformer_options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    _pin_token_refiner_to_sage(transformer_options)
    count = install(model_patcher, backend=backend)
    transformer_options["minimax_h3_attention_backend"] = backend_name
    logging.info("[H3 attention] configured %s on %d blocks", backend_name, count)
    return backend, count


def configure(model_patcher):
    """Backward-compatible standalone SM89 node configuration."""
    return configure_backend(model_patcher, SM89SageMemoryEfficientBackend())
