"""Configure H3-specific consuming attention backends."""

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

    override._h3_backend = "sage"
    transformer_options["optimized_attention_override"] = override


def configure_backend(model_patcher, backend, projector=None):
    """Install one already-preflighted consuming backend.

    Prepared Sage backends pin the small token refiner to registered Sage.  Sol
    can use its own BF16 SDPA dense fallback and therefore preserves the incoming
    token-refiner attention selection instead of making Sage a hidden dependency.
    """
    if backend is None:
        raise TypeError("backend must not be None")
    backend_name = getattr(backend, "name", type(backend).__name__)
    transformer_options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    if bool(getattr(backend, "requires_registered_sage", True)):
        _pin_token_refiner_to_sage(transformer_options)
    count = install(model_patcher, backend=backend, projector=projector)
    transformer_options["minimax_h3_attention_backend"] = backend_name
    logging.info("[H3 attention] configured %s on %d blocks", backend_name, count)
    return backend, count


def configure(model_patcher):
    return configure_backend(model_patcher, SM89SageMemoryEfficientBackend())
