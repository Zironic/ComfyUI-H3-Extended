"""Configure the H3-specific efficient Sage model patch."""

import logging

from comfy.ldm.modules.attention import get_attention_function

from .patch import install
from .sage_mem_eff import SM89SageMemoryEfficientBackend

BACKEND_NAME = "sage_mem_eff"


def _pin_token_refiner_to_sage(transformer_options):
    sage = get_attention_function("sage", default=None)
    if sage is None:
        raise RuntimeError(
            "sage_mem_eff requires the registered 'sage' backend for the H3 token refiner")
    sage_impl = getattr(sage, "__wrapped__", sage)

    def override(_original, *args, **kwargs):
        return sage_impl(*args, **kwargs)

    transformer_options["optimized_attention_override"] = override


def configure(model_patcher):
    """Install the SM89 backend on a cloned ModelPatcher and return diagnostics."""
    transformer_options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    _pin_token_refiner_to_sage(transformer_options)
    backend = SM89SageMemoryEfficientBackend()
    count = install(model_patcher, backend=backend)
    transformer_options["minimax_h3_attention_backend"] = BACKEND_NAME
    logging.info("[H3 attention] configured %s on %d blocks", BACKEND_NAME, count)
    return backend, count
