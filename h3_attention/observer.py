"""Shared observation seam for H3 attention.

Instrumentation used to hang off the module-global `optimized_attention` name in
`comfy.ldm.minimax.model`, which works only while every attention call goes
through that binding. A custom block forward does not - it calls its backend
directly - so the probe would go dark the moment one is installed.

This module is the seam both paths publish to instead:

    legacy adapter (h3_probe.capture)  -> notify_attention(..., layer_index=None)
    H3 block forward (h3_attention)    -> notify_attention(..., layer_index=i)

Observers live in `transformer_options` rather than a module global, so their
lifetime is exactly the diffusion-model invocation that installed them and
concurrent runs cannot see each other's.

Q and K arrive post-RMSNorm and post-RoPE in HND layout, `[batch, heads, seq,
dim]` - what the probe already expects. A backend holding NHD tensors should
pass `.transpose(1, 2)` views; that is free and copies nothing.
"""

import contextlib
import logging

OBSERVER_KEY = "minimax_h3_attention_observers"


def notify_attention(q, k, *, layer_index, transformer_options):
    """Publish one attention call to whatever observers are installed.

    `layer_index` is the DiT block index when the caller knows it, or None when
    it does not and the observer should fall back to counting calls.

    Observation is instrumentation: a failing observer is logged and skipped, it
    never interrupts inference. The no-observer path is a dict lookup.
    """
    if not transformer_options:
        return
    observers = transformer_options.get(OBSERVER_KEY)
    if not observers:
        return
    for observer in observers:
        try:
            observer(q, k, layer_index)
        except Exception:
            logging.exception("[H3 attention] observer failed; continuing inference")


@contextlib.contextmanager
def observing(transformer_options, observer):
    """Install `observer` for the duration of the block, then restore.

    Rebinds the list rather than mutating in place so a nested or concurrent
    invocation that captured the previous list is unaffected.
    """
    previous = transformer_options.get(OBSERVER_KEY)
    transformer_options[OBSERVER_KEY] = list(previous or ()) + [observer]
    try:
        yield
    finally:
        if previous is None:
            transformer_options.pop(OBSERVER_KEY, None)
        else:
            transformer_options[OBSERVER_KEY] = previous
