"""Shared observation seam for H3 attention.

Observers live in ``transformer_options`` so their lifetime is the current
model invocation. The custom block forward supplies explicit layer indices;
the legacy module-global adapter supplies ``None`` and counts calls.
"""

import contextlib
import logging

OBSERVER_KEY = "minimax_h3_attention_observers"
OBSERVED_KEY = "minimax_h3_attention_already_observed"


def notify_attention(q, k, *, layer_index, transformer_options):
    """Publish one attention call without allowing instrumentation to fail inference."""
    if not transformer_options:
        return
    # A custom forward already notified observers before entering a legacy
    # attention adapter. Suppress only that adapter's anonymous observation.
    if layer_index is None and transformer_options.get(OBSERVED_KEY):
        return
    observers = transformer_options.get(OBSERVER_KEY)
    if not observers:
        return
    for observer in tuple(observers):
        try:
            observer(q, k, layer_index)
        except Exception:
            logging.exception("[H3 attention] observer failed; continuing inference")


@contextlib.contextmanager
def observing(transformer_options, observer):
    """Append an observer for one invocation, restoring the previous value."""
    previous = transformer_options.get(OBSERVER_KEY)
    transformer_options[OBSERVER_KEY] = list(previous or ()) + [observer]
    try:
        yield
    finally:
        if previous is None:
            transformer_options.pop(OBSERVER_KEY, None)
        else:
            transformer_options[OBSERVER_KEY] = previous


@contextlib.contextmanager
def marked_observed(transformer_options):
    """Mark that the H3-owned forward already emitted the observer event."""
    previous = transformer_options.get(OBSERVED_KEY)
    transformer_options[OBSERVED_KEY] = True
    try:
        yield
    finally:
        if previous is None:
            transformer_options.pop(OBSERVED_KEY, None)
        else:
            transformer_options[OBSERVED_KEY] = previous
