"""Per-invocation observation seam for H3 activation-memory phases."""

import contextlib
import logging

OBSERVER_KEY = "_h3_activation_memory_observers"


def notify_activation(event, layer_index, transformer_options=None, **payload):
    """Notify observers without allowing diagnostics to interrupt inference."""
    if not transformer_options:
        return
    observers = transformer_options.get(OBSERVER_KEY)
    if not observers:
        return
    record = dict(payload)
    for observer in tuple(observers):
        try:
            observer(event, layer_index, record)
        except Exception:
            logging.exception(
                "[H3 activation memory] observer failed at %s, block %s",
                event,
                layer_index,
            )


@contextlib.contextmanager
def observing(transformer_options, observer):
    """Temporarily attach one observer to a transformer-options dictionary."""
    current = list(transformer_options.get(OBSERVER_KEY, ()))
    transformer_options[OBSERVER_KEY] = current + [observer]
    try:
        yield transformer_options
    finally:
        if current:
            transformer_options[OBSERVER_KEY] = current
        else:
            transformer_options.pop(OBSERVER_KEY, None)
