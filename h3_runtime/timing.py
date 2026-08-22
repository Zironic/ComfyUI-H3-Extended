"""Generic request-local deferred CUDA timing helpers.

The timing implementation is supplied by the owner of a feature, while this
module only publishes and scopes it through ``transformer_options``.  Keeping
the seam generic lets activation and attention code share one request timer
without importing each other's product concepts.
"""

from contextlib import contextmanager
import logging

import torch


TIMING_KEY = "minimax_h3_deferred_cuda_timing"
STAGE_OBSERVERS_KEY = "_minimax_h3_stage_observers"


def publish_timing(transformer_options, timing):
    """Publish a request-local timing object and return it."""
    if transformer_options is not None:
        transformer_options[TIMING_KEY] = timing
    return timing


def get_timing(transformer_options):
    """Retrieve the request-local timing object, if one was published."""
    if not transformer_options:
        return None
    return transformer_options.get(TIMING_KEY)


@contextmanager
def observing_stages(transformer_options, observer):
    """Temporarily attach one request-local eager stage observer."""
    current = list(transformer_options.get(STAGE_OBSERVERS_KEY, ()))
    transformer_options[STAGE_OBSERVERS_KEY] = current + [observer]
    try:
        yield transformer_options
    finally:
        if current:
            transformer_options[STAGE_OBSERVERS_KEY] = current
        else:
            transformer_options.pop(STAGE_OBSERVERS_KEY, None)


@contextmanager
def timed_stage(transformer_options, stage):
    """Measure one stage without synchronizing or allocating when inactive."""
    if torch.compiler.is_compiling():
        yield
        return
    timing = get_timing(transformer_options)
    token = timing.begin(stage) if timing is not None else None
    observers = transformer_options.get(STAGE_OBSERVERS_KEY) if transformer_options else None
    observer_tokens = None
    if observers:
        observer_tokens = []
        for observer in tuple(observers):
            try:
                observer_tokens.append((observer, observer.begin(stage)))
            except Exception:
                logging.exception("[H3 runtime] stage observer failed entering %s", stage)
    try:
        yield
    finally:
        if observer_tokens:
            for observer, observer_token in reversed(observer_tokens):
                try:
                    observer.end(observer_token)
                except Exception:
                    logging.exception("[H3 runtime] stage observer failed leaving %s", stage)
        if timing is not None:
            timing.end(token)
