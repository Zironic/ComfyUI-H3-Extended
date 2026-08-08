"""Generic request-local deferred CUDA timing helpers.

The timing implementation is supplied by the owner of a feature, while this
module only publishes and scopes it through ``transformer_options``.  Keeping
the seam generic lets activation and attention code share one request timer
without importing each other's product concepts.
"""

from contextlib import contextmanager


TIMING_KEY = "minimax_h3_deferred_cuda_timing"


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
def timed_stage(transformer_options, stage):
    """Measure one stage without synchronizing or allocating when inactive."""
    timing = get_timing(transformer_options)
    token = timing.begin(stage) if timing is not None else None
    try:
        yield
    finally:
        if timing is not None:
            timing.end(token)
