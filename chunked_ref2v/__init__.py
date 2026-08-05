"""MiniMax H3 chunked ref2v: the multi-strategy two-chunk experiment harness.

`PLAN.md` is the arbitrary-length production design; the harness plan alongside
it is what this package implements - the experiment that decides which carry
mechanism that production node should use.

Nothing here is the production node. It generates Chunk A once and evaluates one
or more Chunk B strategies against it under controlled conditions.
"""

from .experiments import CATALOG, SUITES, ExperimentSpec, resolve_suite
from .geometry import (
    DEFAULT_GEOMETRY,
    HarnessGeometry,
    UnalignedProfileError,
    find_exact_overlap_slice,
    latent_frame_spans,
)
from .layout_ops import TargetAlignedCondition, insert_target_conditions
from .strategies import STRATEGIES, StrategyDependencies, StrategyUnavailable

__all__ = [
    "CATALOG", "SUITES", "STRATEGIES",
    "DEFAULT_GEOMETRY", "HarnessGeometry", "UnalignedProfileError",
    "ExperimentSpec", "StrategyDependencies", "StrategyUnavailable",
    "TargetAlignedCondition",
    "find_exact_overlap_slice", "insert_target_conditions",
    "latent_frame_spans", "resolve_suite",
]
