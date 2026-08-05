"""Configuration for masked Ref2V measurement.

Only ``measure`` is implemented. The other mode names are reserved so reports
and tests can evolve without silently changing their meaning.
"""

from dataclasses import asdict, dataclass

MODE_MEASURE = "measure"
MODE_FIXED = "fixed"
MODE_DYNAMIC = "dynamic"

MODES = (MODE_MEASURE, MODE_FIXED, MODE_DYNAMIC)
IMPLEMENTED_MODES = (MODE_MEASURE,)

# Wide enough to diagnose the all-active regime seen with the original 0.1
# default. The stored float32 error/source maps permit arbitrary offline sweeps.
THRESHOLD_SWEEP = (
    0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75,
    1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0,
)

SCORE_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


@dataclass(frozen=True)
class MaskedCacheConfig:
    mode: str = MODE_MEASURE
    source_video_ref: int = 1
    burn_in_steps: int = 2
    warmup_steps: int = 2
    refresh_interval: int = 0
    score_threshold: float = 0.1
    score_absolute_floor: float = 1e-3
    tile_h: int = 2
    tile_w: int = 2
    spatial_halo: int = 1
    temporal_halo: int = 1
    dense_fallback_fraction: float = 0.8
    strict: bool = True
    run_tag: str = "h3mask"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError("unknown masked-cache mode %r (expected one of %s)"
                             % (self.mode, ", ".join(MODES)))
        if self.source_video_ref < 1:
            raise ValueError("source_video_ref is one-based; got %d" % self.source_video_ref)
        for name in ("tile_h", "tile_w"):
            if getattr(self, name) < 1:
                raise ValueError("%s must be >= 1" % name)
        for name in ("spatial_halo", "temporal_halo", "burn_in_steps", "refresh_interval"):
            if getattr(self, name) < 0:
                raise ValueError("%s must be >= 0" % name)
        if self.warmup_steps < 1:
            raise ValueError("warmup_steps must be >= 1")
        if self.score_absolute_floor <= 0.0:
            raise ValueError("score_absolute_floor must be > 0")
        if not 0.0 <= self.dense_fallback_fraction <= 1.0:
            raise ValueError("dense_fallback_fraction must be in [0, 1]")

    @property
    def tile(self):
        return (self.tile_h, self.tile_w)

    @property
    def freeze_start(self):
        return self.burn_in_steps

    @property
    def freeze_stop(self):
        return self.burn_in_steps + self.warmup_steps

    def as_dict(self):
        return asdict(self)
