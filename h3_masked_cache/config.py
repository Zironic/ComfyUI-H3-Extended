"""Configuration for masked Ref2V computation.

One frozen dataclass, built by the node and read everywhere else. Nothing
downstream may mutate it, so a run's parameters are exactly what the node was
executed with - a report that says `tile 2x2` cannot be describing a different
tile size than the mask it sits next to.

Only `measure` is implemented at this stage; `fixed` and `dynamic` are declared
here because the report and the state machine already key off the mode, and a
mode string that appears only once the feature lands is a mode string nobody
validated.
"""

from dataclasses import asdict, dataclass

MODE_MEASURE = "measure"
MODE_FIXED = "fixed"
MODE_DYNAMIC = "dynamic"

MODES = (MODE_MEASURE, MODE_FIXED, MODE_DYNAMIC)

# Modes whose forward pass differs from the stock dense one. Stage 0 implements
# none of them: measurement must be provably output-neutral before anything is
# allowed to change what the model computes.
IMPLEMENTED_MODES = (MODE_MEASURE,)

# Thresholds the measure report sweeps. The committed default has to come out of
# these curves rather than the other way round, so the sweep is deliberately
# wider than any threshold anyone would plausibly pick.
THRESHOLD_SWEEP = (0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0)

# Score quantiles reported per observed forward.
SCORE_QUANTILES = (0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


@dataclass(frozen=True)
class MaskedCacheConfig:
    """Immutable parameters for one armed node."""

    mode: str = MODE_MEASURE
    source_video_ref: int = 1          # one-based, over video/video_audio refs only
    warmup_steps: int = 2
    refresh_interval: int = 0
    score_threshold: float = 0.1
    score_absolute_floor: float = 1e-3
    tile_h: int = 2
    tile_w: int = 2
    spatial_halo: int = 1              # in tiles
    temporal_halo: int = 1             # in latent frames
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
        for name in ("spatial_halo", "temporal_halo", "warmup_steps", "refresh_interval"):
            if getattr(self, name) < 0:
                raise ValueError("%s must be >= 0" % name)
        if self.score_absolute_floor <= 0.0:
            # the floor is what keeps the relative score finite on flat latent
            # regions; zero would make empty sky the most active region there is
            raise ValueError("score_absolute_floor must be > 0")

    @property
    def tile(self):
        return (self.tile_h, self.tile_w)

    def as_dict(self):
        return asdict(self)
