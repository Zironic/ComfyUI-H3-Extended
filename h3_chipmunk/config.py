from dataclasses import dataclass

MODES = ("measure", "shadow_validate", "reference_delta")
SCOPES = ("target_video", "all_dynamic")
CACHE_LOCATIONS = ("cpu", "gpu")

# Measurement/validation are diagnostic, so keep the exact dense MLP on the
# already-measured efficient slab size even when an old saved workflow still
# contains the original 128-row prototype default.
MEASURE_MIN_CHUNK_ROWS = 2_048
DEFAULT_CHUNK_ROWS = 2_048
DEFAULT_MEASURE_LAYER_STRIDE = 5

# Depth profile derived from the first real H3 group-delta measurement. Shadow
# validation keeps the real dense output in the model path and tests this profile
# only beside it; none of these fractions alter the generated sample.
SHADOW_DEPTH_PROFILE = (
    (0, 15, 0.30),
    (15, 25, 0.40),
    (25, 30, 0.50),
    (30, 50, 1.00),
)
DEFAULT_SHADOW_LAYER_STRIDE = 5
DEFAULT_SHADOW_SAMPLE_ROWS = 128


@dataclass(frozen=True)
class H3ChipmunkConfig:
    mode: str = "measure"
    enabled: bool = True
    top_fraction: float = 0.25
    refresh_every: int = 6
    first_dense_steps: int = 2
    last_dense_steps: int = 2
    first_dense_layers: int = 2
    layer_start: int = 0
    layer_stop: int = 50
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    token_group_rows: int = 128
    feature_group: int = 256
    scope: str = "target_video"
    cache_location: str = "cpu"
    cache_budget_gb: float = 24.0
    random_groups: float = 0.0
    measure_layer_stride: int = DEFAULT_MEASURE_LAYER_STRIDE
    strict: bool = True
    save_report: bool = True
    run_tag: str = "chipmunk"
    shadow_layer_stride: int = DEFAULT_SHADOW_LAYER_STRIDE
    shadow_sample_rows: int = DEFAULT_SHADOW_SAMPLE_ROWS

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unsupported Chipmunk mode: {self.mode}")
        if self.scope not in SCOPES:
            raise ValueError(f"unsupported Chipmunk scope: {self.scope}")
        if self.cache_location not in CACHE_LOCATIONS:
            raise ValueError(f"unsupported cache_location: {self.cache_location}")
        if not (0.0 < float(self.top_fraction) <= 1.0):
            raise ValueError("top_fraction must be in (0, 1]")
        if int(self.refresh_every) < 1:
            raise ValueError("refresh_every must be >= 1")
        if int(self.chunk_rows) < 1 or int(self.token_group_rows) < 1:
            raise ValueError("chunk_rows/token_group_rows must be positive")
        if int(self.feature_group) <= 0:
            raise ValueError("feature_group must be positive")
        if not (0 <= int(self.layer_start) < int(self.layer_stop) <= 50):
            raise ValueError("layer range must satisfy 0 <= start < stop <= 50")
        if float(self.cache_budget_gb) <= 0:
            raise ValueError("cache_budget_gb must be positive")
        if not (0.0 <= float(self.random_groups) < 1.0):
            raise ValueError("random_groups must be in [0, 1)")
        if int(self.measure_layer_stride) < 1:
            raise ValueError("measure_layer_stride must be >= 1")
        if int(self.shadow_layer_stride) < 1:
            raise ValueError("shadow_layer_stride must be >= 1")
        if int(self.shadow_sample_rows) < 1:
            raise ValueError("shadow_sample_rows must be positive")
        if int(self.shadow_sample_rows) % 32:
            raise ValueError("shadow_sample_rows must be a multiple of 32")

    @property
    def effective_chunk_rows(self):
        if self.mode in ("measure", "shadow_validate"):
            return max(int(self.chunk_rows), MEASURE_MIN_CHUNK_ROWS)
        return int(self.chunk_rows)

    def shadow_fraction_for_layer(self, layer_index: int) -> float:
        layer_index = int(layer_index)
        for start, stop, fraction in SHADOW_DEPTH_PROFILE:
            if start <= layer_index < stop:
                return float(fraction)
        return 1.0

    def shadow_layer_enabled(self, layer_index: int) -> bool:
        layer_index = int(layer_index)
        if not (int(self.layer_start) <= layer_index < int(self.layer_stop)):
            return False
        if self.shadow_fraction_for_layer(layer_index) >= 1.0:
            return False
        sparse_stop = min(int(self.layer_stop), 30)
        last_sparse = sparse_stop - 1
        return (
            layer_index == last_sparse
            or (layer_index - int(self.layer_start)) % int(self.shadow_layer_stride) == 0
        )

    @property
    def shadow_profile(self):
        return tuple(
            (int(start), int(stop), float(fraction))
            for start, stop, fraction in SHADOW_DEPTH_PROFILE
        )

    @property
    def signature(self):
        return (
            self.mode, float(self.top_fraction), int(self.refresh_every),
            int(self.first_dense_steps), int(self.last_dense_steps),
            int(self.first_dense_layers), int(self.layer_start), int(self.layer_stop),
            int(self.chunk_rows), int(self.token_group_rows), int(self.feature_group),
            self.scope, self.cache_location, float(self.cache_budget_gb),
            float(self.random_groups), int(self.measure_layer_stride),
            bool(self.strict), bool(self.save_report), self.run_tag,
            int(self.shadow_layer_stride), int(self.shadow_sample_rows),
        )
