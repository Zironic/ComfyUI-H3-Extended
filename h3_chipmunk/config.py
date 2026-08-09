from dataclasses import dataclass

MODES = ("measure", "reference_delta")
SCOPES = ("target_video", "all_dynamic")
CACHE_LOCATIONS = ("cpu", "gpu")


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
    chunk_rows: int = 128
    token_group_rows: int = 128
    feature_group: int = 256
    scope: str = "target_video"
    cache_location: str = "cpu"
    cache_budget_gb: float = 24.0
    random_groups: float = 0.0
    strict: bool = True
    save_report: bool = True
    run_tag: str = "chipmunk"

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

    @property
    def signature(self):
        return (
            self.mode, float(self.top_fraction), int(self.refresh_every),
            int(self.first_dense_steps), int(self.last_dense_steps),
            int(self.first_dense_layers), int(self.layer_start), int(self.layer_stop),
            int(self.chunk_rows), int(self.token_group_rows), int(self.feature_group),
            self.scope, self.cache_location, float(self.cache_budget_gb),
            float(self.random_groups), bool(self.strict), bool(self.save_report),
            self.run_tag,
        )
