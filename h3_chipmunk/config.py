from dataclasses import dataclass
import math

# Production Comfy node modes only. Diagnostics that require reading CUDA values
# back on the host belong in standalone benchmarks, never in model execution.
MODES = ("measure", "reference_delta")
SCOPES = ("target_video", "all_dynamic")
CACHE_LOCATIONS = ("async_pinned",)
_LEGACY_CACHE_LOCATIONS = ("cpu", "gpu", "async_pinned")
DENSITY_PROFILES = ("depth_safe_v1", "uniform")

MEASURE_MIN_CHUNK_ROWS = 2_048
DEFAULT_CHUNK_ROWS = 2_048
DEFAULT_MEASURE_LAYER_STRIDE = 5
DEFAULT_GPU_STAGING_BUDGET_GB = 1.0
DEFAULT_STAGING_SLOTS = 2

# The first real H3 measurement showed useful MLP delta concentration, but early
# blocks are also the most destructive place to inject approximation error. This
# profile therefore protects blocks 0..10 completely and spends the approximation
# budget progressively later in the stack.
DEPTH_SAFE_V1 = (
    (0, 11, 1.00),
    (11, 20, 0.40),
    (20, 30, 0.50),
    (30, 50, 0.60),
)


@dataclass(frozen=True)
class H3ChipmunkConfig:
    mode: str = "measure"
    enabled: bool = True
    top_fraction: float = 0.25
    refresh_every: int = 6
    first_dense_steps: int = 2
    last_dense_steps: int = 2
    first_dense_layers: int = 0
    layer_start: int = 0
    layer_stop: int = 50
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    token_group_rows: int = 128
    feature_group: int = 256
    scope: str = "target_video"
    cache_location: str = "async_pinned"
    # This is a hard cap on *device staging* used by the cache, not on host
    # backing storage. Persistent state lives in pinned RAM and is DMA'd JIT.
    cache_budget_gb: float = DEFAULT_GPU_STAGING_BUDGET_GB
    random_groups: float = 0.0
    measure_layer_stride: int = DEFAULT_MEASURE_LAYER_STRIDE
    strict: bool = True
    save_report: bool = False
    run_tag: str = "chipmunk"
    density_profile: str = "depth_safe_v1"
    staging_slots: int = DEFAULT_STAGING_SLOTS

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unsupported production Chipmunk mode: {self.mode}")
        if self.scope not in SCOPES:
            raise ValueError(f"unsupported Chipmunk scope: {self.scope}")
        if self.cache_location not in _LEGACY_CACHE_LOCATIONS:
            raise ValueError(f"unsupported cache_location: {self.cache_location}")
        if self.mode == "reference_delta" and self.cache_location != "async_pinned":
            raise ValueError(
                "production reference_delta uses async_pinned JIT offload; the old CPU "
                "path synchronized and the full-GPU cache cannot satisfy the bounded VRAM design"
            )
        if self.density_profile not in DENSITY_PROFILES:
            raise ValueError(f"unsupported density_profile: {self.density_profile}")
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
        if int(self.staging_slots) < 2:
            raise ValueError("staging_slots must be >= 2 for overlap")

    @property
    def effective_chunk_rows(self):
        if self.mode == "measure":
            return max(int(self.chunk_rows), MEASURE_MIN_CHUNK_ROWS)
        return int(self.chunk_rows)

    def fraction_for_layer(self, layer_index: int) -> float:
        layer_index = int(layer_index)
        if self.density_profile == "uniform":
            return float(self.top_fraction)
        for start, stop, fraction in DEPTH_SAFE_V1:
            if int(start) <= layer_index < int(stop):
                return float(fraction)
        return 1.0

    def layer_eligible(self, layer_index: int) -> bool:
        layer_index = int(layer_index)
        if not (int(self.layer_start) <= layer_index < int(self.layer_stop)):
            return False
        if layer_index < int(self.first_dense_layers):
            return False
        return self.fraction_for_layer(layer_index) < 1.0

    def selected_features_for_layer(self, layer_index: int, ffn: int) -> int:
        """Balanced whole-ConvRot-group width for one layer.

        H3's FFN has 56 logical 256-neuron groups, split into two prepacked
        28-group halves. Keeping an equal count from both halves guarantees fixed
        rectangular CUDA tensors without reading selection counts on the host.
        """
        fraction = float(self.fraction_for_layer(layer_index))
        if fraction >= 1.0:
            return int(ffn)
        fg = int(self.feature_group)
        groups = int(ffn) // fg
        if groups % 2:
            raise ValueError("H3 FFN group count must split into two ConvRot halves")
        half_groups = groups // 2
        keep = max(1, min(half_groups, int(math.ceil(half_groups * fraction))))
        if float(self.random_groups) > 0.0 and keep < half_groups:
            keep += min(
                half_groups - keep,
                max(1, int(math.ceil(half_groups * float(self.random_groups)))),
            )
        return int(2 * keep * fg)

    def max_selected_features(self, ffn: int) -> int:
        widths = [
            self.selected_features_for_layer(layer, ffn)
            for layer in range(int(self.layer_start), int(self.layer_stop))
            if self.layer_eligible(layer)
        ]
        return max(widths, default=int(ffn))

    @property
    def profile(self):
        if self.density_profile == "uniform":
            return ((int(self.layer_start), int(self.layer_stop), float(self.top_fraction)),)
        return tuple((int(a), int(b), float(f)) for a, b, f in DEPTH_SAFE_V1)

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
            self.density_profile, int(self.staging_slots),
        )
