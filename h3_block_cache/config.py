from dataclasses import dataclass

MODES = ("observe", "shadow", "fixed_gpu")

@dataclass(frozen=True)
class BlockCacheConfig:
    mode: str = "observe"
    unit_spec: str = "25"
    warmup_steps: int = 2
    refresh_interval: int = 2
    max_reuse_span: int = 1
    force_refresh_last_steps: int = 1
    strict: bool = True
    run_tag: str = "h3block"

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unsupported mode {self.mode!r}; expected one of {MODES}")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.refresh_interval < 1:
            raise ValueError("refresh_interval must be >= 1")
        if self.max_reuse_span < 1:
            raise ValueError("max_reuse_span must be >= 1")
        if self.force_refresh_last_steps < 0:
            raise ValueError("force_refresh_last_steps must be >= 0")
