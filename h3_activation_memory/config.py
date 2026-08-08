"""Validated immutable configuration for H3 activation-memory execution."""

from dataclasses import dataclass

MODE_BF16 = "mlp_chunked_bf16"
MODE_NATIVE = "mlp_chunked_native"
MODES = (MODE_BF16, MODE_NATIVE)
IMPLEMENTED_MODES = frozenset(MODES)
DEFAULT_MODE = MODE_NATIVE

MIN_CHUNK_ROWS = 256
MAX_CHUNK_ROWS = 65_536
DEFAULT_CHUNK_ROWS = 2_048
DEFAULT_ALIGNMENT = 256


@dataclass(frozen=True)
class ActivationMemoryConfig:
    """Configuration closed over by every patched H3 block forward.

    ``mlp_chunked_bf16`` materializes a bounded BF16 SwiGLU slab before the
    down-projection. ``mlp_chunked_native`` asks Comfy's TensorWise-INT8 path to
    fuse SwiGLU into activation quantization when that exact fast path is
    available, and otherwise follows Comfy's eager fallback.
    """

    mode: str = DEFAULT_MODE
    chunk_rows: int = DEFAULT_CHUNK_ROWS
    alignment: int = DEFAULT_ALIGNMENT
    strict: bool = True
    prefer_held_weights: bool = True

    def __post_init__(self):
        if self.mode not in IMPLEMENTED_MODES:
            raise ValueError(
                "activation-memory mode %r is not implemented (available: %s)"
                % (self.mode, ", ".join(sorted(IMPLEMENTED_MODES)))
            )
        if not (MIN_CHUNK_ROWS <= int(self.chunk_rows) <= MAX_CHUNK_ROWS):
            raise ValueError(
                "chunk_rows must be between %d and %d, got %r"
                % (MIN_CHUNK_ROWS, MAX_CHUNK_ROWS, self.chunk_rows)
            )
        if int(self.alignment) <= 0:
            raise ValueError("alignment must be positive")
        if int(self.chunk_rows) < int(self.alignment):
            raise ValueError("chunk_rows must be at least alignment")
        if int(self.chunk_rows) % int(self.alignment):
            raise ValueError(
                "chunk_rows (%d) must be a multiple of alignment (%d)"
                % (self.chunk_rows, self.alignment)
            )

    @property
    def native_swiglu(self):
        return self.mode == MODE_NATIVE

    @property
    def signature(self):
        return (
            self.mode,
            int(self.chunk_rows),
            int(self.alignment),
            bool(self.strict),
            bool(self.prefer_held_weights),
        )
