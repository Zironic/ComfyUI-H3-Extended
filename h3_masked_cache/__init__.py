"""MiniMax H3 masked Ref2V computation.

`../H3 Masked Computation Plan.md` is the full design: remove unchanged target
video tokens from H3's 50 DiT blocks after a dense warm-up, keeping the whole
source/reference stream as context, and clamp the removed region to the source.

This package currently implements **Stage 0 only - measurement**. It answers the
question the rest of the design rests on:

> Does an early predicted-clean source-difference map reliably identify the
> final edited region?

Nothing here changes what the model computes. `blocks.py` and `plan.py` from the
plan's structure do not exist yet, and the node refuses any mode that would need
them.
"""

from .config import MODES, MaskedCacheConfig
from .mask import build_mask, latent_score, token_score
from .session import MaskedCacheRun, MaskedCacheSession
from .source import SourceResolution, resolve_source, video_reference_blocks

__all__ = [
    "MODES", "MaskedCacheConfig",
    "MaskedCacheRun", "MaskedCacheSession",
    "SourceResolution", "resolve_source", "video_reference_blocks",
    "build_mask", "latent_score", "token_score",
]
