"""Two-sided AV bridge experiment.

Does H3 read a second reference video as a *future* constraint, or only as
ordinary conditioning? One held-out interval, four arms, one decisive
comparison (B vs C).

The reference intervals are sliced out of already-encoded C1/C3 latents rather
than built with `ref_builder.encode_video_ref`. See `plan.py` for why.
"""

from .plan import (
    ARMS,
    ARM_ORDER,
    BridgePlan,
    head_latent_slice,
    tail_latent_slice,
)

__all__ = [
    "ARMS",
    "ARM_ORDER",
    "BridgePlan",
    "head_latent_slice",
    "tail_latent_slice",
]
