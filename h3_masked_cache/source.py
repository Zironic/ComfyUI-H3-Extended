"""Resolve which reference video is the *source* the target is edited from.

`MiniMaxH3ReferenceToVideoZi` encodes every reference into a `minimax_refs`
block and core copies those blocks verbatim into `minimax_payload["refs"]`, so
the source latent is already sitting in the payload of every diffusion-model
call. What is missing is the identification: the payload is an ordered list of
mixed images, audio and videos, and nothing in it says which entry the user
meant by "the video I am editing".

The node names it by one-based ordinal *over video blocks only* - `1` is the
first `video`/`video_audio` reference, whether or not images and audio precede
it - because that is how the reference widgets read in the graph.

Everything here fails closed. A reference that is off by one frame, or that the
Ref2V node re-canvased to a different latent size than the target, is not
"close enough": the whole mask rests on `x0` and `source` describing the same
pixel, so a mismatch must produce a reason string, never a resized guess.
"""

from dataclasses import dataclass

import torch

VIDEO_KINDS = ("video", "video_audio")


@dataclass
class SourceResolution:
    """The selected source latent, or the exact reason there is not one."""

    latent: torch.Tensor | None = None
    ref_ordinal: int | None = None          # one-based, over video refs only
    payload_index: int | None = None        # index into payload["refs"]
    kind: str | None = None
    valid: bool = False
    reason: str | None = None

    @classmethod
    def fail(cls, reason):
        return cls(valid=False, reason=reason)

    def describe(self):
        if not self.valid:
            return "unresolved (%s)" % self.reason
        return "video ref %d (payload index %d, kind %s, latent %s)" % (
            self.ref_ordinal, self.payload_index, self.kind,
            list(self.latent.shape) if self.latent is not None else None)


def video_reference_blocks(refs):
    """`[(ordinal, payload_index, block)]` for the video references, in pack order."""
    out = []
    for i, blk in enumerate(refs or []):
        if isinstance(blk, dict) and blk.get("kind") in VIDEO_KINDS:
            out.append((len(out) + 1, i, blk))
    return out


def _normalize_batch(latent, target_batch):
    """Broadcast a batch-1 source up to the target's batch, or refuse."""
    if latent.shape[0] == target_batch:
        return latent, None
    if latent.shape[0] == 1:
        return latent.expand(target_batch, *latent.shape[1:]), None
    return None, ("source batch %d cannot be broadcast to target batch %d"
                  % (latent.shape[0], target_batch))


def resolve_source(payload, target_video_x, ordinal):
    """Pick the source latent for `target_video_x` out of the payload references.

    `target_video_x` is the DiT's video input *before* patch padding, i.e. the
    same geometry the model's video output is cropped back to, which is the
    space the whole mask is defined in.
    """
    refs = (payload or {}).get("refs")
    videos = video_reference_blocks(refs)
    if not videos:
        return SourceResolution.fail("no video reference in the conditioning "
                                     "(masked Ref2V needs one source video)")
    if ordinal < 1 or ordinal > len(videos):
        return SourceResolution.fail(
            "source_video_ref=%d out of range: %d video reference%s available"
            % (ordinal, len(videos), "" if len(videos) == 1 else "s"))

    ref_ordinal, payload_index, blk = videos[ordinal - 1]
    latent = blk.get("latent")
    if not torch.is_tensor(latent):
        return SourceResolution.fail("video ref %d carries no encoded latent" % ref_ordinal)
    if latent.ndim != 5:
        return SourceResolution.fail(
            "video ref %d latent has rank %d, expected 5 [B,C,T,H,W]" % (ref_ordinal, latent.ndim))
    if target_video_x.ndim != 5:
        return SourceResolution.fail(
            "target video latent has rank %d, expected 5 [B,C,T,H,W]" % target_video_x.ndim)

    tb, tc, tt, th, tw = target_video_x.shape
    _, sc, st, sh, sw = latent.shape
    if sc != tc:
        return SourceResolution.fail(
            "channel mismatch: source %d vs target %d" % (sc, tc))
    if st != tt:
        return SourceResolution.fail(
            "temporal mismatch: source latent_t=%d vs target latent_t=%d "
            "(the Ref2V node trims reference videos to a 17n+5 frame count, so a "
            "source clip must be generated at the same length as the target)"
            % (st, tt))
    if (sh, sw) != (th, tw):
        return SourceResolution.fail(
            "spatial mismatch: source latent %dx%d vs target latent %dx%d "
            "(reference videos are re-canvased independently of the requested "
            "generation size)" % (sh, sw, th, tw))

    latent, err = _normalize_batch(latent, tb)
    if err is not None:
        return SourceResolution.fail(err)

    return SourceResolution(latent=latent, ref_ordinal=ref_ordinal,
                            payload_index=payload_index, kind=blk.get("kind"),
                            valid=True)


class SourceCache:
    """One device-resident copy of the source latent, for the length of a run.

    The payload's copy lives wherever the VAE left it (usually CPU/offload); the
    score is computed against it on every observed forward, so it is moved once
    and held, and dropped explicitly at run end rather than at GC's convenience.
    """

    def __init__(self):
        self._latent = None
        self._key = None

    def get(self, resolution, device, dtype=torch.float32):
        key = (resolution.ref_ordinal, str(device), dtype)
        if self._key != key or self._latent is None:
            self._latent = resolution.latent.to(device=device, dtype=dtype).contiguous()
            self._key = key
        return self._latent

    def release(self):
        self._latent = None
        self._key = None
