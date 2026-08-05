"""Chunk geometry and the pixel-frame <-> latent-position mapping.

H3's video latent positions do not cover equal numbers of pixel frames. The DiT
lays them out on the repeating span pattern

    1, 4, 4, 4, 4

so a T=22 target (73 pixel frames) covers frames 0-72 as

    positions  0-14 -> 51 pixel frames
    positions 15-21 -> 22 pixel frames

which is exactly the 73/22/51 profile's stride and overlap. That coincidence is
what makes a *directly reused latent overlap* possible at all, and it is a
property of the profile rather than of the model - so it is computed here and
asserted, never hard-coded. A profile whose stride does not land on a latent
boundary fails loudly (`UnalignedProfileError`) instead of silently rounding to
an approximate position.
"""

from dataclasses import dataclass

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
FPS = 24
AUDIO_LATENT_FPS = 40


class UnalignedProfileError(ValueError):
    """The stride or overlap does not coincide with a latent-position boundary."""


def latent_frame_spans(latent_t):
    """Pixel frames covered by each of `latent_t` video latent positions."""
    if latent_t < 1:
        raise ValueError("latent_t must be at least 1")
    return [FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)] for i in range(latent_t)]


def video_latent_t(frame_count):
    """Target temporal latent length for a frame count on the 17k+5 grid."""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def decoded_frame_count(latent_t):
    """Pixel frames the VAE produces from `latent_t` latent positions."""
    if latent_t < 1:
        raise ValueError("latent_t must be at least 1")
    if latent_t == 1:
        return 1
    groups, remainder = divmod(latent_t - 2, 5)
    return 5 + groups * 17 + min(remainder * 4, 13)


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def find_exact_overlap_slice(latent_t, stride_frames, overlap_frames):
    """Return `(latent_start, latent_count)` for an exactly aligned overlap.

    `latent_start` is the first latent position whose pixel-frame span begins at
    `stride_frames`; `latent_count` covers exactly `overlap_frames`. Raises
    `UnalignedProfileError` when either boundary falls inside a position.
    """
    spans = latent_frame_spans(latent_t)
    total = sum(spans)
    if stride_frames + overlap_frames != total:
        raise UnalignedProfileError(
            "profile mismatch: stride %d + overlap %d = %d pixel frames, but "
            "latent_t=%d covers %d" % (stride_frames, overlap_frames,
                                       stride_frames + overlap_frames, latent_t, total))

    cumulative = 0
    latent_start = None
    for i, span in enumerate(spans):
        if cumulative == stride_frames:
            latent_start = i
            break
        cumulative += span
    if latent_start is None:
        if cumulative == stride_frames:
            latent_start = latent_t          # zero-length overlap
        else:
            raise UnalignedProfileError(
                "stride %d frames does not land on a latent-position boundary "
                "for latent_t=%d (boundaries: %s)"
                % (stride_frames, latent_t, _boundaries(spans)))

    latent_count = latent_t - latent_start
    carried = sum(spans[latent_start:])
    if carried != overlap_frames:
        raise UnalignedProfileError(
            "latent positions %d-%d cover %d pixel frames, not the requested "
            "overlap of %d" % (latent_start, latent_t - 1, carried, overlap_frames))
    return latent_start, latent_count


def _boundaries(spans):
    out, cumulative = [], 0
    for span in spans:
        out.append(cumulative)
        cumulative += span
    out.append(cumulative)
    return ", ".join(str(b) for b in out)


@dataclass(frozen=True)
class HarnessGeometry:
    """One two-chunk experiment profile, with its latent mapping resolved."""

    chunk_frames: int
    overlap_frames: int
    fps: int = FPS

    @property
    def stride_frames(self):
        return self.chunk_frames - self.overlap_frames

    @property
    def target_latent_t(self):
        return video_latent_t(self.chunk_frames)

    @property
    def audio_latent_t(self):
        return round(self.chunk_frames / self.fps * AUDIO_LATENT_FPS)

    @property
    def required_source_frames(self):
        """Source frames needed to cut both chunks at full length."""
        return self.stride_frames + self.chunk_frames

    @property
    def chunk_a_range(self):
        return 0, self.chunk_frames

    @property
    def chunk_b_range(self):
        return self.stride_frames, self.stride_frames + self.chunk_frames

    @property
    def overlap_range(self):
        """Global pixel-frame range shared by both chunks."""
        return self.stride_frames, self.chunk_frames

    def overlap_slice(self):
        """`(latent_start, latent_count)` of the overlap inside a chunk latent.

        Raises `UnalignedProfileError` on a profile that does not align, which is
        what keeps every direct-latent strategy honest.
        """
        return find_exact_overlap_slice(
            self.target_latent_t, self.stride_frames, self.overlap_frames)

    def validate(self):
        """Check the generation grid and the plan's stated latent assertions."""
        if self.chunk_frames % 17 != 5:
            raise ValueError(
                "chunk_frames must satisfy n %% 17 == 5 (got %d; nearest legal "
                "is %d)" % (self.chunk_frames, align_frame_count(self.chunk_frames)))
        if not 0 < self.overlap_frames < self.chunk_frames:
            raise ValueError("overlap_frames must be in (0, chunk_frames)")
        if decoded_frame_count(self.target_latent_t) != self.chunk_frames:
            raise ValueError(
                "latent_t=%d decodes to %d frames, not %d"
                % (self.target_latent_t, decoded_frame_count(self.target_latent_t),
                   self.chunk_frames))

        spans = latent_frame_spans(self.target_latent_t)
        start, count = self.overlap_slice()
        assert sum(spans[:start]) == self.stride_frames
        assert sum(spans[start:start + count]) == self.overlap_frames
        assert start + count == self.target_latent_t
        return self

    def describe(self):
        try:
            start, count = self.overlap_slice()
            mapping = "overlap latent [%d:%d]" % (start, start + count)
        except UnalignedProfileError as exc:
            mapping = "overlap latent unaligned (%s)" % exc
        return ("C=%d O=%d S=%d T=%d at %d fps; source frames >= %d; %s"
                % (self.chunk_frames, self.overlap_frames, self.stride_frames,
                   self.target_latent_t, self.fps, self.required_source_frames, mapping))


DEFAULT_GEOMETRY = HarnessGeometry(chunk_frames=73, overlap_frames=22)
