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
VAE_GROUP_POSITIONS = len(FRAME_PER_TOKEN)
VAE_GROUP_FRAMES = sum(FRAME_PER_TOKEN)


def splitmix64(seed):
    z = (seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF


def chunk_seed(seed, index):
    """Per-chunk noise seed derived from the run seed.

    Lives here rather than in `harness` so the prompt planner and the resume
    scan can derive the same value without importing torch. Both sides deriving
    it independently is exactly how a resume ends up rejecting chunks it should
    have kept, so there is one definition.
    """
    return splitmix64(int(seed) + 1000 + int(index))
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


def audio_boundary_is_exact(stride_frames, fps=FPS, audio_latent_fps=AUDIO_LATENT_FPS):
    """Whether a chunk boundary lands on a whole audio latent position.

    Video frames are 1/24 s and audio latents 1/40 s, so the two grids only
    coincide every three frames. On a stride that misses, the audio carry slice
    rounds to the nearest latent and the model is handed a continuation that is
    up to a third of a latent (8.3 ms) out of step with its own video, while the
    video carry stays exact. Nothing downstream detects that.
    """
    return (int(stride_frames) * int(audio_latent_fps)) % int(fps) == 0


def aligned_overlap_frames(chunk_frames, overlap_frames, fps=FPS):
    """Nearest overlap that is exact on the video *and* audio grids.

    Returns the requested overlap unchanged when it already aligns. Ties prefer
    the larger overlap: more carried context is the safer direction to move a
    profile that the caller asked to have corrected.
    """
    chunk_frames = int(chunk_frames)
    overlap_frames = int(overlap_frames)
    candidates = []
    for candidate in range(1, chunk_frames):
        stride = chunk_frames - candidate
        if stride < 1 or not audio_boundary_is_exact(stride, fps=fps):
            continue
        try:
            find_exact_overlap_slice(
                video_latent_t(chunk_frames), stride, candidate
            )
        except UnalignedProfileError:
            continue
        candidates.append(candidate)
    if not candidates:
        raise UnalignedProfileError(
            "no overlap aligns video and audio for chunk_frames=%d" % chunk_frames
        )
    if overlap_frames in candidates:
        return overlap_frames
    return min(candidates, key=lambda c: (abs(c - overlap_frames), -c))


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

    @property
    def total_frames(self):
        """Frames the two chunks cover end to end - the monolithic equivalent."""
        return self.stride_frames + self.chunk_frames

    @property
    def supports_monolithic(self):
        """True when the two-chunk span is itself a legal generation length.

        `total = S + C` and `C % 17 == 5`, so `total % 17 == 5` exactly when
        `S % 17 == 0`. Only those profiles can be compared against a single run
        of the same length without snapping one side to a different duration and
        making the comparison inexact.
        """
        return self.total_frames % 17 == 5

    @property
    def monolithic_latent_t(self):
        return video_latent_t(self.total_frames)

    def tail_range(self, tail_frames=17):
        """Global frame range of the tail both renderings should agree on.

        The tail is the far end of the second chunk - the frames furthest from
        the carried state, and therefore where a carry that failed to hold the
        trajectory shows up worst.
        """
        tail = min(tail_frames, self.chunk_frames)
        return self.total_frames - tail, self.total_frames

    def tail_in_chunk_b(self, tail_frames=17):
        """Same tail, as local frame indices inside chunk B."""
        tail = min(tail_frames, self.chunk_frames)
        return self.chunk_frames - tail, self.chunk_frames

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
