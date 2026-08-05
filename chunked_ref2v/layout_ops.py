"""Insert target-aligned condition rows into a stock H3 `PackedLayout`.

Core builds the ordinary ref2v pack:

    [text | references | target audio | target video]

and this turns it into

    [text | target-aligned conditions | references | target audio | target video]

without reimplementing `PackedLayout`. The conditions go physically where core
puts keyframe `cond` rows - immediately after text - so the row order that
`_cond_video_rows` consumes stays `conditions, then refs`, and every existing row
index after the insertion point is shifted.

The point of the transform is the *temporal* coordinate. Core pins a first-frame
keyframe to `cond_t = float(text_len)`, but when references are present it walks
the cursor past every reference block before laying down the target video grid
(`comfy/ldm/minimax/model.py:335-388`), so a keyframe ends up sharing a temporal
origin with the first reference rather than with target frame 0. The
`copy_target` policy sidesteps the arithmetic entirely: it copies the exact
`(t, h, w)` rows the target position already has. The `stock` policy reproduces
the uncorrected placement so the two can be compared with everything else held
identical.

Nothing here mutates the layout it is handed.
"""

from dataclasses import dataclass, field

import torch

POSITION_POLICIES = ("copy_target", "stock")


@dataclass
class TargetAlignedCondition:
    """A latent clip conditioned at a known target latent position.

    Deliberately not "a keyframe": one frame, a seven-position overlap and a
    future interior clip are the same object with a different `latent` and
    `target_latent_start`.
    """

    latent: torch.Tensor                 # [B, 24, T, H, W], B == 1
    target_latent_start: int
    label: str = ""
    position_policy: str = "copy_target"

    @property
    def latent_t(self):
        return int(self.latent.shape[2])

    def describe(self):
        return "%s: latent t=%d at target position %d (%s)" % (
            self.label or "condition", self.latent_t,
            self.target_latent_start, self.position_policy)


@dataclass
class TransformedLayout:
    """A `PackedLayout`-shaped object with condition rows spliced in.

    Duck-typed rather than subclassed: `MiniMaxH3Model._forward` only reads
    `seq_len`, `position_ids`, `img_pos`, `img_update`, `audio_pos`,
    `audio_update`, `segments` and `signature`.

    `signature` is carried through unchanged on purpose. `_forward` rebuilds the
    layout from scratch when the signature does not match the tensors it is
    denoising, so a transformed layout that reported its own inflated row count
    would be silently discarded on the first step.
    """

    seq_len: int
    position_ids: torch.Tensor
    img_pos: torch.Tensor
    img_update: torch.Tensor
    audio_pos: torch.Tensor
    audio_update: torch.Tensor
    segments: list
    signature: tuple
    condition_rows: int = 0
    condition_segments: list = field(default_factory=list)   # [(start, stop, label)]


def _segment_range(segments, kind, last=False):
    found = None
    for a, b, k in segments:
        if k == kind:
            found = (a, b)
            if not last:
                return found
    if found is None:
        raise ValueError("packed layout has no '%s' segment" % kind)
    return found


def frame_rows_of(layout):
    """Rows one target latent frame occupies, from the layout's own signature."""
    _, _, latent_h, latent_w = layout.signature[:4]
    return (latent_h // 2) * (latent_w // 2)


def _stock_position_ids(base, condition, frame_rows, text_len):
    """Pre-reference placement: the condition timeline starts at `text_len`.

    Reproduces the coordinate a stock first-frame keyframe would have received,
    generalized past one position by advancing the normal video spans from that
    origin.
    """
    from comfy.ldm.minimax.model import _video_t_grid

    target_start = _segment_range(base.segments, "video", last=True)[0]
    spatial = base.position_ids[target_start:target_start + frame_rows, 1:]
    t_grid = _video_t_grid(condition.latent_t, text_len)

    out = torch.empty(condition.latent_t * frame_rows, 3, dtype=base.position_ids.dtype)
    view = out.view(condition.latent_t, frame_rows, 3)
    view[:, :, 0] = t_grid[:, None].to(base.position_ids.dtype)
    view[:, :, 1:] = spatial[None]
    return out


def _copy_target_position_ids(base, condition, frame_rows):
    """Exact target placement: copy the rows the target position already has."""
    target_start = _segment_range(base.segments, "video", last=True)[0]
    row_start = target_start + condition.target_latent_start * frame_rows
    row_stop = row_start + condition.latent_t * frame_rows
    return base.position_ids[row_start:row_stop].clone()


def insert_target_conditions(base_layout, conditions, *, position_policy="copy_target"):
    """Return a new layout with `conditions` spliced in after the text segment.

    `position_policy` is the default for conditions that do not carry their own.
    With no conditions the base layout is returned untouched.
    """
    if not conditions:
        return base_layout
    if position_policy not in POSITION_POLICIES:
        raise ValueError("unknown position policy %r (expected one of %s)"
                         % (position_policy, ", ".join(POSITION_POLICIES)))

    text_len, target_latent_t = base_layout.signature[0], base_layout.signature[1]
    frame_rows = frame_rows_of(base_layout)
    text_start, text_stop = _segment_range(base_layout.segments, "text")
    if text_start != 0:
        raise ValueError("expected the text segment to lead the pack")
    target_start, target_stop = _segment_range(base_layout.segments, "video", last=True)

    validate_conditions(base_layout, conditions)

    insert_at = text_stop
    pieces, seg_rows, cond_segments = [], [], []
    row = insert_at
    for condition in conditions:
        policy = condition.position_policy or position_policy
        if policy == "copy_target":
            pos = _copy_target_position_ids(base_layout, condition, frame_rows)
        elif policy == "stock":
            pos = _stock_position_ids(base_layout, condition, frame_rows, text_len)
        else:
            raise ValueError("unknown position policy %r on condition %r"
                             % (policy, condition.label))
        n = condition.latent_t * frame_rows
        if pos.shape != (n, 3):
            raise AssertionError("condition %r produced %s position rows, expected (%d, 3)"
                                 % (condition.label, tuple(pos.shape), n))
        pieces.append(pos)
        seg_rows.append(n)
        cond_segments.append((row, row + n, condition.label))
        row += n

    inserted = sum(seg_rows)

    position_ids = torch.cat(
        [base_layout.position_ids[:insert_at], *pieces, base_layout.position_ids[insert_at:]])

    # Image rows are consumed in segment order, and the conditions now lead:
    # `all_video_rows[~img_update] = cond_video_rows` fills conditions then refs.
    cond_pos = torch.arange(insert_at, insert_at + inserted)
    shifted_img_pos = base_layout.img_pos + (base_layout.img_pos >= insert_at) * inserted
    img_pos = torch.cat([cond_pos, shifted_img_pos])
    img_update = torch.cat(
        [torch.zeros(inserted, dtype=torch.bool), base_layout.img_update])

    audio_pos = base_layout.audio_pos + (base_layout.audio_pos >= insert_at) * inserted

    segments = [(text_start, text_stop, "text")]
    offset = insert_at
    for n in seg_rows:
        segments.append((offset, offset + n, "cond"))
        offset += n
    for a, b, kind in base_layout.segments:
        if kind == "text":
            continue
        segments.append((a + inserted, b + inserted, kind))

    layout = TransformedLayout(
        seq_len=base_layout.seq_len + inserted,
        position_ids=position_ids,
        img_pos=img_pos,
        img_update=img_update,
        audio_pos=audio_pos,
        audio_update=base_layout.audio_update.clone(),
        segments=segments,
        signature=base_layout.signature,
        condition_rows=inserted,
        condition_segments=cond_segments,
    )
    _assert_consistent(layout, base_layout, conditions, frame_rows,
                       target_start + inserted, target_latent_t)
    return layout


def validate_conditions(base_layout, conditions):
    """Check every condition against the target it claims to align to."""
    _, target_latent_t, latent_h, latent_w = base_layout.signature[:4]
    seen = []
    for condition in conditions:
        t = condition.latent_t
        if condition.latent.ndim != 5:
            raise ValueError("condition %r latent must be [B, C, T, H, W], got %s"
                             % (condition.label, tuple(condition.latent.shape)))
        if condition.latent.shape[0] != 1:
            raise ValueError("condition %r must have batch size 1" % condition.label)
        if condition.target_latent_start < 0:
            raise ValueError("condition %r has a negative target position" % condition.label)
        if condition.target_latent_start + t > target_latent_t:
            raise ValueError(
                "condition %r covers target positions %d-%d but the target is only "
                "%d positions long" % (condition.label, condition.target_latent_start,
                                       condition.target_latent_start + t - 1, target_latent_t))
        if tuple(condition.latent.shape[-2:]) != (latent_h, latent_w):
            raise ValueError(
                "condition %r is %dx%d but the target canvas is %dx%d - the harness "
                "pins one canvas for exactly this reason"
                % (condition.label, condition.latent.shape[-2], condition.latent.shape[-1],
                   latent_h, latent_w))
        span = (condition.target_latent_start, condition.target_latent_start + t)
        for other_label, other in seen:
            if span[0] < other[1] and other[0] < span[1]:
                raise ValueError("conditions %r and %r overlap target positions"
                                 % (other_label, condition.label))
        seen.append((condition.label, span))
    return True


def _assert_consistent(layout, base, conditions, frame_rows, target_start, target_latent_t):
    """The invariants from the plan's §7.5, checked before any sampling happens.

    A failure here is an implementation bug, not a model result, so it must stop
    the experiment rather than produce an image nobody can interpret.
    """
    if layout.position_ids.shape[0] != layout.seq_len:
        raise AssertionError("position_ids has %d rows for seq_len %d"
                             % (layout.position_ids.shape[0], layout.seq_len))
    if layout.img_pos.shape[0] != layout.img_update.shape[0]:
        raise AssertionError("img_pos and img_update disagree (%d vs %d)"
                             % (layout.img_pos.shape[0], layout.img_update.shape[0]))
    if layout.audio_pos.shape[0] != layout.audio_update.shape[0]:
        raise AssertionError("audio_pos and audio_update disagree (%d vs %d)"
                             % (layout.audio_pos.shape[0], layout.audio_update.shape[0]))

    expected_cond_rows = sum(c.latent_t for c in conditions) * frame_rows
    base_cond_rows = int((~base.img_update).sum())
    if int((~layout.img_update).sum()) != base_cond_rows + expected_cond_rows:
        raise AssertionError(
            "condition row count mismatch: layout holds %d non-target image rows, "
            "expected %d base + %d inserted"
            % (int((~layout.img_update).sum()), base_cond_rows, expected_cond_rows))

    total = sum(b - a for a, b, _ in layout.segments)
    if total != layout.seq_len:
        raise AssertionError("segments cover %d rows, seq_len is %d" % (total, layout.seq_len))
    offset = 0
    for a, b, _ in layout.segments:
        if a != offset:
            raise AssertionError("segment table is not contiguous at row %d" % a)
        offset = b

    # exact alignment: each copy_target condition must equal its target rows in
    # the *shifted* layout, which is the property the whole transform exists for
    for condition, (start, stop, _) in zip(conditions, layout.condition_segments):
        if (condition.position_policy or "copy_target") != "copy_target":
            continue
        row_start = target_start + condition.target_latent_start * frame_rows
        row_stop = row_start + condition.latent_t * frame_rows
        if not torch.equal(layout.position_ids[start:stop],
                           layout.position_ids[row_start:row_stop]):
            raise AssertionError(
                "condition %r does not carry its target's position ids" % condition.label)

    if layout.signature[1] != target_latent_t:
        raise AssertionError("signature drifted during transform")


def condition_latents(conditions):
    """Condition latents in the row order `_cond_video_rows` consumes them."""
    return [c.latent for c in conditions]


def describe_layout(layout):
    counts = {}
    for a, b, kind in layout.segments:
        counts[kind] = counts.get(kind, 0) + (b - a)
    parts = ["seq_len=%d" % layout.seq_len]
    for kind in ("text", "cond", "ref_img", "ref_audio", "audio", "video"):
        if kind in counts:
            parts.append("%s=%d" % (kind, counts[kind]))
    return " ".join(parts)
