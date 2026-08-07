"""Bridge geometry, arm catalog and the prompt wording the arms share.

`17k+5` constrains an *independently encoded* H3 segment, not a reference
interval sliced out of one that is already valid. `ref_builder.snap_video_frames`
exists because a standalone `vae.encode(n)` normalizes onto the presentation
lattice; it is not a statement about what the DiT can be handed as a reference.

So this experiment never encodes a 51-frame clip. It encodes C1 and C3 whole -
both legal 17k+5 lengths - and slices the reference intervals out of the
resulting latents, the way `_slice_dynamic_av_reference` already does for N+1.
Two reasons that matters:

- The stock builder head-trims (`frames[:n]`), which on a *left* reference drops
  the frames nearest the seam - the only ones the experiment is about - and it
  does not trim the paired audio to match the snapped video, so a 51-frame AV
  reference silently desynchronizes by 12 frames.
- The eventual regeneration system will use the N+1 slicing path. Testing
  against the stock builder would measure an implementation difference rather
  than the model.

Reference lengths therefore only need to be exact on both grids, which for
R=51 they are: 51 latent-covers a whole number of positions from either end of a
141-frame encode, and 51/24*40 = 85 audio latents.
"""

from dataclasses import dataclass, field

from ..geometry import (
    AUDIO_LATENT_FPS,
    FPS,
    UnalignedProfileError,
    audio_boundary_is_exact,
    find_exact_overlap_slice,
    latent_frame_spans,
    video_latent_t,
)

RIGHT_REF_POLICIES = ("none", "natural", "counterfactual")
PROMPT_POLICIES = ("continuation", "bridge", "bridge_swapped", "neutral")
REF_ORDERS = ("past_first", "future_first")


def head_latent_slice(latent_t, frames):
    """`(0, count)` covering exactly `frames` from the start of an encode.

    `find_exact_overlap_slice` answers the same question from the tail. A head
    slice is the right reference: C3's opening, not its ending.
    """
    spans = latent_frame_spans(latent_t)
    cumulative = 0
    for i, span in enumerate(spans):
        cumulative += span
        if cumulative == frames:
            return 0, i + 1
        if cumulative > frames:
            break
    raise UnalignedProfileError(
        "%d frames does not land on a latent-position boundary from the head "
        "for latent_t=%d" % (frames, latent_t))


def tail_latent_slice(latent_t, total_frames, frames):
    """`(start, count)` covering exactly the final `frames` of an encode."""
    return find_exact_overlap_slice(latent_t, total_frames - frames, frames)


def audio_latent_span(start_frame, frames, fps=FPS):
    """`(start, count)` audio latent positions for a frame interval.

    Raises rather than rounding. A left reference that is a third of a latent
    out of step with its own video is exactly the defect this experiment would
    otherwise blame on the model.
    """
    for label, value in (("start", start_frame), ("length", frames)):
        if (int(value) * AUDIO_LATENT_FPS) % int(fps):
            raise UnalignedProfileError(
                "reference %s of %d frames is not audio exact at %d fps"
                % (label, value, fps))
    start = int(start_frame) * AUDIO_LATENT_FPS // int(fps)
    count = int(frames) * AUDIO_LATENT_FPS // int(fps)
    return start, count


@dataclass(frozen=True)
class BridgeArm:
    arm_id: str
    display_name: str
    right_ref: str = "none"
    prompt_policy: str = "continuation"
    ref_order: str = "past_first"
    notes: str = ""

    def as_dict(self):
        return {
            "arm_id": self.arm_id,
            "display_name": self.display_name,
            "right_ref": self.right_ref,
            "prompt_policy": self.prompt_policy,
            "ref_order": self.ref_order,
            "notes": self.notes,
        }


ARMS = {}


def _register(arm):
    if arm.right_ref not in RIGHT_REF_POLICIES:
        raise ValueError("bad right_ref policy on %s" % arm.arm_id)
    if arm.prompt_policy not in PROMPT_POLICIES:
        raise ValueError("bad prompt policy on %s" % arm.arm_id)
    if arm.ref_order not in REF_ORDERS:
        raise ValueError("bad ref order on %s" % arm.arm_id)
    # A swapped presentation with unswapped definitions would tell the model
    # that the following part is the preceding one. That is a different (and
    # useless) experiment, so it is refused rather than allowed by accident.
    swapped_order = arm.ref_order == "future_first"
    swapped_text = arm.prompt_policy == "bridge_swapped"
    if arm.right_ref != "none" and swapped_order != swapped_text:
        raise ValueError(
            "%s mixes ref_order=%s with prompt_policy=%s; the definitions would "
            "no longer describe the clips" % (arm.arm_id, arm.ref_order,
                                              arm.prompt_policy))
    ARMS[arm.arm_id] = arm
    return arm


_register(BridgeArm(
    arm_id="A_left_only",
    display_name="A - left continuation only",
    right_ref="none",
    prompt_policy="continuation",
    notes="Control. Whatever landing A achieves is what C1 alone implies, and "
          "C3 is C1's natural future - so A already lands near C3.",
))

_register(BridgeArm(
    arm_id="B_natural",
    display_name="B - natural right reference",
    right_ref="natural",
    prompt_policy="bridge",
    notes="C3's opening as <Video 2>, described as chronologically after the "
          "target.",
))

_register(BridgeArm(
    arm_id="C_counterfactual",
    display_name="C - counterfactual right reference",
    right_ref="counterfactual",
    prompt_policy="bridge",
    notes="Byte-identical prompt to B. Only the <Video 2>/<Audio 2> content "
          "differs. B vs C is the whole experiment.",
))

_register(BridgeArm(
    arm_id="B_swapped",
    display_name="B swapped - future presented first",
    right_ref="natural",
    prompt_policy="bridge_swapped",
    ref_order="future_first",
    notes="Same two clips as B, slots exchanged, definitions rewritten so they "
          "still describe the clips truthfully. B produced <Video 2>'s content "
          "then <Video 1>'s - a reverse-order replay rather than a bridge. If "
          "that is keyed to slot position this arm inverts it; if it is keyed to "
          "the text roles or to content, the output stays as it was.",
))

_register(BridgeArm(
    arm_id="D_no_chronology",
    display_name="D - natural right reference, no chronology wording",
    right_ref="natural",
    prompt_policy="neutral",
    notes="Optional. Separates 'the footage did it' from 'the wording did it'. "
          "Drop this arm first if budget is tight.",
))


# Cheapest first: A carries one reference block, the rest carry two. An OOM on a
# two-reference arm then cannot destroy the control.
ARM_ORDER = ["A_left_only", "B_natural", "B_swapped", "C_counterfactual",
             "D_no_chronology"]

SUITES = {
    "swap": ["B_natural", "B_swapped"],
    "decisive": ["A_left_only", "B_natural", "C_counterfactual"],
    "full": list(ARM_ORDER),
    "plumbing": ["B_natural"],
}
SUITE_NAMES = list(SUITES) + ["custom"]


# ---------------------------------------------------------------- prompts

# The tag vocabulary is the one the source run actually used. Inventing a
# variant here would be an untested variable riding along with the experiment.
_TAG = "[video continuation + reference generation]"

_SUBJECT_DEF = "<Subject 1> is the woman shown in <Video 1>."

_CONTINUATION_DEFS = (
    _SUBJECT_DEF + "\n"
    "<Video 1> is the immediately preceding part of the same continuous source "
    "video. Its final frame and motion state occur immediately before the target "
    "video begins.\n"
    "<Audio 1> is the synchronized audio of <Video 1> and ends immediately "
    "before the target audio begins."
)

_BRIDGE_DEFS = _CONTINUATION_DEFS + (
    "\n<Video 2> is the immediately following part of the same continuous source "
    "video. Its first frame and motion state occur immediately after the target "
    "video ends.\n"
    "<Audio 2> is the synchronized audio of <Video 2> and begins immediately "
    "after the target audio ends."
)

# B_swapped: the same two clips in exchanged slots. Every role statement is
# inverted with them, so the text still describes what is actually there - only
# the slot each clip occupies has changed.
_SWAPPED_DEFS = (
    _SUBJECT_DEF + "\n"
    "<Video 1> is the immediately following part of the same continuous source "
    "video. Its first frame and motion state occur immediately after the target "
    "video ends.\n"
    "<Audio 1> is the synchronized audio of <Video 1> and begins immediately "
    "after the target audio ends.\n"
    "<Video 2> is the immediately preceding part of the same continuous source "
    "video. Its final frame and motion state occur immediately before the target "
    "video begins.\n"
    "<Audio 2> is the synchronized audio of <Video 2> and ends immediately "
    "before the target audio begins."
)

_SWAPPED_SUMMARY = (
    _TAG + " The target video fills the missing continuous interval between "
    "<Video 2> and <Video 1>. It continues directly forward from the end of "
    "<Video 2> and reaches a compatible final visual and motion state that "
    "continues directly into the beginning of <Video 1>."
)

_SWAPPED_TAIL = (
    "The target begins immediately after the final moment of <Video 2>, "
    "preserving its subject state, camera trajectory, motion direction and "
    "phase, environment, and lighting. The action proceeds continuously through "
    "the requested intermediate events. By the end of the target, the subject "
    "state, spatial arrangement, camera position and motion, and ongoing action "
    "approach the state immediately preceding the beginning of <Video 1>, so "
    "that playback can continue directly into <Video 1> without a visible "
    "restart."
)

# Arm D: the same two references, presented without any chronological claim.
_NEUTRAL_DEFS = _CONTINUATION_DEFS + (
    "\n<Video 2> is a reference video of the same subject and environment.\n"
    "<Audio 2> is the synchronized audio of <Video 2>."
)

_CONTINUATION_SUMMARY = (
    _TAG + " The target video continues directly forward from the end of "
    "<Video 1>."
)

_BRIDGE_SUMMARY = (
    _TAG + " The target video fills the missing continuous interval between "
    "<Video 1> and <Video 2>. It continues directly forward from the end of "
    "<Video 1> and reaches a compatible final visual and motion state that "
    "continues directly into the beginning of <Video 2>."
)

_NEUTRAL_SUMMARY = _CONTINUATION_SUMMARY

_RETENTION = (
    "<Subject 1> fully_preserved - Her identity, outfit, hair and appearance "
    "remain consistent with <Video 1>."
)

_CONTINUATION_TAIL = (
    "The target begins immediately after the final moment of <Video 1>, "
    "preserving its subject state, camera trajectory, motion direction and "
    "phase, environment, and lighting."
)

_BRIDGE_TAIL = _CONTINUATION_TAIL + (
    " The action proceeds continuously through the requested intermediate "
    "events. By the end of the target, the subject state, spatial arrangement, "
    "camera position and motion, and ongoing action approach the state "
    "immediately preceding the beginning of <Video 2>, so that playback can "
    "continue directly into <Video 2> without a visible restart."
)

_POLICY_TEXT = {
    "continuation": (_CONTINUATION_DEFS, _CONTINUATION_SUMMARY, _CONTINUATION_TAIL),
    "bridge": (_BRIDGE_DEFS, _BRIDGE_SUMMARY, _BRIDGE_TAIL),
    "bridge_swapped": (_SWAPPED_DEFS, _SWAPPED_SUMMARY, _SWAPPED_TAIL),
    "neutral": (_NEUTRAL_DEFS, _NEUTRAL_SUMMARY, _CONTINUATION_TAIL),
}


def is_formatted(prompt):
    """Whether a prompt is already a full H3 chunk prompt.

    A real chunk prompt carries its own `summary:` and `detailed_description:`
    headers. Wrapping one in another `detailed_description:` would nest the whole
    thing and destroy it, so the two cases have to be told apart.
    """
    body = (prompt or "").lower()
    return "summary:" in body and "detailed_description:" in body


def build_prompt(base_prompt, policy):
    """Assemble an arm's prompt in the house format.

    Two shapes go in. A bare description gets wrapped in the four sections the
    long-form runs use. An already-formatted chunk prompt is left completely
    alone except for the `subject_definitions:` block, which is where the
    `<Video n>` roles have to be declared and which a chunk prompt never carries
    (the long-form node supplies it separately as the global prompt).

    B and C both resolve to `bridge`, so their prompts are identical strings by
    construction - the only difference between those arms is what the second
    reference contains. `assert_shared_wording` proves that at run time.
    """
    try:
        defs, summary, tail = _POLICY_TEXT[policy]
    except KeyError:
        raise ValueError("unknown prompt policy %r" % policy) from None
    body = (base_prompt or "").strip()

    if is_formatted(body):
        return "subject_definitions:\n%s\n\n%s" % (defs, body)

    return (
        "subject_definitions:\n%s\n\n"
        "summary:\n%s\n\n"
        "retention_analysis:\n%s\n\n"
        "detailed_description:\n%s\n%s" % (defs, summary, _RETENTION, body, tail)
    ).strip()


def assert_shared_wording(prompts_by_arm):
    """B and C must differ only in reference content, never in text."""
    b = prompts_by_arm.get("B_natural")
    c = prompts_by_arm.get("C_counterfactual")
    if b is not None and c is not None and b != c:
        raise RuntimeError(
            "arms B and C have different prompts; the comparison would confound "
            "wording with reference content")


# ---------------------------------------------------------------- plan

@dataclass(frozen=True)
class BridgePlan:
    """One held-out interval and the two intervals bracketing it.

        |---- C1 (chunk_frames) ----|---- target ----|---- C3 ----|
                       [left R]      held out         [right R]

    `bridge_start` is the first held-out frame; C1 is the `chunk_frames` before
    it and C3 the `chunk_frames` after, so both are legal standalone encodes.
    """

    chunk_frames: int
    ref_frames: int
    bridge_start: int
    fps: int = FPS
    counterfactual_start: int = -1
    notes: list = field(default_factory=list)

    @property
    def context_latent_t(self):
        return video_latent_t(self.chunk_frames)

    @property
    def left_start(self):
        return self.bridge_start - self.chunk_frames

    @property
    def right_start(self):
        return self.bridge_start + self.chunk_frames

    @property
    def required_frames(self):
        return self.right_start + self.chunk_frames

    def left_video_slice(self):
        return tail_latent_slice(self.context_latent_t, self.chunk_frames,
                                 self.ref_frames)

    def right_video_slice(self):
        return head_latent_slice(self.context_latent_t, self.ref_frames)

    def left_audio_slice(self):
        return audio_latent_span(self.chunk_frames - self.ref_frames,
                                 self.ref_frames, self.fps)

    def right_audio_slice(self):
        return audio_latent_span(0, self.ref_frames, self.fps)

    def describe(self):
        lv, lc = self.left_video_slice()
        rv, rc = self.right_video_slice()
        la, lac = self.left_audio_slice()
        ra, rac = self.right_audio_slice()
        return (
            "C=%d R=%d, held out [%d:%d]\n"
            "  left  pixels C1[%d:%d]  video latent [%d:%d] (%d)  audio [%d:%d] (%d)\n"
            "  right pixels C3[0:%d]   video latent [%d:%d] (%d)  audio [%d:%d] (%d)"
            % (self.chunk_frames, self.ref_frames, self.bridge_start,
               self.bridge_start + self.chunk_frames,
               self.chunk_frames - self.ref_frames, self.chunk_frames,
               lv, lv + lc, lc, la, la + lac, lac,
               self.ref_frames, rv, rv + rc, rc, ra, ra + rac, rac))


def resolve_plan(*, chunk_frames, ref_frames, bridge_start, total_frames,
                 counterfactual_start=-1, fps=FPS):
    """Validate a bridge geometry, or explain precisely why it is illegal."""
    chunk_frames = int(chunk_frames)
    ref_frames = int(ref_frames)
    bridge_start = int(bridge_start)

    if chunk_frames % 17 != 5:
        raise UnalignedProfileError(
            "chunk_frames=%d is not a legal H3 generation length (needs "
            "n %% 17 == 5)" % chunk_frames)
    if not audio_boundary_is_exact(chunk_frames, fps=fps):
        raise UnalignedProfileError(
            "chunk_frames=%d does not end on an audio latent boundary "
            "(needs n %% 3 == 0 at %d fps)" % (chunk_frames, fps))
    if not 0 < ref_frames < chunk_frames:
        raise ValueError("ref_frames must be in (0, chunk_frames)")

    plan = BridgePlan(chunk_frames=chunk_frames, ref_frames=ref_frames,
                      bridge_start=bridge_start, fps=fps,
                      counterfactual_start=int(counterfactual_start))

    # Both ends and both grids, up front. A geometry error here is silent and
    # asymmetric once it reaches the model.
    plan.left_video_slice()
    plan.right_video_slice()
    plan.left_audio_slice()
    plan.right_audio_slice()

    if plan.left_start < 0:
        raise ValueError(
            "bridge_start=%d leaves no room for a %d-frame C1"
            % (bridge_start, chunk_frames))
    if plan.required_frames > int(total_frames):
        raise ValueError(
            "source has %d frames; this plan needs %d (C1 + target + C3)"
            % (int(total_frames), plan.required_frames))
    if bridge_start % 3:
        raise UnalignedProfileError(
            "bridge_start=%d is not on the shared audio grid (needs a multiple "
            "of 3 at 24 fps)" % bridge_start)
    return plan


def resolve_suite(suite, custom_arms=""):
    if suite == "custom":
        ids = [x.strip() for x in custom_arms.split(",") if x.strip()]
        if not ids:
            raise ValueError("arm_suite is 'custom' but custom_arms is empty")
    else:
        try:
            ids = list(SUITES[suite])
        except KeyError:
            raise ValueError("unknown suite %r (known: %s)"
                             % (suite, ", ".join(sorted(SUITES)))) from None
    unknown = [i for i in ids if i not in ARMS]
    if unknown:
        raise ValueError("unknown arm id(s): %s (known: %s)"
                         % (", ".join(unknown), ", ".join(sorted(ARMS))))
    order = {name: i for i, name in enumerate(ARM_ORDER)}
    return sorted(dict.fromkeys(ids), key=lambda i: order.get(i, len(order)))
