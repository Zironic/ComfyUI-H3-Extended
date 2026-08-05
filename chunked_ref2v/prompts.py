"""Prompt variants per policy.

Kept deliberately thin. The unmodified prompt is the control, and every variant
*appends* to it rather than rewriting it, so a difference between two arms is
the added sentence and not a reworded description of the edit.

`keyframe_completion` is the one variant Qwen cannot see the evidence for: the
carried latent rides as DiT condition rows only, never as a `<Video N>` item, so
the text is the only channel telling the model that the opening is already
decided. `video2` and `composite` describe references Qwen *does* see, so their
job is only to say what the extra footage is.
"""

KEYFRAME_COMPLETION_SUFFIX = (
    "The opening moment of this clip is already fixed; continue it smoothly from "
    "that state, preserving the motion already in progress."
)

VIDEO2_SUFFIX = (
    "<Video 2> is the already-generated opening of this clip. Continue directly "
    "from it, matching its appearance and motion, then follow <Video 1>."
)

COMPOSITE_SUFFIX = ""


def build_prompt(base_prompt, policy):
    """The prompt text for one policy.

    `composite` returns the prompt unchanged: `<Video 1>` is still the sole edit
    source, so the numbering the user wrote stays valid. That its opening frames
    are generated rather than original is recorded in the report instead - a
    property of the run, not something the prompt should be asserting.
    """
    base = (base_prompt or "").rstrip()
    if policy == "original":
        return base_prompt
    if policy == "composite":
        return base_prompt
    if policy == "keyframe_completion":
        suffix = KEYFRAME_COMPLETION_SUFFIX
    elif policy == "video2":
        suffix = VIDEO2_SUFFIX
    else:
        raise ValueError("unknown prompt policy %r" % policy)
    return ("%s\n\n%s" % (base, suffix)) if base else suffix


# Which Qwen encode a policy consumes. Policies sharing a key share one encode -
# that is the whole reason Phase D groups its work instead of re-running the
# 14.6 GB encoder per experiment.
PROMPT_ENCODE_KEY = {
    "original": "chunk_b",
    "keyframe_completion": "prompted",
    "video2": "video2",
    "composite": "composite",
}


def encode_key(policy):
    try:
        return PROMPT_ENCODE_KEY[policy]
    except KeyError:
        raise ValueError("unknown prompt policy %r" % policy) from None
