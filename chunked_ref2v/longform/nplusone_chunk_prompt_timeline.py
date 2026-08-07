"""N+1-aware prompt timeline for native long-form continuation nodes.

Unlike the overlap-based ``MiniMax H3 Chunk Prompt Timeline (Zi)``, this plan
uses full generated chunks with no overlap/stride semantics.  It stores only the
user-authored global/chunk instructions plus a continuation policy.  The
consuming long-form node resolves the dynamic <Video N+1>/<Audio M+1> reference
numbers after the previous chunk has actually been generated.
"""

from __future__ import annotations

import json
import math

from comfy_api.latest import ComfyExtension, io
from ..audio_boundary_profile import (
    legal_reference_tail_frames,
    validate_av_continuation_chunk_frames,
)
from ..geometry import chunk_seed as derive_chunk_seed

FPS = 24
# 3: the plan owns the run seed and the derived per-chunk seeds. Seed lives with
# the thing being edited, so a requeue after a prompt edit keeps it - which is
# what makes resuming an existing run possible at all.
PLAN_VERSION = 3
POLICY_AV_CONTINUATION = "previous_av_continuation"
REFERENCE_MIN_SECONDS = 2
REFERENCE_MAX_SECONDS = 15
REFERENCE_FRAMES_MIN = FPS * REFERENCE_MIN_SECONDS
REFERENCE_FRAMES_MAX = FPS * REFERENCE_MAX_SECONDS

NPlusOneChunkPromptPlan = io.Custom("H3_N_PLUS_ONE_CHUNK_PROMPT_PLAN")


def _nearest(value, candidates, *, prefer_larger_on_tie=True):
    if not candidates:
        raise ValueError("no legal reference-tail candidate")
    value = int(value)
    if prefer_larger_on_tie:
        return min(candidates, key=lambda item: (abs(item - value), -item))
    return min(candidates, key=lambda item: (abs(item - value), item))


def _resolve_reference_frames(chunk_frames, reference_frames):
    validate_av_continuation_chunk_frames(chunk_frames)
    candidates = legal_reference_tail_frames(chunk_frames)
    bounded = [
        value
        for value in candidates
        if REFERENCE_FRAMES_MIN <= value <= REFERENCE_FRAMES_MAX
    ]
    return _nearest(
        reference_frames,
        bounded or candidates,
        prefer_larger_on_tie=True,
    )


def _parse_prompt_store(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("N+1 chunk prompt data is not valid JSON") from exc
    if isinstance(value, dict):
        value = value.get("prompts", [])
    if not isinstance(value, list):
        raise ValueError("N+1 chunk prompt data must contain a prompts list")
    return [str(item or "") for item in value]


def _chunk_count(target_frames, chunk_frames):
    target_frames = int(target_frames)
    chunk_frames = int(chunk_frames)
    if target_frames <= 0 or chunk_frames <= 0:
        raise ValueError("target_frames and chunk_frames must be positive")
    return int(math.ceil(target_frames / float(chunk_frames)))


def build_nplusone_chunk_prompt_plan(
    *,
    output_seconds,
    chunk_frames,
    global_prompt="",
    chunk_prompts_json="",
    reference_frames=90,
    seed=0,
    continuation_policy=POLICY_AV_CONTINUATION,
):
    output_seconds = int(output_seconds)
    chunk_frames = int(chunk_frames)
    reference_frames = int(reference_frames)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    if output_seconds <= 0:
        raise ValueError("output_seconds must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if reference_frames <= 0:
        raise ValueError("reference_frames must be positive")
    if continuation_policy != POLICY_AV_CONTINUATION:
        raise ValueError("unsupported N+1 continuation policy %r" % continuation_policy)

    target_frames = output_seconds * FPS
    chunk_count = _chunk_count(target_frames, chunk_frames)
    stored = _parse_prompt_store(chunk_prompts_json)
    prompts = stored[:chunk_count]
    prompts.extend("" for _ in range(chunk_count - len(prompts)))
    resolved_reference_frames = _resolve_reference_frames(chunk_frames, reference_frames)
    return {
        "version": PLAN_VERSION,
        "schedule": "full_chunks",
        "continuation_policy": continuation_policy,
        "fps": FPS,
        "output_seconds": output_seconds,
        "target_frames": target_frames,
        "chunk_frames": chunk_frames,
        "chunk_count": chunk_count,
        "global_prompt": str(global_prompt or ""),
        "chunk_prompts": prompts,
        "reference_frames": resolved_reference_frames,
        "seed": seed,
        "chunk_seeds": [derive_chunk_seed(seed, index) for index in range(chunk_count)],
    }


def validate_nplusone_chunk_prompt_plan(
    plan,
    *,
    output_seconds,
    chunk_frames,
    reference_frames=90,
    continuation_policy=POLICY_AV_CONTINUATION,
):
    if not isinstance(plan, dict):
        raise TypeError(
            "N+1 chunk prompt plan must come from MiniMax H3 N+1 Chunk Prompt Timeline (Zi)"
        )
    if int(plan.get("version", -1)) != PLAN_VERSION:
        raise ValueError("unsupported N+1 chunk prompt plan version")
    if plan.get("schedule") != "full_chunks":
        raise ValueError("N+1 chunk prompt plan must use full_chunks scheduling")
    if plan.get("continuation_policy") != continuation_policy:
        raise ValueError(
            "N+1 plan continuation_policy=%r does not match required policy=%r"
            % (plan.get("continuation_policy"), continuation_policy)
        )

    expected = build_nplusone_chunk_prompt_plan(
        output_seconds=output_seconds,
        chunk_frames=chunk_frames,
        reference_frames=reference_frames,
        continuation_policy=continuation_policy,
    )
    for key in (
        "fps",
        "output_seconds",
        "target_frames",
        "chunk_frames",
        "chunk_count",
        "reference_frames",
    ):
        if int(plan.get(key, -1)) != int(expected[key]):
            raise ValueError(
                "N+1 chunk prompt plan %s=%r does not match downstream node %s=%r"
                % (key, plan.get(key), key, expected[key])
            )

    prompts = plan.get("chunk_prompts")
    if not isinstance(prompts, list) or len(prompts) != expected["chunk_count"]:
        raise ValueError(
            "N+1 chunk prompt plan contains %d prompts; expected exactly %d"
            % (
                len(prompts) if isinstance(prompts, list) else 0,
                expected["chunk_count"],
            )
        )

    seed = int(plan.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF
    chunk_seeds = plan.get("chunk_seeds")
    expected_seeds = [
        derive_chunk_seed(seed, index) for index in range(expected["chunk_count"])
    ]
    if list(chunk_seeds or []) != expected_seeds:
        # A plan whose seeds do not follow from its own seed would make the
        # resume scan reject chunks that are actually valid, or keep ones that
        # are not.
        raise ValueError(
            "N+1 chunk prompt plan chunk_seeds do not match seed=%d over %d chunks"
            % (seed, expected["chunk_count"])
        )

    normalized = dict(plan)
    normalized["global_prompt"] = str(plan.get("global_prompt") or "")
    normalized["chunk_prompts"] = [str(item or "") for item in prompts]
    normalized["seed"] = seed
    normalized["chunk_seeds"] = expected_seeds
    return normalized


def compile_nplusone_chunk_prompts(plan, fallback_prompt=""):
    global_prompt = str(plan.get("global_prompt") or "").strip()
    fallback = str(fallback_prompt or "").strip()
    compiled = []
    for local in plan["chunk_prompts"]:
        local = str(local or "").strip()
        if global_prompt and local:
            text = global_prompt + "\n\n" + local
        else:
            text = global_prompt or local or fallback
        compiled.append(text)
    return compiled


def prompts_for_av_continuation_plan(
    plan,
    fallback_prompt,
    *,
    output_seconds,
    chunk_frames,
    reference_frames=90,
):
    if plan is None:
        count = _chunk_count(int(output_seconds) * FPS, int(chunk_frames))
        _resolve_reference_frames(chunk_frames, reference_frames)
        return [str(fallback_prompt or "")] * count
    normalized = validate_nplusone_chunk_prompt_plan(
        plan,
        output_seconds=output_seconds,
        chunk_frames=chunk_frames,
        reference_frames=reference_frames,
        continuation_policy=POLICY_AV_CONTINUATION,
    )
    return compile_nplusone_chunk_prompts(normalized, fallback_prompt)


class MiniMaxH3NPlusOneChunkPromptTimeline(io.ComfyNode):
    """Build one base instruction per full generated N+1 continuation chunk."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3NPlusOneChunkPromptTimelineZi",
            display_name="MiniMax H3 N+1 Chunk Prompt Timeline (Zi)",
            category="model/video/minimax/testing",
            description=(
                "Builds a no-overlap chunk timeline for N+1 continuation. The node "
                "stores user-authored base prompts only; the downstream continuation "
                "node injects the generated <Video N+1>/<Audio M+1> relationship at "
                "runtime after the preceding chunk exists."
            ),
            inputs=[
                io.Int.Input(
                    "output_seconds",
                    default=30,
                    min=1,
                    max=3600,
                    tooltip="Desired final duration at MiniMax H3's fixed 24 fps.",
                ),
                io.Int.Input(
                    "chunk_frames",
                    default=141,
                    min=22,
                    max=362,
                    step=17,
                    tooltip=(
                        "Frames generated by each continuation invocation. Every chunk "
                        "contributes all generated frames; there is no overlap."
                    ),
                ),
                io.String.Input(
                    "global_prompt",
                    default="",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "Optional text prepended to every chunk's requested action. "
                        "Do not manually describe <Video N+1>/<Audio M+1>; the "
                        "continuation node adds those dynamic references at runtime."
                    ),
                ),
                io.String.Input(
                    "chunk_prompts_json",
                    default='{"version":2,"prompts":[]}',
                    multiline=True,
                    dynamic_prompts=False,
                    tooltip="Internal timeline storage managed by the node editor.",
                ),
                io.Int.Input(
                    "reference_frames",
                    default=90,
                    min=1,
                    max=362,
                    tooltip=(
                        "Tail frames from each generated chunk used as dynamic "
                        "references for the next chunk. This is resolved to the "
                        "same AV-aligned set used by continuation execution."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "Run seed. Per-chunk seeds derive from it, and it travels "
                        "in the plan. Keep it fixed to requeue after editing a "
                        "chunk prompt: a changed seed rerolls every chunk, while an "
                        "unchanged one lets completed chunks be reused."
                    ),
                ),
            ],
            outputs=[
                NPlusOneChunkPromptPlan.Output(
                    "n_plus_one_prompt_plan",
                    display_name="N+1 prompt plan",
                ),
                io.Int.Output("output_seconds", display_name="output seconds"),
                io.Int.Output("chunk_frames", display_name="chunk frames"),
                io.String.Output("report", display_name="report"),
                io.Int.Output("reference_frames", display_name="reference frames"),
            ],
        )

    @classmethod
    def execute(
        cls,
        output_seconds=30,
        chunk_frames=141,
        global_prompt="",
        chunk_prompts_json="",
        reference_frames=90,
    ) -> io.NodeOutput:
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=output_seconds,
            chunk_frames=chunk_frames,
            reference_frames=reference_frames,
            global_prompt=global_prompt,
            chunk_prompts_json=chunk_prompts_json,
        )
        lines = [
            "MiniMax H3 N+1 Chunk Prompt Timeline",
            "mode      previous <Video N+1> + <Audio M+1> AV continuation",
            "output    %d s / %d frames at %d fps"
            % (plan["output_seconds"], plan["target_frames"], plan["fps"]),
            "profile   C=%d; no overlap; stride=%d; R=%d"
            % (
                plan["chunk_frames"],
                plan["chunk_frames"],
                plan["reference_frames"],
            ),
            "chunks    %d" % plan["chunk_count"],
        ]
        for index, prompt in enumerate(plan["chunk_prompts"]):
            start = index * plan["chunk_frames"]
            stop = min(start + plan["chunk_frames"], plan["target_frames"])
            preview = " ".join(str(prompt or "").strip().split())
            if len(preview) > 72:
                preview = preview[:69] + "..."
            dynamic = "static refs only" if index == 0 else "runtime N+1 AV reference"
            lines.append(
                "chunk %03d frames %d-%d (%.3f-%.3f s) [%s]: %s"
                % (
                    index,
                    start,
                    max(start, stop - 1),
                    start / plan["fps"],
                    stop / plan["fps"],
                    dynamic,
                    preview or "<global/fallback only>",
                )
            )
        return io.NodeOutput(
            plan,
            int(output_seconds),
            int(chunk_frames),
            "\n".join(lines),
            int(plan["reference_frames"]),
        )


class MiniMaxH3NPlusOneChunkPromptTimelineExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3NPlusOneChunkPromptTimeline]


__all__ = [
    "MiniMaxH3NPlusOneChunkPromptTimeline",
    "MiniMaxH3NPlusOneChunkPromptTimelineExtension",
    "NPlusOneChunkPromptPlan",
    "PLAN_VERSION",
    "POLICY_AV_CONTINUATION",
    "build_nplusone_chunk_prompt_plan",
    "compile_nplusone_chunk_prompts",
    "prompts_for_av_continuation_plan",
    "validate_nplusone_chunk_prompt_plan",
]
