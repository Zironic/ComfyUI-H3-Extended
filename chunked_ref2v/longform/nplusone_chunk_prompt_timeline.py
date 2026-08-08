"""N+1-aware prompt timeline for native long-form continuation nodes.

Unlike the overlap-based ``MiniMax H3 Chunk Prompt Timeline (Zi)``, this plan
uses full generated chunks with no overlap/stride semantics. It owns the chunk
geometry, run seed, derived per-chunk seeds, and user-authored prompt timeline.
The consuming node resolves dynamic <Video N+1>/<Audio M+1> reference numbers
after the previous chunk has actually been generated.
"""

from __future__ import annotations

import json
import math

from comfy_api.latest import ComfyExtension, io
from ..geometry import AUDIO_LATENT_FPS, chunk_seed as derive_chunk_seed
from .nplusone_resume import legal_reference_frames, prompt_digest

FPS = 24
# 4: video and audio reference lengths are independent. The plan also owns the
# run seed, derived per-chunk seeds, and compiled-prompt digests.
PLAN_VERSION = 4
POLICY_AV_CONTINUATION = "previous_av_continuation"
NPlusOneChunkPromptPlan = io.Custom("H3_N_PLUS_ONE_CHUNK_PROMPT_PLAN")


def _validate_video_chunk_frames(chunk_frames):
    chunk_frames = int(chunk_frames)
    if chunk_frames < 22 or (chunk_frames - 5) % 17:
        raise ValueError("chunk_frames=%d is not a legal H3 VAE length" % chunk_frames)


def _nearest(value, candidates, *, prefer_larger_on_tie=True):
    if not candidates:
        raise ValueError("no legal reference-tail candidate")
    value = int(value)
    if prefer_larger_on_tie:
        return min(candidates, key=lambda item: (abs(item - value), -item))
    return min(candidates, key=lambda item: (abs(item - value), item))


def resolve_nplusone_reference_frames(chunk_frames, video_reference_frames):
    _validate_video_chunk_frames(chunk_frames)
    candidates = [
        value
        for value in legal_reference_frames(chunk_frames) + [int(chunk_frames)]
    ]
    return _nearest(
        video_reference_frames,
        candidates,
        prefer_larger_on_tie=True,
    )


def normalize_audio_reference_seconds(audio_reference_seconds):
    seconds = float(audio_reference_seconds)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("audio_reference_seconds must be positive")
    latents = int(round(seconds * AUDIO_LATENT_FPS))
    if latents <= 0:
        raise ValueError("audio_reference_seconds is below one audio latent")
    return latents


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


def _compiled_prompt(global_prompt, local_prompt, fallback_prompt=""):
    global_prompt = str(global_prompt or "").strip()
    local_prompt = str(local_prompt or "").strip()
    fallback_prompt = str(fallback_prompt or "").strip()
    if global_prompt and local_prompt:
        return global_prompt + "\n\n" + local_prompt
    return global_prompt or local_prompt or fallback_prompt


def build_nplusone_chunk_prompt_plan(
    *,
    output_seconds,
    chunk_frames,
    global_prompt="",
    chunk_prompts_json="",
    video_reference_frames=90,
    audio_reference_seconds=4.0,
    seed=0,
    continuation_policy=POLICY_AV_CONTINUATION,
):
    output_seconds = int(output_seconds)
    chunk_frames = int(chunk_frames)
    video_reference_frames = int(video_reference_frames)
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    if output_seconds <= 0:
        raise ValueError("output_seconds must be positive")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if video_reference_frames <= 0:
        raise ValueError("video_reference_frames must be positive")
    if continuation_policy != POLICY_AV_CONTINUATION:
        raise ValueError("unsupported N+1 continuation policy %r" % continuation_policy)

    target_frames = output_seconds * FPS
    chunk_count = _chunk_count(target_frames, chunk_frames)
    stored = _parse_prompt_store(chunk_prompts_json)
    prompts = stored[:chunk_count]
    prompts.extend("" for _ in range(chunk_count - len(prompts)))
    resolved_reference_frames = resolve_nplusone_reference_frames(
        chunk_frames, video_reference_frames,
    )
    requested_audio_reference_latents = normalize_audio_reference_seconds(audio_reference_seconds)
    audio_capacity = round(chunk_frames / float(FPS) * AUDIO_LATENT_FPS)
    audio_reference_latents = min(requested_audio_reference_latents, audio_capacity)
    plan = {
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
        "video_reference_frames": resolved_reference_frames,
        "audio_reference_seconds": float(audio_reference_seconds),
        "audio_reference_latents": audio_reference_latents,
        "seed": seed,
        "chunk_seeds": [derive_chunk_seed(seed, index) for index in range(chunk_count)],
    }
    plan["chunk_digests"] = [
        prompt_digest(_compiled_prompt(plan["global_prompt"], local))
        for local in plan["chunk_prompts"]
    ]
    return plan


def validate_nplusone_chunk_prompt_plan(
    plan,
    *,
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

    output_seconds = int(plan.get("output_seconds", -1))
    chunk_frames = int(plan.get("chunk_frames", -1))
    target_frames = int(plan.get("target_frames", -1))
    chunk_count = int(plan.get("chunk_count", -1))
    video_reference_frames = int(plan.get("video_reference_frames", -1))
    audio_reference_latents = int(plan.get("audio_reference_latents", -1))
    if int(plan.get("fps", -1)) != FPS:
        raise ValueError("N+1 chunk prompt plan fps is invalid")
    if output_seconds <= 0 or target_frames != output_seconds * FPS:
        raise ValueError("N+1 chunk prompt plan duration is invalid")
    if chunk_count != _chunk_count(target_frames, chunk_frames):
        raise ValueError("N+1 chunk prompt plan chunk geometry is invalid")
    if (
        resolve_nplusone_reference_frames(chunk_frames, video_reference_frames)
        != video_reference_frames
    ):
        raise ValueError("N+1 chunk prompt plan reference geometry is invalid")
    audio_capacity = round(chunk_frames / float(FPS) * AUDIO_LATENT_FPS)
    requested_audio_reference_latents = normalize_audio_reference_seconds(
        plan.get("audio_reference_seconds", audio_reference_latents / AUDIO_LATENT_FPS)
    )
    if audio_reference_latents <= 0 or audio_reference_latents > audio_capacity:
        raise ValueError("N+1 chunk prompt plan audio reference geometry is invalid")
    if audio_reference_latents != min(requested_audio_reference_latents, audio_capacity):
        raise ValueError("N+1 chunk prompt plan audio reference geometry is invalid")

    prompts = plan.get("chunk_prompts")
    if not isinstance(prompts, list) or len(prompts) != chunk_count:
        raise ValueError(
            "N+1 chunk prompt plan contains %d prompts; expected exactly %d"
            % (
                len(prompts) if isinstance(prompts, list) else 0,
                chunk_count,
            )
        )

    seed = int(plan.get("seed", 0)) & 0xFFFFFFFFFFFFFFFF
    chunk_seeds = plan.get("chunk_seeds")
    expected_seeds = [
        derive_chunk_seed(seed, index) for index in range(chunk_count)
    ]
    if list(chunk_seeds or []) != expected_seeds:
        # A plan whose seeds do not follow from its own seed would make the
        # resume scan reject chunks that are actually valid, or keep ones that
        # are not.
        raise ValueError(
            "N+1 chunk prompt plan chunk_seeds do not match seed=%d over %d chunks"
            % (seed, chunk_count)
        )

    compiled = [
        _compiled_prompt(plan.get("global_prompt"), item)
        for item in prompts
    ]
    expected_digests = [prompt_digest(text) for text in compiled]
    if list(plan.get("chunk_digests") or []) != expected_digests:
        raise ValueError(
            "N+1 chunk prompt plan chunk_digests do not match its compiled prompts"
        )

    normalized = dict(plan)
    normalized["global_prompt"] = str(plan.get("global_prompt") or "")
    normalized["chunk_prompts"] = [str(item or "") for item in prompts]
    normalized["seed"] = seed
    normalized["chunk_seeds"] = expected_seeds
    normalized["chunk_digests"] = expected_digests
    normalized["video_reference_frames"] = video_reference_frames
    normalized["audio_reference_latents"] = audio_reference_latents
    normalized["audio_reference_seconds"] = float(
        plan.get("audio_reference_seconds", audio_reference_latents / AUDIO_LATENT_FPS)
    )
    return normalized


def compile_nplusone_chunk_prompts(plan, fallback_prompt=""):
    return [
        _compiled_prompt(plan.get("global_prompt"), local, fallback_prompt)
        for local in plan["chunk_prompts"]
    ]


def prompts_for_av_continuation_plan(
    plan,
    fallback_prompt,
    *,
    output_seconds,
    chunk_frames,
    video_reference_frames=90,
    audio_reference_seconds=4.0,
):
    if plan is None:
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=output_seconds,
            chunk_frames=chunk_frames,
            global_prompt=fallback_prompt,
            video_reference_frames=video_reference_frames,
            audio_reference_seconds=audio_reference_seconds,
        )
    normalized = validate_nplusone_chunk_prompt_plan(plan)
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
                    "video_reference_frames",
                    default=90,
                    min=1,
                    max=362,
                    tooltip=(
                        "Tail video frames from each generated chunk used as dynamic "
                        "references for the next chunk. This is resolved to a legal "
                        "H3 VAE-group tail independently of audio."
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
                io.Float.Input(
                    "audio_reference_seconds",
                    default=4.0,
                    min=0.025,
                    max=60.0,
                    step=0.025,
                    tooltip=(
                        "Audio history in seconds. It is normalized once to the "
                        "integer 40 Hz audio_reference_latents in the plan."
                    ),
                ),
            ],
            outputs=[
                NPlusOneChunkPromptPlan.Output(
                    "n_plus_one_prompt_plan",
                    display_name="N+1 prompt plan",
                ),
                io.Int.Output("output_seconds", display_name="output seconds (legacy)"),
                io.Int.Output("chunk_frames", display_name="chunk frames (legacy)"),
                io.String.Output("report", display_name="report"),
                io.Int.Output(
                    "video_reference_frames", display_name="video reference frames"
                ),
                io.Int.Output("seed", display_name="seed"),
            ],
        )

    @classmethod
    def execute(
        cls,
        output_seconds=30,
        chunk_frames=141,
        global_prompt="",
        chunk_prompts_json="",
        video_reference_frames=90,
        audio_reference_seconds=4.0,
        seed=0,
    ) -> io.NodeOutput:
        plan = build_nplusone_chunk_prompt_plan(
            output_seconds=output_seconds,
            chunk_frames=chunk_frames,
            video_reference_frames=video_reference_frames,
            audio_reference_seconds=audio_reference_seconds,
            global_prompt=global_prompt,
            chunk_prompts_json=chunk_prompts_json,
            seed=seed,
        )
        lines = [
            "MiniMax H3 N+1 Chunk Prompt Timeline",
            "mode      previous <Video N+1> + <Audio M+1> AV continuation",
            "output    %d s / %d frames at %d fps"
            % (plan["output_seconds"], plan["target_frames"], plan["fps"]),
            "profile   C=%d; no overlap; stride=%d; video tail=%d; audio=%d latents"
            % (
                plan["chunk_frames"],
                plan["chunk_frames"],
                plan["video_reference_frames"],
                plan["audio_reference_latents"],
            ),
            "chunks    %d" % plan["chunk_count"],
            "seed      %d (per-chunk seeds derived; keep fixed to resume)"
            % plan["seed"],
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
            int(plan["video_reference_frames"]),
            int(plan["seed"]),
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
    "prompt_digest",
    "prompts_for_av_continuation_plan",
    "resolve_nplusone_reference_frames",
    "normalize_audio_reference_seconds",
    "validate_nplusone_chunk_prompt_plan",
]
