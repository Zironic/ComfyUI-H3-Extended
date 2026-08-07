"""The bridge experiment node.

One node for the same reason the Ref2V harness is one node: the arm count
varies and a Comfy graph cannot grow an output socket per arm. Outputs are a
seam playback clip for the selected arm, the report, and the manifest path.
"""

import logging

import torch

import torchaudio

import comfy.model_management
import nodes as comfy_nodes
from comfy_api.latest import ComfyExtension, io

try:
    from ...cond_cache import MODES as COND_CACHE_MODES
except ImportError:  # self-tests import the package as a top-level module
    from cond_cache import MODES as COND_CACHE_MODES

from .. import comparison, ref_builder
from ..geometry import FPS
from ..longform import audio_output
from . import runner
from .plan import ARMS, SUITE_NAMES, resolve_plan, resolve_suite

LOG_PREFIX = "[H3 Extended] bridge"


def _fmt(value, spec="%.4f"):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return spec % value
    return str(value)


def format_report(plan, results, notes, *, seed, arm_ids, columns=None,
                  selected=None, window=None):
    lines = [
        "MiniMax H3 - two-sided AV bridge",
        "=" * 64,
        plan.describe(),
        "seed %d, arms: %s" % (seed, ", ".join(arm_ids)),
        "",
    ]
    if columns:
        lines.append("comparison columns, left to right:")
        for i, label in enumerate(columns):
            lines.append("  %d. %s" % (i + 1, label))
        if window:
            lines.append("  each column: %d real frames | %d generated | %d real"
                         % (window, plan.chunk_frames, window))
            lines.append("  so the seams sit at frame %d and frame %d in every column"
                         % (window, window + plan.chunk_frames))
        lines.append("")
    if selected:
        lines.extend(["selected_arm / selected_arm_audio show: %s" % selected, ""])
    lines.extend("  " + note for note in notes)
    lines.append("")

    header = ("%-20s %10s %10s %10s %10s %10s"
              % ("arm", "L seam", "L ratio", "R seam", "end dx", "dx err"))
    lines.append(header)
    lines.append("-" * len(header))
    for arm_id in arm_ids:
        record = results.get(arm_id)
        if not record:
            continue
        if record.get("status") != "completed":
            lines.append("%-20s  %s" % (arm_id, record.get("error", record.get("status"))))
            continue
        m = record["metrics"]
        lines.append("%-20s %10s %10s %10s %10s %10s" % (
            arm_id,
            _fmt(m.get("left_seam_mae")),
            _fmt(m.get("left_seam_ratio"), "%.2f"),
            _fmt(m.get("right_seam_mae_natural")),
            _fmt(m.get("ending_dx"), "%+.2f"),
            _fmt(m.get("dx_error_natural"), "%.2f"),
        ))

    decisive = results.get("_decisive")
    lines.extend(["", "B vs C - the decisive comparison", "-" * 40])
    if not decisive:
        lines.append("  not available (needs both B_natural and C_counterfactual)")
    else:
        lines.append("  natural future dx        %s" % _fmt(decisive["natural_dx"], "%+.2f"))
        lines.append("  counterfactual future dx %s" % _fmt(decisive["counterfactual_dx"], "%+.2f"))
        lines.append("  B ending dx              %s" % _fmt(decisive["b_ending_dx"], "%+.2f"))
        lines.append("  C ending dx              %s" % _fmt(decisive["c_ending_dx"], "%+.2f"))
        lines.append("  swing C-B                %s px" % _fmt(decisive["dx_swing_px"], "%+.2f"))
        lines.append("  swing / target spread    %s" % _fmt(decisive["dx_swing_normalized"], "%+.3f"))
        lines.append("  left seam preserved      %s" % _fmt(decisive["left_seam_preserved"]))
        normalized = decisive["dx_swing_normalized"]
        if normalized is None:
            verdict = ("the two candidate futures do not differ in horizontal "
                       "motion; pick a counterfactual that does")
        elif normalized > 0.25:
            verdict = ("C's ending moved toward the counterfactual future. "
                       "Evidence that <Video 2> acts as a future constraint.")
        elif normalized < -0.25:
            verdict = ("C's ending moved away from the counterfactual. "
                       "Unexpected; inspect before drawing conclusions.")
        else:
            verdict = ("no detectable swing. Consistent with <Video 2> being "
                       "ordinary conditioning - but a null at one seed is weak "
                       "evidence; replicate before closing the branch.")
        lines.extend(["", "  " + verdict])

    lines.extend([
        "",
        "ground-truth MAE is reported per arm in the manifest. It is a sanity "
        "reading, not a success criterion.",
    ])
    return "\n".join(lines)


def _seam_playback(frames, context, window):
    """Left tail + a bridge + right head, so both seams sit in one clip.

    Every column is built the same way, so the two seams land on the same frame
    index in every column and a scrub lines them up.
    """
    parts = [context.left["pixels"][-window:], frames,
             context.right_natural["pixels"][:window]]
    return torch.cat([p.to("cpu", torch.float32) for p in parts])


def _comparison(context, results, arm_ids, window, max_long_edge):
    """Ground truth beside every completed arm, in one IMAGE batch."""
    clips = [("ground truth (held out)",
              _seam_playback(context.ground_truth, context, window))]
    for arm_id in arm_ids:
        record = results.get(arm_id)
        if record and record.get("status") == "completed":
            clips.append((arm_id, _seam_playback(record["pixels"], context, window)))
    return comparison.columns(clips, max_long_edge=max_long_edge)


def _match_audio(waveform, channels, rate, source_rate):
    if waveform is None:
        return None
    wave = waveform.to("cpu", torch.float32)
    if wave.ndim == 2:
        wave = wave[None]
    if source_rate and rate and int(source_rate) != int(rate):
        wave = torchaudio.functional.resample(wave, int(source_rate), int(rate))
    have = wave.shape[1]
    if have == channels:
        return wave
    if have == 1:
        return wave.repeat(1, channels, 1)
    return wave.mean(dim=1, keepdim=True).repeat(1, channels, 1)


def _seam_audio(context, record, audio_vae, window):
    """Context audio, the generated audio, then context audio again.

    A right-seam pop is the one failure mode no still frame shows and no latent
    metric fully captures, so the arm has to be listenable.
    """
    if record is None or record.get("audio_latent") is None:
        return None
    generated = audio_output.decode_audio_chunk(
        audio_vae, record["audio_latent"].to(comfy.model_management.get_torch_device()))
    rate = audio_output.audio_sample_rate(audio_vae)
    channels = int(generated.shape[1])

    plan = context.plan
    lead = context.playback_audio(plan.bridge_start - window, window)
    trail = context.playback_audio(plan.right_start, window)
    source_rate = (context.source_audio or {}).get("sample_rate")

    parts = [_match_audio(p["waveform"] if isinstance(p, dict) else p,
                          channels, rate, source_rate)
             for p in (lead, None, trail)]
    parts[1] = generated.to("cpu", torch.float32)
    parts = [p for p in parts if p is not None and p.shape[-1] > 0]
    if not parts:
        return None
    return {"waveform": torch.cat(parts, dim=-1), "sample_rate": rate}


class MiniMaxH3BridgeExperiment(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3BridgeExperimentZi",
            display_name="MiniMax H3 Two-Sided Bridge Experiment (Zi)",
            category="model/conditioning/minimax",
            description=(
                "Holds out one interval of a source video and regenerates it "
                "from a left AV continuation, with and without a right-side "
                "reference. Tests whether H3 reads <Video 2> as a future "
                "constraint."),
            inputs=[
                io.Model.Input("model", tooltip="Apply the (Zi) sigma shift first, "
                                                "as with the Ref2V harness."),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("source_video", tooltip="Continuous 24 fps source. "
                                                       "Needs C1 + target + C3 frames."),
                io.String.Input("base_prompt", multiline=True, dynamic_prompts=True,
                                tooltip="A broad, compatible instruction only - do not "
                                        "describe what actually happens in the held-out "
                                        "interval on the first run."),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             tooltip="Shared by every arm. The arms differ only in "
                                     "what <Video 2> contains."),
                io.Int.Input("chunk_frames", default=141, min=22, max=362, step=17,
                             tooltip="Target length and the length of each context "
                                     "segment. Must satisfy n %% 17 == 5 and n %% 3 == 0."),
                io.Int.Input("ref_frames", default=51, min=3, max=180, step=3,
                             tooltip="Reference interval per side, sliced from the "
                                     "encoded context - not independently encoded, so "
                                     "17k+5 does not apply. Must land on a latent "
                                     "boundary from both ends and be audio exact."),
                io.Int.Input("bridge_start", default=141, min=0, max=1 << 20, step=3,
                             tooltip="First held-out frame. C1 is the chunk_frames "
                                     "before it, C3 the chunk_frames after."),
                io.Combo.Input("arm_suite", options=SUITE_NAMES, default="decisive"),
                io.String.Input("custom_arms", default="", multiline=False),
                io.Int.Input("counterfactual_start", default=-1, min=-1, max=1 << 20,
                             tooltip="Start frame in the source for arm C's right "
                                     "reference. Ignored when a counterfactual clip is "
                                     "connected. -1 requires the clip input."),
                io.Combo.Input("cond_cache", options=COND_CACHE_MODES, default="auto"),
                io.Int.Input("width", default=0, min=0, max=comfy_nodes.MAX_RESOLUTION,
                             step=32, tooltip="0 uses the model's native canvas."),
                io.Int.Input("height", default=0, min=0, max=comfy_nodes.MAX_RESOLUTION,
                             step=32),
                io.Boolean.Input("save_artifacts", default=True),
                io.Boolean.Input("continue_after_failure", default=True),
                io.String.Input("preview_arm", default="", multiline=False,
                                tooltip="Which arm the single-arm outputs show. "
                                        "Blank picks the last completed arm. The "
                                        "comparison always shows every arm."),
                io.Int.Input("playback_context", default=24, min=0, max=180, step=3,
                             tooltip="Frames of real source shown before and after "
                                     "each bridge, so both seams are visible and "
                                     "audible. 24 = 1 second."),
                io.Int.Input("comparison_long_edge", default=1024, min=256, max=4096,
                             step=64,
                             tooltip="Long edge of each comparison column. Lower it "
                                     "if the combined batch gets unwieldy."),
                io.Audio.Input("source_audio", optional=True),
                io.Image.Input("counterfactual_video", optional=True,
                               tooltip="Arm C's right reference. Same subject and "
                                       "environment, clearly incompatible future - "
                                       "opposite travel direction reads best in the "
                                       "signed metric."),
                io.Audio.Input("counterfactual_audio", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="comparison"),
                io.Image.Output(display_name="selected_arm"),
                io.Audio.Output(display_name="selected_arm_audio"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="manifest_path"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, source_video, base_prompt,
                sampler, sigmas, seed, chunk_frames, ref_frames, bridge_start,
                arm_suite, custom_arms, counterfactual_start, cond_cache,
                width, height, save_artifacts, continue_after_failure, preview_arm,
                playback_context, comparison_long_edge,
                source_audio=None, counterfactual_video=None,
                counterfactual_audio=None) -> io.NodeOutput:

        arm_ids = resolve_suite(arm_suite, custom_arms)
        plan = resolve_plan(
            chunk_frames=chunk_frames, ref_frames=ref_frames,
            bridge_start=bridge_start, total_frames=int(source_video.shape[0]),
            counterfactual_start=counterfactual_start, fps=FPS)

        canvas = ref_builder.pin_canvas(source_video, width=width, height=height)
        logging.info("%s canvas %dx%d\n%s", LOG_PREFIX, canvas[0], canvas[1],
                     plan.describe())

        needs_counterfactual = any(
            ARMS[a].right_ref == "counterfactual" for a in arm_ids)
        context = runner.BridgeContext(plan, canvas, cond_cache=cond_cache)
        notes = context.prepare(
            video_vae=video_vae, audio_vae=audio_vae, source_frames=source_video,
            source_audio=source_audio, counterfactual_frames=counterfactual_video,
            counterfactual_audio=counterfactual_audio,
            need_counterfactual=needs_counterfactual)

        directory = (runner.new_run_directory(plan, seed)
                     if save_artifacts else None)
        results = runner.run_arms(
            context, arm_ids, model=model, clip=clip, video_vae=video_vae,
            sampler=sampler, sigmas=sigmas, seed=seed, base_prompt=base_prompt,
            cond_cache=cond_cache,
            continue_after_failure=continue_after_failure, save_to=directory)

        window = int(playback_context)
        grid, labels = _comparison(context, results, arm_ids, window,
                                   int(comparison_long_edge))

        completed = [a for a in arm_ids
                     if results.get(a, {}).get("status") == "completed"]
        selected = preview_arm.strip() or (completed[-1] if completed else "")
        record = results.get(selected)
        if record and record.get("status") == "completed":
            single = _seam_playback(record["pixels"], context, window)
            audio = _seam_audio(context, record, audio_vae, window)
        else:
            single = _seam_playback(context.ground_truth, context, window)
            audio = context.playback_audio(
                plan.bridge_start - window, plan.chunk_frames + 2 * window)
            selected = "ground truth"

        report = format_report(plan, results, notes, seed=seed, arm_ids=arm_ids,
                               columns=labels, selected=selected, window=window)
        logging.info("%s\n%s", LOG_PREFIX, report)

        manifest_path = ""
        if directory:
            manifest_path = runner.write_manifest(
                directory, plan=plan, arm_ids=arm_ids, seed=seed,
                base_prompt=base_prompt, results=results, notes=notes)

        if grid is None:
            grid = single
        return io.NodeOutput(grid, single, audio, report, manifest_path)


class MiniMaxH3BridgeExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3BridgeExperiment]


async def comfy_entrypoint() -> MiniMaxH3BridgeExtension:
    return MiniMaxH3BridgeExtension()


__all__ = ["MiniMaxH3BridgeExperiment", "MiniMaxH3BridgeExtension"]
