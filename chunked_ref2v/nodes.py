"""The harness node.

One node, because the experiment count varies and a Comfy graph cannot grow an
output socket per arm. Every complete output lands in the artifact directory;
the node returns a comparison video, one selected preview, the report text and
the path.
"""

import logging

import torch

import nodes
from comfy_api.latest import ComfyExtension, io

try:
    from .. import run_context
    from ..cond_cache import MODES as COND_CACHE_MODES
except ImportError:  # self-tests import the package as a top-level module
    import run_context
    from cond_cache import MODES as COND_CACHE_MODES

from . import artifacts, comparison, harness, memory, report
from .experiments import CATALOG, SUITE_NAMES, resolve_suite
from .geometry import HarnessGeometry

LOG_PREFIX = "[H3 Extended] harness"


class MiniMaxH3Ref2VExperimentHarness(io.ComfyNode):
    """Generate Chunk A once, then evaluate Chunk B carry strategies against it."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Ref2VExperimentHarnessZi",
            display_name="MiniMax H3 Ref2V Experiment Harness (Zi)",
            category="model/video/minimax/testing",
            is_experimental=True,
            description=(
                "Two-chunk experiment harness for MiniMax H3 Ref2V. Generates Chunk A "
                "once, derives every carry asset from it, and runs the selected Chunk B "
                "strategies under one seed and one sigma schedule. Chunk A is cached on "
                "its own identity, so adding an experiment never regenerates it."
            ),
            inputs=[
                io.Model.Input("model", tooltip="Apply the (Zi) sigma shift node first for the "
                                                "sigma shifts and the VRAM guard. Attention is armed "
                                                "here instead, by the capability resolver - leave the "
                                                "shift node's attention_backend on 'comfy'. A model "
                                                "already patched by MiniMaxH3MemoryOptimizerZi is "
                                                "inherited rather than re-armed."),
                io.Clip.Input("clip"),
                io.Vae.Input("video_vae"),
                io.Vae.Input("audio_vae"),
                io.Image.Input("source_video", tooltip="Source frames at 24 fps. Needs at "
                                                       "least stride + chunk frames (124 at the default profile)."),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                             control_after_generate=True,
                             tooltip="Chunk A noise, Chunk B noise, conditioning augmentation "
                                     "and clamp noise are derived from this with SplitMix64. "
                                     "Every Chunk B arm shares one Chunk B seed."),
                io.Int.Input("chunk_frames", default=73, min=22, max=362, step=17,
                             tooltip="Must satisfy n %% 17 == 5. 73 is the memory-safe default; "
                                     "90 is the compute optimum with no headroom."),
                io.Int.Input("overlap_frames", default=22, min=5, max=180,
                             tooltip="Shared frames between the chunks. 22 aligns exactly with "
                                     "latent positions at C=73; a profile that does not align "
                                     "fails rather than approximating."),
                io.Combo.Input("experiment_suite", options=SUITE_NAMES, default="minimal"),
                io.String.Input("custom_experiments", default="", multiline=False,
                                tooltip="Comma-separated experiment ids, used when the suite is 'custom'."),
                io.Combo.Input("ref_image_size", options=["native", "match", "max"], default="native"),
                io.Combo.Input("cond_cache", options=COND_CACHE_MODES, default="auto"),
                io.Boolean.Input("monolithic_reference", default=False,
                                 tooltip="Also generate the whole span in ONE run and score every "
                                         "arm's tail against it. This is the only measurement here "
                                         "against ground truth rather than against the harness's "
                                         "own first chunk. Costs one extra sample of "
                                         "stride+chunk frames. Requires S %% 17 == 0."),
                io.Int.Input("tail_frames", default=17, min=2, max=120,
                             tooltip="How many frames at the far end to compare against the "
                                     "monolithic run. The tail is furthest from the carried "
                                     "state, so it is where a carry that lost the trajectory "
                                     "shows up."),
                io.Combo.Input("attention", options=list(memory.ATTENTION_MODES),
                               default="auto",
                               tooltip="Armed once for the whole run and shared by Chunk A and "
                                       "every arm, so peak-VRAM and runtime stay comparable "
                                       "across arms. 'auto' picks the prepared-QKV Sage backend "
                                       "matching this GPU; 'existing' leaves the incoming "
                                       "attention alone for an A/B."),
                io.Combo.Input("attention_fallback", options=list(memory.FALLBACK_MODES),
                               default="allow",
                               tooltip="'allow' preserves existing attention when no optimized "
                                       "adapter fits, and the report says so - arm-to-arm "
                                       "comparison survives, absolute resource figures stop being "
                                       "attributable. 'error' refuses to start the run instead."),
                io.Combo.Input("activation", options=list(memory.ACTIVATION_MODES),
                               default="mlp_chunked_bf16",
                               tooltip="MLP activation chunking, armed with attention. 'off' "
                                       "disables it."),
                io.Boolean.Input("cuda_async_soft_gc", default=False,
                                 tooltip="Under cudaMallocAsync, ask the default pool to return "
                                         "memory above a soft threshold on later syncs. Worth "
                                         "considering for an unattended multi-hour suite; it is "
                                         "not a hard limit and does not empty the cache."),
                io.Float.Input("cuda_async_release_threshold_gib", default=11.0,
                               min=0.25, max=256.0, step=0.25,
                               tooltip="Soft retention threshold, ignored unless soft GC is on."),
                io.String.Input("reuse_run", default="", multiline=False,
                                tooltip="Prior run id to reuse Chunk A from. Blank searches by "
                                        "identity and reuses a match automatically."),
                io.Int.Input("width", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32,
                             tooltip="0 pins the canvas from the source, which is what keeps "
                                     "Chunk A and Chunk B latents sliceable against each other."),
                io.Int.Input("height", default=0, min=0, max=nodes.MAX_RESOLUTION, step=32),
                io.Boolean.Input("save_latents", default=True),
                io.Boolean.Input("save_frames", default=True),
                io.Boolean.Input("continue_after_failure", default=True),
                io.String.Input("preview_experiment", default="", multiline=False,
                                tooltip="Experiment id to return as selected_preview. "
                                        "Blank picks the last completed arm."),
                io.Autogrow.Input("ref_images", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("ref_image"),
                                      prefix="ref_image_", min=0, max=9)),
                io.Audio.Input("source_audio", optional=True),
            ],
            outputs=[
                io.Image.Output(display_name="comparison_video"),
                io.Image.Output(display_name="selected_preview"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="artifact_path"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, clip, video_vae, audio_vae, source_video, prompt,
                sampler, sigmas, seed, chunk_frames=73, overlap_frames=22,
                experiment_suite="minimal", custom_experiments="",
                ref_image_size="native", cond_cache="auto",
                monolithic_reference=False, tail_frames=17,
                attention="auto", attention_fallback="allow",
                activation="mlp_chunked_bf16", cuda_async_soft_gc=False,
                cuda_async_release_threshold_gib=11.0, reuse_run="",
                width=0, height=0, save_latents=True, save_frames=True,
                continue_after_failure=True, preview_experiment="",
                ref_images=None, source_audio=None) -> io.NodeOutput:

        geometry = HarnessGeometry(chunk_frames=chunk_frames,
                                   overlap_frames=overlap_frames).validate()
        experiment_ids = resolve_suite(experiment_suite, custom_experiments)
        seeds = harness.SeedSet(seed)
        canvas = _pin_canvas(source_video, width, height)

        # Armed before anything is sampled, and before the VAE work in Phase A,
        # so an `error` fallback refuses the run in the first second rather than
        # after Chunk A has been generated.
        model, memory_status = memory.arm(
            model, attention=attention, attention_fallback=attention_fallback,
            activation=activation, cuda_async_soft_gc=cuda_async_soft_gc,
            cuda_async_release_threshold_gib=cuda_async_release_threshold_gib)
        logging.info("%s %s", LOG_PREFIX, memory.describe(memory_status))

        run_context.record(
            "MiniMax H3 Ref2V Experiment Harness (Zi)", run_context.node_id(cls),
            [("canvas", "%dx%d" % canvas),
             ("profile", geometry.describe()),
             ("suite", "%s -> %s" % (experiment_suite, ", ".join(experiment_ids))),
             ("ref_image_size", ref_image_size),
             ("cond_cache", cond_cache),
             ("memory optimizer", memory.describe(memory_status)),
             ("source_video", "%d frames %dx%d" % (source_video.shape[0],
                                                   source_video.shape[2], source_video.shape[1])),
             ("prompt", "%d chars" % len(prompt))],
            # Both chunks share one target shape, so the VRAM guard's cancel log
            # can match its record against whatever is being denoised.
            video_latent_shape=(1, 24, geometry.target_latent_t,
                                canvas[1] // 16, canvas[0] // 16),
        )

        context = harness.HarnessContext(geometry, seeds, canvas)

        # Phase A - VAE resident once
        notes = harness.phase_a_vae(
            context, video_vae=video_vae, audio_vae=audio_vae,
            source_frames=source_video, ref_images=ref_images,
            ref_image_size=ref_image_size, source_audio=source_audio,
            monolithic=monolithic_reference)

        identity = artifacts.chunk_a_identity(
            source_frames=context.source_chunk_a_pixels,
            prompt=prompt,
            ref_pixels=[b["latent"] for b in context.static_ref_blocks],
            canvas=canvas, geometry=geometry, seed=seeds.chunk_a_noise,
            sampler_name=type(sampler).__name__, sigmas=sigmas,
            checkpoint=_checkpoint_identity(model),
            attention=memory_status.get("attention_selected"))

        root = artifacts.resolve_root()
        run_id = reuse_run.strip() or artifacts.find_reusable_run(root, identity) \
            or artifacts.new_run_id(identity)
        store = artifacts.RunStore(root, run_id)
        store.write_text("prompt.txt", prompt)

        # Phase B - Qwen resident once
        harness.phase_b_qwen(context, clip=clip, prompt=prompt, cond_cache=cond_cache,
                             monolithic=monolithic_reference)

        # Phase C - DiT resident, Chunk A generated at most once ever
        reused = harness.phase_c_chunk_a(
            context, model=model, sampler=sampler, sigmas=sigmas,
            store=store, identity=identity, video_vae=video_vae)
        mono_reused = None
        if monolithic_reference:
            mono_reused = harness.phase_c_monolithic(
                context, model=model, sampler=sampler, sigmas=sigmas,
                store=store, identity=identity, video_vae=video_vae)

        store.write_manifest(chunk_a_identity=identity, profile=geometry.describe(),
                             canvas=list(canvas), seeds=seeds.as_dict())

        # Phase D - only what the selected strategies asked for
        dependencies, dynamic_notes = harness.phase_d_dynamic(
            context, experiment_ids=experiment_ids, video_vae=video_vae,
            audio_vae=audio_vae, clip=clip, cond_cache=cond_cache,
            store=store, identity=identity)

        # Phase E - one payload per arm
        results = harness.phase_e_experiments(
            context, experiment_ids=experiment_ids, model=model, sampler=sampler,
            sigmas=sigmas, video_vae=video_vae, store=store,
            continue_after_failure=continue_after_failure,
            save_latents=save_latents, save_frames=save_frames,
            tail_frames=tail_frames)

        document = report.build(
            run_id=run_id, geometry=geometry, seeds=seeds, canvas=canvas,
            experiment_ids=experiment_ids, results=results,
            chunk_a_reused=reused, dependencies=dependencies,
            notes=notes + dynamic_notes, memory_status=memory_status,
            monolithic_reused=mono_reused, tail_frames=tail_frames)
        store.write_text("report.json", report.to_json(document))
        text = report.to_text(document)
        store.write_text("report.txt", text)
        store.write_manifest(experiments=list(document["experiments"]))

        comparison_video, preview = _outputs(context, results, geometry, preview_experiment)
        logging.info("%s finished run %s: %s", LOG_PREFIX, run_id,
                     ", ".join("%s=%s" % (r["experiment_id"], r["status"]) for r in results))
        return io.NodeOutput(comparison_video, preview, text, store.root)


def _pin_canvas(source_video, width, height):
    from . import ref_builder
    return ref_builder.pin_canvas(source_video, width, height)


def _checkpoint_identity(model):
    """Best-effort model provenance for the Chunk A key.

    Unknown provenance is recorded as such rather than guessed - a wrong reuse
    is much worse than a redundant regeneration.
    """
    try:
        config = model.model.model_config
        return "%s/%s" % (type(config).__name__, type(model.model).__name__)
    except Exception:
        return None


def _outputs(context, results, geometry, preview_experiment):
    completed = [r for r in results if r.get("pixels") is not None]
    fallback = torch.zeros(1, 64, 64, 3)

    baseline = next((r for r in completed if r["experiment_id"] == "baseline_none"), None)
    selected = None
    if preview_experiment:
        selected = next((r for r in completed
                         if r["experiment_id"] == preview_experiment.strip()), None)
    if selected is None:
        selected = completed[-1] if completed else None

    if selected is None:
        return fallback, fallback

    comparison_video, _ = comparison.overlap_comparison(
        source_pixels=context.source_chunk_a_pixels,
        chunk_a_pixels=context.chunk_a_output_pixels,
        baseline_pixels=None if baseline is None else baseline["pixels"],
        experiment_pixels=selected["pixels"],
        geometry=geometry)

    preview = selected.get("boundary")
    if preview is None:
        preview = selected["pixels"]
    return (comparison_video if comparison_video is not None else fallback), preview


class MiniMaxH3HarnessExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3Ref2VExperimentHarness]


async def comfy_entrypoint() -> MiniMaxH3HarnessExtension:
    return MiniMaxH3HarnessExtension()


__all__ = ["MiniMaxH3Ref2VExperimentHarness", "MiniMaxH3HarnessExtension", "CATALOG"]
