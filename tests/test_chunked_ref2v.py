"""CPU self-test for the chunked Ref2V experiment harness.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_chunked_ref2v.py
"""

import os
import shutil
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

from comfy.model_management import InterruptProcessingException  # noqa: E402
from chunked_ref2v import experiments, geometry as geo, layout_ops, prompts, strategies  # noqa: E402
from chunked_ref2v import artifacts, comparison, harness, metrics, report  # noqa: E402

TEXT_LEN = 40
LATENT_H, LATENT_W = 16, 24
AUDIO_T = 30
_failures = []


def check(value, message):
    if value:
        print("  ok: %s" % message)
    else:
        print("  FAIL: %s" % message)
        _failures.append(message)


def raises(exc_type, fn, message):
    try:
        fn()
    except exc_type:
        print("  ok: %s" % message)
        return
    except Exception as exc:
        print("  FAIL: %s (raised %s)" % (message, type(exc).__name__))
        _failures.append(message)
        return
    print("  FAIL: %s (did not raise)" % message)
    _failures.append(message)


def build_layout(latent_t=22, refs=True):
    from comfy.ldm.minimax.model import PackedLayout
    blocks = None
    if refs:
        blocks = [
            {"kind": "image", "latent_h": 8, "latent_w": 8},
            {"kind": "video_audio", "latent_t": latent_t,
             "latent_h": LATENT_H, "latent_w": LATENT_W,
             "ref_audio_t": 12},
        ]
    return PackedLayout(TEXT_LEN, latent_t, LATENT_H, LATENT_W, AUDIO_T,
                        refs=blocks)


def condition(t, start=0, policy="copy_target", label="condition"):
    return layout_ops.TargetAlignedCondition(
        latent=torch.zeros(1, 24, t, LATENT_H, LATENT_W),
        target_latent_start=start, label=label, position_policy=policy)


def test_geometry():
    print("geometry")
    g = geo.HarnessGeometry(73, 22).validate()
    check(g.stride_frames == 51, "73/22 gives stride 51")
    check(g.target_latent_t == 22, "73 frames gives latent T=22")
    check(g.overlap_slice() == (15, 7), "73/22/51 maps to latent 15:22")
    check(geo.HarnessGeometry(90, 22).validate().overlap_slice() == (20, 7),
          "90/22/68 maps to latent 20:27")
    raises(geo.UnalignedProfileError,
           lambda: geo.find_exact_overlap_slice(22, 50, 23),
           "unaligned profiles fail rather than round")


def test_layout_transform():
    print("layout transform")
    base = build_layout()
    frame_rows = layout_ops.frame_rows_of(base)
    before = base.position_ids.clone()
    out = layout_ops.insert_target_conditions(base, [condition(7)])
    check(out.seq_len == base.seq_len + 7 * frame_rows,
          "seven latent positions insert seven grids")
    check(torch.equal(base.position_ids, before), "base layout is not mutated")
    check(out.segments[1][2] == "cond", "conditions physically follow text")
    first_ref = next(a for a, _, kind in out.segments if kind == "ref_img")
    check(out.condition_segments[0][1] <= first_ref,
          "conditions precede references in row order")
    video_start = next(a for a, _, kind in out.segments if kind == "video")
    a, b, _ = out.condition_segments[0]
    check(torch.equal(out.position_ids[a:b],
                      out.position_ids[video_start:video_start + 7 * frame_rows]),
          "copy_target duplicates exact target position rows")
    check(out.img_pos.shape[0] == out.img_update.shape[0],
          "image row indices and masks stay aligned")
    check(out.audio_pos.shape[0] == out.audio_update.shape[0],
          "audio row indices and masks stay aligned")

    stock = layout_ops.insert_target_conditions(
        base, [condition(1, policy="stock")])
    corrected = layout_ops.insert_target_conditions(base, [condition(1)])
    stock_t = stock.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 0]
    corrected_t = corrected.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 0]
    check(bool((stock_t == float(TEXT_LEN)).all()),
          "stock diagnostic reproduces cond_t=text_len")
    check(not torch.equal(stock_t, corrected_t),
          "stock and corrected temporal positions differ with refs")


def test_row_consumption():
    print("row consumption")
    from comfy.ldm.minimax.model import patchify_video

    base = build_layout()
    conditions = [condition(7)]
    out = layout_ops.insert_target_conditions(base, conditions)
    ref_latents = [
        torch.rand(1, 24, 1, 8, 8),
        torch.rand(1, 24, 22, LATENT_H, LATENT_W),
    ]
    rows = torch.cat([
        patchify_video(z.float(), (1, 2, 2))
        for z in layout_ops.condition_latents(conditions) + ref_latents
    ])
    check(rows.shape[0] == int((~out.img_update).sum()),
          "conditions and refs exactly fill non-updating image rows")


def test_dependencies_and_prompt_lookup():
    print("dependencies and prompt lookup")
    prompted = experiments.CATALOG["frame_direct_prompted"]
    check(prompted.dependencies().needs_dynamic_qwen,
          "prompted direct-latent arm explicitly requires Qwen")
    check(not prompted.dependencies().needs_dynamic_video_vae,
          "prompted direct-latent arm does not require VAE preprocessing")
    check(experiments.union_dependencies(["frame_direct_prompted"]).needs_dynamic_qwen,
          "dependency union retains prompt-driven Qwen work")

    context = harness.HarnessContext(geo.DEFAULT_GEOMETRY, harness.SeedSet(1),
                                     (128, 128))
    context.base_prompt = "prompt"
    context.conditionings["chunk_b"] = object()
    check(context.conditioning_for("original") is context.conditionings["chunk_b"],
          "original policy uses base Chunk B conditioning")
    raises(strategies.StrategyUnavailable,
           lambda: context.conditioning_for("keyframe_completion"),
           "missing prompted conditioning cannot fall back silently")
    context.conditionings["prompted"] = object()
    check(context.conditioning_for("keyframe_completion") is
          context.conditionings["prompted"],
          "prompted policy uses its own Qwen result")


def test_strategy_isolation():
    print("strategy isolation")
    context = harness.HarnessContext(geo.DEFAULT_GEOMETRY, harness.SeedSet(1),
                                     (128, 128))
    context.base_prompt = "prompt"
    context.conditionings["chunk_b"] = [["cond", {}]]
    context.qwen_ref_items_b = [{"type": "video"}]
    block = {"kind": "video", "latent": torch.zeros(1, 24, 22, 4, 4)}
    context.dit_ref_blocks_b = [block]
    context.source_ref_block_b = block
    context.overlap_latent = torch.zeros(1, 24, 7, 4, 4)
    context.direct_frame_latent = torch.zeros(1, 24, 1, 4, 4)

    overlap = strategies.get_strategy("direct_latent_overlap").prepare(
        context, experiments.CATALOG["aligned_overlap_direct"])
    frame = strategies.get_strategy("direct_latent_frame").prepare(
        context, experiments.CATALOG["frame_direct_corrected"])
    overlap.dit_ref_blocks.append({"kind": "poison"})
    overlap.target_conditions.append(condition(1))
    check(len(frame.dit_ref_blocks) == 1 and len(frame.target_conditions) == 1,
          "one arm cannot mutate another")


def test_composite_qwen_replacement():
    print("composite Qwen replacement")
    context = harness.HarnessContext(geo.DEFAULT_GEOMETRY, harness.SeedSet(1),
                                     (128, 128))
    audio = {"type": "audio"}
    video = {"type": "video", "name": "original"}
    replacement = {"type": "video", "name": "composite"}
    context.qwen_ref_items_b = [audio, video]
    out = harness._replace_source_item(context, [replacement])
    check([item["type"] for item in out] == ["audio", "video"],
          "composite swap keeps exactly one audio item")
    check(out[-1] is replacement, "composite video replaces the original video")


def test_comparison_cap():
    print("comparison cap")
    # Long-edge fixtures without allocating source-video-sized gigabytes.
    frames = torch.zeros(2, 32, 1024, 3)
    out, _ = comparison.columns([("a", frames), ("b", frames)])
    check(out.shape[1] <= 512 and out.shape[2] <= 1024,
          "two-column preview is bounded to 512 pixels per tile")
    boundary = comparison.boundary_playback(
        chunk_a_pixels=torch.zeros(73, 32, 1024, 3),
        chunk_b_pixels=torch.zeros(73, 32, 1024, 3),
        geometry=geo.DEFAULT_GEOMETRY)
    check(max(boundary.shape[1:3]) <= 512,
          "boundary preview is also bounded")


def test_artifact_reuse_policy():
    print("artifact reuse policy")
    root = tempfile.mkdtemp(prefix="h3_harness_test_")
    old = os.environ.pop(artifacts.AUTO_REUSE_ENV, None)
    try:
        kwargs = dict(
            source_frames=torch.rand(73, 8, 8, 3), prompt="p", ref_pixels=[],
            canvas=(128, 128), geometry=geo.DEFAULT_GEOMETRY, seed=7,
            sampler_name="euler", sigmas=torch.linspace(1, 0, 11),
            checkpoint="ckpt")
        identity = artifacts.chunk_a_identity(**kwargs)
        run_id = artifacts.new_run_id(identity)
        store = artifacts.RunStore(root, run_id)
        store.write_manifest(chunk_a_identity=identity)
        check(artifacts.find_reusable_run(root, identity) is None,
              "automatic reuse is disabled by default")
        os.environ[artifacts.AUTO_REUSE_ENV] = "1"
        check(artifacts.find_reusable_run(root, identity) == run_id,
              "operator can explicitly opt into automatic reuse")
    finally:
        if old is None:
            os.environ.pop(artifacts.AUTO_REUSE_ENV, None)
        else:
            os.environ[artifacts.AUTO_REUSE_ENV] = old
        shutil.rmtree(root, ignore_errors=True)


def test_interruption_propagates():
    print("interruption propagation")
    original = harness.sample_experiment

    class Store:
        pass

    def interrupted(*args, **kwargs):
        raise InterruptProcessingException()

    harness.sample_experiment = interrupted
    try:
        context = harness.HarnessContext(geo.DEFAULT_GEOMETRY,
                                         harness.SeedSet(1), (128, 128))
        raises(InterruptProcessingException,
               lambda: harness.phase_e_experiments(
                   context, experiment_ids=["baseline_none"], model=None,
                   sampler=None, sigmas=torch.ones(2), video_vae=None,
                   store=Store(), continue_after_failure=True),
               "Cancel/VRAM interrupt is never swallowed")
    finally:
        harness.sample_experiment = original


def test_sample_saved_before_decode():
    print("sample persistence before decode")
    original_sample = harness.sample_experiment
    original_decode = harness.decode_video

    class Store:
        def __init__(self):
            self.saved = []
        def save_experiment_tensors(self, experiment_id, name, tensors):
            self.saved.append((experiment_id, name, sorted(tensors)))
            return "/tmp/recovery.safetensors"
        def experiment_dir(self, experiment_id):
            return tempfile.gettempdir()

    def sampled(*args, **kwargs):
        return {
            "latent": torch.zeros(1, 24, 22, 4, 4),
            "audio_latent": torch.zeros(1, 32, 2, 122),
            "prepared": {},
        }

    def decode_failure(*args, **kwargs):
        raise RuntimeError("decode failed")

    harness.sample_experiment = sampled
    harness.decode_video = decode_failure
    store = Store()
    try:
        context = harness.HarnessContext(geo.DEFAULT_GEOMETRY,
                                         harness.SeedSet(1), (128, 128))
        results = harness.phase_e_experiments(
            context, experiment_ids=["baseline_none"], model=None,
            sampler=None, sigmas=torch.ones(2), video_vae=None, store=store,
            continue_after_failure=True, save_latents=True, save_frames=False)
        check(store.saved and store.saved[0][1] == "sampled_output",
              "sampled latent is stored before decode")
        check(results[0]["status"] == "decode_failed",
              "decode failure is distinguished from sampling failure")
        check(results[0].get("recovery_latent"),
              "decode failure reports the recovery latent")
        check("latent" not in results[0],
              "recovery path replaces the in-memory latent after decode failure")
    finally:
        harness.sample_experiment = original_sample
        harness.decode_video = original_decode


def test_metrics_and_report():
    print("metrics and report")
    a = torch.rand(73, 8, 8, 3)
    b = torch.cat([a[51:73], torch.rand(51, 8, 8, 3)])
    values = metrics.pixel_overlap_metrics(a, b, geo.DEFAULT_GEOMETRY)
    check(abs(values["pixel_overlap_mae"]) < 1e-6,
          "identical overlap has zero pixel MAE")

    class Seeds:
        def as_dict(self):
            return {"node_seed": 1}

    doc = report.build(
        run_id="r", geometry=geo.DEFAULT_GEOMETRY, seeds=Seeds(),
        canvas=(128, 128), experiment_ids=["baseline_none"],
        results=[{"experiment_id": "baseline_none", "status": "completed",
                  "metrics": values}], chunk_a_reused=False,
        dependencies=strategies.StrategyDependencies())
    check("baseline_none" in report.to_text(doc), "text report names experiment")


def main():
    tests = (
        test_geometry,
        test_layout_transform,
        test_row_consumption,
        test_dependencies_and_prompt_lookup,
        test_strategy_isolation,
        test_composite_qwen_replacement,
        test_comparison_cap,
        test_artifact_reuse_policy,
        test_interruption_propagates,
        test_sample_saved_before_decode,
        test_metrics_and_report,
    )
    for test in tests:
        test()
        print()
    if _failures:
        print("%d FAILURE(S):" % len(_failures))
        for failure in _failures:
            print("  - %s" % failure)
        return 1
    print("all chunked_ref2v tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
