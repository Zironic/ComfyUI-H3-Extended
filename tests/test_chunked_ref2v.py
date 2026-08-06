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


class FakePatcher:
    """Enough ModelPatcher for the memory-optimizer wiring."""

    def __init__(self, transformer_options=None):
        self.model_options = {
            "transformer_options": dict(transformer_options or {})
        }
        self.clones = 0

    def clone(self):
        copy = FakePatcher(self.model_options["transformer_options"])
        copy.clones = self.clones + 1
        return copy


def fake_environment(capability=(8, 9)):
    from h3_memory_optimizer.attention import RuntimeEnvironment
    return RuntimeEnvironment(
        cuda_available=capability is not None,
        device_index=0 if capability is not None else None,
        capability=capability,
        device_name="fake GPU" if capability is not None else "no CUDA device")


def make_resolver(selected, backend, reason="synthetic", capability=(8, 9)):
    from h3_memory_optimizer.attention import AttentionDecision

    def resolver(requested, fallback, environment=None):
        return AttentionDecision(
            requested=requested, selected=selected, backend=backend,
            adapter=selected, reason=reason,
            environment=environment or fake_environment(capability))
    return resolver


def make_applier(calls):
    """Stands in for h3_memory_optimizer.patch.apply, status recording included."""
    from h3_memory_optimizer.patch import STATUS_KEY

    class Result:
        def __init__(self, decision, config):
            self.attention_requested = config.attention
            self.attention_selected = decision.selected
            self.attention_reason = decision.reason
            self.attention_blocks = 50 if decision.backend is not None else 0
            self.activation_mode = config.activation
            self.activation_blocks = 50
            self.architecture = decision.environment.architecture
            self.device_name = decision.environment.device_name

    def applier(model, config=None, decision=None, pool_policy=None):
        calls.append((model, config, decision, pool_policy))
        result = Result(decision, config)
        options = model.model_options["transformer_options"] = dict(
            model.model_options.get("transformer_options", {}))
        options[STATUS_KEY] = {
            "attention_requested": result.attention_requested,
            "attention_selected": result.attention_selected,
            "attention_reason": result.attention_reason,
            "attention_blocks": result.attention_blocks,
            "activation_mode": result.activation_mode,
            "activation_blocks": result.activation_blocks,
            "architecture": result.architecture,
            "device_name": result.device_name,
        }
        return result
    return applier


def null_pool(enabled, threshold, device_index=None):
    return None


def test_memory_arming():
    print("memory optimizer arming")
    from chunked_ref2v import memory

    calls = []
    model = FakePatcher()
    armed, status = memory.arm(
        model, attention="auto", activation="mlp_chunked_bf16",
        resolver=make_resolver("efficient_sage_sm89", object()),
        applier=make_applier(calls), pool_configurer=null_pool)

    check(armed is not model and armed.clones == 1,
          "arming clones rather than mutating the incoming MODEL")
    check(len(calls) == 1, "the optimizer is applied exactly once for the run")
    check(status["attention_selected"] == "efficient_sage_sm89",
          "the resolved adapter is recorded")
    check(status["armed_by"] == "harness", "the report can tell who armed it")
    check(memory.is_optimized(status), "an SM-matched backend counts as optimized")
    check("efficient_sage_sm89" in memory.describe(status),
          "the one-line description names the backend")

    raises(ValueError,
           lambda: memory.arm(FakePatcher(), attention="not_a_mode"),
           "an unknown attention mode is rejected before anything is patched")


def test_memory_inherits_existing_patch():
    print("memory optimizer inheritance")
    from chunked_ref2v import memory
    from h3_memory_optimizer.patch import STATUS_KEY

    calls = []
    inherited = {
        "attention_selected": "efficient_sage_sm89",
        "attention_requested": "auto",
        "activation_mode": "mlp_chunked_bf16",
    }
    model = FakePatcher({STATUS_KEY: inherited})
    armed, status = memory.arm(
        model, resolver=make_resolver("efficient_sage_sm89", object()),
        applier=make_applier(calls), pool_configurer=null_pool)

    # Re-applying the same backend is a no-op and a different one raises inside
    # the installer; neither is worth risking once a run has started.
    check(armed is model, "an already-optimized MODEL is used as-is")
    check(not calls, "the optimizer is not applied a second time")
    check(status["armed_by"] == "incoming model",
          "the report distinguishes inherited from harness-armed")
    check(status["attention_selected"] == "efficient_sage_sm89",
          "the inherited selection is what gets reported")


def test_memory_disabled_and_fallback():
    print("memory optimizer disabled and fallback")
    from chunked_ref2v import memory

    calls = []
    model = FakePatcher()
    armed, status = memory.arm(
        model, attention="existing", activation="off",
        applier=make_applier(calls), pool_configurer=null_pool)
    check(armed is model and not calls,
          "fully disabled leaves the MODEL completely untouched")
    check(status["armed_by"] == "disabled" and not memory.is_optimized(status),
          "a deliberate A/B is recorded as disabled, not as a failure")

    calls = []
    armed, status = memory.arm(
        FakePatcher(), attention="auto",
        resolver=make_resolver(memory.ATTENTION_EXISTING, None,
                               reason="unsupported GPU", capability=(7, 5)),
        applier=make_applier(calls), pool_configurer=null_pool)
    check(len(calls) == 1,
          "a capability fallback still installs activation chunking")
    check(not memory.is_optimized(status),
          "a fallback is not reported as optimized")
    check("unsupported GPU" in memory.describe(status),
          "the fallback reason survives into the description")


def test_report_attributes_the_backend():
    print("report backend attribution")
    from chunked_ref2v import memory

    class Seeds:
        def as_dict(self):
            return {"node_seed": 1}

    def document_for(status):
        return report.build(
            run_id="r", geometry=geo.DEFAULT_GEOMETRY, seeds=Seeds(),
            canvas=(128, 128), experiment_ids=["baseline_none"],
            results=[{"experiment_id": "baseline_none", "status": "completed",
                      "metrics": {"pixel_overlap_mae": 0.1},
                      "resources": {"peak_reserved_mb": 9000}}],
            chunk_a_reused=False, dependencies=strategies.StrategyDependencies(),
            memory_status=status)

    optimized = document_for({
        "attention_selected": "efficient_sage_sm89",
        "attention_requested": "auto", "activation_mode": "mlp_chunked_bf16",
        "armed_by": "harness", "architecture": "sm89"})
    entry = optimized["experiments"]["baseline_none"]
    check(optimized["runtime"]["attention_selected"] == "efficient_sage_sm89",
          "the run records which backend produced its numbers")
    check(entry["resources"]["attention_selected"] == "efficient_sage_sm89",
          "each arm's resources carry the backend alongside peak VRAM")
    check("WARNING" not in report.to_text(optimized),
          "an optimized run needs no caveat")

    fell_back = document_for({
        "attention_selected": memory.ATTENTION_EXISTING,
        "attention_requested": "auto", "attention_reason": "unsupported GPU",
        "activation_mode": "mlp_chunked_bf16", "armed_by": "harness"})
    text = report.to_text(fell_back)
    check("WARNING" in text and "NOT attributable" in text,
          "a silent fallback is stated loudly next to the resource figures")
    check("still valid" in text,
          "the caveat says what remains valid, not just what broke")

    disabled = document_for({
        "attention_selected": memory.ATTENTION_EXISTING,
        "armed_by": "disabled", "activation_mode": "off"})
    check("NOTE:" in report.to_text(disabled),
          "a deliberate A/B reads as a note rather than a warning")


def test_identity_covers_the_backend():
    print("chunk A identity covers the backend")
    kwargs = dict(
        source_frames=torch.rand(73, 8, 8, 3), prompt="p", ref_pixels=[],
        canvas=(128, 128), geometry=geo.DEFAULT_GEOMETRY, seed=7,
        sampler_name="euler", sigmas=torch.linspace(1, 0, 11), checkpoint="ckpt")
    optimized = artifacts.chunk_a_identity(attention="efficient_sage_sm89", **kwargs)
    existing = artifacts.chunk_a_identity(attention="existing", **kwargs)
    # Sage quantizes Q/K to INT8, so Chunk A carries the backend's numerics into
    # every overlap metric that compares it against Chunk B.
    check(optimized != existing,
          "a Chunk A from one attention backend is not reused against another")
    check(artifacts.chunk_a_identity(attention="efficient_sage_sm89", **kwargs)
          == optimized, "the backend-aware identity is still stable")


def test_monolithic_geometry():
    print("monolithic reference geometry")
    g = geo.HarnessGeometry(73, 22).validate()
    check(g.total_frames == 124, "two chunks at 73/22/51 span exactly 124 frames")
    check(g.supports_monolithic, "124 is on the 17k+5 grid, so one run can match it")
    check(g.monolithic_latent_t == 37, "124 frames is latent T=37")
    check(geo.decoded_frame_count(37) == 124, "T=37 decodes back to 124 - exact, not snapped")
    check(g.tail_range(17) == (107, 124), "the tail is global frames 107-123")
    check(g.tail_in_chunk_b(17) == (56, 73), "the same tail is chunk B local 56-72")

    # total = S + C and C % 17 == 5, so a legal total needs S % 17 == 0.
    check(not geo.HarnessGeometry(73, 17).supports_monolithic,
          "O=17 spans 129 frames, off-grid, so it cannot be compared like for like")
    check(geo.HarnessGeometry(73, 39).supports_monolithic,
          "O=39 spans 107 frames, which is on-grid")
    check(all(geo.HarnessGeometry(73, o).supports_monolithic
              for o in (5, 22, 39, 56)),
          "the S %% 17 == 0 family is exactly the monolithic-comparable one")


def test_monolithic_tail_metric():
    print("monolithic tail metric")
    g = geo.HarnessGeometry(73, 22).validate()
    mono = torch.rand(124, 8, 8, 3)
    # A chunk B whose tail is copied from the monolithic run should score ~0.
    chunk_b = torch.rand(73, 8, 8, 3)
    chunk_b[56:73] = mono[107:124]
    out = metrics.monolithic_tail_metrics(mono, chunk_b, g, 17)
    check(abs(out["monolithic_tail_mae"]) < 1e-6,
          "a tail identical to the single run scores ~0")
    check(out["monolithic_tail_frames"] == [107, 124],
          "the metric reports which frames it compared")
    check(len(out["monolithic_tail_mae_per_frame"]) == 17,
          "per-frame values cover the whole tail")
    check(out["monolithic_chunk_b_mae"] > out["monolithic_tail_mae"],
          "the body still disagrees, so the tail is measured separately from it")

    unrelated = metrics.monolithic_tail_metrics(mono, torch.rand(73, 8, 8, 3), g, 17)
    check(unrelated["monolithic_tail_mae"] > out["monolithic_tail_mae"],
          "an unrelated tail scores worse than a matching one")

    short = metrics.monolithic_tail_metrics(mono[:50], chunk_b, g, 17)
    check(short.get("monolithic_tail_note"),
          "too few decoded frames reports why rather than a wrong number")

    collected = metrics.collect(
        geometry=g, chunk_a_latent=None, chunk_a_pixels=None,
        chunk_b_latent=None, chunk_b_pixels=chunk_b, monolithic_pixels=mono)
    check("monolithic_tail_mae" in collected,
          "collect() surfaces the ground-truth metric when a reference exists")
    check("monolithic_tail_mae" not in metrics.collect(
        geometry=g, chunk_a_latent=None, chunk_a_pixels=None,
        chunk_b_latent=None, chunk_b_pixels=chunk_b),
        "and omits it entirely when there is none")


def test_monolithic_report_section():
    print("monolithic report section")

    class Seeds:
        def as_dict(self):
            return {"node_seed": 1}

    def doc(mono_reused):
        return report.build(
            run_id="r", geometry=geo.HarnessGeometry(73, 22).validate(), seeds=Seeds(),
            canvas=(608, 352), experiment_ids=["baseline_none", "aligned_overlap_direct"],
            results=[
                {"experiment_id": "baseline_none", "status": "completed",
                 "metrics": {"pixel_overlap_mae": 0.03, "monolithic_tail_mae": 0.20}},
                {"experiment_id": "aligned_overlap_direct", "status": "completed",
                 "metrics": {"pixel_overlap_mae": 0.01, "monolithic_tail_mae": 0.12,
                             "monolithic_tail_mae_vs_baseline": 0.6}},
            ],
            chunk_a_reused=False, dependencies=strategies.StrategyDependencies(),
            monolithic_reused=mono_reused)

    d = doc(False)
    ref = d["common_assets"]["monolithic_reference"]
    check(ref["total_frames"] == 124 and ref["tail_range"] == [107, 124],
          "the report records the span and tail it compared")
    text = report.to_text(d)
    check("ground truth" in text, "the ground-truth ranking is printed")
    check(text.index("aligned_overlap_direct") < text.index("baseline_none",
                                                            text.index("ground truth")),
          "the better arm ranks first against the single run")
    check("never reaches zero" in text,
          "the report states why a perfect score is impossible")

    without = report.build(
        run_id="r", geometry=geo.DEFAULT_GEOMETRY, seeds=Seeds(), canvas=(608, 352),
        experiment_ids=["baseline_none"],
        results=[{"experiment_id": "baseline_none", "status": "completed",
                  "metrics": {"pixel_overlap_mae": 0.03}}],
        chunk_a_reused=False, dependencies=strategies.StrategyDependencies())
    check(without["common_assets"]["monolithic_reference"] is None,
          "no reference means the section is absent, not empty")
    check("ground truth" not in report.to_text(without),
          "and the text report does not mention it")


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
        test_memory_arming,
        test_memory_inherits_existing_patch,
        test_memory_disabled_and_fallback,
        test_report_attributes_the_backend,
        test_identity_covers_the_backend,
        test_monolithic_geometry,
        test_monolithic_tail_metric,
        test_monolithic_report_section,
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
