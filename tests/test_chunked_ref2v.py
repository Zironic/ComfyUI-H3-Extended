"""Self-test for the chunked-ref2v experiment harness. CPU only, no checkpoint.

Run from the ComfyUI root so `comfy` is importable:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_chunked_ref2v.py

This is the plan's §18 coverage: the temporal mapping, the layout transformation
against a *real* `PackedLayout`, strategy dependency declarations, conditioning
isolation between arms, and artifact reuse. It never allocates on the GPU and is
safe to run while a generation is in flight.
"""

import os
import shutil
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                  # the package
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # ComfyUI root

# `comfy.model_management` initializes a CUDA context at import time, which on a
# 12 GB card is not free while a generation is in flight. `--cpu` makes
# `get_torch_device` return CPU before anything touches the driver, so this test
# can run against the real `PackedLayout` without going near the GPU.
# `comfy.cli_args` only reads argv when args parsing is enabled, so both have to
# be set before the first comfy import.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

from chunked_ref2v import (  # noqa: E402
    experiments, geometry as geo, layout_ops, metrics, prompts, report, strategies)

TEXT_LEN = 40
LATENT_H, LATENT_W = 16, 24
AUDIO_T = 30

_failures = []


def check(cond, msg):
    if cond:
        print("  ok: %s" % msg)
    else:
        print("  FAIL: %s" % msg)
        _failures.append(msg)


def raises(exc_type, fn, msg):
    try:
        fn()
    except exc_type:
        print("  ok: %s" % msg)
        return
    except Exception as exc:
        print("  FAIL: %s (raised %s instead)" % (msg, type(exc).__name__))
        _failures.append(msg)
        return
    print("  FAIL: %s (did not raise)" % msg)
    _failures.append(msg)


def build_layout(latent_t, refs=True):
    from comfy.ldm.minimax.model import PackedLayout
    ref_blocks = None
    if refs:
        ref_blocks = [
            {"kind": "image", "latent_h": 8, "latent_w": 8},
            {"kind": "video_audio", "latent_t": latent_t, "latent_h": LATENT_H,
             "latent_w": LATENT_W, "ref_audio_t": 12},
        ]
    return PackedLayout(TEXT_LEN, latent_t, LATENT_H, LATENT_W, AUDIO_T, refs=ref_blocks)


def cond(latent_t, start=0, policy="copy_target", label="c"):
    return layout_ops.TargetAlignedCondition(
        latent=torch.zeros(1, 24, latent_t, LATENT_H, LATENT_W),
        target_latent_start=start, label=label, position_policy=policy)


# ---------------------------------------------------------------------------

def test_temporal_mapping():
    print("temporal mapping")
    check(geo.latent_frame_spans(7) == [1, 4, 4, 4, 4, 1, 4], "spans repeat 1,4,4,4,4")

    g = geo.HarnessGeometry(chunk_frames=73, overlap_frames=22).validate()
    check(g.stride_frames == 51, "C=73 O=22 -> S=51")
    check(g.target_latent_t == 22, "C=73 -> T=22")
    check(g.overlap_slice() == (15, 7), "C=73/S=51/O=22 -> latent slice 15:22")

    g90 = geo.HarnessGeometry(chunk_frames=90, overlap_frames=22).validate()
    check(g90.target_latent_t == 27, "C=90 -> T=27")
    check(g90.overlap_slice() == (20, 7), "C=90/S=68/O=22 -> latent slice 20:27")

    spans = geo.latent_frame_spans(22)
    check(sum(spans[:15]) == 51, "positions 0-14 cover the 51-frame stride")
    check(sum(spans[15:22]) == 22, "positions 15-21 cover the 22-frame overlap")

    # An overlap that does not begin on a boundary must fail loudly rather than
    # silently pick the nearest position.
    raises(geo.UnalignedProfileError,
           lambda: geo.find_exact_overlap_slice(22, 50, 23),
           "an unaligned stride raises UnalignedProfileError")
    raises(ValueError,
           lambda: geo.HarnessGeometry(chunk_frames=70, overlap_frames=22).validate(),
           "a chunk length off the 17k+5 grid is rejected")

    check(geo.decoded_frame_count(22) == 73, "latent_t=22 decodes to 73 frames")
    check(geo.DEFAULT_GEOMETRY.required_source_frames == 124,
          "the default profile needs 124 source frames")


def test_layout_no_conditions():
    print("layout transformation - identity")
    base = build_layout(22)
    out = layout_ops.insert_target_conditions(base, [])
    check(out is base, "no conditions leaves the base layout untouched")


def test_layout_single_condition():
    print("layout transformation - one condition")
    base = build_layout(22)
    frame_rows = layout_ops.frame_rows_of(base)
    before = (base.seq_len, base.position_ids.clone(), base.img_pos.clone(),
              base.img_update.clone(), base.audio_pos.clone(), list(base.segments))

    out = layout_ops.insert_target_conditions(base, [cond(1)])

    check(out.seq_len == base.seq_len + frame_rows, "one condition inserts one spatial grid")
    check(out.position_ids.shape[0] == out.seq_len, "position_ids matches seq_len")
    check(out.segments[1][2] == "cond", "the condition segment follows text")
    check(out.segments[0] == (0, TEXT_LEN, "text"), "text still leads the pack")
    check(out.signature == base.signature, "signature is carried through unchanged")

    check(base.seq_len == before[0] and torch.equal(base.position_ids, before[1])
          and torch.equal(base.img_pos, before[2])
          and torch.equal(base.img_update, before[3])
          and torch.equal(base.audio_pos, before[4])
          and list(base.segments) == before[5],
          "the original layout is not mutated")


def test_layout_overlap_condition():
    print("layout transformation - seven-position condition")
    base = build_layout(22)
    frame_rows = layout_ops.frame_rows_of(base)
    out = layout_ops.insert_target_conditions(base, [cond(7)])

    check(out.seq_len == base.seq_len + 7 * frame_rows, "seven positions insert seven grids")
    check(int((~out.img_update).sum()) == int((~base.img_update).sum()) + 7 * frame_rows,
          "condition rows appear as non-target image rows")
    check(bool((~out.img_update[:7 * frame_rows]).all()),
          "the condition rows lead the image-row order, before the references")

    video_start = next(a for a, _, k in out.segments if k == "video")
    cond_start, cond_stop, _ = out.condition_segments[0]
    check(torch.equal(out.position_ids[cond_start:cond_stop],
                      out.position_ids[video_start:video_start + 7 * frame_rows]),
          "condition positions equal the corresponding target positions")


def test_layout_shifting():
    print("layout transformation - index shifting")
    base = build_layout(22)
    frame_rows = layout_ops.frame_rows_of(base)
    out = layout_ops.insert_target_conditions(base, [cond(1)])
    inserted = frame_rows

    base_video = next(a for a, _, k in base.segments if k == "video")
    out_video = next(a for a, _, k in out.segments if k == "video")
    check(out_video == base_video + inserted, "target video rows shift by the insertion")

    base_refs = [(a, b) for a, b, k in base.segments if k == "ref_img"]
    out_refs = [(a, b) for a, b, k in out.segments if k == "ref_img"]
    check(all(o == (a + inserted, b + inserted) for (a, b), o in zip(base_refs, out_refs)),
          "reference rows shift by the insertion")

    check(torch.equal(out.audio_pos, base.audio_pos + inserted),
          "audio row indices shift correctly")
    check(torch.equal(out.audio_update, base.audio_update),
          "audio update mask is unchanged")
    check(out.img_pos.shape[0] == out.img_update.shape[0],
          "img_pos and img_update stay aligned")
    check(int(out.img_pos.max()) < out.seq_len, "no image row points past the sequence")

    # Every image row index must still land inside a video-bearing segment.
    kinds = {}
    for a, b, k in out.segments:
        for row in range(a, b):
            kinds[row] = k
    check(all(kinds[int(r)] in ("cond", "ref_img", "video") for r in out.img_pos),
          "every img_pos row lands in a video-bearing segment")
    check(all(kinds[int(r)] in ("ref_audio", "audio") for r in out.audio_pos),
          "every audio_pos row lands in an audio segment")


def test_layout_stock_policy():
    print("layout transformation - stock position policy")
    base = build_layout(22)
    frame_rows = layout_ops.frame_rows_of(base)
    corrected = layout_ops.insert_target_conditions(base, [cond(1, policy="copy_target")])
    stock = layout_ops.insert_target_conditions(base, [cond(1, policy="stock")])

    check(stock.seq_len == corrected.seq_len, "both policies insert the same row count")
    t_stock = stock.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 0]
    check(bool((t_stock == float(TEXT_LEN)).all()),
          "stock places the condition at cond_t = text_len")

    t_corrected = corrected.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 0]
    check(not torch.equal(t_stock, t_corrected),
          "with references present the two placements genuinely differ")
    check(torch.equal(stock.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 1:],
                      corrected.position_ids[TEXT_LEN:TEXT_LEN + frame_rows, 1:]),
          "only the temporal coordinate differs; the spatial grid is shared")


def test_layout_validation():
    print("layout transformation - validation")
    base = build_layout(22)
    raises(ValueError,
           lambda: layout_ops.insert_target_conditions(base, [cond(7, start=20)]),
           "a condition running past the target is rejected")
    raises(ValueError,
           lambda: layout_ops.insert_target_conditions(base, [cond(1, start=-1)]),
           "a negative target position is rejected")
    raises(ValueError,
           lambda: layout_ops.insert_target_conditions(
               base, [layout_ops.TargetAlignedCondition(
                   latent=torch.zeros(1, 24, 1, 8, 8), target_latent_start=0, label="small")]),
           "a condition on a different canvas is rejected")
    raises(ValueError,
           lambda: layout_ops.insert_target_conditions(
               base, [cond(4, start=0, label="a"), cond(4, start=2, label="b")]),
           "overlapping conditions are rejected")


def test_row_consumption():
    """The invariant the whole transform rests on, checked the way the DiT does.

    `_forward` fills `all_video_rows[~img_update] = cond_video_rows`, so the
    patchified condition latents must supply exactly the non-target image rows,
    in exactly the order the segment table walks them. Getting this wrong does
    not raise - it silently pairs each condition with the wrong rows.
    """
    print("condition row consumption")
    from comfy.ldm.minimax.model import patchify_video

    base = build_layout(22)
    conditions = [cond(7)]
    ref_latents = [
        torch.rand(1, 24, 1, 8, 8),        # the image ref block
        torch.rand(1, 24, 22, LATENT_H, LATENT_W),   # the video_audio ref block
    ]
    out = layout_ops.insert_target_conditions(base, conditions)

    cond_video_latents = layout_ops.condition_latents(conditions) + ref_latents
    rows = torch.cat([patchify_video(z.to(torch.float32), (1, 2, 2))
                      for z in cond_video_latents])

    check(rows.shape[0] == int((~out.img_update).sum()),
          "patchified condition + reference rows fill exactly the non-target image rows")

    target_rows = 22 * layout_ops.frame_rows_of(base)
    check(rows.shape[0] + target_rows == out.img_update.shape[0],
          "condition, reference and target rows together cover every image row")

    # The segment walk that assembles the hidden state must consume the same
    # rows in the same order.
    consumed = sum(b - a for a, b, k in out.segments
                   if k in ("cond", "ref_img", "video"))
    check(consumed == out.img_update.shape[0],
          "the segment walk consumes every image row exactly once")

    # A condition placed after the refs would still count correctly but pair
    # with the wrong latents; assert the ordering explicitly.
    cond_span = out.condition_segments[0]
    first_ref = next(a for a, _, k in out.segments if k == "ref_img")
    check(cond_span[1] <= first_ref,
          "condition rows precede every reference row, matching the latent order")


def test_strategy_dependencies():
    print("strategy dependencies")
    def deps(strategy_id):
        return strategies.get_strategy(strategy_id).dependencies()

    d = deps("direct_latent_overlap")
    check(d.needs_chunk_a_latent and not d.needs_dynamic_qwen
          and not d.needs_dynamic_video_vae,
          "direct overlap needs neither Qwen nor a dynamic VAE pass")

    d = deps("generated_overlap_video2")
    check(d.needs_dynamic_qwen and d.needs_dynamic_video_vae and d.needs_chunk_a_pixels,
          "Video 2 requests both dynamic passes")

    d = deps("composite_source")
    check(d.needs_dynamic_qwen and d.needs_dynamic_video_vae,
          "composite source requests both dynamic passes")

    d = deps("frame_reencode")
    check(d.needs_anchor_reencode and d.needs_dynamic_video_vae,
          "the re-encoded frame requests a dynamic VAE pass")

    d = deps("clamped_target_overlap")
    check(d.needs_sampler_intervention, "clamping requests a sampler intervention")

    prompted = experiments.CATALOG["frame_direct_prompted"]
    check(not prompted.dependencies().needs_dynamic_video_vae,
          "the prompted frame arm needs no dynamic video VAE")
    check(prompts.encode_key(prompted.prompt_policy) != "chunk_b",
          "the prompted frame arm needs its own Qwen encode")

    union = experiments.union_dependencies(["baseline_none", "frame_direct_corrected"])
    check(not union.needs_dynamic_qwen and not union.needs_dynamic_video_vae,
          "the direct-latent arms together need no dynamic preprocessing")
    union = experiments.union_dependencies(["baseline_none", "composite_source"])
    check(union.needs_dynamic_qwen, "the union picks up a dynamic dependency from one arm")


def test_suites():
    print("suites")
    ids = experiments.resolve_suite("minimal")
    check(ids == ["baseline_none", "frame_reencode_corrected",
                  "frame_direct_corrected", "frame_direct_stock_position"],
          "the minimal suite runs in lower-memory-first order")
    check(experiments.resolve_suite("reference")[0] == "baseline_none",
          "the baseline runs first in the reference suite")
    check(experiments.resolve_suite("custom", "aligned_overlap_direct,baseline_none")
          == ["baseline_none", "aligned_overlap_direct"],
          "a custom list is reordered into run order")
    raises(ValueError, lambda: experiments.resolve_suite("custom", "nope"),
           "an unknown experiment id fails before Chunk A is generated")
    raises(ValueError, lambda: experiments.resolve_suite("custom", ""),
           "an empty custom list is rejected")
    check("target_overlap_clamped" not in experiments.SUITES["all"],
          "'all' excludes the experiment that is not implemented")


def build_context():
    """A real `HarnessContext` with Phase A/B/C results filled in by hand.

    Using the real object rather than a stub is the point: a fake that answers
    `require()` differently would pass this test while the harness refused to
    run, or worse, ran anyway.
    """
    from chunked_ref2v.harness import HarnessContext, SeedSet

    context = HarnessContext(geo.DEFAULT_GEOMETRY, SeedSet(1), (128, 128))
    context.base_prompt = "a prompt"
    context.conditionings = {"chunk_b": [["cond", {"shared": True}]]}
    context.qwen_ref_items_b = [{"type": "video"}]
    context.dit_ref_blocks_b = [{"kind": "video", "latent": torch.zeros(1, 24, 22, 4, 4)}]
    context.source_ref_block_b = context.dit_ref_blocks_b[0]
    context.overlap_latent = torch.zeros(1, 24, 7, 4, 4)
    context.direct_frame_latent = torch.zeros(1, 24, 1, 4, 4)
    return context


def test_conditioning_isolation():
    print("conditioning isolation")
    context = build_context()
    a = strategies.get_strategy("direct_latent_overlap").prepare(
        context, experiments.CATALOG["aligned_overlap_direct"])
    b = strategies.get_strategy("direct_latent_frame").prepare(
        context, experiments.CATALOG["frame_direct_corrected"])

    a.qwen_ref_items.append({"type": "poison"})
    a.dit_ref_blocks.append({"kind": "poison"})
    a.target_conditions.append(cond(1))

    check(len(b.qwen_ref_items) == 1, "one arm cannot mutate another's Qwen ref items")
    check(len(b.dit_ref_blocks) == 1, "one arm cannot mutate another's reference blocks")
    check(len(b.target_conditions) == 1, "one arm cannot mutate another's target conditions")
    check(len(context.qwen_ref_items_b) == 1 and len(context.dit_ref_blocks_b) == 1,
          "the shared context is not mutated by preparing an arm")

    stock = strategies.get_strategy("direct_latent_frame").prepare(
        context, experiments.CATALOG["frame_direct_stock_position"])
    check(stock.position_policy == "stock" and b.position_policy == "copy_target",
          "position policy comes from the spec, not from the strategy")
    check(stock.target_conditions[0].latent is b.target_conditions[0].latent,
          "the stock diagnostic carries the identical latent source")

    raises(strategies.StrategyUnavailable,
           lambda: strategies.get_strategy("clamped_target_overlap").prepare(
               context, experiments.CATALOG["target_overlap_clamped"]),
           "the unimplemented clamp refuses rather than approximating")
    raises(strategies.StrategyUnavailable,
           lambda: strategies.get_strategy("composite_source").prepare(
               context, experiments.CATALOG["composite_source"]),
           "a strategy whose dynamic asset is missing refuses rather than sampling")

    # On a profile whose overlap does not land on a latent boundary the direct
    # assets are deliberately absent, and every direct-latent arm must refuse
    # rather than condition on an approximate slice.
    unaligned = build_context()
    unaligned.overlap_latent = None
    unaligned.direct_frame_latent = None
    for experiment_id, strategy_id in (("aligned_overlap_direct", "direct_latent_overlap"),
                                       ("frame_direct_corrected", "direct_latent_frame")):
        raises(strategies.StrategyUnavailable,
               lambda s=strategy_id, e=experiment_id: strategies.get_strategy(s).prepare(
                   unaligned, experiments.CATALOG[e]),
               "%s refuses when the direct latent asset is absent" % experiment_id)


def test_prompt_policies():
    print("prompt policies")
    base = "make it snow"
    check(prompts.build_prompt(base, "original") == base, "original leaves the prompt alone")
    check(prompts.build_prompt(base, "composite") == base,
          "composite leaves the prompt alone - <Video 1> is still the sole edit source")
    check(base in prompts.build_prompt(base, "keyframe_completion")
          and len(prompts.build_prompt(base, "keyframe_completion")) > len(base),
          "keyframe_completion appends rather than rewriting")
    check(prompts.encode_key("original") == "chunk_b"
          and prompts.encode_key("video2") != prompts.encode_key("composite"),
          "policies needing different encodes get different keys")


def test_metrics():
    print("metrics")
    g = geo.DEFAULT_GEOMETRY
    a_pixels = torch.rand(73, 8, 8, 3)
    b_pixels = a_pixels[51:73].clone()
    b_pixels = torch.cat([b_pixels, torch.rand(51, 8, 8, 3)])

    out = metrics.pixel_overlap_metrics(a_pixels, b_pixels, g)
    check(abs(out["pixel_overlap_mae"]) < 1e-6, "an identical overlap scores ~0 pixel MAE")
    check(len(out["pixel_overlap_mae_per_frame"]) == 22, "per-frame MAE has one value per frame")

    a_latent = torch.rand(1, 24, 22, 4, 4)
    b_latent = torch.cat([a_latent[:, :, 15:22], torch.rand(1, 24, 15, 4, 4)], dim=2)
    out = metrics.latent_overlap_metrics(a_latent, b_latent, g)
    check(abs(out["latent_overlap_mae"]) < 1e-6, "an identical latent overlap scores ~0")

    frozen = torch.zeros(22, 8, 8, 3)
    out = metrics.source_adherence(frozen, torch.rand(22, 8, 8, 3))
    check(out["motion_energy_ratio"] == 0.0,
          "a frozen output reports zero motion energy - the failure a low MAE can hide")

    unaligned = geo.HarnessGeometry(chunk_frames=73, overlap_frames=23)
    out = metrics.latent_overlap_metrics(a_latent, b_latent, unaligned)
    check(out["latent_overlap_mae"] is None and out.get("latent_overlap_note"),
          "an unaligned profile reports why rather than an approximate number")


def test_artifacts():
    print("artifact reuse")
    from chunked_ref2v import artifacts

    root = tempfile.mkdtemp(prefix="h3_harness_test_")
    try:
        g = geo.DEFAULT_GEOMETRY
        frames = torch.rand(73, 8, 8, 3)
        sigmas = torch.linspace(1.0, 0.0, 11)
        kwargs = dict(source_frames=frames, prompt="p", ref_pixels=[],
                      canvas=(128, 128), geometry=g, seed=7,
                      sampler_name="euler", sigmas=sigmas, checkpoint="ckpt")

        identity = artifacts.chunk_a_identity(**kwargs)
        check(artifacts.chunk_a_identity(**kwargs) == identity, "the identity is stable")
        check(artifacts.chunk_a_identity(**{**kwargs, "seed": 8}) != identity,
              "changing the Chunk A seed invalidates Chunk A")
        check(artifacts.chunk_a_identity(**{**kwargs, "prompt": "q"}) != identity,
              "changing the prompt invalidates Chunk A")
        check(artifacts.chunk_a_identity(
            **{**kwargs, "source_frames": torch.rand(73, 8, 8, 3)}) != identity,
              "changing the source frames invalidates Chunk A")
        check(artifacts.chunk_a_identity(
            **{**kwargs, "geometry": geo.HarnessGeometry(90, 22)}) != identity,
              "changing the geometry invalidates Chunk A")

        run_id = artifacts.new_run_id(identity)
        store = artifacts.RunStore(root, run_id)
        store.save_tensors("common", "chunk_a_output",
                           {"video_latent": torch.rand(1, 24, 22, 4, 4)}, identity=identity)
        store.write_manifest(chunk_a_identity=identity)

        reopened = artifacts.RunStore(root, run_id)
        check(reopened.load_tensors("common", "chunk_a_output", identity) is not None,
              "a stored asset reloads under the same identity")
        check(reopened.load_tensors("common", "chunk_a_output", "deadbeef") is None,
              "an asset from a different identity is refused")
        check(artifacts.find_reusable_run(root, identity) == run_id,
              "adding a Chunk B experiment finds and reuses the stored Chunk A")
        check(artifacts.find_reusable_run(root, "nomatch") is None,
              "a run with a different identity is not reused")

        path = os.path.join(reopened.common, "chunk_a_output.safetensors")
        with open(path, "wb") as fh:
            fh.write(b"not a safetensors file")
        check(reopened.load_tensors("common", "chunk_a_output", identity) is None,
              "a corrupt asset is rejected rather than partially loaded")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_report():
    print("report")
    g = geo.DEFAULT_GEOMETRY

    class Seeds:
        def as_dict(self):
            return {"node_seed": 1}

    document = report.build(
        run_id="r", geometry=g, seeds=Seeds(), canvas=(128, 128),
        experiment_ids=["baseline_none", "aligned_overlap_direct"],
        results=[
            {"experiment_id": "baseline_none", "status": "completed",
             "metrics": {"pixel_overlap_mae": 0.2, "motion_energy_ratio": 1.0}},
            {"experiment_id": "aligned_overlap_direct", "status": "cancelled",
             "note": "vram guard"},
        ],
        chunk_a_reused=True, dependencies=strategies.StrategyDependencies())

    check(document["profile"]["overlap_latent_start"] == 15, "the report records the mapping")
    check(document["experiments"]["aligned_overlap_direct"]["metrics"] == {},
          "a cancelled arm carries no invented metrics")
    text = report.to_text(document)
    check("baseline_none" in text and "cancelled" in text,
          "the text report names every arm and its status")


def main():
    for test in (test_temporal_mapping, test_layout_no_conditions,
                 test_layout_single_condition, test_layout_overlap_condition,
                 test_layout_shifting, test_layout_stock_policy,
                 test_layout_validation, test_row_consumption,
                 test_strategy_dependencies, test_suites,
                 test_conditioning_isolation, test_prompt_policies, test_metrics,
                 test_artifacts, test_report):
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
