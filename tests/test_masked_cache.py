"""Self-test for the H3 masked-Ref2V measurement stage. CPU only, no checkpoint.

Run from the ComfyUI root so `comfy` is importable:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_masked_cache.py

Covers the plan's §11 items that Stage 0 reaches: source resolution, the score
chain against a planted edit whose location is known in advance, tile expansion
and halos on grids small enough to write the answer out by hand, sigma-boundary
mask promotion, and the fail-closed paths.

The end-to-end check drives the real diffusion-model wrapper with a fake
executor and a `CONST` flow sampling object, so it exercises the actual
`calculate_denoised` relation rather than a re-derivation of it - including the
requirement that `measure` mode hand back the model's output untouched.

Never allocates on the GPU; safe to run while a generation is in flight.
"""

import json
import os
import shutil
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                                  # the package
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # ComfyUI root

# `--cpu` before the first comfy import keeps `model_management` from opening a
# CUDA context; see the note in test_chunked_ref2v.py.
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
sys.argv = [sys.argv[0], "--cpu"]

from h3_masked_cache import mask as mask_ops  # noqa: E402
from h3_masked_cache import report as report_mod  # noqa: E402
from h3_masked_cache import source as source_mod  # noqa: E402
from h3_masked_cache import wrappers  # noqa: E402
from h3_masked_cache.config import MaskedCacheConfig  # noqa: E402
from h3_masked_cache.session import MaskedCacheSession  # noqa: E402
from h3_probe import layout as h3_layout  # noqa: E402

TEXT_LEN = 40
LATENT_T, LATENT_H, LATENT_W = 7, 16, 24
AUDIO_T = 30
CHANNELS = 24

# the planted edit, in latent cells
EDIT_T = slice(2, 4)
EDIT_H = slice(6, 10)
EDIT_W = slice(8, 12)

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


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_source(t=LATENT_T, h=LATENT_H, w=LATENT_W, c=CHANNELS, batch=1):
    torch.manual_seed(7)
    return torch.randn(batch, c, t, h, w)


def video_block(latent, kind="video"):
    return {"kind": kind, "latent_t": latent.shape[2],
            "latent_h": latent.shape[3], "latent_w": latent.shape[4],
            "ref_audio_t": 0, "latent": latent, "audio_latent": None}


def build_layout(refs=None):
    from comfy.ldm.minimax.model import PackedLayout
    packed = PackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, refs=refs)
    return h3_layout.from_packed_layout(packed)


class FakeConstSampling:
    """`comfy.model_sampling.CONST.calculate_denoised`, standing in for the real one."""

    def calculate_denoised(self, sigma, model_output, model_input):
        sigma = sigma.reshape(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        return model_input - model_output * sigma


# --------------------------------------------------------------------------
# 11.1 source resolution
# --------------------------------------------------------------------------

def test_source_resolution():
    print("source resolution")
    src = make_source()
    target = torch.zeros_like(src)

    img = {"kind": "image", "latent_h": 8, "latent_w": 8, "latent": torch.zeros(1, 24, 1, 8, 8)}
    aud = {"kind": "audio", "ref_audio_t": 4, "audio_latent": torch.zeros(1, 32, 4)}
    vid = video_block(src)
    vid2 = video_block(src + 1.0, kind="video_audio")

    r = source_mod.resolve_source({"refs": [img, aud, vid]}, target, 1)
    check(r.valid and r.payload_index == 2, "one-based ordinal counts video refs only, not images/audio")

    r = source_mod.resolve_source({"refs": [vid, img, vid2]}, target, 2)
    check(r.valid and r.kind == "video_audio" and r.payload_index == 2,
          "second video reference selected, video_audio counts as a video")

    r = source_mod.resolve_source({"refs": [img]}, target, 1)
    check(not r.valid and "no video reference" in r.reason, "images alone -> no source, with a reason")

    r = source_mod.resolve_source({"refs": [vid]}, target, 2)
    check(not r.valid and "out of range" in r.reason, "index past the last video reference refuses")

    r = source_mod.resolve_source({}, target, 1)
    check(not r.valid, "empty payload refuses rather than sampling unmasked")

    short = video_block(make_source(t=LATENT_T - 1))
    r = source_mod.resolve_source({"refs": [short]}, target, 1)
    check(not r.valid and "temporal mismatch" in r.reason, "temporal mismatch named exactly")

    wide = video_block(make_source(w=LATENT_W + 2))
    r = source_mod.resolve_source({"refs": [wide]}, target, 1)
    check(not r.valid and "spatial mismatch" in r.reason, "spatial mismatch named exactly")

    chan = video_block(make_source(c=16))
    r = source_mod.resolve_source({"refs": [chan]}, target, 1)
    check(not r.valid and "channel mismatch" in r.reason, "channel mismatch named exactly")

    r = source_mod.resolve_source({"refs": [{"kind": "video", "latent": None}]}, target, 1)
    check(not r.valid and "no encoded latent" in r.reason, "a video block without a latent refuses")

    # the sampler can hand the model a batched target; a batch-1 source broadcasts
    r = source_mod.resolve_source({"refs": [vid]}, target.expand(2, *target.shape[1:]), 1)
    check(r.valid and r.latent.shape[0] == 2, "batch-1 source broadcasts to a batched target")

    two = video_block(src.repeat(2, 1, 1, 1, 1))
    r = source_mod.resolve_source({"refs": [two]}, target, 1)
    check(not r.valid and "broadcast" in r.reason, "a batch-2 source into a batch-1 target refuses")


# --------------------------------------------------------------------------
# 11.2 score calculation
# --------------------------------------------------------------------------

def test_scores():
    print("score calculation")
    src = make_source()
    x0 = src.clone()
    x0[:, :, EDIT_T, EDIT_H, EDIT_W] += 5.0

    score = mask_ops.latent_score(x0, src, 1e-3)
    check(tuple(score.shape) == (LATENT_T, LATENT_H, LATENT_W), "latent score is [T,H,W]")

    planted = torch.zeros(LATENT_T, LATENT_H, LATENT_W, dtype=torch.bool)
    planted[EDIT_T, EDIT_H, EDIT_W] = True
    check(float(score[~planted].abs().max()) == 0.0, "identical cells score exactly zero")
    check(float(score[planted].min()) > 0.0, "every planted cell scores above zero")

    # channel RMS, computed the long way on one cell
    cell = (x0 - src)[0, :, 2, 6, 8]
    want = float(cell.pow(2).mean().sqrt()) / (float(src[0, :, 2, 6, 8].pow(2).mean().sqrt()) + 1e-3)
    check(abs(float(score[2, 6, 8]) - want) < 1e-5, "channel RMS ratio matches a hand calculation")

    tokens = mask_ops.token_score(score)
    layout = build_layout()
    check(tuple(tokens.shape) == tuple(layout.video_shape),
          "token grid matches the packed layout's video shape %s" % (layout.video_shape,))

    active = tokens >= 0.5
    want_tokens = torch.zeros_like(active)
    want_tokens[EDIT_T, 3:5, 4:6] = True        # the 2x2 patches covering the planted cells
    check(bool((active == want_tokens).all()),
          "2x2 pooling maps the planted cuboid to exactly its own tokens, no neighbours")

    # one changed cell must keep its whole patch: max, not mean
    single = src.clone()
    single[:, :, 0, 1, 1] += 5.0
    st = mask_ops.token_score(mask_ops.latent_score(single, src, 1e-3))
    check(float(st[0, 0, 0]) > 0.0 and float(st[0].sum()) == float(st[0, 0, 0]),
          "a single changed cell activates its patch and only its patch")

    # odd latent sizes pad the way core pads before patching
    odd = mask_ops.token_score(torch.zeros(2, 7, 9))
    check(tuple(odd.shape) == (2, 4, 5), "odd latent grids round up to a full token grid")


# --------------------------------------------------------------------------
# 11.3 tile expansion
# --------------------------------------------------------------------------

def test_tiles():
    print("tile expansion")
    m = torch.zeros(1, 8, 12, dtype=torch.bool)
    m[0, 3, 4] = True

    tiles = mask_ops.to_tiles(m, 2, 2)
    check(tuple(tiles.shape) == (1, 4, 6) and int(tiles.sum()) == 1, "one active token -> one active 2x2 tile")
    back = mask_ops.from_tiles(tiles, 2, 2, 8, 12)
    want = torch.zeros_like(m)
    want[0, 2:4, 4:6] = True
    check(bool((back == want).all()), "the tile activates all four of its tokens")

    tiles4 = mask_ops.to_tiles(m, 4, 4)
    back4 = mask_ops.from_tiles(tiles4, 4, 4, 8, 12)
    want4 = torch.zeros_like(m)
    want4[0, 0:4, 4:8] = True
    check(bool((back4 == want4).all()), "a 4x4 tile activates its full 16 tokens")

    check(bool((mask_ops.from_tiles(mask_ops.to_tiles(m, 1, 1), 1, 1, 8, 12) == m).all()),
          "1x1 tiles are the identity")

    # odd grids: the edge tile is partial, kept whole, and cropped back
    odd = torch.zeros(1, 7, 9, dtype=torch.bool)
    odd[0, 6, 8] = True                                  # the far corner, in a partial tile
    t = mask_ops.to_tiles(odd, 2, 2)
    check(tuple(t.shape) == (1, 4, 5), "edge tiles are padded up, not dropped")
    b = mask_ops.from_tiles(t, 2, 2, 7, 9)
    check(tuple(b.shape) == (1, 7, 9) and bool(b[0, 6, 8]) and int(b.sum()) == 1,
          "a token alone in a partial edge tile survives the round trip")

    full = torch.ones(1, 7, 9, dtype=torch.bool)
    rt = mask_ops.from_tiles(mask_ops.to_tiles(full, 2, 2), 2, 2, 7, 9)
    check(int(rt.sum()) == 7 * 9, "no token is lost at odd grid sizes")


# --------------------------------------------------------------------------
# 11.4 halos
# --------------------------------------------------------------------------

def test_halos():
    print("spatial and temporal halo")
    m = torch.zeros(1, 5, 5, dtype=torch.bool)
    m[0, 2, 2] = True

    d1 = mask_ops.dilate_spatial(m, 1)
    want = torch.zeros_like(m)
    want[0, 1:4, 1:4] = True
    check(bool((d1 == want).all()), "spatial halo 1 grows one cell in every direction")
    check(bool((mask_ops.dilate_spatial(m, 0) == m).all()), "halo 0 is the identity")

    corner = torch.zeros(1, 5, 5, dtype=torch.bool)
    corner[0, 0, 0] = True
    dc = mask_ops.dilate_spatial(corner, 1)
    check(int(dc.sum()) == 4, "a corner cell dilates into the grid, not off it")

    t = torch.zeros(5, 2, 2, dtype=torch.bool)
    t[2, 0, 0] = True
    dt = mask_ops.dilate_temporal(t, 1)
    check(bool(dt[1, 0, 0]) and bool(dt[3, 0, 0]) and int(dt.sum()) == 3,
          "temporal halo 1 reaches the neighbouring latent frames only")
    check(int(mask_ops.dilate_temporal(t, 2).sum()) == 5, "temporal halo 2 reaches two frames out")
    check(bool((mask_ops.dilate_temporal(t, 0) == t).all()), "temporal halo 0 is the identity")

    # the full chain, on a grid where the answer is writable by hand
    scores = torch.zeros(3, 4, 4)
    scores[1, 0, 0] = 1.0
    core, expanded, tiles = mask_ops.build_mask(scores, 0.5, 2, 2, 1, 1)
    check(int(core.sum()) == 1, "threshold alone keeps one token")
    check(int(tiles.sum()) == 3 * 4, "one tile, dilated to 2x2 tiles and 3 frames")
    check(int(expanded.sum()) == 3 * 4 * 4, "which is 4x4 tokens on every frame")

    check(mask_ops.jaccard(core, core) == 1.0, "jaccard of a mask with itself is 1")
    check(mask_ops.jaccard(core, torch.zeros_like(core)) == 0.0, "disjoint masks score 0")
    check(mask_ops.jaccard(torch.zeros_like(core), torch.zeros_like(core)) == 1.0,
          "two empty masks are identical, not undefined")
    check(mask_ops.escaped_fraction(expanded, core) > 0.0
          and mask_ops.escaped_fraction(core, expanded) == 0.0,
          "escaped_fraction is directional: a subset escapes nothing")


# --------------------------------------------------------------------------
# 11.9 sigma-boundary promotion
# --------------------------------------------------------------------------

def test_sigma_promotion():
    print("sigma-boundary mask promotion")
    cfg = MaskedCacheConfig(run_tag="promo")
    session = MaskedCacheSession(cfg, tempfile.gettempdir())
    run = session.begin()

    a = torch.zeros(1, 2, 2, dtype=torch.bool)
    a[0, 0, 0] = True
    b = torch.zeros(1, 2, 2, dtype=torch.bool)
    b[0, 1, 1] = True

    check(run.observe_sigma(1.0), "the first call opens a sigma")
    run.stage_mask(a)
    check(run.active_mask is None, "a mask inferred at sigma A is not active at sigma A")
    check(not run.observe_sigma(1.0), "the second condition at sigma A is not a new sigma")
    check(run.active_mask is None, "the uncond branch at sigma A still sees no mask")

    check(run.observe_sigma(0.8), "sigma B opens a new sigma")
    check(run.active_mask is not None and bool((run.active_mask == a).all()),
          "sigma A's candidate becomes active at sigma B")
    check(run.pending_mask is None, "promotion clears the pending mask")

    run.stage_mask(b)
    run.observe_sigma(0.6)
    check(int(run.active_mask.sum()) == 1 and bool(run.active_mask[0, 1, 1]),
          "each sigma's candidate replaces the last as the active mask")
    check(int(run.union_mask.sum()) == 2, "the union keeps every token ever active")

    run.stage_mask(a)
    run.stage_mask(b)
    check(int(run.pending_mask.sum()) == 2, "two observations at one sigma union rather than replace")

    session.sources.release()
    session.run = None


# --------------------------------------------------------------------------
# end to end: the real wrapper, a fake executor
# --------------------------------------------------------------------------

def _drive(payload, strict=True, sigmas=(1.0, 0.8), out_dir=None, cond_and_uncond=True):
    """Run the diffusion-model wrapper over a synthetic schedule.

    Returns `(session, run, calls)` where `calls` records what the executor was
    handed and what the wrapper gave back, so output neutrality is checkable.
    """
    cfg = MaskedCacheConfig(score_threshold=0.5, tile_h=1, tile_w=1,
                            spatial_halo=0, temporal_halo=0, strict=strict,
                            run_tag="e2e")
    session = MaskedCacheSession(cfg, out_dir or tempfile.gettempdir(),
                                 model_sampling=FakeConstSampling())
    run = session.begin()
    wrapper = wrappers.make_diffusion_wrapper(session)

    src = make_source()
    x0 = src.clone()
    x0[:, :, EDIT_T, EDIT_H, EDIT_W] += 5.0
    audio_x = torch.zeros(1, 32, AUDIO_T)
    context = torch.zeros(1, TEXT_LEN, 5120)
    calls = []

    for sigma in sigmas:
        for cu in ((0, 1) if cond_and_uncond else (0,)):
            sigma_t = torch.tensor([sigma])
            # the velocity whose denoised result is exactly the planted x0
            video_out = (src - x0) / sigma
            out = [video_out.clone(), torch.zeros_like(audio_x)]
            transformer_options = {
                "sigmas": sigma_t,
                "sample_sigmas": torch.tensor(list(sigmas) + [0.0]),
                "cond_or_uncond": [cu],
            }

            def executor(*args, **kwargs):
                return out

            got = wrapper(executor, [src, audio_x], sigma_t * 1000, context,
                          transformer_options, minimax_payload=payload)
            calls.append((out, got))
    return session, run, calls


def test_end_to_end(tmp):
    print("end to end (fake executor, real wrapper)")
    src = make_source()
    payload = {"refs": [video_block(src)]}
    session, run, calls = _drive(payload, out_dir=tmp)

    check(all(got is out for out, got in calls),
          "measure mode returns the model's own output object, unmodified")
    check(len(run.steps) == 2, "one row per conditional forward; the uncond branch is not scored")
    check(run.disabled_reason is None, "a well-formed run is never disabled")
    check(run.sigma_count == 2, "two distinct sigmas observed across four forwards")

    row = run.steps[0]
    # sigma cancels out of (src - x0)/sigma, so the recovered x0 is exact and the
    # active set is the planted cuboid's tokens: 2 frames x 2 x 1 tokens
    total = LATENT_T * (LATENT_H // 2) * (LATENT_W // 2)
    check(abs(row["active_core"] - (2 * 2 * 2) / total) < 1e-9,
          "the recovered mask is exactly the planted cuboid (%.2f%% of the target)"
          % (100.0 * row["active_core"]))
    check(row["jaccard_prev"] is None, "the first row has no previous mask to compare against")
    check(run.steps[1]["jaccard_prev"] == 1.0, "an unchanged edit gives J=1 between steps")
    check(run.steps[1]["escaped_union"] == 0.0, "nothing escapes a union that already contains it")
    check(len(row["threshold_sweep"]) == len(report_mod.THRESHOLD_SWEEP),
          "every swept threshold is reported")
    check(row["threshold_sweep"][0]["active_core"] >= row["threshold_sweep"][-1]["active_core"],
          "the sweep is monotonically decreasing in threshold")


def test_report(tmp):
    print("report artifacts")
    src = make_source()
    session, run, _ = _drive({"refs": [video_block(src)]}, out_dir=tmp)
    path = session.end()

    check(path is not None and os.path.exists(path), "report.txt written")
    for name in ("summary.json", "steps.jsonl", "mask.npz"):
        check(os.path.exists(os.path.join(run.out_dir, name)), "%s written" % name)

    with open(os.path.join(run.out_dir, "summary.json"), encoding="utf-8") as f:
        data = json.load(f)
    check(data["config"]["mode"] == "measure", "the config is recorded alongside the numbers")
    check(len(data["steps"]) == 2 and data["summary"]["distinct_sigmas"] == 2,
          "summary.json carries every step and the sigma count")
    check(data["source"]["ref_ordinal"] == 1, "the resolved source is named in the report")

    with open(os.path.join(run.out_dir, "steps.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    check(len(rows) == 2, "steps.jsonl has one line per observed forward")

    import numpy as np
    with np.load(os.path.join(run.out_dir, "mask.npz")) as z:
        check("mask_union" in z and any(k.startswith("score_") for k in z.files),
              "mask.npz carries the score maps and the union mask")

    text = open(path, encoding="utf-8").read()
    check("THRESHOLD SWEEP" in text and "MiniMax H3 masked Ref2V" in text,
          "report.txt renders the sweep")
    check("status:      complete" in text, "a finished report is labelled complete")


def test_progress_write(tmp):
    """A run in flight must be distinguishable from a finished one, and the
    per-step write must not rebuild the arrays or the policy sweep."""
    print("progress writes and completion status")
    import numpy as np

    src = make_source()
    out = os.path.join(tmp, "progress")
    session, run, _ = _drive({"refs": [video_block(src)]}, out_dir=out)
    npz = os.path.join(run.out_dir, "mask.npz")

    report_mod.write_run(run, complete=False)
    with open(os.path.join(run.out_dir, "summary.json"), encoding="utf-8") as f:
        data = json.load(f)
    check(data["status"] == "running", "a progress write is labelled running")
    check(data["summary"]["policy_sweep"] is None,
          "the progress write skips the costly policy sweep")
    check(not os.path.exists(npz),
          "the progress write does not recompress mask.npz")
    check("RUNNING" in open(os.path.join(run.out_dir, "report.txt"),
                            encoding="utf-8").read(),
          "a partial report.txt says so in its header")

    report_mod.write_run(run, complete=False, arrays=True)
    check(os.path.exists(npz), "an array checkpoint preserves the raw maps mid-run")
    with np.load(npz) as z:
        check(json.loads(str(z["index"]))["status"] == "running",
              "a checkpointed archive is labelled running, not complete")

    session.end()
    with open(os.path.join(run.out_dir, "summary.json"), encoding="utf-8") as f:
        data = json.load(f)
    check(data["status"] == "complete", "the closing write flips status to complete")
    check(data["summary"]["policy_sweep"] is not None,
          "the closing write carries the policy sweep")
    with np.load(npz) as z:
        check(json.loads(str(z["index"]))["status"] == "complete",
              "the closing archive is labelled complete")


def test_wall_times(tmp):
    """The dense baseline any future compact execution has to beat."""
    print("wall-clock accounting")
    src = make_source()
    _, run, _ = _drive({"refs": [video_block(src)]},
                       out_dir=os.path.join(tmp, "timing"))

    check(all(s["dense_wall_s"] is not None and s["dense_wall_s"] >= 0.0
              for s in run.steps),
          "every guided row carries the dense DiT wall time")
    check(run.steps[0]["step_wall_s"] is None,
          "step wall time is a delta, so the first observation has none")
    check(run.steps[1]["step_wall_s"] is not None and run.steps[1]["step_wall_s"] >= 0.0,
          "later observations carry the sync-accurate per-step wall time")
    check(run.pending_dense_wall_s > 0.0,
          "an unscored uncond forward leaves its time pending for the next prediction")

    # ...and when the scored conditional branch is last, nothing is left over.
    _, cond_only, _ = _drive({"refs": [video_block(src)]}, cond_and_uncond=False,
                             out_dir=os.path.join(tmp, "timing_cond"))
    check(cond_only.pending_dense_wall_s == 0.0,
          "the accumulator is cleared once a prediction has consumed it")


# --------------------------------------------------------------------------
# 11.10 fail-closed
# --------------------------------------------------------------------------

def test_fail_closed(tmp):
    print("fail-closed paths")
    raises(RuntimeError, lambda: _drive({"refs": []}, strict=True, out_dir=tmp),
           "strict: no source video reference stops the run")
    raises(RuntimeError,
           lambda: _drive({"refs": [video_block(make_source(t=LATENT_T - 1))]},
                          strict=True, out_dir=tmp),
           "strict: a source of the wrong length stops the run")

    session, run, calls = _drive({"refs": []}, strict=False, out_dir=tmp)
    check(all(got is out for out, got in calls), "non-strict: the dense output still comes back")
    check(run.disabled_reason is not None and not run.steps,
          "non-strict: measurement disables itself and records nothing rather than guessing")
    path = session.end()
    text = open(path, encoding="utf-8").read()
    check("MEASUREMENT DISABLED" in text,
          "the report says so loudly - a fallback cannot be mistaken for a measurement")

    # a run whose sigma has collapsed is not measurable either
    raises(RuntimeError,
           lambda: _drive({"refs": [video_block(make_source())]}, strict=True,
                          sigmas=(1e-9,), out_dir=tmp),
           "strict: a sigma below the stability floor stops the run")

    # config validation is fail-closed too
    raises(ValueError, lambda: MaskedCacheConfig(mode="compact"), "an unknown mode is rejected")
    raises(ValueError, lambda: MaskedCacheConfig(source_video_ref=0),
           "a zero source_video_ref is rejected (the widget is one-based)")
    raises(ValueError, lambda: MaskedCacheConfig(score_absolute_floor=0.0),
           "a zero score floor is rejected")


def main():
    tmp = tempfile.mkdtemp(prefix="h3_masked_cache_test_")
    try:
        test_source_resolution()
        test_scores()
        test_tiles()
        test_halos()
        test_sigma_promotion()
        test_end_to_end(tmp)
        test_report(tmp)
        test_progress_write(tmp)
        test_wall_times(tmp)
        test_fail_closed(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if _failures:
        print("\n%d masked-cache self-test failure(s):" % len(_failures))
        for f in _failures:
            print("  - %s" % f)
        raise SystemExit(1)
    print("\nall masked-cache self-tests passed")


if __name__ == "__main__":
    main()
