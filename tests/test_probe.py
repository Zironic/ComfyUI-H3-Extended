"""Self-test for the H3 attention probe. No model or checkpoint required.

Run from the ComfyUI root so `comfy` is importable:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_probe.py

Builds a real `PackedLayout`, drives `_block_stats` with synthetic Q/K whose
attention target is known in advance, and checks that the reported masses are
conserved and land where they should.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # the package
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # ComfyUI root

from h3_attention.observer import OBSERVER_KEY, notify_attention, observing  # noqa: E402
from h3_probe import capture, layout as h3_layout, metrics  # noqa: E402

TEXT_LEN = 50
LATENT_T, LATENT_H, LATENT_W = 7, 16, 24
AUDIO_T = 30
HEADS, DIM = 4, 16


def build_layout(with_refs=True):
    from comfy.ldm.minimax.model import PackedLayout
    refs = [{"kind": "image", "latent_h": 8, "latent_w": 8}] if with_refs else None
    packed = PackedLayout(TEXT_LEN, LATENT_T, LATENT_H, LATENT_W, AUDIO_T, refs=refs)
    return h3_layout.from_packed_layout(packed)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def test_layout():
    print("layout")
    lay = build_layout()
    ph, pw = LATENT_H // 2, LATENT_W // 2
    check(lay.video_shape == (LATENT_T, ph, pw), "video_shape is (latent_t, h/2, w/2)")
    check(lay.frame_rows == ph * pw, "frame_rows = patch_h * patch_w")
    check(lay.video_range[1] - lay.video_range[0] == LATENT_T * ph * pw, "video rows = t * frame_rows")
    check(lay.audio_range[1] - lay.audio_range[0] == AUDIO_T * 2, "audio rows = 2 * audio_t (stereo, channel-major)")
    check(lay.text_range == (0, TEXT_LEN), "text is the leading segment")
    check(lay.video_range[1] == lay.seq_len, "target video is the final segment")
    check(lay.audio_range[1] == lay.video_range[0], "target audio packs directly before target video")
    check(len(lay.reference_ranges) == 1 and lay.reference_ranges[0][0] == "ref_img",
          "reference image surfaced as a ref_img range")

    covered = sum(b - a for a, b, _ in lay.segments)
    check(covered == lay.seq_len, "segments tile the sequence with no gaps")

    f0, f1 = lay.video_frame_range(3)
    check(lay.frame_of_row(f0) == 3 and lay.frame_of_row(f1 - 1) == 3, "frame_of_row inverts video_frame_range")
    check(lay.frame_of_row(0) is None, "text rows are not video frames")
    return lay


def test_selection(lay):
    print("selection")
    check(capture.resolve_indices("auto", 50) == [5, 24, 44], "auto layers -> early/middle/late")
    check(capture.resolve_indices("0,7,-1", 20) == [0, 7, 19], "explicit spec with negative index")
    check(capture.resolve_indices("99", 20) == [], "out-of-range indices dropped")

    qs = capture.select_query_blocks(lay, n_time=3, n_spatial=2, block=32)
    vid = [q for q in qs if q["kind"] == "video"]
    check(len(vid) == 6, "3 time positions x 2 spatial positions")
    check(all(q["stop"] - q["start"] == 32 for q in vid), "video query blocks are one block wide")
    for q in vid:
        check(lay.frame_of_row(q["start"]) == q["frame"]
              and lay.frame_of_row(q["stop"] - 1) == q["frame"],
              "query block t=%d stays inside its own frame" % q["frame"])
    check(any(q["kind"] == "audio" for q in qs), "an audio query block is included")


def test_block_stats(lay):
    """Synthetic Q/K: every query points at one key direction, so the mass must
    concentrate on the tokens carrying it."""
    print("block stats")
    seq = lay.seq_len
    torch.manual_seed(0)

    target_t = 4
    t0, t1 = lay.video_frame_range(target_t)

    direction = torch.zeros(DIM)
    direction[0] = 1.0
    k = torch.randn(1, HEADS, seq, DIM) * 0.01
    k[:, :, t0:t1, :] += direction * 8.0          # frame `target_t` is the only strong key
    q = torch.zeros(1, HEADS, seq, DIM)
    q[:, :, :, :] = direction * 8.0

    qs_, qe_ = lay.video_frame_range(2)
    qe_ = qs_ + 32
    stats = capture._block_stats(q, k, lay, qs_, qe_, block=32, head_chunk=2)

    bm = stats["block_mass"]
    cm = stats["cat_mass"]
    fm = stats["frame_mass"]
    check(bm.shape == (HEADS, stats["n_blocks"]), "block_mass is [heads, n_blocks]")
    check(torch.allclose(bm.sum(-1), torch.ones(HEADS), atol=1e-4), "block masses sum to 1 per head")
    check(torch.allclose(cm.sum(-1), torch.ones(HEADS), atol=1e-4), "category masses sum to 1 per head")
    check(torch.allclose(fm.sum(-1), cm[:, metrics.KINDS.index("video")], atol=1e-4),
          "per-frame masses sum to the video category mass")
    check(torch.allclose(stats["spatial_mass"].sum(), fm.mean(0).sum(), atol=1e-4),
          "per-position masses sum to the same video mass")

    peak = int(fm.mean(0).argmax())
    check(peak == target_t, "mass concentrates on the planted frame (t=%d)" % target_t)
    check(fm.mean(0)[target_t] > 0.9, "planted frame holds >90%% of mass")
    return stats


def test_metrics(lay, stats):
    print("metrics")
    rec = dict(stats)
    rec.update({"kind": "video", "frame": 2, "spatial_offset": 0,
                "start": lay.video_frame_range(2)[0], "stop": lay.video_frame_range(2)[0] + 32,
                "layer": 24, "step": 7, "sigma": 0.5, "cond_or_uncond": 0})
    a = metrics.analyze(rec, lay, adjacent=1)

    total = a["mandatory"] + a["target_video"]
    check(abs(total - 1.0) < 1e-3, "mandatory + target video accounts for all mass")
    check(abs(a["current_frame"] + a["adjacent_frames"] + a["other_frames"] - a["target_video"]) < 1e-3,
          "current/adjacent/other partition the video mass")
    check(abs(sum(a["by_temporal_distance"].values()) - a["target_video"]) < 1e-3,
          "temporal-distance buckets partition the video mass")
    check(abs(a["same_spatial_region"] + a["other_spatial"] - a["target_video"]) < 1e-3,
          "spatial split partitions the video mass")

    check(a["local_blocks"] >= a["local_exact"] - 1e-6,
          "block-granular local mask retains at least the exact local mass")
    ks = sorted(a["topk"])
    vals = [a["topk"][k] for k in ks]
    check(all(vals[i] <= vals[i + 1] + 1e-6 for i in range(len(vals) - 1)),
          "top-k coverage is monotonic in k")
    check(vals[-1] <= 1.0 + 1e-3, "coverage never exceeds 100%")
    check(a["local_exact"] < 0.5, "planted distant frame is correctly *not* covered by the local mask")
    check(vals[0] > 0.9, "top-4 distant blocks recover the planted frame")

    summary = metrics.summarize([a])
    check(summary["n"] == 1 and "topk" in summary, "summary aggregates")
    check(metrics.recommend(summary, target=0.99) in (None, 4, 8, 16, 32), "recommend returns a valid budget")

    from h3_probe import report
    text = report.render_record(a)
    check("local mask retained" in text and "local + top-4" in text, "report renders the decision lines")
    print("\n--- sample report record ---\n%s\n---" % text)


def test_patch():
    """The interception must be H3-scoped, idempotent, and transparent when idle."""
    print("patch")
    import comfy.ldm.minimax.model as mm
    from comfy.ldm.modules.attention import optimized_attention as core_attn

    original = mm.optimized_attention
    capture.install()
    capture.install()                                  # idempotent
    check(mm.optimized_attention is not original, "H3 module binding is swapped")
    check(core_attn is not mm.optimized_attention, "the shared attention entry point is untouched")

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 64, 16) for _ in range(3))
    got = mm.optimized_attention(q, k, v, 2, mask=None, skip_reshape=True)
    want = original(q, k, v, 2, mask=None, skip_reshape=True)
    check(torch.equal(got, want), "idle probe delegates bit-identically")

    # no transformer_options at all is the token-refiner / bare-call shape
    got = mm.optimized_attention(q, k, v, 2, mask=None, skip_reshape=True,
                                 transformer_options={})
    check(torch.equal(got, want), "empty transformer_options delegates bit-identically")

    capture.uninstall()
    check(mm.optimized_attention is original, "uninstall restores the original binding")


def test_observer_seam():
    """Both attention paths must reach the probe through one seam."""
    print("observer seam")
    seen = []

    def observer(q, k, layer_index):
        seen.append((tuple(q.shape), layer_index))

    to = {}
    check(OBSERVER_KEY not in to, "no observer key before install")
    notify_attention(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8),
                     layer_index=3, transformer_options=to)
    check(seen == [], "notify with no observers installed is a no-op")

    with observing(to, observer):
        check(len(to[OBSERVER_KEY]) == 1, "observer installed into transformer_options")
        notify_attention(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8),
                         layer_index=7, transformer_options=to)
    check(OBSERVER_KEY not in to, "observing() removes the key it added")
    check(seen == [((1, 2, 4, 8), 7)], "observer received shape and explicit layer index")

    # a failing observer must not reach the caller
    def boom(q, k, layer_index):
        raise RuntimeError("observer blew up")

    with observing(to, boom):
        notify_attention(torch.zeros(1, 2, 4, 8), torch.zeros(1, 2, 4, 8),
                         layer_index=0, transformer_options=to)
    check(True, "observer exception is isolated from inference")

    # nesting restores the outer list rather than clearing it
    with observing(to, observer):
        with observing(to, boom):
            check(len(to[OBSERVER_KEY]) == 2, "nested observers stack")
        check(len(to[OBSERVER_KEY]) == 1, "inner observer removed, outer retained")


def test_layer_attribution(lay):
    """Counted and explicit layer identification, and no double capture."""
    print("layer attribution")

    class FakeRun:
        def __init__(self, layers):
            self.layers = layers
            self.records = []
            self.n_time, self.n_spatial, self.block = 1, 1, 32
            self.include_audio, self.include_text = False, False

    seq = lay.seq_len
    q = torch.zeros(1, HEADS, seq, DIM)
    k = torch.zeros(1, HEADS, seq, DIM)
    short = torch.zeros(1, HEADS, 8, DIM)

    # counted: layer index comes from call order, refiner calls do not advance it
    probe = capture.ForwardProbe(FakeRun({2}), lay, step=0, sigma=1.0, cond_or_uncond=0)
    for _ in range(5):
        probe.observe(short, short)                    # token refiner, wrong seq_len
    check(probe.layer == -1, "short-sequence calls never advance the layer counter")
    for _ in range(4):
        probe.observe(q, k)
    check(probe.layer == 3, "counted path advances once per full-sequence call")
    check([r["layer"] for r in probe.run.records] == [2], "only the selected layer recorded")

    # explicit: the block index is taken from the caller, not the call order
    probe = capture.ForwardProbe(FakeRun({0, 49}), lay, step=0, sigma=1.0, cond_or_uncond=0)
    probe.observe(q, k, layer_index=49)
    probe.observe(q, k, layer_index=0)
    check([r["layer"] for r in probe.run.records] == [49, 0],
          "explicit indices are recorded as given, out of order")
    check(probe.layer == -1, "explicit path leaves the counter untouched")

    # once a caller identifies itself, counted observations are ignored
    probe = capture.ForwardProbe(FakeRun({0, 1}), lay, step=0, sigma=1.0, cond_or_uncond=0)
    probe.observe(q, k, layer_index=0)
    probe.observe(q, k)                                 # would be counted layer 0 -> duplicate
    check([r["layer"] for r in probe.run.records] == [0], "no double capture of the same layer")


def main():
    lay = test_layout()
    print("  layout: %s" % lay.describe())
    test_selection(lay)
    stats = test_block_stats(lay)
    test_metrics(lay, stats)
    test_patch()
    test_observer_seam()
    test_layer_attribution(lay)
    print("\nall probe self-tests passed")


if __name__ == "__main__":
    main()
