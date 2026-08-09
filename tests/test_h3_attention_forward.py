"""Parity self-test for the H3-owned block attention forward. CPU only.

Run from the ComfyUI root so `comfy` is importable:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_h3_attention_forward.py

Drives core's real `comfy.ldm.minimax.model.Attention` and the replacement
forward over the same weights and inputs, and requires the outputs to be
bit-identical. Commit 2 changes no numerics, so anything other than exact
equality means the fork has drifted from core.

The fused RMSNorm/RoPE op has an eager backend that runs on CPU, so the rotary
path is covered here without touching the GPU.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                       # the package
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))  # ComfyUI root

import comfy.ops  # noqa: E402
from comfy.ldm.minimax.model import Attention, rope_rotation_table  # noqa: E402

from h3_attention import forward as h3_forward  # noqa: E402
from h3_attention import patch as h3_patch  # noqa: E402
from h3_attention.observer import observing  # noqa: E402
from h3_attention.hybrid.stats import DeferredCudaTiming  # noqa: E402
from h3_runtime.timing import publish_timing  # noqa: E402

HIDDEN, HEADS, HEAD_DIM = 96, 2, 128
SEQ = 16
ROT_PAIRS = 48                       # rot_dim 96 over a 128 head_dim: partial rotary
DTYPE = torch.bfloat16


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def build_attention(seed=0):
    torch.manual_seed(seed)
    attn = Attention(HIDDEN, HEADS, HEAD_DIM, eps=1e-6, dtype=DTYPE, device="cpu",
                     operations=comfy.ops.disable_weight_init)
    for p in attn.parameters(recurse=True):
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, dtype=torch.float32).to(DTYPE) * 0.05)
        # ComfyUI loads inference weights without grad; the fused in-place RoPE
        # refuses to run if any operand (including the RMSNorm scales, which
        # cast_to passes through untouched) still requires it
        p.requires_grad_(False)
    return attn


def build_inputs(seed=1):
    torch.manual_seed(seed)
    x = (torch.randn(SEQ, HIDDEN, dtype=torch.float32) * 0.1).to(DTYPE)
    angles = torch.randn(SEQ, ROT_PAIRS * 2, dtype=torch.float32)
    rope = rope_rotation_table(angles, DTYPE)
    return x, rope


def test_parity_with_rope():
    print("parity (rotary path)")
    attn = build_attention()
    x, rope = build_inputs()

    want = attn.forward(x.clone(), rope_freqs=rope, transformer_options={})
    fn = h3_forward.make_forward(attn, layer_index=0)
    got = fn(x.clone(), rope_freqs=rope, transformer_options={})

    check(got.shape == want.shape == (SEQ, HIDDEN), "output is [seq, hidden] %s" % (tuple(got.shape),))
    check(got.dtype == want.dtype == DTYPE, "output dtype preserved (%s)" % got.dtype)
    check(torch.equal(got, want), "custom forward is bit-identical to core (rotary)")


def test_parity_without_rope():
    print("parity (no-rope path)")
    attn = build_attention(seed=3)
    x, _ = build_inputs(seed=4)

    want = attn.forward(x.clone(), rope_freqs=None, transformer_options={})
    fn = h3_forward.make_forward(attn, layer_index=7)
    got = fn(x.clone(), rope_freqs=None, transformer_options={})

    check(torch.equal(got, want), "custom forward is bit-identical to core (q_norm/k_norm)")


def test_views_not_copies():
    """The NHD->HND transposes must stay views: PLAN.md §1.3 turns on this."""
    print("layout is views")
    attn = build_attention(seed=5)
    x, rope = build_inputs(seed=6)
    q, k, v = h3_forward.project_qkv(attn, x, rope)

    check(q.data_ptr() != k.data_ptr() != v.data_ptr(), "q, k, v are distinct spans")
    base = q.untyped_storage().data_ptr()
    check(k.untyped_storage().data_ptr() == base and v.untyped_storage().data_ptr() == base,
          "q, k, v share one fused QKV allocation")

    qh, kh, vh = h3_forward.to_hnd(q, k, v)
    check(qh.shape == (1, HEADS, SEQ, HEAD_DIM), "HND shape is [1, heads, seq, dim]")
    check(qh.data_ptr() == q.data_ptr(), "HND transpose copies nothing")
    check(not qh.is_contiguous(), "HND view is strided, as core's is")

    stride = q.stride(0)
    check(stride == HEADS * HEAD_DIM * 3,
          "sequence stride is 3 * heads * head_dim (%d) - the fused-QKV stride "
          "the int32 overflow turns on" % stride)


def test_v_stride_guard():
    """The u32 guard relocated out of ComfyUI's tree. PLAN.md §2.3."""
    print("v stride guard")
    attn = build_attention(seed=13)
    x, rope = build_inputs(seed=14)
    q, k, v = h3_forward.project_qkv(attn, x, rope)
    _, _, vh = h3_forward.to_hnd(q, k, v)

    guarded = h3_forward.guard_v_stride(vh)
    check(guarded is vh, "short sequences are passed through untouched (no copy)")

    # the real H3 geometry: stride 21504, unsigned wrap at 2^32
    real_stride = 56 * 128 * 3
    limit_row = h3_forward.V_OFFSET_LIMIT // real_stride
    check(limit_row == 199728,
          "u32 ceiling at H3's fused stride is row 199,728 (got %d)" % limit_row)

    # a tensor whose max offset actually exceeds the limit must be copied, but
    # allocating one is impossible here - drive the predicate directly instead
    def max_offset(shape, stride):
        return sum((n - 1) * s for n, s in zip(shape, stride))

    under = max_offset((1, 56, 199_000, 128), (0, 128, real_stride, 1))
    over = max_offset((1, 56, 200_000, 128), (0, 128, real_stride, 1))
    check(under <= h3_forward.V_OFFSET_LIMIT < over,
          "the guard's threshold brackets the measured 199,728-row ceiling")


def test_observer():
    print("observation")
    seen = []
    attn = build_attention(seed=7)
    x, rope = build_inputs(seed=8)
    fn = h3_forward.make_forward(attn, layer_index=42)

    to = {}
    with observing(to, lambda q, k, layer_index: seen.append((tuple(q.shape), layer_index))):
        fn(x.clone(), rope_freqs=rope, transformer_options=to)

    check(len(seen) == 1, "one observation per forward")
    shape, layer = seen[0]
    check(shape == (1, HEADS, SEQ, HEAD_DIM), "observer sees HND q %s" % (shape,))
    check(layer == 42, "observer sees the real block index, not a counter")

    # unobserved is the normal case and must cost nothing
    fn(x.clone(), rope_freqs=rope, transformer_options={})
    check(len(seen) == 1, "no observers installed -> no capture")


def test_attention_injection():
    print("attention injection")
    attn = build_attention(seed=9)
    x, rope = build_inputs(seed=10)
    calls = []

    def fake_attention(q, k, v, heads, **kwargs):
        calls.append((tuple(q.shape), heads, kwargs.get("skip_reshape")))
        return torch.zeros(1, SEQ, HEADS * HEAD_DIM, dtype=q.dtype)

    fn = h3_forward.make_forward(attn, layer_index=1, attention=fake_attention)
    out = fn(x.clone(), rope_freqs=rope, transformer_options={})

    check(len(calls) == 1, "attention backend called once")
    shape, heads, skip = calls[0]
    check(shape == (1, HEADS, SEQ, HEAD_DIM), "backend receives HND %s" % (shape,))
    check(heads == HEADS and skip is True, "backend receives heads and skip_reshape=True")
    check(out.shape == (SEQ, HIDDEN), "out_proj applied to the backend result")


def test_projected_attention_injection():
    print("fused projection injection")
    attn = build_attention(seed=18)
    x, rope = build_inputs(seed=19)
    calls = []
    carrier = object()

    class Projector:
        name = "fake_fused_qkv"

        def project(self, module, value, rope_freqs, *, layer_index, transformer_options):
            calls.append(("project", module is attn, value is x, rope_freqs is rope, layer_index))
            return carrier

    class Backend:
        name = "fake_projected_backend"

        def prepare_projected(self, projected, *, layer_index, transformer_options):
            calls.append(("prepare_projected", projected is carrier, layer_index))
            return projected

        def execute(self, prepared):
            calls.append(("execute", prepared is carrier))
            return torch.zeros((1, HEADS, SEQ, HEAD_DIM), dtype=x.dtype)

    fn = h3_forward.make_forward(
        attn,
        layer_index=23,
        backend=Backend(),
        projector=Projector(),
    )
    out = fn(x, rope_freqs=rope, transformer_options={})
    check([item[0] for item in calls] == ["project", "prepare_projected", "execute"],
          "fused carrier flows directly from projector to consuming backend")
    check(calls[0][1:] == (True, True, True, 23),
          "fused projector receives the exact H3 call contract")
    check(out.shape == (SEQ, HIDDEN), "out_proj consumes projected-backend HND output")


def test_projector_observer_fallback():
    print("fused projection observer fallback")
    attn = build_attention(seed=20)
    x, rope = build_inputs(seed=21)
    calls = []
    seen = []

    class Projector:
        name = "must_not_run"

        def project(self, *args, **kwargs):
            raise AssertionError("observer path must retain BF16 Q/K visibility")

    class Backend:
        name = "fake_observed_backend"

        def prepare(self, q, k, v, *, layer_index, transformer_options):
            calls.append((tuple(q.shape), layer_index))
            return object()

        def execute(self, prepared):
            return torch.zeros((1, HEADS, SEQ, HEAD_DIM), dtype=x.dtype)

    fn = h3_forward.make_forward(
        attn,
        layer_index=24,
        backend=Backend(),
        projector=Projector(),
    )
    options = {}
    with observing(options, lambda q, k, layer_index: seen.append(layer_index)):
        fn(x, rope_freqs=rope, transformer_options=options)
    check(seen == [24], "observer still receives the BF16 Q/K path")
    check(calls == [((1, HEADS, SEQ, HEAD_DIM), 24)],
          "observed calls use the backend's established BF16 preparation path")


def test_deferred_projection_stage_counts():
    print("deferred attention stage timing")
    attn = build_attention(seed=16)
    x, rope = build_inputs(seed=17)
    clock = [0.0]
    events = []

    class FakeTimingEvent:
        def __init__(self):
            self.value = None

        def record(self):
            clock[0] += 1.0
            self.value = clock[0]

        def synchronize(self):
            events.append("synchronize")

        def elapsed_time(self, other):
            return other.value - self.value

    timing = DeferredCudaTiming(True, event_factory=lambda **kwargs: FakeTimingEvent())
    timing.begin_request(5, cuda=True)
    options = {}
    publish_timing(options, timing)

    with torch.no_grad():
        h3_forward.make_forward(attn, layer_index=16)(
            x.clone(), rope_freqs=rope, transformer_options=options
        )
    summary = timing.resolve(1.0)
    for stage in ("qkv_proj", "qk_rmsnorm_rope", "out_proj"):
        check(summary["stages"][stage]["count"] == 1,
              "%s runs once per attention forward" % stage)
    check(events.count("synchronize") == 1,
          "attention timing synchronizes once at request end")


# --------------------------------------------------------------------------
# patch installation
# --------------------------------------------------------------------------

class FakeBlock(torch.nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn


class FakePatcher:
    def __init__(self, blocks):
        self._blocks = torch.nn.ModuleList(blocks)
        self.object_patches = {}

    def get_model_object(self, name):
        if name == h3_patch.BLOCKS_ATTR:
            return self._blocks
        raise AttributeError(name)

    def add_object_patch(self, name, obj):
        self.object_patches[name] = obj


def build_patcher(n=3):
    return FakePatcher([FakeBlock(build_attention(seed=i)) for i in range(n)])


def test_patch_install():
    print("patch install")
    p = build_patcher()
    n = h3_patch.install(p)
    check(n == 3, "patched every main block (%d)" % n)
    check(set(p.object_patches) == {h3_patch.key_for(i) for i in range(3)},
          "patch keys address blocks.<i>.attn.forward")
    check(all(getattr(f, "_h3_layer_index") == i
              for i, f in ((i, p.object_patches[h3_patch.key_for(i)]) for i in range(3))),
          "each patched forward carries its own block index")

    check(h3_patch.install(p) == 0, "re-installing is idempotent")


def test_patch_configuration_identity():
    print("patch configuration identity")

    class Backend:
        name = "hybrid_sparse"

        def __init__(self, budget):
            self.installation_signature = (self.name, float(budget))

    p = build_patcher()
    h3_patch.install(p, backend=Backend(0.5))
    check(h3_patch.install(p, backend=Backend(0.5)) == 0,
          "equivalent backend signatures remain idempotent")
    try:
        h3_patch.install(p, backend=Backend(0.25))
    except h3_patch.H3PatchError as exc:
        check("already patched" in str(exc),
              "changed backend signature raises a configuration conflict")
    else:
        raise AssertionError("changed backend signature retained stale closures")


def test_patch_conflict():
    print("patch conflict")
    p = build_patcher()
    p.add_object_patch(h3_patch.key_for(1), lambda *a, **k: None)
    try:
        h3_patch.install(p)
    except h3_patch.H3PatchError as exc:
        check("blocks.1.attn.forward" in str(exc), "conflict names the contested key")
    else:
        raise AssertionError("a foreign patch on the same forward must raise")


def test_patch_validation():
    print("patch validation")
    p = build_patcher()
    del p._blocks[0].attn.q_norm
    try:
        h3_patch.validate(p)
    except h3_patch.H3PatchError as exc:
        check("q_norm" in str(exc), "missing attribute is named in the error")
    else:
        raise AssertionError("a drifted core Attention must raise")

    class Empty:
        object_patches = {}

        def get_model_object(self, name):
            raise AttributeError(name)

    try:
        h3_patch.validate(Empty())
    except h3_patch.H3PatchError as exc:
        check("MiniMax H3" in str(exc), "a non-H3 model is rejected by name")
    else:
        raise AssertionError("a non-H3 model must raise")


def test_training_rejected():
    print("inference only")
    import comfy.model_management as mm
    attn = build_attention(seed=11)
    x, rope = build_inputs(seed=12)
    fn = h3_forward.make_forward(attn, layer_index=0)

    original = mm.in_training
    mm.in_training = True
    try:
        fn(x.clone(), rope_freqs=rope, transformer_options={})
    except RuntimeError as exc:
        check("inference-only" in str(exc), "training mode raises rather than silently diverging")
    else:
        raise AssertionError("training mode must raise")
    finally:
        mm.in_training = original


def main():
    # the fused in-place RoPE refuses to run under autograd, so sampling always
    # happens inside no_grad; reproduce that rather than working around it
    with torch.no_grad():
        test_parity_with_rope()
        test_parity_without_rope()
        test_views_not_copies()
        test_v_stride_guard()
        test_observer()
        test_attention_injection()
        test_projected_attention_injection()
        test_projector_observer_fallback()
        test_deferred_projection_stage_counts()
        test_patch_install()
        test_patch_configuration_identity()
        test_patch_conflict()
        test_patch_validation()
        test_training_rejected()
    print("\nall H3 attention forward self-tests passed")


if __name__ == "__main__":
    main()
