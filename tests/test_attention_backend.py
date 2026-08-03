"""Self-test for the H3 attention-backend override.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_attention_backend.py

SageAttention is a compiled CUDA dependency that may not be installed, so the
routing mechanism is exercised with a stand-in registered under a test name.
The stand-in is `wrap_attn`-decorated exactly like `attention_sage`, so the
`__wrapped__` unwrapping path this relies on is the real one.
"""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.ldm.modules.attention as A  # noqa: E402
from nodes_minimax_h3 import _set_h3_attention_backend  # noqa: E402

CALLS = []


@A.wrap_attn
def _spy_attention(q, k, v, heads, mask=None, attn_precision=None,
                   skip_reshape=False, skip_output_reshape=False, **kwargs):
    CALLS.append(dict(kwargs))
    return A.attention_pytorch(q, k, v, heads, mask=mask, skip_reshape=skip_reshape,
                               skip_output_reshape=skip_output_reshape, **kwargs)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def main():
    A.register_attention_function("h3_selftest", _spy_attention)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 64, 32) for _ in range(3))

    print("comfy passthrough")
    to = {}
    _set_h3_attention_backend(to, "comfy")
    check("optimized_attention_override" not in to, "'comfy' writes no override key")

    print("unavailable backend")
    try:
        _set_h3_attention_backend({}, "definitely_not_installed")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        check("not available" in str(e), "unavailable backend raises rather than falling back")
        check("h3_selftest" in str(e), "error lists the registered backends")

    print("routing")
    to = {}
    _set_h3_attention_backend(to, "h3_selftest")
    check("optimized_attention_override" in to, "override key written")

    CALLS.clear()
    got = A.optimized_attention(q, k, v, 4, mask=None, skip_reshape=True, transformer_options=to)
    check(len(CALLS) == 1, "override routed attention to the selected backend (%d call)" % len(CALLS))
    want = A.attention_pytorch(q, k, v, 4, mask=None, skip_reshape=True)
    check(torch.equal(got, want), "routed result matches the backend called directly")

    print("no re-entry")
    check(CALLS[0].get("_inside_attn_wrapper") is True,
          "wrap_attn's re-entry guard reaches the impl, so nested calls skip the override")
    CALLS.clear()
    A.optimized_attention(q, k, v, 4, mask=None, skip_reshape=True, transformer_options=to)
    check(len(CALLS) == 1, "still exactly one dispatch (no recursion)")

    print("probe composition")
    from h3_probe import capture
    import comfy.ldm.minimax.model as mm
    capture.install()
    try:
        CALLS.clear()
        out = mm.optimized_attention(q, k, v, 4, mask=None, skip_reshape=True, transformer_options=to)
        check(len(CALLS) == 1, "probe delegation still reaches the selected backend")
        check(torch.equal(out, want), "probe + backend override produce the backend's result")
    finally:
        capture.uninstall()

    print("explicit pytorch baseline")
    to_pt = {}
    _set_h3_attention_backend(to_pt, "pytorch")
    check("optimized_attention_override" in to_pt,
          "'pytorch' pins a dense baseline regardless of --use-sage-attention")
    got_pt = A.optimized_attention(q, k, v, 4, mask=None, skip_reshape=True, transformer_options=to_pt)
    check(torch.equal(got_pt, A.attention_pytorch(q, k, v, 4, mask=None, skip_reshape=True)),
          "pinned pytorch matches attention_pytorch exactly")

    print("real sage backend")
    if "sage" not in A.REGISTERED_ATTENTION_FUNCTIONS:
        print("  SKIP: sageattention not installed in this environment")
    elif not torch.cuda.is_available():
        print("  SKIP: no CUDA device")
    else:
        to_sage = {}
        _set_h3_attention_backend(to_sage, "sage")
        # H3 calls attention as [1, heads, seq, dim] with skip_reshape=True
        qs, ks, vs = (torch.randn(1, 8, 2048, 64, device="cuda", dtype=torch.float16) for _ in range(3))
        got = A.optimized_attention(qs, ks, vs, 8, mask=None, skip_reshape=True,
                                    transformer_options=to_sage)
        ref = A.attention_pytorch(qs, ks, vs, 8, mask=None, skip_reshape=True)
        check(got.shape == ref.shape, "sage output shape matches pytorch (%s)" % (tuple(got.shape),))
        d = (got.float() - ref.float()).abs()
        rms = ref.float().pow(2).mean().sqrt()
        rel = float(d.mean() / rms)
        check(rel < 0.05, "sage matches pytorch within INT8 quantization error (rel %.4f)" % rel)
        # a silent kernel fallback would make benchmarks meaningless, so prove it did not happen
        check(not torch.equal(got, ref), "sage did NOT silently fall back to pytorch (results differ)")

    print("\nall attention-backend self-tests passed")


if __name__ == "__main__":
    main()
