"""Self-test for the int64 Q/K quantizer. PLAN.md §5, goal 1 (Q/K half).

    python custom_nodes/ComfyUI-H3-Extended/tests/test_triton_i64.py
    python custom_nodes/ComfyUI-H3-Extended/tests/test_triton_i64.py --overflow

Install/uninstall and the offset arithmetic are CPU-only. The numerical checks
need CUDA and print SKIP without it.

The bit-parity check compares against `quant_per_thread.py.orig` - the unpatched
stock kernel saved beside the site-packages file - not against the installed one,
which on this machine already carries the int64 fix and would therefore compare
int64 against int64 and prove nothing.

`--overflow` drives a sequence past the signed-int32 row limit at H3's real
stride. It allocates ~5.4 GB and must be run in a subprocess: a wrapping kernel
produces an access violation that kills the interpreter rather than raising.
"""

import argparse
import importlib.machinery
import importlib.util
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_attention import triton_i64  # noqa: E402

HEADS, HEAD_DIM = 56, 128
FUSED_STRIDE = HEADS * HEAD_DIM * 3          # 21504


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


def load_stock():
    """Import the unpatched kernel saved beside the site-packages file."""
    import sageattention.triton.quant_per_thread as patched
    orig = patched.__file__ + ".orig"
    if not os.path.exists(orig):
        return None
    # the .orig suffix means the normal path-based finder will not load it
    loader = importlib.machinery.SourceFileLoader("h3_stock_quant_per_thread", orig)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def fused_qk(seq, device, seed=0):
    """q and k as strided views into one fused QKV buffer, exactly as H3 makes them."""
    torch.manual_seed(seed)
    q, k, _ = torch.empty(seq, FUSED_STRIDE, dtype=torch.bfloat16,
                          device=device).normal_().split(HEADS * HEAD_DIM, dim=-1)
    q = q.view(seq, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0)
    k = k.view(seq, HEADS, HEAD_DIM).transpose(0, 1).unsqueeze(0)
    return q, k


def test_offsets():
    print("offset arithmetic")
    check(triton_i64.first_wrapping_row(FUSED_STRIDE) == 99865,
          "first wrapping row index at H3's 21504 stride is 99,865")
    check(triton_i64.first_wrapping_row(HEADS * HEAD_DIM) == 299594,
          "a contiguous 7168 stride pushes the first wrapping row to 299,594")

    # off-by-one that a length-99,865 tensor hides: it covers rows 0..99,864,
    # all of which still fit. The wrap needs one more row.
    safe = (1, HEADS, 99865, HEAD_DIM)
    over = (1, HEADS, 99866, HEAD_DIM)
    stride = (0, HEAD_DIM, FUSED_STRIDE, 1)
    check(not triton_i64.wraps(safe, stride), "S=99,865 does not wrap (max row 99,864)")
    check(triton_i64.wraps(over, stride), "S=99,866 wraps (includes row 99,865)")


def test_install():
    print("install")
    try:
        import sageattention.core as sage_core
    except ImportError:
        print("  SKIP: sageattention not installed")
        return

    original = sage_core.per_thread_int8_triton
    check(not triton_i64.is_installed(), "not installed before install()")
    check(triton_i64.install() is True, "install() reports success")
    check(triton_i64.is_installed(), "is_installed() sees the swap")
    check(sage_core.per_thread_int8_triton is triton_i64.per_thread_int8_i64,
          "sageattention.core binding now points at the int64 quantizer")
    check(triton_i64.install() is True, "install() is idempotent")

    triton_i64.uninstall()
    check(sage_core.per_thread_int8_triton is original, "uninstall() restores the original")
    check(not triton_i64.is_installed(), "is_installed() is False after uninstall")


def test_bit_parity(device):
    """Below the overflow the int64 kernel must be bit-identical to stock."""
    print("bit parity vs stock")
    stock = load_stock()
    if stock is None:
        print("  SKIP: no quant_per_thread.py.orig beside the installed file")
        return

    for seq in (4096, 8192):
        q, k = fused_qk(seq, device, seed=seq)
        want = stock.per_thread_int8(q, k, None, BLKQ=128, WARPQ=32, BLKK=64,
                                     WARPK=64, tensor_layout="HND")
        got = triton_i64.per_thread_int8_i64(q, k, None, BLKQ=128, WARPQ=32, BLKK=64,
                                             WARPK=64, tensor_layout="HND")
        names = ("q_int8", "q_scale", "k_int8", "k_scale")
        for name, a, b in zip(names, want, got):
            check(torch.equal(a, b), "S=%d %s bit-identical to stock" % (seq, name))

    # km (smooth_k) path, though ComfyUI passes smooth_k=False for H3
    q, k = fused_qk(2048, device, seed=7)
    km = k.mean(dim=-2, keepdim=True)
    want = stock.per_thread_int8(q, k, km, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64,
                                 tensor_layout="HND")
    got = triton_i64.per_thread_int8_i64(q, k, km, BLKQ=128, WARPQ=32, BLKK=64,
                                         WARPK=64, tensor_layout="HND")
    check(all(torch.equal(a, b) for a, b in zip(want, got)), "km path bit-identical to stock")


def test_contiguous_equivalence(device):
    """A strided view and its contiguous copy must quantize identically."""
    print("stride independence")
    q, k = fused_qk(4096, device, seed=11)
    strided = triton_i64.per_thread_int8_i64(q, k, None, tensor_layout="HND")
    packed = triton_i64.per_thread_int8_i64(q.contiguous(), k.contiguous(), None,
                                            tensor_layout="HND")
    check(torch.equal(strided[0], packed[0]), "q_int8 independent of input stride")
    check(torch.equal(strided[2], packed[2]), "k_int8 independent of input stride")


def test_overflow(device):
    """Past the signed-i32 row limit at H3's real stride. Allocates ~5.4 GB."""
    # 99,866 not 99,865: rows are 0-indexed, so the shorter tensor stops exactly
    # one row below the first wrapping index and proves nothing
    seq = 99866
    print("overflow (S=%d, first wrapping row %d)"
          % (seq, triton_i64.first_wrapping_row(FUSED_STRIDE)))
    free = torch.cuda.mem_get_info(device)[0] / 1024 ** 3
    need = seq * FUSED_STRIDE * 2 / 1024 ** 3 + 2 * seq * HEADS * HEAD_DIM / 1024 ** 3
    print("  free %.2f GB, need ~%.2f GB" % (free, need))
    if free < need * 1.25:
        print("  SKIP: not enough free VRAM")
        return

    q, k = fused_qk(seq, device, seed=3)
    max_off = sum((n - 1) * s for n, s in zip(q.shape, q.stride()))
    check(triton_i64.wraps(q.shape, q.stride()),
          "q's max element offset %d exceeds signed i32 (%d)" % (max_off, 2 ** 31 - 1))

    q8, q_scale, k8, k_scale = triton_i64.per_thread_int8_i64(
        q, k, None, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64, tensor_layout="HND")
    torch.cuda.synchronize(device)
    check(q8.shape == q.shape and k8.shape == k.shape, "shapes preserved past the limit")
    check(torch.isfinite(q_scale).all() and (q_scale > 0).all(), "q scales finite and positive")
    check(torch.isfinite(k_scale).all() and (k_scale > 0).all(), "k scales finite and positive")
    check(int(q8.abs().max()) > 0, "q_int8 is not all zeros (the kernel actually wrote)")
    print("H3_I64_OVERFLOW_OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overflow", action="store_true",
                    help="also drive a sequence past the int32 limit (~5.4 GB)")
    args = ap.parse_args()

    test_offsets()
    test_install()

    if not torch.cuda.is_available():
        print("\nSKIP: no CUDA device; numerical checks not run")
        return 0
    device = torch.device("cuda")
    test_bit_parity(device)
    test_contiguous_equivalence(device)
    if args.overflow:
        test_overflow(device)
    else:
        print("\n(skipping the overflow case; pass --overflow to run it)")

    print("\nall int64 quantizer self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
