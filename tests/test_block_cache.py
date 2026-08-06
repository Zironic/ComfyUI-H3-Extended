"""CPU self-test for H3 block/range cache scheduling and residual semantics."""
import os, sys, tempfile
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_block_cache.config import BlockCacheConfig
from h3_block_cache.session import BlockCacheSession
from h3_block_cache.units import parse_unit_spec

failures = []
def check(c, m):
    print(("  ok: " if c else "  FAIL: ") + m)
    if not c: failures.append(m)

def block(add):
    def original(args):
        args["img"].add_(add)
        return {"img": args["img"]}
    return original

def test_units():
    u = parse_unit_spec("2,5-7,10")
    check([x.key for x in u] == ["2", "5-7", "10"], "unit parser preserves ordered units")
    try: parse_unit_spec("2-4,4-6")
    except ValueError: check(True, "overlap rejected")
    else: check(False, "overlap rejected")

def test_range_refresh_and_reuse():
    cfg = BlockCacheConfig(mode="fixed_gpu", unit_spec="1-2", warmup_steps=0,
                           refresh_interval=2, max_reuse_span=1,
                           force_refresh_last_steps=0)
    s = BlockCacheSession(cfg, tempfile.gettempdir()).begin()
    opts = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, .5, 0.0]),
            "cond_or_uncond": [0], "prefetch_dynamic_vbars": True}
    s.prepare_forward(opts)
    x = torch.zeros(3, 4)
    x = s.block(1, {"img": x}, block(1.0))
    x = s.block(2, {"img": x}, block(2.0))
    check(torch.allclose(x, torch.full_like(x, 3.0)), "range refresh executes both blocks")
    check(opts["prefetch_dynamic_vbars"] is False,
          "static vbar prefetch disabled while AIMDO on-demand remains available")

    opts["sigmas"] = torch.tensor([.5])
    s.prepare_forward(opts)
    y = torch.full((3, 4), 10.0)
    y = s.block(1, {"img": y}, block(100.0))
    y = s.block(2, {"img": y}, block(100.0))
    check(torch.allclose(y, torch.full_like(y, 13.0)), "range reuse applies one aggregate residual")
    s.end()

def test_cfg_separation():
    cfg = BlockCacheConfig(mode="fixed_gpu", unit_spec="3", warmup_steps=0,
                           refresh_interval=2, max_reuse_span=1,
                           force_refresh_last_steps=0)
    s = BlockCacheSession(cfg).begin()
    base = {"sigmas": torch.tensor([1.0]), "sample_sigmas": torch.tensor([1.0, .5, 0.0])}
    for branch, add in [(0, 1.0), (1, 5.0)]:
        opts = dict(base, cond_or_uncond=[branch])
        s.prepare_forward(opts)
        s.block(3, {"img": torch.zeros(2, 2)}, block(add))
    check(len(s.entries) == 2, "conditional and unconditional caches are distinct")
    s.end()

if __name__ == "__main__":
    test_units(); test_range_refresh_and_reuse(); test_cfg_separation()
    if failures:
        raise SystemExit("%d failures" % len(failures))
    print("all block-cache CPU tests passed")
