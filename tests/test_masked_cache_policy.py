"""CPU-only checks for masked-cache burn-in and threshold policy reporting.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_masked_cache_policy.py
"""

import os
import sys
import tempfile

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_masked_cache import mask as mask_ops  # noqa: E402
from h3_masked_cache import report  # noqa: E402
from h3_masked_cache.config import MaskedCacheConfig  # noqa: E402
from h3_masked_cache.session import MaskedCacheSession  # noqa: E402


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    print("  ok: %s" % message)


def score_at(*positions, baseline=0.0):
    score = torch.full((1, 4, 4), baseline, dtype=torch.float32)
    for h, w in positions:
        score[0, h, w] = 2.0
    return score


def record(session, run, score, i):
    run.observe_sigma(1.0 - i * 0.05)
    component = torch.zeros(1, 8, 8)
    return session.record_step(
        run, score, component, component, i, 1.0 - i * 0.05,
        source_kind="guided",
    )


def test_burn_in_freeze():
    print("burn-in freeze")
    cfg = MaskedCacheConfig(
        burn_in_steps=2,
        warmup_steps=2,
        score_threshold=1.0,
        tile_h=1,
        tile_w=1,
        spatial_halo=0,
        temporal_halo=0,
        run_tag="policy",
    )
    session = MaskedCacheSession(cfg, tempfile.gettempdir())
    run = session.begin()

    # The two broad early predictions must not contaminate the frozen mask.
    record(session, run, torch.full((1, 4, 4), 2.0), 0)
    record(session, run, torch.full((1, 4, 4), 2.0), 1)
    check(run.frozen_mask is None, "burn-in predictions do not freeze a mask")

    record(session, run, score_at((1, 1)), 2)
    check(run.frozen_mask is None, "freeze waits for the complete warmup window")
    record(session, run, score_at((1, 2)), 3)

    check(run.frozen_range == (2, 4), "frozen range starts after burn-in")
    check(int(run.frozen_mask.sum()) == 2, "frozen mask unions only observations 2 and 3")
    check(bool(run.frozen_mask[0, 1, 1]) and bool(run.frozen_mask[0, 1, 2]),
          "both stable warmup edits are retained")
    check(not bool(run.frozen_mask[0, 0, 0]),
          "discarded broad burn-in activity does not leak into the frozen mask")

    record(session, run, score_at((3, 3)), 4)
    check(abs(run.steps[-1]["escaped_frozen"] - 1.0) < 1e-9,
          "a later new edit is reported as escaping the immutable frozen mask")


def test_excess_mass_and_policy_sweep():
    print("excess mass and threshold policy sweep")
    cfg = MaskedCacheConfig(
        burn_in_steps=2,
        warmup_steps=2,
        score_threshold=1.0,
        tile_h=1,
        tile_w=1,
        spatial_halo=0,
        temporal_halo=0,
        run_tag="policy",
    )
    session = MaskedCacheSession(cfg, tempfile.gettempdir())
    run = session.begin()

    for i, score in enumerate((
        torch.full((1, 4, 4), 2.0),
        torch.full((1, 4, 4), 2.0),
        score_at((1, 1)),
        score_at((1, 2)),
        score_at((3, 3)),
    )):
        record(session, run, score, i)

    final_score = score_at((1, 1), (3, 3), baseline=0.5)
    component = torch.zeros(1, 8, 8)
    session.record_final(run, final_score, component, component)

    missed = mask_ops.missed_score_mass(final_score, run.frozen_mask, threshold=1.0)
    check(abs(missed - 0.5) < 1e-9,
          "missed mass counts only above-threshold excess, not the 0.5 background")

    summary = report.aggregate(run)
    policy = next(x for x in summary["policy_sweep"] if x["threshold"] == 1.0)
    check(abs(policy["frozen_active"] - 2 / 16) < 1e-9,
          "threshold sweep recomputes the burn-in-aware frozen mask")
    check(abs(policy["final_active"] - 2 / 16) < 1e-9,
          "threshold sweep reports final active fraction")
    check(abs(policy["final_escaped"] - 0.5) < 1e-9,
          "threshold sweep reports final escape from frozen")
    check(abs(policy["final_missed_excess_mass"] - 0.5) < 1e-9,
          "threshold sweep reports missed final excess score")


def main():
    test_burn_in_freeze()
    test_excess_mass_and_policy_sweep()
    print("\nall masked-cache policy tests passed")


if __name__ == "__main__":
    main()
