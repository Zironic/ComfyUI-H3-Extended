"""CPU tests for the H3 FirstBlockCache coordinator."""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_block_cache.config import FirstBlockCacheConfig  # noqa: E402
from h3_block_cache.coordinator import FirstBlockCacheCoordinator  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def options(step):
    snapshot = RuntimeSnapshot(
        request_id=0,
        step_index=step,
        total_steps=4,
        sigma=1.0 - step / 4,
        branch=(0,),
        layout=None,
        layout_signature=None,
        compute_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return {RUNTIME_KEY: snapshot}


def main():
    coordinator = FirstBlockCacheCoordinator(
        FirstBlockCacheConfig(
            mode="first_block",
            threshold=0.08,
            warmup_steps=1,
            collective=False,
        )
    )
    head0 = torch.ones(4, 8)
    opts0 = options(0)
    input0 = torch.zeros_like(head0)
    check(not coordinator.after_head(input0, head0, opts0), "warmup head computes the tail")
    tail = torch.full_like(head0, 0.25)
    coordinator.finish_compute(head0 + tail, opts0)

    head1 = head0 * 1.01
    opts1 = options(1)
    input1 = torch.zeros_like(head1)
    check(coordinator.after_head(input1, head1, opts1), "small residual change skips remaining blocks")
    cached = coordinator.apply_cached_tail(head1.clone(), opts1)
    check(torch.equal(cached, head1 + tail), "skip applies the cached total tail residual once")
    coordinator.finish_skip(opts1)

    head2 = head1 * 1.5
    opts2 = options(2)
    input2 = torch.zeros_like(head2)
    check(not coordinator.after_head(input2, head2, opts2), "large residual change recomputes the tail")
    coordinator.finish_compute(head2 + tail * 2, opts2)
    status = coordinator.as_status()
    check(status["skipped_tails"] == 1 and status["computed_tails"] == 2, "coordinator records compute/skip counts")
    check(status["cache_bytes"] > 0, "cache memory is reported")

    report = coordinator.report(seconds=12.5)
    check(
        [row["decision"] for row in report["steps"]] == ["compute", "skip", "compute"],
        "the recorder keeps the per-step decision trace",
    )
    check(
        report["steps"][0]["diff"] is None
        and report["steps"][1]["diff"] is not None,
        "the first step has no diff and later steps carry the scalar",
    )
    check(
        report["steps"][2]["reason"] == "above_threshold",
        "a recomputed step records why it was not skipped",
    )
    check(
        report["skip_fraction"] == 1 / 3 and report["sampler_seconds"] == 12.5,
        "the report carries skip fraction and wall time",
    )
    check(
        all("residual" not in key for key in report),
        "the report holds scalars only, never residual tensors",
    )

    coordinator.on_request_reset(1)
    check(not coordinator.states, "new request releases all cached tensors")
    check(not coordinator.report()["steps"], "a new request starts an empty trace")

    zero = FirstBlockCacheCoordinator(
        FirstBlockCacheConfig(
            mode="first_block", threshold=0.0, warmup_steps=0, collective=False
        )
    )
    base = torch.ones(4, 8)
    zero.after_head(torch.zeros_like(base), base, options(0))
    zero.finish_compute(base + tail, options(0))
    moved = base * 1.0001
    check(
        not zero.after_head(torch.zeros_like(moved), moved, options(1)),
        "threshold zero never skips, so it measures wrapper overhead alone",
    )
    print("\nall FirstBlockCache tests passed")


if __name__ == "__main__":
    main()
