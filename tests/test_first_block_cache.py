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

    coordinator.on_request_reset(1)
    check(not coordinator.states, "new request releases all cached tensors")
    print("\nall FirstBlockCache tests passed")


if __name__ == "__main__":
    main()
