"""End-to-end CPU test for the composable FirstBlockCache block wrappers."""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_block_cache.config import FirstBlockCacheConfig  # noqa: E402
from h3_block_cache.patch import install  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class Block:
    def __init__(self, amount):
        self.amount = amount
        self.calls = 0

    def forward(self, x, t_emb, mod_segments, rope_freqs, transformer_options={}):
        self.calls += 1
        return x.add_(self.amount)


class Patcher:
    def __init__(self, blocks):
        self.blocks = blocks
        self.object_patches = {}
        self.model_options = {"transformer_options": {}}

    def get_model_object(self, name):
        if name == "diffusion_model.blocks":
            return self.blocks
        raise KeyError(name)

    def add_object_patch(self, key, value):
        self.object_patches[key] = value


def snapshot(step):
    return RuntimeSnapshot(
        request_id=0,
        step_index=step,
        total_steps=2,
        sigma=1.0 - step * 0.5,
        branch=(0,),
        layout=None,
        layout_signature=None,
        compute_dtype=torch.float32,
        device=torch.device("cpu"),
    )


def run(patcher, step):
    x = torch.zeros(2, 3)
    options = {RUNTIME_KEY: snapshot(step)}
    for index in range(len(patcher.blocks)):
        fn = patcher.object_patches["diffusion_model.blocks.%d.forward" % index]
        x = fn(x, None, [], None, transformer_options=options)
    return x


def main():
    blocks = [Block(1.0), Block(2.0), Block(3.0)]
    patcher = Patcher(blocks)
    coordinator, count = install(
        patcher,
        FirstBlockCacheConfig(
            mode="first_block",
            threshold=0.08,
            warmup_steps=0,
            collective=False,
        ),
    )
    check(count == 3, "all H3 blocks are wrapped")
    first = run(patcher, 0)
    check(torch.equal(first, torch.full_like(first, 6.0)), "computed pass produces full block-stack output")
    calls = [block.calls for block in blocks]
    second = run(patcher, 1)
    check(torch.equal(second, first), "cached pass reconstructs the same synthetic output")
    check(blocks[0].calls == calls[0] + 1, "block 0 always runs")
    check(blocks[1].calls == calls[1] and blocks[2].calls == calls[2], "tail blocks are skipped")
    check(coordinator.as_status()["skipped_tails"] == 1, "skip is recorded once")
    check(patcher.model_options["transformer_options"]["prefetch_dynamic_vbars"] is False, "dynamic weight prefetch is disabled")
    print("\nall FirstBlockCache patch tests passed")


if __name__ == "__main__":
    main()
