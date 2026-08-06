"""Simulate H3 FirstBlockCache decisions from captured or synthetic block-0 residuals.

A real run exposes live counters through transformer_options under
``minimax_h3_first_block_cache``.  For offline threshold sweeps, save a tensor of
shape ``[steps, rows, hidden]`` containing block-0 residuals and pass ``--residuals``.
"""

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from h3_block_cache.config import FirstBlockCacheConfig  # noqa: E402
from h3_block_cache.coordinator import FirstBlockCacheCoordinator  # noqa: E402
from h3_runtime.context import RUNTIME_KEY, RuntimeSnapshot  # noqa: E402


def synthetic(steps, rows, hidden, seed):
    generator = torch.Generator().manual_seed(seed)
    base = torch.randn(rows, hidden, generator=generator, dtype=torch.float32)
    values = []
    current = base
    for step in range(steps):
        scale = 0.12 / (step + 1)
        current = current + torch.randn(
            rows, hidden, generator=generator, dtype=torch.float32
        ) * scale
        values.append(current.clone())
    return torch.stack(values)


def snapshot(step, total, tensor):
    return RuntimeSnapshot(
        request_id=0,
        step_index=step,
        total_steps=total,
        sigma=1.0 - step / max(1, total),
        branch=(0,),
        layout=None,
        layout_signature=None,
        compute_dtype=tensor.dtype,
        device=tensor.device,
    )


def simulate(residuals, threshold, warmup):
    config = FirstBlockCacheConfig(
        mode="first_block",
        threshold=threshold,
        warmup_steps=warmup,
        collective=False,
    )
    coordinator = FirstBlockCacheCoordinator(config)
    decisions = []
    for step, residual in enumerate(residuals):
        options = {RUNTIME_KEY: snapshot(step, len(residuals), residual)}
        original = torch.zeros_like(residual)
        skipped = coordinator.after_head(original, residual, options)
        decisions.append(
            {
                "step": step,
                "skip": bool(skipped),
                "diff": coordinator.states[(0,)].last_diff,
            }
        )
        if skipped:
            coordinator.finish_skip(options)
        else:
            # The exact tail is irrelevant to the decision simulation.  Keep a
            # stable nonzero residual so the coordinator becomes cache-ready.
            coordinator.finish_compute(residual + residual.mul(0.05), options)
    return decisions, coordinator.as_status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residuals", default=None)
    parser.add_argument("--heads", default=None, help="Deprecated alias for --residuals")
    parser.add_argument("--thresholds", default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=9182)
    args = parser.parse_args()

    path = args.residuals or args.heads
    if path:
        residuals = torch.load(path, map_location="cpu", weights_only=True)
        if residuals.ndim < 3:
            raise SystemExit("--residuals must contain [steps, rows, hidden]")
    else:
        residuals = synthetic(args.steps, args.rows, args.hidden, args.seed)

    results = []
    for raw in args.thresholds.split(","):
        threshold = float(raw)
        decisions, status = simulate(residuals, threshold, args.warmup_steps)
        results.append(
            {
                "threshold": threshold,
                "skipped_steps": [row["step"] for row in decisions if row["skip"]],
                "skip_fraction": sum(row["skip"] for row in decisions) / len(decisions),
                "decisions": decisions,
                "status": status,
            }
        )
    print(json.dumps({"shape": list(residuals.shape), "results": results}, indent=2))


if __name__ == "__main__":
    main()
