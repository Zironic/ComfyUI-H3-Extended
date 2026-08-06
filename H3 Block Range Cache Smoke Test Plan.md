# MiniMax H3 Block/Range Cache + AIMDO Smoke-Test Plan

**Repository:** `Zironic/ComfyUI-H3-Extended`  
**Branch implementation status:** Stages 0–2 scaffolding and fixed GPU cache prototype implemented in `h3_block_cache/`; host cache, quantization, AIMDO transfer instrumentation, and filtered asynchronous prefetch remain gated stages.

## 0. Objective

The smoke test must answer four independent questions:

1. Can an H3 block or contiguous block range reuse its previous denoising-step residual without unacceptable error?
2. Does grouped refresh or upstream refresh propagation prevent stale-cache error from contaminating later blocks?
3. Can the reuse path avoid faulting skipped AIMDO weights?
4. Is cache transfer/dequantization/addition cheaper than AIMDO weight transfer plus block computation?

AIMDO is mandatory for all full-model tests on the target machine. The prototype never disables AIMDO. It disables only H3's static `prefetch_dynamic_vbars` all-block lookahead queue during an armed forward, because that queue runs before block replacements and would otherwise transfer weights for blocks that the cache later skips. Executed blocks still fault through AIMDO on demand.

## 1. Initial implementation boundary

Implemented now:

- output-neutral `observe` mode;
- output-neutral `shadow` mode with previous-cache hypothetical error;
- `fixed_gpu` active reuse mode;
- whole-block residuals;
- aggregate contiguous range residuals;
- fixed warm-up/refresh/reuse schedules;
- maximum reuse-span and final-step refresh guards;
- separate conditional/unconditional caches with a shared schedule;
- strict shape checks and cache-miss refresh;
- report output under `output/h3_block_cache/`;
- CPU self-tests.

Not claimed by the current code:

- pinned-RAM cache storage;
- FP8/INT8 cache storage;
- asynchronous cache transfer;
- AIMDO fault-byte instrumentation;
- filtered asynchronous AIMDO prefetch;
- adaptive similarity policy;
- masked-sequence composition;
- production memory efficiency at C=90.

The active GPU prototype deliberately stores only explicitly selected units. It is a correctness and error-propagation probe, not the consumer implementation.

## 2. Cache semantics

For a block `i`:

```text
R_i(t) = h_(i+1)(t) - h_i(t)
reuse: h_(i+1)(t+1) ~= h_i(t+1) + R_i(t)
```

For an aggregate range `[a,b]`:

```text
R_[a,b](t) = h_(b+1)(t) - h_a(t)
reuse at block a: h <- h + R_[a,b]
blocks a+1 ... b: no-op
```

Only one cache tensor is needed per independently skippable unit and CFG branch.

The current smoke prototype captures a range residual with a clone of the range input. This is intentionally simple and expensive. A production implementation must instead accumulate the gated attention and MLP residual branches while the range executes, avoiding a second full hidden-state tensor.

## 3. Runtime invariants

- AIMDO must be enabled for full H3 node execution.
- Disabled node means no wrappers and no block replacements.
- Caches are cleared at every outer sampling-run boundary.
- Conditional and unconditional branches never share residual values.
- A missing, stale, or shape-mismatched cache causes refresh.
- Cache-unit ranges are ordered, contiguous internally, and non-overlapping.
- Any foreign H3 `double_block` replacement causes installation failure.
- `observe` and `shadow` always return the real dense block output.
- Static all-block vbar prefetch is suppressed only after the diffusion wrapper has armed the current forward.
- Executed layers continue using AIMDO on-demand weight faulting.

## 4. Package layout

```text
h3_block_cache/
    __init__.py
    config.py       immutable smoke-test configuration
    units.py        block/range parsing and validation
    cache.py        GPU residual cache entry
    policy.py       fixed refresh/reuse decisions
    metrics.py      residual and shadow-error metrics
    session.py      per-run/per-sigma/per-CFG state
    wrappers.py     OUTER_SAMPLE, DIFFUSION_MODEL, and block replacements
    report.py       JSON/JSONL output
    nodes.py        Comfy model-patch node

tests/test_block_cache.py
H3 Block Range Cache Smoke Test Plan.md
```

## 5. Stage 0 — dense stability survey

Extend `observe` mode to record per-block input/output/residual statistics, deterministic sketches, per-segment drift, CUDA duration, and AIMDO residency/transfer data. Do not retain all 50 full residuals. Derive candidate range sketches for lengths 1, 2, 5, and 10. Test debug, C=22, C=39, and the largest safe production-like profile across static-subject, moving-subject, moving-camera, large-replacement, and global-change Ref2V cases.

**Gate:** at least one unit shows useful adjacent-step stability and meaningful skipped compute/weight traffic.

## 6. Stage 1 — shadow full-residual measurement

Shortlist units from Stage 0. At the next sigma compute `approx = current_input + previous_residual`, execute the real unit, record local and propagated error, and return the real output. `shadow` remains output-neutral.

**Gate:** at least one block or range has acceptable one-step shadow error without catastrophic downstream amplification.

## 7. Stage 2 — fixed GPU BF16 reuse

The current `fixed_gpu` implementation provides the first active path. Test refresh-only, one block, repeated one-block reuse, independent blocks, aggregate five-block and ten-block ranges, and middle-step alternating schedules. Replace range-input cloning with an H3-owned gated-residual accumulator before large profiles.

## 8. Stage 3 — refresh compensation

Compare independent schedules, whole-unit refresh, and refresh-front propagation from an upstream anchor through the deepest requested refresh. Prefer whole-unit refresh unless measured error requires the more complex policy.

## 9. Stage 4 — AIMDO instrumentation

Instrument benchmark-only seams around `cast_modules_with_vbar`, `cast_bias_weight`, and `resolve_cast_module_with_vbar`. Map calls to block/submodule and record faults, resident hits, bytes, and duration.

**Gate:** a reused unit produces zero weight faults for every skipped module.

## 10. Stage 5 — pinned-RAM BF16 cache

Allocate pinned CPU cache entries. On refresh, asynchronously write the accumulated residual GPU-to-host. On reuse, transfer and add in token slabs, initially 4,096 rows, using double-buffered staging. Never read old cache during refresh and never transfer skipped weights during reuse.

## 11. Stage 6 — INT8 cache

Start with symmetric per-row INT8 residuals and FP16/BF16 row scales. Measure quantization/dequantization time, transfer bytes, reconstruction error, propagated error, and decoded quality. INT8 must reduce total reuse-path time, not merely storage.

## 12. Stage 7 — filtered asynchronous AIMDO prefetch

The current smoke node suppresses H3's static queue because it contains all 50 blocks before replacement decisions. After the economics pass, add an H3-specific filtered loop or a small Comfy execution-controller seam:

```text
plan forward before prefetch
executing blocks -> AIMDO weight queue
reused units     -> cache-transfer queue
```

The cache queue remains separate from AIMDO's model-weight virtual-address system.

## 13. Stage 8 — adaptive policy

Only after fixed schedules pass. Use previous completed-sigma metrics so the next plan exists before prefetch. Refresh on cache miss, warm-up, final region, maximum reuse span, either CFG branch exceeding threshold, NaN/Inf, or unfavorable measured cache-versus-compute cost.

## 14. Composition with masked computation

When target-token compaction becomes active, include compact sequence length and mask hash in cache keys, invalidate on mask changes, force refresh on dense mask-refresh steps, allocate compact-sized caches, and rerun break-even profiling.

Expected order:

```text
dense warm-up -> infer mask
compact sequence -> plan block/range reuse
per unit -> stream weights or cache, never both
```

## 15. Tests

CPU tests cover unit parsing, overlap rejection, aggregate-range residual semantics, fixed scheduling, CFG separation, and suppression of only static vbar prefetch. GPU tests still required include refresh-only identity, shadow and active reuse, Sage composition, AIMDO fault accounting, interrupted cleanup, and decoded comparisons.

## 16. Final smoke-test success condition

Proceed toward a consumer implementation only when all three statements are measured true:

```text
a useful H3 block/range is reusable
its cache replaces rather than supplements AIMDO weight traffic
cache transfer/decode/add is cheaper than weight transfer plus compute
```
