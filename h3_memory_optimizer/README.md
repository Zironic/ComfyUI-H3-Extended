# MiniMax H3 Memory Optimizer

`MiniMax H3 Memory Optimizer (Zi)` combines the lossless prepared-QKV attention
and bounded-MLP paths with optional Sol-Engine experiments behind one model
patch. The ordinary defaults remain conservative:

```text
attention = auto                # prepared dense Sage
activation = mlp_chunked_bf16
adaln_precompute = off
block_cache = off
```

Approximate features are never selected automatically.

## Dense attention support

`attention=auto` mirrors SageAttention 2.2's architecture families:

| capability | prepared path |
| --- | --- |
| SM80 | per-thread INT8 Q/K, FP16 V, CUDA FP32 accumulation |
| SM86 | per-block INT8 Q/K, FP16 V, Triton attention |
| SM89 | per-thread INT8 Q/K, FP8 V, CUDA FP32+FP16 accumulation |
| SM90 | per-thread INT8 Q/K, FP8 V, Hopper CUDA accumulation |
| SM120/121 | per-warp INT8 Q/K, FP8 V, Blackwell path |

Incomplete wheels and unsupported devices preserve the incoming attention
backend when `attention_fallback=allow`. Fallback occurs before model mutation.

## Sol-Attn

`attention=sol_attn` is an explicit approximate speed experiment. It requires
the released Sol-Attn Python package from NVlabs/Sana's `sol-engine` branch. If
that package is not installed as `sol_attn`, point `H3_SOL_ATTN_ROOT` at the
folder containing the package.

The H3 policy follows the released integration:

- `tau=1.0`, `thresh_type=diag`;
- first 10 denoising steps dense;
- first 2 transformer blocks dense;
- no token/Morton reordering;
- one sparse call over the packed sequence;
- everything before the target-video tail is an exact KV sink by default;
- sink query rows are recomputed with dense SDPA;
- a real-QKV all-exact correctness gate runs once per shape;
- route density and decline reasons are retained in diagnostics.

Full-reference long-form chunks can place most of the sequence before the
target-video tail. The default `sol_max_sink_fraction=0.5` therefore declines
to the dense prepared backend when the exact prefix exceeds half of the packed
sequence. Raising the limit is an explicit benchmark choice, not a safe default.

Eligible sparse calls need independent contiguous BF16 Q/K/V, so Sol-Attn is a
speed sibling of prepared Sage, not an additive layer on top of its INT8/FP8
buffers. Dense warmup calls use the architecture-selected prepared-Sage backend
when available, otherwise a consuming contiguous-BF16 SDPA fallback. The token
refiner keeps the incoming Comfy attention selection.

## AdaLN trajectory precompute

`adaln_precompute=auto|on` computes each block's modulation outputs for the full
sampler sigma trajectory once and replaces later projection calls with exact
table lookups. The implementation deliberately keeps the original checkpoint
weights available and offloadable; it does not permanently delete or replace
checkpoint tensors as a serving-only runtime might.

`auto` declines when the table is not smaller than the detected projection
weights or exceeds `adaln_max_table_gib`. Curve-form checkpoints are therefore
expected to decline because Comfy already replaces the huge full-width AdaLN
weights with a small shared basis.

Compatible long-form chunks reuse the same table. A new sampler request rewinds
the table cursor but does not discard a table whose schedule, layout, device,
dtype, shifts, and augmentation signature are unchanged.

## FirstBlockCache

`block_cache=first_block` is approximate. Block 0 always runs. The change in its
residual (output minus input) relative to the previous computed step decides
whether blocks 1-49 run or a cached total tail residual is reused. State is
isolated by CFG branch and sampling request. Dynamic weight prefetch is disabled
for the cloned model so skipped blocks do not pull their weights into VRAM
before the decision is known.

This cache is mutually exclusive in intent with whole-step caches such as
EasyCache. When EasyCache is detected, FirstBlockCache declines rather than
stacking two reuse policies.

## Shared runtime context

Sol-Attn, AdaLN and FirstBlockCache share two wrappers:

- an `OUTER_SAMPLE` wrapper creates an explicit request boundary for every
  sampler invocation, including identical one-step long-form chunks;
- a diffusion wrapper publishes request and denoising-step indices, CFG branch,
  packed text/reference/audio/video ranges, device, and compute dtype.

The explicit boundary clears Sol's request clock and FirstBlockCache state.
Sigma, step, and layout inference remains only as a fallback for direct model
calls that bypass the sampler wrapper. Missing layout or step data produces a
named dense/cache decline; strict modes turn those declines into errors.

## Measurement isolation

`MiniMax H3 Masked Ref2V Cache (Zi)` treats its Stage-0 trajectory as a dense
reference. It rejects or disables measurement when EasyCache, Sol-Attn, or
FirstBlockCache is active. Prepared Sage, BF16 MLP chunking, and exact AdaLN
precompute remain valid measurement baselines.

## Diagnostics and benchmarks

```bash
python custom_nodes/ComfyUI-H3-Extended/tests/test_h3_runtime_context.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_sol_attention.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_adaln_precompute.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_first_block_cache.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_memory_optimizer.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_sol_runtime_compat.py

python custom_nodes/ComfyUI-H3-Extended/benchmarks/bench_sol_attention.py --frames 90
python custom_nodes/ComfyUI-H3-Extended/benchmarks/bench_adaln_precompute.py --ckpt <checkpoint>
python custom_nodes/ComfyUI-H3-Extended/benchmarks/bench_first_block_cache.py
python custom_nodes/ComfyUI-H3-Extended/benchmarks/compare_h3_outputs.py \
  --reference-frames <dir> --candidate-frames <dir>
```

Only SM89 prepared Sage has been hardware-validated in this repository. Sol,
AdaLN and FirstBlockCache remain explicit experiments until full trajectory,
audio and reference-identity comparisons establish useful operating points.
