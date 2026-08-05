# MiniMax H3 activation-memory plan

**Status:** design, pre-implementation.

**Basis:** MiniMax H3 core at ComfyUI commit
`6f7cd7fceaaf60d2669b554936394a7412c6fde5`, and this repository after the
H3-owned attention-forward work in `h3_attention/`.

**Purpose:** determine which sequence-scaled BF16 activations can be removed,
chunked, fused, or only then reduced in precision, before considering CPU
transient offload.

The first implementation target is deliberately narrower than “quantize all
activations”: keep the persistent residual stream in BF16, preserve the model's
mathematical graph, and stop materializing full-sequence BF16 staging tensors
when the consumer can operate on token slabs.

---

## 0. Decision and boundary

This is a **separate experiment from `h3_attention`**, but the two plans are meant
to compose.

| owner | responsibility |
| --- | --- |
| `h3_attention/` | Attention-specific Q/K/V representation, Sage kernels, overflow safety, fused-QKV lifetime, and any prequantized or rectangular-attention API. |
| `h3_activation_memory/` | DiT-block execution order, token-slab planning, MLP chunking, normalization/modulation slabs, projection-output consumption, residual accumulation, activation-memory measurement, and optional storage-precision ablations. |

The patch points do not conflict:

```text
h3_attention patches:
    diffusion_model.blocks.N.attn.forward

h3_activation_memory patches:
    diffusion_model.blocks.N.forward
```

The activation-memory block forward must continue to call `block.attn(...)` in
its initial stages. That preserves the current attention patch, observer seam,
backend selection, and future `sage_mem_eff` work. Attention-side slab production
is added only through an explicit capability from `h3_attention`; this plan must
not reach into Sage internals opportunistically.

### First shipping candidate

The first candidate is:

```text
BF16 residual stream
+ existing attention backend / h3_attention forward
+ exact-graph token-chunked MLP
+ immediate BF16 gated residual accumulation
```

No FP8 activation storage is required for that result.

---

## 1. Why this work matters after QKV release

At `C=90`, `S=45,990`, H3 uses:

```text
hidden size       5,376
attention width   56 * 128 = 7,168
FFN width         14,336
fc1 width         2 * 14,336 = 28,672
```

The measured sequence-dependent block transient is 5.085 GB. The H3 attention
work has proven that dropping the Q/K/V views makes the complete fused BF16 QKV
allocation reusable:

```text
fused QKV at C=90:       1.842 GB
measured block transient: 5.085 GB
remainder after release:  3.243 GB
```

The release is real in isolation, but it can shift the peak into the MLP rather
than reduce the end-to-end block peak by the same 1.842 GB. The current MLP
materializes a full `[S, 28,672]` `fc1` output before the down-projection:

```text
BF16 fc1 output at C=90: 2.456 GB
```

An approximate MLP live set before kernel workspaces is therefore:

| tensor | C=90 size |
| --- | ---: |
| residual `x`, `[S, 5376]` BF16 | 0.461 GB |
| normalized/modulated `h`, `[S, 5376]` BF16 | 0.461 GB |
| `fc1` output, `[S, 28672]` BF16 | 2.456 GB |
| `fc2` output, `[S, 5376]` BF16 | 0.461 GB |
| **subtotal** | **3.839 GB** |

The exact MLP peak is still unknown because weight residency, activation
quantization, kernel workspaces, allocator granularity, and the lifetime of the
SwiGLU/down-projection intermediates have not yet been traced. But the arithmetic
is enough to establish the risk:

```text
post-QKV attention estimate: ~3.243 GB
unoptimized MLP subtotal:    ~3.839 GB before workspace
```

Therefore MLP chunking is not merely a second independent saving. It may be the
change required for the attention plan's proven QKV release to appear in the
end-to-end peak at all.

### What a 4,096-row slab changes

At the initial experimental chunk size of 4,096 rows:

| BF16 tensor | full C=90 | 4,096-row slab |
| --- | ---: | ---: |
| hidden-width `h` or projection output | 0.461 GB | 0.041 GB |
| fused QKV | 1.842 GB | 0.164 GB |
| attention-width output | 0.614 GB | 0.055 GB |
| MLP `fc1` expansion | 2.456 GB | 0.219 GB |
| materialized SwiGLU result, if needed | 1.228 GB | 0.109 GB |

These are storage bounds, not promised peak reductions. The real result is what
the measurement stages below exist to establish.

---

## 2. Core facts and unknowns

### 2.1 Verified from source

| claim | source | status |
| --- | --- | --- |
| The main DiT has 50 sequential blocks. | `MiniMaxH3Model.blocks` | **verified** |
| `DiTBlock.forward` creates full-sequence `norm1` and `norm2` outputs, calls attention and MLP, then gates their full outputs into `x`. | [`model.py:255-273`] | **verified** |
| MLP computation is tokenwise: `fc2(swiglu(fc1(x)))`; there is no cross-token dependency. | [`model.py:184-192`] | **verified** |
| RMSNorm operates independently per token over the hidden dimension. | PyTorch/Comfy RMSNorm semantics | **verified** |
| AdaLN modulation is piecewise constant over contiguous `mod_segments`. | `_mod_scale_shift`, `_mod_gate` | **verified** |
| `_mod_gate` accumulates into the persistent residual stream in place. | [`model.py:215-219`] | **verified** |
| `linear_input_act` can fold SwiGLU into the TensorWise-INT8 down-projection path. | `comfy/ops.py` | **verified** |
| The existing H3 attention patch owns only `attn.forward`; a block-forward patch can compose with it. | `h3_attention/{forward,patch}.py` | **verified** |
| Dropping Q/K/V returns 1.518 GB at C=73 and 1.842 GB at C=90 to the in-process async pool. | `benchmarks/measure_qkv_release.py` | **measured** |
| `memory_reserved` and driver-free memory are not valid release metrics under the current `cudaMallocAsync` setup. | `h3_attention/PLAN.md` §2.1 | **measured** |

### 2.2 Unknown and blocking

| question | why it matters | stage |
| --- | --- | --- |
| What operation establishes the real MLP peak? | Determines whether MLP chunking unlocks the full QKV saving. | Stage 0 |
| Does calling `fc1`/`fc2` per chunk fault, cast, or transfer weights per chunk? | Repeated weight streaming could erase all runtime gains. | Stage 0 |
| What activation-quantization granularity does the current `fc2` fast path use on the active checkpoint? | Chunk-local scales may change numerics relative to the full-sequence call. | Stage 0 |
| Are all 50 blocks homogeneous in weight layout and patch metadata? | The implementation cannot assume one block represents the checkpoint. | Stage 0 |
| Which chunk size minimizes peak without turning large GEMMs into inefficient small GEMMs? | Sets the default and runtime overhead. | Stage 1 |
| Does the future `sage_mem_eff` backend expose prequantized Q/K/V or rectangular Q against full K/V? | Required for attention-side slab production and output streaming. | Stages 3-4 |
| Does exact-graph chunking remain acceptably close after 50 blocks and all denoising steps? | Chunked GEMM shapes and quantization may change reduction order/scales. | Stages 1 and 6 |

[`model.py:184-192`]: ../../../comfy/ldm/minimax/model.py
[`model.py:215-219`]: ../../../comfy/ldm/minimax/model.py
[`model.py:255-273`]: ../../../comfy/ldm/minimax/model.py

---

## 3. Precision policy

The initial experiment is about **lifetime and materialization**, not aggressive
precision reduction.

### 3.1 Keep in BF16

- The persistent residual stream `x` across all 50 blocks.
- Residual accumulation.
- Per-slab RMSNorm and AdaLN results.
- Per-slab projection outputs before immediate consumption, unless an existing
  kernel already uses a lower internal representation.
- The final block output handed to the next block.

The residual stream is the worst place to start an FP8 experiment: it is updated
100 times across 50 attention and 50 MLP residual branches, and any storage error
is carried forward. It is also only 0.461 GB at C=90, much smaller than QKV or the
MLP expansion.

### 3.2 Use the backend's existing compact formats

For Sage attention, the destination representation remains:

```text
Q: INT8
K: INT8
V: FP8 on the current SM89 path
```

The improvement is to produce those buffers from bounded BF16 slabs, not to add a
second incompatible quantization scheme.

### 3.3 Optional precision experiments come last

FP8 storage for normalized inputs, attention outputs, or MLP expansions belongs
in Stage 5 only. Every such mode must be marked approximate and measured against
the BF16-slab implementation. It must never be silently enabled as a side effect
of selecting chunked execution.

### 3.4 Terminology

“Exact” in this plan means the same mathematical graph and activation dtype, not
necessarily bit-identical output. Different GEMM shapes can select different
kernels or reduction orders. If chunking also changes activation-quantization
scale granularity, that mode is not called exact; it is recorded as a numerical
variant and gated separately.

---

## 4. Proposed package and patch architecture

```text
h3_activation_memory/
    PLAN.md
    __init__.py
    config.py          immutable runtime configuration and validation
    chunks.py          segment-aware token-slab planner
    observer.py        optional phase/memory observer carried in transformer_options
    linear.py          held-weight sessions and linear-input-activation adapters
    forward.py         H3-owned DiTBlock forward variants
    patch.py           reversible block.forward object patches
    stats.py           per-run counters and log-once diagnostics
    nodes.py           experimental MODEL patch node, added only after GPU gates

benchmarks/
    profile_h3_activations.py
    benchmark_h3_mlp_chunks.py
    benchmark_h3_block_memory.py

 tests/
    test_activation_chunks.py
    test_activation_linear.py
    test_activation_forward.py
    test_activation_patch.py
    test_activation_memory_gpu.py
```

### 4.1 Patch installation

`patch.py` installs on:

```text
diffusion_model.blocks.0.forward
...
diffusion_model.blocks.49.forward
```

It follows the existing `h3_attention.patch` rules:

- validate every attribute the replacement reads;
- patch only the 50 main DiT blocks;
- leave the token refiner untouched;
- inference-only initially;
- idempotent when all existing patches are ours;
- raise on a foreign patch at the same key;
- use `ModelPatcher.add_object_patch`, never mutate the shared model;
- attach `_h3_activation_memory`, mode, and chunk-size metadata to each closure;
- log the exact number of patched blocks and active mode.

A block patch and the current attention patch are compatible because they own
different object paths. Tests must exercise both installation orders.

### 4.2 Segment-aware chunk planner

`chunks.py` validates that `mod_segments`:

- cover `[0, S)` contiguously;
- contain no gaps, overlap, negative spans, or out-of-range modulation rows;
- preserve their existing order.

It then yields slabs that never cross a modulation boundary:

```python
for start, stop, mod_row in iter_mod_chunks(mod_segments, max_rows):
    ...
```

Splitting at segment boundaries removes conditional logic from the hot loop and
ensures each slab has one shift, scale, and gate row. Slabs should otherwise be as
large as possible, with an optional alignment requirement supplied by the
consumer. The initial general alignment is 256 rows; Sage-specific producers may
require a different multiple.

### 4.3 Observer and metrics seam

`observer.py` follows the attention observer pattern: the observer lives in
`transformer_options`, not a module global. In ordinary sampling it is absent and
costs one dictionary lookup per block. In benchmark mode it records named phase
boundaries:

```text
block_enter
attention_norm_ready
attention_returned
attention_gated
mlp_chunk_enter
mlp_fc1_ready
mlp_fc2_ready
mlp_chunk_gated
block_exit
```

CUDA synchronization is forbidden in the shipping path. Benchmarks may request a
synchronized observer explicitly.

---

## 5. Stage 0 — characterize the real checkpoint and peak

Do not implement the shipping loop until these measurements exist. This stage is
cheap relative to full generation and prevents optimizing the wrong allocation.

### 5.1 Checkpoint inventory

For every main block, record:

- module class for `norm1`, `norm2`, `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`,
  and `mlp.fc2`;
- weight shape, storage dtype, logical/original dtype;
- `QuantizedTensor` layout class and parameters;
- presence of `weight_function`, `bias_function`, low-VRAM functions, LoRA or
  other patches;
- whether the weight is resident, vbar-backed, or ordinary CPU storage;
- whether all 50 blocks agree.

Output a JSON report and a concise log summary. A heterogeneous checkpoint is not
an error, but it means dispatch must be per module rather than selected once from
block 0.

### 5.2 Phase memory trace

`profile_h3_activations.py` runs one real `DiTBlock` with the current checkpoint
at the packed shapes already used by the VRAM probe:

```text
C=22   S=13,617
C=73   S=37,898
C=90   S=45,990
```

Collect:

- `torch.cuda.memory_allocated()` at every named phase;
- `torch.cuda.max_memory_allocated()` per attention and MLP half;
- optional `torch.cuda.memory._snapshot()` allocation history in a standalone
  process;
- CUDA-event duration for norm/modulation, projections, attention, and MLP;
- allocation sizes and stack labels where the private snapshot API provides them;
- current and peak weight bytes resident during each phase.

Use `memory_allocated`, not `memory_reserved`, for release deltas under
`cudaMallocAsync`. Driver-free memory is recorded only as context.

### 5.3 Weight-streaming probe

Wrap `cast_bias_weight`, `uncast_bias_weight`, or the narrowest available Comfy
seam to count, for one block:

- acquisitions per module;
- bytes transferred/cast;
- whether the returned weight remains quantized;
- whether calling `fc1` and `fc2` for N slabs causes N acquisitions;
- whether the async offload stream waits correctly after the last slab.

Test three execution forms:

1. one full-sequence stock MLP call;
2. per-slab ordinary module calls;
3. one held-weight session spanning all slabs.

This determines whether a held session is mandatory. The expected result is that
holding both MLP weights concurrently costs hundreds of MB at most on the active
quantized checkpoint, which is small against the multi-GB expansion it removes;
that expectation must be measured rather than assumed.

### 5.4 Activation-quantization granularity probe

On the active `fc2` layout, compare:

```text
full linear_input_act(fc2, full_fc1, "swiglu")
vs.
concatenate(linear_input_act(fc2, fc1_slab, "swiglu") for slab)
```

Run at several chunk boundaries and inspect any exposed input scales. Establish:

- whether quantization is per tensor, per row/token, per block, or otherwise;
- whether slab boundaries change scales;
- output max absolute error, mean absolute error, relative L2, cosine similarity,
  NaN/Inf counts;
- whether the result changes when the same slab partition is repeated.

If the fast path is scale-local and chunking changes numerics materially, retain
two candidate modes for Stage 1:

```text
native_quantized: preserve the current fast kernel per slab; fastest, numerical variant
bf16_swiglu: materialize only a BF16 SwiGLU slab and call fc2; higher precision, possibly slower
```

Do not hide that distinction behind one mode name.

### 5.5 Stage 0 gate

Proceed to Stage 1 when all are true:

- the full `fc1` output is confirmed live at the MLP peak;
- its removal is capable of bringing the MLP peak to or below the post-QKV
  attention peak, within 0.25 GB;
- the active weight layouts are supported by an explicit dispatch path;
- a held-weight strategy exists that avoids one weight transfer/cast per slab;
- the numerical behavior of chunk-local activation quantization is known.

If the MLP is already safely below the post-QKV attention peak, stop after the
measurement report and move directly to attention-output work; do not build an
optimization that cannot move the combined peak.

---

## 6. Stage 1 — token-chunked MLP

This is the highest-confidence implementation because the complete MLP branch is
row-independent.

### 6.1 Replacement block flow

The initial block forward keeps the attention half structurally identical and
replaces only the MLP half:

```python
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

# unchanged attention branch
h = core_mod_scale_shift(block.norm1(x), shift_msa, scale_msa, mod_segments)
attn_out = block.attn(h, rope_freqs=rope_freqs, transformer_options=transformer_options)
x = core_mod_gate(x, gate_msa, attn_out, mod_segments)

# chunked MLP branch
with held_mlp_weights(block.mlp, x) as mlp:
    for a, b, row in iter_mod_chunks(mod_segments, chunk_rows):
        h = block.norm2(x[a:b])
        h.mul_(1.0 + scale_mlp[row].to(h.dtype)).add_(shift_mlp[row].to(h.dtype))
        expanded = mlp.fc1(h)
        out = mlp.fc2_swiglu(expanded)
        x[a:b].addcmul_(out, gate_mlp[row].to(x.dtype))

return x
```

At no point does this branch allocate full-sequence `h`, `fc1`, SwiGLU, or `fc2`
outputs.

### 6.2 Held-weight session

`linear.py` owns a context manager that acquires `fc1` and `fc2` once for the
whole slab loop and releases them in reverse order afterwards.

Requirements:

- use Comfy's cast/offload APIs rather than `.to(device)`;
- preserve weight functions, LoRA patches, pre-quant scales, ConvRot metadata,
  and `want_requant` behavior;
- retain `QuantizedTensor` dispatch when the current module would retain it;
- hold the offload handles until all slab kernels have been enqueued;
- let `uncast_bias_weight` establish the stream dependency before reuse/offload;
- release already-acquired weights if acquiring the second weight raises;
- report whether each weight executed quantized, dequantized, or eager fallback;
- never silently fall back in benchmark/strict mode.

Holding both MLP weights at once is a deliberate trade: modestly more concurrent
weight residency in exchange for eliminating repeated streaming and a 2.456 GB
activation. Stage 0 measures the actual net result.

### 6.3 Down-projection adapters

The held session exposes one of these explicit paths:

| path | condition | behavior |
| --- | --- | --- |
| `tensorwise_int8_native` | `fc2` remains TensorWiseINT8 and the CK kernel is available | Call the same INT8 linear kernel with `input_act="swiglu"`, using the already acquired weight. |
| `quantized_dispatch_bf16_swiglu` | another supported `QuantizedTensor` layout | Compute BF16 SwiGLU for the slab, then call `F.linear` with the quantized weight and let layout dispatch handle it. |
| `eager_bf16` | returned weight is ordinary BF16/FP16 due to layout, LoRA, or patching | Compute BF16 SwiGLU and ordinary `F.linear`. |
| `unsupported` | semantics cannot be preserved | Raise in strict mode; optionally log and use the stock full-sequence MLP in non-strict mode. |

Do not copy the current `linear_input_act` implementation blindly. Keep the
adapter small, test it against the public helper on a full input, and document the
ComfyUI revision it mirrors.

### 6.4 Chunk-size sweep

One-block benchmarks sweep:

```text
1,024
2,048
4,096
8,192
full sequence
```

For each size and C profile collect:

- MLP peak allocated memory;
- full block peak allocated memory;
- MLP and block CUDA-event time;
- number of slab launches;
- weight acquisitions/transfers;
- numerical metrics against the stock MLP.

The expected starting point is 4,096 rows because the QKV release benchmark
already uses that size to bound temporary FP32 work. It is not the default until
the sweep shows that its runtime/memory trade is best.

### 6.5 Stage 1 gates

At C=90, the selected mode must satisfy:

- no full `[S, 28,672]` allocation appears in the memory trace;
- isolated MLP peak falls by at least 1.75 GB;
- each MLP weight is acquired at most once per block;
- full block peak is no longer established by the unchunked MLP;
- MLP runtime overhead is no more than 8% in the one-block benchmark;
- full denoising runtime overhead is provisionally no more than 5%;
- no NaN/Inf regression;
- numerical gates in §11 pass.

If `native_quantized` is faster but exceeds the numerical gate while
`bf16_swiglu` passes, ship only the BF16 path and keep the native mode
experimental.

---

## 7. Stage 2 — remove full projection outputs around residual accumulation

Once the MLP loop is stable, extend the same principle to outputs that exist only
so `_mod_gate` can consume them.

### 7.1 MLP output

Stage 1 already performs `fc2` and gated residual accumulation per slab, so the
full 0.461 GB MLP output disappears automatically.

### 7.2 Attention `out_proj` output

The current `Attention.forward` returns a complete `[S, 5,376]` projection, and
only then does `DiTBlock.forward` gate it into `x`. Eliminating that allocation
requires coordination with `h3_attention`: the block forward must receive the
pre-`out_proj` attention result, project it in slabs with the `out_proj` weight
held once, and accumulate each slab immediately.

This is **not** implemented by intercepting `out_proj.forward`, because a normal
module forward still has to return a complete output tensor. It requires an
explicit attention capability.

Proposed boundary, to be finalized by the attention implementation rather than
hard-coded here:

```python
result = attention_service.compute_preprojected(...)
# result exposes [S, attention_width] or an iterator of query slabs

with held_linear(block.attn.out_proj, sample=result) as out_proj:
    for a, b, row, attn_slab in result.iter_slabs(chunk_rows):
        projected = out_proj(attn_slab)
        x[a:b].addcmul_(projected, gate_msa[row].to(x.dtype))
```

The service must retain observer behavior and identify the executing backend.
The ordinary attention return path stays available for `sage`, `comfy`, and
`pytorch` baselines.

### 7.3 Stage 2 gate

- eliminate the full `[S, 5,376]` attention projection allocation;
- no more than one `out_proj` weight acquisition per block;
- no numerical difference beyond the chunked projection's established tolerance;
- recover at least 0.30 GB of peak if that output is live at the actual peak;
- no more than 3% additional full-run overhead beyond Stage 1.

If the full projection is not live at the peak, retain the design as groundwork
for Stage 4 but do not claim a memory win.

---

## 8. Stage 3 — bounded BF16 QKV production into Sage buffers

This stage crosses into attention-specific work. The implementation belongs in
`h3_attention`, while this package supplies the segment/slab planner and block
coordination.

### 8.1 Target dataflow

```text
allocate final compact Q/K/V destinations once
hold qkv_proj weight once

for each aligned token slab:
    norm1 + AdaLN slab                    BF16
    qkv_proj slab                         BF16
    Q/K RMSNorm + RoPE on matching slice  BF16 in place
    quantize/write Q                      INT8 destination slice
    quantize/write K                      INT8 destination slice
    transform/quantize/write V            FP8 destination slice
    discard all BF16 slab references

run dense Sage attention over compact destinations
```

This changes the BF16 QKV working allocation at C=90 from 1.842 GB to about
0.164 GB at 4,096 rows, while retaining the full compact attention inputs Sage
actually needs.

### 8.2 Requirements

- the custom Sage backend must accept prequantized Q/K/V and their scale metadata;
- chunk boundaries must align with every quantizer tile or scale domain that
  affects output;
- destination indexing must use safe offsets and preserve the int64 Q/K work;
- RoPE receives the exact `rope_freqs[a:b]` slice and original logical row
  indices;
- compact output must be bit-identical to full-buffer quantization below the
  overflow boundary wherever the quantizer's scale domain permits it;
- if a quantizer scale spans the complete sequence, either preserve that scale
  explicitly or classify the slabbed producer as a numerical variant;
- the current V guard and padded layout remain correct;
- Q/K/V destination allocation and scale buffers are included in the measured
  peak, not treated as free.

### 8.3 Stage 3 gate

- no full BF16 QKV allocation appears;
- compact Q/K/V and scales match the full producer under the defined numerical
  contract;
- end-to-end attention output passes the existing `sage_mem_eff` tests;
- block peak improves beyond early-release-only mode or equal memory is achieved
  with lower allocation pressure/fragmentation;
- runtime overhead is no more than 5% relative to early-release `sage_mem_eff`.

If the full compact-buffer API makes this slower without reducing peak, retain
early released QKV and stop. The slab producer is not mandatory for shipping the
first activation-memory result.

---

## 9. Stage 4 — query-slab attention and immediate output consumption

This is the most invasive exact-memory stage and must follow the simpler wins.

### 9.1 Feasibility test first

Establish whether the active dense Sage kernels correctly and efficiently support:

```text
Q length = query slab
K length = full sequence
V length = full sequence
```

Test rectangular attention against the full-call output at small and real shapes.
Do not infer support from an API signature; run the kernels and verify that no
fallback occurred.

### 9.2 Target dataflow

```text
full compact K/V resident
full compact Q or a reproducible Q-slab producer
held out_proj weight

for each query slab:
    dense attention(Q_slab, K_full, V_full)
    BF16 attention result slab
    out_proj slab
    gated add into x slab
    discard both outputs
```

Potential C=90 allocations removed:

```text
full BF16 attention result: 0.614 GB
full BF16 out_proj result:   0.461 GB
```

The actual saving depends on Sage workspace and whether full compact Q remains.

### 9.3 Performance risk

Multiple attention launches may lose scheduling efficiency, repeat setup, and
prevent the kernel from optimizing query tiles globally. K/V are logically read
for every query tile in either design, but separate launches can still materially
increase traffic and overhead. This stage is rejected if it buys memory by making
multi-hour sampling substantially slower.

### 9.4 Stage 4 gate

- rectangular path executes the intended Sage kernel with no fallback;
- full and slabbed attention outputs meet the numerical gate;
- remove both full BF16 output tensors from the allocation trace;
- recover at least 0.50 GB of additional peak;
- attention runtime overhead no more than 8%;
- complete denoising overhead no more than 5% beyond the Stage 1/2 configuration.

---

## 10. Stage 5 — explicit storage-precision ablations

Only run these after the BF16 slab path is stable. Each candidate is a separate
mode with its own report.

### 10.1 Candidates

| candidate | possible saving | initial position |
| --- | ---: | --- |
| FP8 normalized/modulated slab | only reduces slab size | low value after chunking; test only if larger slabs improve throughput |
| FP8 attention result before `out_proj` | up to half of a 0.614 GB full tensor, less when query-chunked | redundant if Stage 4 works |
| FP8 MLP expansion before `fc2` | up to half of 2.456 GB if kept full | inferior to BF16 chunking, but may allow larger/faster chunks |
| FP8 MLP output before residual add | up to half of 0.461 GB full tensor | redundant with immediate chunked accumulation |
| FP8 persistent residual `x` | 0.230 GB | **out of scope initially** |

### 10.2 Scale contracts

Every FP8 mode declares its scale domain:

```text
per-tensor global
per-slab
per-row/token
fixed checkpoint scale
block-scaled format
```

Changing the scale domain is a numerical change. Reports must include clipping
rate, scale distribution, saturation count, and error by block and denoising
step.

### 10.3 Acceptance

An FP8 storage mode proceeds only if it:

- saves at least 0.25 GB beyond the best BF16-slab configuration;
- does not increase runtime;
- passes stricter layerwise and end-to-end quality review than the ordinary Sage
  Q/K approximation already accepted;
- is opt-in and visibly labelled experimental.

No FP8 activation mode is required for the main plan to succeed.

---

## 11. Correctness and numerical validation

### 11.1 Unit-level CPU tests

Use tiny dimensions and ordinary FP32/BF16 weights to prove graph structure:

- chunk planner coverage and boundary splitting;
- full versus chunked RMSNorm/AdaLN;
- full versus chunked MLP;
- gate accumulation with multiple modulation segments;
- odd sequence lengths and chunks smaller/larger than segments;
- zero-length/invalid segment rejection;
- idempotent patch install and foreign-patch conflict;
- both block/attention patch installation orders;
- training-mode rejection;
- observer failures cannot interrupt inference.

For FP32 tiny tests, require ordinary tight `allclose` tolerances. Where operations
are rowwise and use the same kernel, test bit identity opportunistically but do
not make it the general contract.

### 11.2 GPU block tests

At C=22 and C=73, compare stock versus each candidate on the same real block
input, timestep embedding, modulation table, RoPE table, and checkpoint weights.
Record:

- maximum and mean absolute error;
- relative L2 error;
- cosine similarity;
- per-token error percentiles;
- NaN/Inf counts;
- error by packed segment kind;
- repeatability across two identical runs.

Provisional gates for BF16 exact-graph chunking:

```text
relative L2 <= 1e-3
cosine similarity >= 0.999999
no NaN/Inf mismatch
```

These are provisional because the active quantized `fc2` path may use a scale
domain changed by chunking. Stage 0 establishes the baseline and may tighten the
gate. It may only relax it with a written justification and end-to-end evidence.

### 11.3 Multi-block drift

Run captured inputs through:

```text
1 block
10 blocks
50 blocks
```

Compare after every block. A small error that grows monotonically or concentrates
in one segment is a stop signal even if the final aggregate metric remains under
the provisional threshold.

### 11.4 Denoising-level validation

Fixed seed, conditioning, sampler, scheduler, and step count. Capture model outputs
and latent state at each step for baseline and candidate. Report:

- per-step video/audio model-output error;
- per-step latent error;
- final latent error;
- decoded video contact sheet;
- audio waveform/spectrogram comparison when audio is present;
- wall-clock and peak memory.

Generation outputs can diverge from tiny numerical differences, so image metrics
alone are not a sufficient proof. The layer and step traces identify whether a
visible difference came from controlled numerical drift or a broken graph.

---

## 12. Benchmark matrix

Avoid a combinatorial end-to-end matrix. Use cheap block tests for sweeps and only
a few complete runs.

### 12.1 One-block sweep

| profile | modes | chunk sizes |
| --- | --- | --- |
| C=22 | stock, MLP chunked paths | 1,024 / 2,048 / 4,096 / 8,192 / full |
| C=73 | stock, MLP chunked paths | 2,048 / 4,096 / 8,192 / full |
| C=90 | stock, selected MLP path | selected size / full |

### 12.2 Combined block peak

At C=73 and C=90:

```text
stock sage
sage_mem_eff only
sage_mem_eff + MLP chunking
sage_mem_eff + best exact activation configuration
```

This is the matrix that determines whether the QKV release was being hidden by an
MLP peak.

### 12.3 Complete sampling runs

Only after block gates pass:

| profile | configuration | purpose |
| --- | --- | --- |
| C=22 | stock vs combined | cheap correctness and instrumentation check; not generation-quality evidence |
| C=73 | stock vs combined | current shipping-profile quality and runtime |
| C=90 | stock vs combined | decisive memory result and potential shipping-profile restoration |

Collect completion, peak allocated memory, driver/process context, wall-clock,
packed sequence length, per-step metrics, final checksums, decoded artifacts, and
logs identifying every backend/activation path.

---

## 13. Runtime configuration and node wiring

Do not add UI before Stage 1 GPU gates pass. Initial integration is a separate
experimental patch node so A/B testing remains explicit:

```text
MiniMax H3 Activation Memory (Zi)
```

Proposed inputs:

```text
model
mode:
    off
    mlp_chunked_bf16
    mlp_chunked_native_quantized      [only if numerical gate passes]
    full_chunked_exact                [later stages]
    fp8_experimental                  [Stage 5 only]
chunk_rows:
    1024..16384, default selected by benchmark
strict:
    true by default
```

The node clones the model, installs block patches, writes immutable configuration
into `transformer_options`, and returns the clone. `off` returns an unmodified
clone for graph symmetry.

### 13.1 Composition requirements

Test with:

- `MiniMaxH3SigmaShiftZi` and all existing attention backends;
- the H3 attention forward patch in both node orders;
- `h3_probe` observer active;
- `h3_masked_cache` measurement wrapper active;
- EasyCache active, noting that it skips complete forwards rather than changing a
  block that does run;
- LoRA-patched and unpatched weights;
- absent SageAttention, where MLP-only mode should still function.

No node may silently overwrite another block-forward patch. If future masked
computation also needs to own `DiTBlock.forward`, the two projects must merge into
one ordered block executor rather than race through object patches.

---

## 14. Success criteria

### Minimum success — worth shipping experimentally

- Full-sequence MLP `h`, `fc1`, and output allocations are absent.
- At C=90 the isolated MLP peak falls by at least 1.75 GB.
- The combined `sage_mem_eff + MLP chunking` block peak recovers at least 1.50 GB
  relative to stock Sage, demonstrating that the MLP no longer hides most of the
  QKV release.
- Full-run overhead is at most 5%.
- BF16 exact-graph numerical gates pass.
- No repeated MLP weight transfer/cast per slab.
- C=90 completes with useful guard margin on the same workload that previously sat
  at the ceiling.

### Full success

- Combined sequence-dependent C=90 block peak is at or near the post-QKV attention
  peak rather than the MLP peak.
- Bounded QKV production and attention output consumption remove the remaining
  avoidable full-sequence BF16 staging tensors.
- The resulting C=90 profile is stable enough to revise `chunked_ref2v/PLAN.md`
  back from C=73 to the measured compute optimum, with an explicit unattended-run
  margin.

### Stop conditions

Stop or retain the work as benchmark-only when any of these hold:

- MLP chunking does not move the combined peak after QKV release;
- weight streaming repeats per slab and a held session cannot preserve patch/LoRA
  semantics;
- the only fast path introduces unacceptable numerical drift;
- selected chunk sizes add more than 5% complete-run time for less than 0.5 GB
  combined benefit;
- attention query chunking adds more than 8% attention time;
- a lower-precision storage mode saves less memory than BF16 chunking while adding
  more error.

---

## 15. Commit sequence

| # | commit | contents | gate |
| ---: | --- | --- | --- |
| 0 | Activation-memory plan | `h3_activation_memory/PLAN.md` | review boundary and targets |
| 1 | Baseline activation observer and profiler | `observer.py`, `benchmarks/profile_h3_activations.py`, memory/weight inventory | Stage 0 facts recorded; no model behavior change |
| 2 | Segment-aware chunk planner | `chunks.py`, `tests/test_activation_chunks.py` | complete/gap-free coverage and segment parity |
| 3 | Held linear sessions and dispatch adapters | `linear.py`, `tests/test_activation_linear.py` | full-input parity with stock helpers; one acquisition per module |
| 4 | Chunked MLP block forward | `forward.py`, `patch.py`, forward/patch tests | Stage 1 memory, runtime, and numerical gates |
| 5 | Experimental node and reports | `config.py`, `stats.py`, `nodes.py`, root extension wiring | C=22/C=73 end-to-end completion; default remains off |
| 6 | Combined QKV-release/MLP validation | benchmark matrix and report | >=1.50 GB C=90 combined saving and <=5% overhead |
| 7 | Attention preprojection bridge | small capability addition in `h3_attention`, block-side consumer | Stage 2 gate; ordinary backend path unchanged |
| 8 | Slabbed QKV producer | implementation primarily in `h3_attention`, shared chunk planner | Stage 3 gate; no full BF16 QKV allocation |
| 9 | Query-slab attention | rectangular-kernel probe and output/gate pipeline | Stage 4 memory and runtime gates |
| 10 | Precision ablations | opt-in FP8 modes and reports | Stage 5 acceptance; never default automatically |
| 11 | Shipping decision | README and `chunked_ref2v/PLAN.md` profile update | complete C=90 evidence |

Commits 1-6 are the main experiment. Commits 7-10 are conditional follow-ups, not
prerequisites for the first useful result.

---

## 16. Open questions

1. **What is the actual MLP peak after the attention QKV release?** The source
   arithmetic says it can become dominant; only the phase trace settles it.
2. **Does the active TensorWise-INT8 path quantize activations over the complete
   tensor or a local row/block domain?** This decides whether native slab execution
   is numerically equivalent or an approximation.
3. **Can Comfy's cast/offload API safely hold two quantized linears across the
   complete slab loop without defeating vbar reuse?** The implementation must use
   the existing stream protocol rather than invent another one.
4. **How much GEMM efficiency is lost below 4,096 or 8,192 rows on SM89?** The best
   memory size is not automatically the best throughput size.
5. **Does `sage_mem_eff` expose a stable prequantized-input service?** If not,
   bounded QKV production remains inside the attention project until such an API
   exists.
6. **Do Sage kernels support rectangular query lengths without fallback or a large
   performance loss?** Unknown until Stage 4's direct probe.
7. **Will future masked block computation also need to own `DiTBlock.forward`?** If
   yes, both features need one composable block executor rather than independent
   object patches.
8. **Does `torch.compile` appear in the actual H3 path?** The first implementation
   is Python-loop inference and should raise or disable itself under unsupported
   compilation rather than silently graph-break.
9. **Is any FP8 activation storage still useful after exact slab execution?** It
   may only be valuable to permit larger, faster slabs; that is an empirical Stage
   5 question, not an assumption.

---

## 17. Expected conclusion

The most likely useful policy is:

```text
persistent residual state:       BF16
normalization/modulation:         BF16 slabs
MLP expansion and outputs:        BF16 slabs, immediately consumed
attention Q/K/V:                  existing INT8 / INT8 / FP8 destinations
QKV source:                       bounded BF16 slabs once Sage exposes the path
residual accumulation:            BF16
CPU transient offload:            not attempted unless bounded execution is insufficient
```

This preserves precision where state persists, uses lower precision where the
attention backend already requires it, and attacks the main memory problem as a
lifetime problem rather than assuming every large tensor must be stored in a
smaller dtype.