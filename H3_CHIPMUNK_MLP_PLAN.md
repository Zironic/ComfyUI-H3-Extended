# H3 Chipmunk MLP implementation plan

## 1. Recommended scope

The first implementation should **not port Chipmunk wholesale**.

It should:

* Keep the existing **20% Sparse Sage attention** path.
* Leave QKV projection, normalization, AdaLN, attention, and residual gating unchanged.
* Replace only H3’s dense MLP execution with a **Chipmunk-style cached sparse-delta MLP**.
* Initially sparsify only **target-video token rows**. Keep text, references, conditioning rows, and audio rows dense.
* Treat full Chipmunk attention as out of scope because Sparse Sage already attacks the attention matrix, while Chipmunk attention would not eliminate H3’s QKV projections anyway.

The practical architecture is therefore:

```text
H3 block
    exact AdaLN + norm
    exact QKV
    20% Sparse Sage attention
    exact attention residual
    exact norm + MLP AdaLN
    Chipmunk-style MLP delta
    exact current-step MLP gate
```

This directly attacks the remaining MLP cost without disturbing the sparse-attention work.

## 2. Why the upstream MLP cannot simply be inserted

H3’s main DiT consists of 50 blocks with hidden width 5,376 and FFN width 14,336. Its MLP is bias-free SwiGLU:

```text
fc1: 5,376 → 28,672
split into:
    gate: 14,336
    value: 14,336
SwiGLU:
    SiLU(gate) × value
fc2:
    14,336 → 5,376
```

The current optimized checkpoint path uses per-output-channel INT8 ConvRot-256 weights, while the current Chipmunk MLP wrapper expects a conventional single-activation MLP and supports BF16 or FP8 weights. The upstream implementation also directly expects an `fc1.bias`, which H3 does not have.

The Chipmunk paper says its assumed GELU can conceptually be replaced by another activation, so the algorithm is applicable to SwiGLU. The kernel implementation still has to be rewritten because one H3 logical MLP neuron depends on **two fc1 rows**:

[
a_j=\operatorname{SiLU}(g_j),u_j
]

Selecting neuron (j) requires recomputing:

```text
fc1 row j             → gate_j
fc1 row 14,336 + j    → value_j
fc2 input column j
```

Chipmunk’s general idea transfers; its existing MLP operation does not. ([arXiv][1])

## 3. Exact H3 delta formulation

For one token slab in one H3 block, after the current-step normalization and AdaLN modulation:

[
h_t=\operatorname{AdaLN}_t(\operatorname{RMSNorm}(x_t))
]

The dense H3 MLP is:

[
g_t,u_t=\operatorname{split}(W_1h_t)
]

[
a_t=\operatorname{SiLU}(g_t)\odot u_t
]

[
y_t=W_2a_t
]

[
x_t \leftarrow x_t+\operatorname{gate}_t\odot y_t
]

A dense refresh stores:

```text
activation_cache = a_t
output_cache     = y_t
selector_state   = a compact summary used to choose changing neurons
```

On a sparse step, for selected logical neurons (J):

[
a_t[J]=
\operatorname{SiLU}(W_{1,g}[J]h_t)
\odot
W_{1,u}[J]h_t
]

[
\Delta a[J]=a_t[J]-a_{\text{cache}}[J]
]

[
y_{\text{cache}}
\leftarrow
y_{\text{cache}}+W_2[:,J]\Delta a[J]
]

[
a_{\text{cache}}[J]\leftarrow a_t[J]
]

The block then uses:

[
x_t\leftarrow x_t+
\operatorname{gate}*t\odot y*{\text{cache}}
]

Two details are critical.

First, **cache the raw MLP output before the MLP gate**. H3’s AdaLN gate changes with the current video and audio timesteps. Reusing a previously gated residual would apply the old timestep modulation.

Second, when a neuron becomes selected after several skipped updates, its delta must be calculated against the **last value actually represented by the output cache**, not merely the immediately preceding diffusion step. This is why full dynamic Chipmunk normally keeps an activation cache.

This is the mechanism that addresses the issue found during token-masking experiments: nonselected components are not replaced by zero and the complete previous output remains present. Only their newest small correction is postponed.

## 4. The main H3-specific complication: ConvRot

H3’s INT8 weights use a 256-wide groupwise Hadamard rotation. The weight is rotated offline and the input activation is rotated online before the INT8 matrix multiplication. H3’s current ConvRot executor explicitly requires dimensions aligned to 256 and calls the ConvRot INT8 operation for both fc1 and fc2.

### 4.1 fc1 is manageable

ConvRot rotates along the **input dimension** of fc1. Selecting some fc1 output rows does not inherently break that rotation.

A sparse fc1 kernel can:

1. Rotate and quantize the complete 5,376-wide input row.
2. Gather selected fc1 output rows.
3. Run an INT8 GEMM producing only the selected gate and value features.

The selected rows must be gathered in SwiGLU pairs.

### 4.2 fc2 naturally couples 256 logical neurons

For fc2, the 14,336-wide SwiGLU activation is the input being rotated. A change in one original logical neuron spreads across the whole 256-feature Hadamard group in the rotated basis.

Therefore, **inference from the ConvRot math**: the existing fc2 weight layout naturally supports sparse updates at a granularity of **one complete 256-neuron group**, not arbitrary isolated neurons.

H3 has:

[
14,336 / 256=56
]

logical MLP feature groups per block.

The most direct production implementation should therefore route:

```text
C target tokens × selected 256-neuron groups
```

rather than upstream Chipmunk’s:

```text
C target tokens × selected individual neurons
```

At 25% active features, the kernel would select 14 of the 56 groups:

```text
14 × 256 = 3,584 logical SwiGLU features
7,168 selected fc1 output rows:
    3,584 gate rows
    3,584 value rows
```

The existing H3 activation-memory experiments already use 3,584-wide feature tiles, which makes this a particularly natural initial geometry.

### 4.3 Why this must be measured first

The Chipmunk paper found that coarse block sparsity had materially greater approximation error than fine column sparsity. At its reported MLP setting, 75% block sparsity had 61.3% unexplained change, versus 29.1% for 75% column sparsity, despite similar kernel runtime. A 256-neuron H3 group is not exactly the paper’s block shape, but the result is a warning that ConvRot-compatible grouping may discard too much useful change. ([arXiv][1])

The first experiment must answer:

> Does the cross-step MLP change remain concentrated when H3’s 14,336 logical neurons are aggregated into 56 ConvRot groups?

That determines the entire production kernel direction.

## 5. Cache-size constraint

A naïve Chipmunk cache is extremely large on H3.

For sequence length (S), each block would ordinarily store:

```text
activation cache: [S, 14,336]
output cache:     [S, 5,376]
```

At the repository’s documented benchmark sequence lengths:

| Sequence | BF16 activation | BF16 output | Per block | All 50 blocks |
| -------: | --------------: | ----------: | --------: | ------------: |
|   45,990 |        1.23 GiB |    0.46 GiB |  1.69 GiB |      84.4 GiB |
|   63,448 |        1.69 GiB |    0.64 GiB |  2.33 GiB |     116.5 GiB |

Even an INT8 activation cache plus a BF16 output cache is approximately:

```text
45,990 tokens: 1.07 GiB/block, 53.7 GiB total
63,448 tokens: 1.48 GiB/block, 74.1 GiB total
```

This excludes selector state, additional branches, staging buffers, and allocator overhead.

The cache architecture is consequently as important as the sparse kernel.

## 6. Two cache models to implement

### 6.1 Full-dynamic cache

This is closest to upstream Chipmunk.

For every token row and logical feature, retain the most recently represented activation:

```text
full activation cache: [rows, 14,336]
output cache:          [rows, 5,376]
```

Advantages:

* The selected neuron groups may change on every step.
* A newly selected group can subtract its correct last cached contribution.
* It most closely follows the paper’s algorithm.

Disadvantages:

* Very large host-RAM footprint.
* Every layer requires large cache transfers.
* CPU-to-GPU cache traffic may erase the MLP compute savings.
* Two CFG branches would require separate caches.

This should be implemented as a research/reference mode, not the first production default.

### 6.2 Fixed-window sparse cache

This is the recommended first production architecture.

The selected feature groups remain fixed between dense refreshes:

```text
dense refresh
    select active feature groups J
    cache only activation values for J
    cache full MLP output

sparse step 1
    update J

sparse step 2
    update J

...

next dense refresh
    recompute complete MLP
    choose a new J
```

Because the mask changes only during a dense refresh, there is no need to cache all 14,336 activations. Only the selected groups need activation storage.

At 25% active features:

```text
selected activation cache ≈ 25% of full activation cache
full output cache remains unavoidable
```

At 45,990 tokens, using INT8 selected activations and BF16 output:

```text
selected activation: ~0.15 GiB/block
output:              ~0.46 GiB/block
total:               ~0.61 GiB/block
50 blocks:           ~30.7 GiB
```

This is still substantial, but it is much more plausible for system-RAM backing.

The cost is that the sparse set is less dynamic. Dense refresh intervals must be short enough that a previously quiet group cannot become important for too long.

## 7. INT8 delta arithmetic

There is another H3-specific problem: H3 dynamically quantizes fc2 input activations per token row.

In exact real-valued arithmetic:

[
W_2(a_{\text{new}}-a_{\text{old}})
==================================

W_2a_{\text{new}}-W_2a_{\text{old}}
]

With independently chosen INT8 activation scales:

[
Q(a_{\text{new}})-Q(a_{\text{old}})
]

does not generally equal:

[
Q(a_{\text{new}}-a_{\text{old}})
]

A sparse delta computed using a new dynamic scale would therefore not reproduce the difference between the two dense INT8 operations.

### Recommended solution: frozen fc2 activation scales

During a dense refresh:

1. Compute the complete rotated SwiGLU activation.
2. Determine the current per-token activation scale.
3. Quantize the complete activation with that scale.
4. Compute and cache the exact dense output.
5. Preserve the scale for the following sparse window.

During sparse steps:

1. Compute the new selected 256-feature groups.
2. Rotate each selected group.
3. Quantize it using the **same frozen row scale**.
4. Compare it to the cached selected INT8 activation.
5. Update the cached output using the old and new quantized contributions.

Conceptually:

[
y_{\text{cache}}
\leftarrow
y_{\text{cache}}
----------------

W_2Q_s(a_{\text{old}}[J])
+
W_2Q_s(a_{\text{new}}[J])
]

where (Q_s) uses the scale fixed at the refresh step.

This makes the sparse update internally consistent with a frozen-scale quantized MLP.

If a new value exceeds the representable range of the frozen scale:

```text
saturation detected
→ abandon sparse execution for that chunk
→ run a dense refresh
→ establish a new scale
```

### Sparse fc2 implementation choices

There are two practical implementations.

**Two INT8 sparse GEMMs**

```text
current selected contribution
minus
cached selected contribution
```

This preserves the frozen-scale INT8 representation and should reuse tensor-core integer arithmetic, but it doubles the selected fc2 GEMM work.

**One BF16 delta GEMM**

```text
BF16 delta activation × dequantized INT8 weight tile
```

This requires a W8A16/BF16 sparse kernel and may be less computationally efficient, but it performs one selected fc2 pass and has cleaner linear delta semantics.

Both must be benchmarked against H3’s current W8A8 ConvRot baseline. Chipmunk’s published sparse MLP results were not measured against this exact baseline, and the released HunyuanVideo and WAN example configurations leave MLP sparsity disabled. H3 MLP acceleration is therefore a new experiment rather than a direct reuse of their demonstrated video configuration.

## 8. Phase 0: measurement-only feasibility probe

This should be implemented before any approximate execution.

### 8.1 Node

Add:

```text
MiniMax H3 Chipmunk MLP Probe (Zi)
```

The probe must run the existing exact H3 MLP and never alter its output.

### 8.2 Observation points

The current activation-memory forward already exposes the correct seam:

```text
attention residual complete
norm2
MLP AdaLN scale and shift
fc1
SwiGLU
fc2
MLP residual gate
```

The probe should observe:

```text
h_t         exact current MLP input after norm2/AdaLN
g_t, u_t    fc1 halves
a_t         SwiGLU activation
y_t         current exact MLP output before gate
gate_t      current MLP residual gate
```

The existing forward patch already processes MLP rows in bounded, modulation-aligned slabs, so the probe can be added without allocating a full `[sequence, 28,672]` expansion.

### 8.3 Sampling strategy

Do not retain all intermediate activations for every layer.

Start with layers:

```text
0, 1, 2, 5, 10, 20, 30, 40, 47, 49
```

For each selected layer, sample:

* All diffusion steps.
* Several target-video token tiles.
* At least one audio tile for comparison.
* One or two reference/text tiles.
* Low-motion, ordinary-motion, and high-motion generations.
* FL2VA and Ref2VA, because their reference-token layouts differ.
* The existing 20% Sparse Sage attention setting.

### 8.4 Candidate sparsity geometries

Evaluate:

```text
individual logical neurons       oracle upper bound
64-neuron groups
128-neuron groups
256-neuron ConvRot groups
512-neuron groups
```

Token grouping candidates:

```text
64 rows
128 rows
192 rows
256 rows
```

Never allow one token group to cross:

* A `mod_segment` boundary.
* A modality boundary.
* A target/reference boundary.

The paper found voxel reordering useful for attention but nearly irrelevant for MLP approximation. Do not initially reorder H3’s complete packed sequence, because that would also disturb existing attention and layout assumptions. Spatially local MLP-only grouping can be investigated without changing the attention token order. ([arXiv][1])

### 8.5 Selector candidates

Measure all of the following rather than choosing one by intuition.

#### Oracle activation delta

[
s_j=\lvert a_t[j]-a_{t-1}[j]\rvert
]

This gives the maximum plausible quality for a given active fraction.

#### W2-norm-weighted activation delta

[
s_j=
\lvert\Delta a_j\rvert
,
\lVert W_2[:,j]\rVert_2
]

This better estimates the potential output contribution.

#### Paired preactivation proxy

For the SwiGLU pair:

[
s_j=
\lvert g_t[j]-g_{\text{cache}}[j]\rvert+
\lvert u_t[j]-u_{\text{cache}}[j]\rvert
]

This is cheap but ignores the SwiGLU nonlinearity.

#### Approximate activation proxy

Using token-group means:

[
\bar h=\operatorname{mean}_{\text{token group}}(h)
]

[
\bar a=
\operatorname{SiLU}(W_{1,g}\bar h)
\odot
W_{1,u}\bar h
]

[
s_j=\lvert\bar a_t[j]-\bar a_{\text{cache}}[j]\rvert
]

This is the closest H3 analogue to Chipmunk’s block-mean selector.

#### First-order SwiGLU proxy

Approximate:

[
\Delta a
\approx
\operatorname{SiLU}'(g),u,\Delta g+
\operatorname{SiLU}(g),\Delta u
]

This may identify important features more accurately without calculating the complete token-level activation.

### 8.6 Metrics

For active fractions:

```text
10%, 15%, 20%, 25%, 30%, 40%, 50%, 60%, 75%
```

measure:

1. Explained variance in activation change.
2. Explained variance in exact MLP output change.
3. Relative L2 error in the current block MLP output.
4. Relative L2 error after applying the current gate.
5. Cosine similarity.
6. Maximum absolute error.
7. Error by layer.
8. Error by diffusion step.
9. Error by modality.
10. Error after several consecutive simulated sparse steps.
11. Required active fraction for individual-neuron versus 256-group selection.
12. Correlation between selector proxy and oracle selection.
13. Selector cost.
14. Cache bandwidth implied by each design.

The important metric is not merely:

```text
How much of |Δa| is selected?
```

It is:

```text
How accurately does the selected delta reproduce the gated block output
after several consecutive sparse steps?
```

### 8.7 Probe report

Write:

```text
output/h3_chipmunk_probe/<run>/
├── summary.json
├── layer_step_metrics.jsonl
├── selector_sweeps.npz
├── drift_windows.jsonl
├── timings.json
└── report.txt
```

The report should include heatmaps for:

```text
layer × step active fraction
layer × step explained delta
layer × step group-vs-column quality loss
layer × step projected speedup
```

### 8.8 First go/no-go decision

Suggested experimental decision rule:

* Proceed with ConvRot-group execution if approximately 25–35% of the 256-neuron groups consistently captures enough gated MLP output change to survive 4–8 sparse steps between refreshes.
* Proceed with fine-column execution if individual-neuron sparsity succeeds but 256-group sparsity does not.
* Stop the MLP project if neither granularity exhibits substantial concentration on H3.

The exact quality threshold must come from generated outputs and block-error distributions; it should not be copied from HunyuanVideo.

## 9. Phase 1: mathematically correct reference implementation

Create a pure PyTorch implementation that prioritizes correctness rather than speed.

Suggested module:

```python
class H3SwiGLUDeltaReference:
    def dense_refresh(...)
    def sparse_update(...)
    def reset(...)
```

Implement two modes.

### 9.1 BF16 reference

Use dequantized weights and ordinary BF16 operations.

Tests:

* 100% selected features reproduce the dense BF16 MLP.
* Repeated fixed-mask updates equal explicitly accumulated selected deltas.
* A feature selected after several skipped steps subtracts its correct cached value.
* Cache reset restores dense behavior.
* Changing the current MLP gate does not corrupt the raw output cache.

### 9.2 Frozen-scale ConvRot reference

Simulate:

* 256-wide Hadamard groups.
* INT8 activation values.
* Fixed activation scale during sparse windows.
* Separate old and new selected contributions.
* Saturation-triggered refresh.

Tests:

* A 100% active update reproduces the frozen-scale dense reference.
* Updating a complete 256-feature group preserves the rotated representation.
* Changing the group mask occurs only during a dense refresh in fixed-window mode.
* Saturation never silently clamps without reporting.
* NaN/Inf detection forces a dense fallback.

This phase establishes the algorithm before writing custom kernels.

## 10. Phase 2: one-block performance benchmark

Before integrating all 50 blocks, load one real H3 block using the same checkpoint-loading approach as the existing activation-memory benchmark.

Add:

```text
benchmarks/benchmark_h3_chipmunk_mlp.py
```

Inputs:

```text
checkpoint
block index
sequence rows
token slab sizes
active fractions
refresh intervals
token group sizes
cache dtypes
kernel backend
```

Benchmark:

1. Current dense ConvRot MLP.
2. Current two-slice ConvRot MLP.
3. BF16 reference delta.
4. ConvRot 256-group delta.
5. Fine-column alternate-W2 delta, if implemented.
6. Selector alone.
7. Cache load/store alone.
8. Complete sparse update including selector and cache I/O.

Sweep:

```text
rows:             1,024 / 2,048 / 4,096 / 8,192
active fraction:  10% through 75%
refresh interval: 2 / 4 / 6 / 8 / 10
```

The actual condition for using sparse MLP must be:

[
T_{\text{selector}}
+
T_{\text{cache I/O}}
+
T_{\text{sparse fc1}}
+
T_{\text{sparse fc2}}
<
T_{\text{dense H3 MLP}}
]

H3’s dense baseline is already quantized and optimized. A sparse implementation that beats a BF16 reference but loses to H3’s current ConvRot path is not useful.

## 11. Phase 3: ConvRot-native group-delta kernel

This is the recommended first production kernel if the probe supports 256-neuron grouping.

### 11.1 Token and feature geometry

Use:

```text
token group C:      initially 128
feature group:      fixed 256
feature groups:     56
MLP slab rows:      initially 2,048
token groups/slab:  16
```

The sparse group mask for one slab is only:

```text
[16 token groups, 56 feature groups]
```

At 25% active features:

```text
14 selected groups per token group
```

### 11.2 Dense-refresh kernel

For each slab:

1. Compute exact current `h`.
2. Execute complete fc1.
3. Apply SwiGLU.
4. Rotate activation in 256-feature groups.
5. Quantize using a per-token scale.
6. Execute complete fc2.
7. Write BF16 `output_cache`.
8. Generate selector summaries.
9. Select feature groups for the following sparse window.
10. Store selected rotated INT8 activation values.
11. Store frozen scales.
12. Clear saturation/error state.

This dense path should use the same quantized weight representation as the current H3 executor.

### 11.3 Sparse fc1 kernel

Inputs:

```text
h
fc1 qdata and scales
selected group indices
selected counts
```

Process:

1. Apply the standard online ConvRot input transformation.
2. Dynamically quantize `h` exactly as the dense fc1 path does.
3. Gather the selected gate rows.
4. Gather the corresponding value rows.
5. Compute both with packed dense tensor-core tiles.
6. Apply SwiGLU.
7. Produce `[C, selected_groups × 256]`.

Because selected logical groups correspond to contiguous 256-row regions in each fc1 half, this should be more hardware-friendly than arbitrary single-row gathers.

### 11.4 Selected activation update

For each selected group:

1. Apply the 256-wide activation rotation.
2. Quantize with the frozen fc2 scale.
3. Check saturation.
4. Load its cached old INT8 activation.
5. Make both old and current values available to sparse fc2.
6. Replace the cache with the current value after the update completes.

### 11.5 Sparse fc2 kernel

Recommended initial implementation:

```text
accumulator = BF16/FP32 output_cache

for selected group:
    accumulator -= old_quantized_activation × selected W2 columns
    accumulator += new_quantized_activation × selected W2 columns

write accumulator back to output_cache
```

Fuse the subtraction and addition into one persistent kernel if possible, but internally it may issue two integer MMA operations.

The alternative W8A16 implementation would:

```text
delta = dequantized(new - old)
accumulator += delta × selected W2 columns
```

Both should use selected 256-column weight tiles.

### 11.6 Residual application

After sparse MLP execution:

```python
x_chunk.addcmul_(
    output_cache,
    current_gate_mlp[mod_row],
)
```

The gate is always the current step’s gate.

## 12. Cache data structures

For fixed-window group mode, one slab cache could be:

```python
@dataclass
class H3ChipmunkChunkCache:
    state: CacheState

    selected_groups: Tensor       # [token_groups, max_groups], uint8/int16
    selected_counts: Tensor       # [token_groups], uint8

    activation_q: Tensor          # packed selected activation groups, int8
    activation_scale: Tensor      # [rows], fp16/fp32
    output_cache: Tensor          # [rows, 5376], bf16

    selector_summary: Tensor      # [token_groups, 56], fp16/fp32
    sentinel_summary: Tensor | None

    refresh_step: int
    last_update_step: int
    saturation_count: int
    fallback_reason: str | None
```

A kernel-friendly activation layout would be approximately:

```text
[token_group, selected_group_slot, C, 256]
```

or its column-major equivalent, depending on the selected GEMM layout.

Full-dynamic mode would instead use:

```text
activation_q: [rows, 14,336]
```

and allow group indices to change each step.

## 13. Cache residency and offloading

### 13.1 First kernel prototype: GPU-resident selected layers

Do not begin with CPU offloading.

Enable Chipmunk on only 2–4 blocks and keep their caches in VRAM. This isolates:

* Sparse kernel performance.
* Selector quality.
* Quantization drift.
* Visual impact.

At the documented 45,990-token shape, a full BF16 activation/output cache is approximately 1.69 GiB per block. A selected-only INT8 activation plus BF16 output cache at 25% active is around 0.61 GiB per block.

This makes a 4-block prototype plausible on a high-VRAM device without introducing PCIe effects.

### 13.2 Slab-streamed CPU cache

Once the kernel works, use the existing 2,048-row MLP slabs.

For one 2,048-row slab:

```text
full BF16 activation: ~56 MiB
BF16 output:          ~21 MiB
full INT8 activation: ~28 MiB
```

For selected-only INT8 activation at 25%:

```text
selected activation: ~7 MiB
BF16 output:          ~21 MiB
total:                ~28 MiB/slab
```

Use two GPU cache slots:

```text
slot A: current compute
slot B: asynchronously loading the next slab / storing the previous slab
```

The host backing store contains all layer/chunk caches, but active GPU cache memory remains bounded.

### 13.3 Streams

Use dedicated streams for:

* Cache H2D prefetch.
* Cache D2H writeback.
* Main MLP computation.

The pipeline should be:

```text
load chunk n+1 cache
    while
compute chunk n
    while
store chunk n-1 cache
```

The upstream Chipmunk storage layer similarly uses pinned CPU backing, a small number of shared GPU slots, and asynchronous load/offload streams. Its design can be adapted, although H3’s cache geometry and interaction with weight staging differ substantially.

### 13.4 Coordinate with weight staging

H3 may already be streaming block weights through AIMDO/prefetch infrastructure. Cache transfers and weight transfers must not independently saturate PCIe.

The scheduler should expose a single layer pipeline:

```text
prefetch next block weights
prefetch next block cache slab
compute current block
write current cache slab
release previous block weights
```

Measure:

* Effective PCIe throughput.
* Overlap percentage.
* Time blocked waiting for cache.
* Time blocked waiting for weights.

A sparse kernel that saves compute but causes additional serial PCIe traffic is not a successful optimization.

### 13.5 Cache compression

Implement in this order:

1. BF16 output, INT8 activation.
2. FP8 output experiment.
3. INT8 or block-FP8 output experiment.
4. Error-feedback or residual compensation only if output compression is otherwise viable.

Output-cache compression is likely to be more sensitive because its complete 5,376-dimensional vector is inserted into the current block residual on every sparse step.

## 14. Scheduler and dense refreshes

Use a per-layer state machine:

```text
EMPTY
  ↓
DENSE_WARMUP_0
  ↓
DENSE_WARMUP_1 + choose mask
  ↓
SPARSE_1
  ↓
SPARSE_2
  ↓
...
  ↓
DENSE_REFRESH + choose new mask
```

### 14.1 Initial schedule

For measurement, use an explicit schedule:

```text
step 0: dense
step 1: dense
steps 2–5: sparse
step 6: dense
steps 7–10: sparse
...
```

Do not copy upstream’s FLUX defaults directly. Its example uses 70% MLP sparsity and a dense step every ten calls, but H3 uses a different activation, quantization layout, sequence composition, and baseline kernel.

### 14.2 Per-layer decisions

Each layer should independently choose dense or sparse.

Force that layer dense when:

* Its selected density exceeds the measured sparse-kernel crossover.
* Its selector confidence is low.
* Saturation occurs.
* A sentinel group changes unexpectedly.
* Cache data is unavailable.
* The current step is part of a mandatory refresh schedule.

This matters because some layers may have much less concentrated change than others.

### 14.3 Adaptive refresh

After the fixed schedule is validated, add an adaptive controller.

Inputs:

* Current selector-score distribution.
* Saturation count.
* Change in group summaries.
* Random or round-robin sentinel groups.
* Sigma distance from the last actual model call.
* Number of sparse updates since refresh.
* Historical approximation error in diagnostic mode.

Actions:

```text
continue sparse
increase active fraction
force next layer refresh
force complete model refresh
```

### 14.4 Sentinel groups

Fixed-window caching cannot notice an unselected group becoming important unless it measures something outside the mask.

Reserve a small sentinel budget:

```text
1–3 random or round-robin feature groups per token group
```

Compute their cheap selector proxy on sparse steps. If their change exceeds the selected-group threshold:

```text
mark cache stale
run dense at the next safe point
```

Sentinel values do not need to be inserted into the current output unless the kernel supports dynamic activation caching.

### 14.5 Late-step conservatism

A conservative preset should force:

* The first two model evaluations dense.
* The final one or two evaluations dense.
* More frequent refreshes when the activation-delta distribution becomes less concentrated.

Whether H3 is actually less sparse early or late is unknown; the probe must establish that.

## 15. Runtime lifecycle

The repository already tracks:

* Sampling request identity.
* Step index.
* Total steps.
* Sigma.
* CFG branch.
* Packed-layout signature.
* Device and compute dtype.
* Request reset and request end.

Reuse this runtime rather than inventing a second sampler tracker.

Cache key:

```text
request_id
branch
layout_signature
layer_index
chunk_index
```

Add an actual **model-call ordinal**, because some samplers can evaluate the network multiple times within one nominal step.

Force dense reset on:

* New outer sample request.
* Layout change.
* Branch change without a corresponding cache.
* Sigma direction reversal.
* Non-monotonic sampler behavior.
* Missing or invalid step information.
* Jump of more than the configured sigma/step distance.
* Device or dtype change.
* Interrupted or failed model call.
* N+1/long-form transition to the next sampling request.

Do not carry MLP caches between long-form chunks. Their references, prompt, packed layout, and diffusion trajectory may differ even when tensor geometry is identical.

## 16. Modality policy

H3’s packed sequence contains text, references or condition rows, target audio, and target video.

Recommended first policy:

```text
text:             dense
reference image:  dense
reference video:  dense
reference audio:  dense
target audio:     dense
target video:     Chipmunk candidate
```

Reasons:

* Target video dominates the token count.
* MLP is tokenwise, so row-specific execution does not require changing attention geometry.
* Keeping conditioning and audio rows dense makes early quality diagnosis easier.
* Audio can still be affected indirectly because the next block’s attention mixes approximate video states with audio states, so audio evaluation remains necessary.

Later experiments can test:

* Sparse target audio with stricter thresholds.
* Sparse reference-video rows.
* Different active fractions by modality.
* Different refresh intervals by modality.

## 17. Repository integration

### 17.1 New package

```text
h3_chipmunk/
├── __init__.py
├── config.py
├── state.py
├── selector.py
├── reference.py
├── cache.py
├── storage.py
├── scheduler.py
├── executor.py
├── ops.py
├── patch.py
├── probe.py
├── report.py
├── nodes.py
└── kernels/
    ├── triton_selector.py
    ├── triton_sparse_fc1.py
    ├── triton_group_delta_fc2.py
    └── csrc/
```

### 17.2 Existing files to change

#### `h3_activation_memory/forward.py`

Refactor the current exact MLP section behind an executor interface:

```python
class H3MlpExecutor(Protocol):
    def run_chunk(
        self,
        *,
        layer_index: int,
        chunk_index: int,
        h: torch.Tensor,
        mod_row: int,
        gate: torch.Tensor,
        block,
        snapshot,
        transformer_options,
    ) -> torch.Tensor:
        ...
```

Implement:

```text
ExactNativeExecutor
ExactConvRotExecutor
ChipmunkMeasureExecutor
ChipmunkGroupDeltaExecutor
ChipmunkFineColumnExecutor
```

The block forward should continue owning:

* AdaLN.
* Attention.
* Chunk construction.
* Residual gating.

The executor owns only the MLP computation and cache state.

This is preferable to installing a second independent block-forward patch. The current activation-memory installer rejects foreign owners of `blocks.N.forward`, so an independent Chipmunk block patch would conflict with the existing system.

#### `h3_activation_memory/config.py`

Either extend the internal MLP backend configuration or introduce a composite configuration:

```python
@dataclass(frozen=True)
class H3MlpExecutionConfig:
    exact_mode: ...
    chipmunk: ChipmunkConfig | None
```

Avoid duplicating chunk size, alignment, strictness, and held-weight options.

#### `h3_runtime/context.py`

Add:

```text
model_call_ordinal
previous_step_index per branch
step_gap
sigma_gap
successful_forward flag
```

Notify Chipmunk listeners on failed forward so partially updated caches are invalidated.

#### `h3_runtime/block_dispatch.py`

Coordinate cache prefetch with weight binding/prefetch.

#### `h3_runtime/block_compile.py`

Do not modify this during the initial reference implementation. Add compiled support only after the stateful kernel is stable.

#### Root registration

Register:

```text
MiniMax H3 Chipmunk MLP Probe (Zi)
MiniMax H3 Chipmunk MLP (Zi)
```

The upstream Chipmunk source is MIT-licensed; copied or adapted kernel code must retain its copyright and license notice.

## 18. Custom operation boundaries

Suggested custom operations:

```text
minimax_h3::chipmunk_select_groups
minimax_h3::chipmunk_sparse_swiglu_fc1
minimax_h3::chipmunk_group_delta_fc2
minimax_h3::chipmunk_dense_cache_refresh
```

Each needs:

* CUDA/Triton implementation.
* Fake/meta implementation.
* Explicit mutation declaration for cache tensors.
* Shape and dtype validation.
* BF16-only initial compute contract.
* ConvRot-256 validation.
* Static maximum selected-group count.

Use fixed-size `indices` arrays and a dynamic `counts` value to avoid graph recompilation when the active fraction changes.

The upstream Chipmunk MLP operation is deliberately excluded from `torch.compile` because Inductor inserted an expensive intermediate copy between its two sparse kernels. An initial graph break around the H3 custom MLP operation is acceptable if the operation contains all expensive computation internally.

## 19. Shared-block compilation

The current repository can compile one shared H3 block graph and call it repeatedly with different weight carriers. The graph currently includes the ConvRot fc1 and fc2 operations.

Integration should happen in two stages.

### Stage A: compiled attention, eager cached MLP

Split the block execution conceptually into:

```text
compiled:
    AdaLN
    QKV
    Sparse Sage
    attention projection and residual

eager/custom op:
    norm2/AdaLN
    cache staging
    sparse-delta MLP
    MLP residual
```

This is the simplest stateful implementation.

### Stage B: explicit cache tensors in the shared graph

Once stable:

* Pass loaded GPU cache slabs as explicit tensor inputs.
* Pass indices and counts as explicit tensors.
* Call one opaque custom MLP operation inside the shared graph.
* Return the updated residual tensor.
* Write cache tensors back outside the graph.

Do not pass Python layer IDs or registry identities into Dynamo. The existing shared-block architecture intentionally avoids layer-identity specialization.

## 20. Public node design

Initial node:

```text
MiniMax H3 Chipmunk MLP (Zi)
```

Keep the ordinary UI limited to:

```text
model
enabled
preset
cache_budget_gb
scope
strict
save_report
```

Presets:

```text
measure
conservative
balanced
aggressive
manual
```

Advanced/manual fields:

```text
active fraction
refresh interval
layer start/stop
token group rows
cache mode
cache location
activation cache dtype
output cache dtype
sentinel fraction
dense fallback fraction
```

Do not merge this into the general Memory Optimizer until it demonstrates an end-to-end improvement over the current exact ConvRot MLP.

## 21. Correctness test plan

### Mathematical tests

```text
test_swiglu_pair_selection
test_dense_refresh_matches_reference
test_full_selection_matches_dense_reference
test_sparse_delta_accumulation
test_newly_selected_feature_uses_correct_old_cache
test_current_gate_is_not_cached
test_fixed_scale_int8_update
test_saturation_forces_refresh
test_group_rotation_roundtrip
```

### Cache lifecycle tests

```text
test_new_request_resets_all_caches
test_layout_change_resets
test_branch_caches_are_isolated
test_sigma_reversal_resets
test_failed_forward_invalidates_cache
test_step_gap_forces_dense
test_longform_next_chunk_resets
```

### H3 integration tests

```text
test_only_target_video_rows_are_sparse
test_mod_segments_never_share_selector_group
test_audio_rows_remain_dense
test_reference_rows_remain_dense
test_disabled_path_is_unchanged
test_foreign_block_patch_is_rejected
test_sparse_attention_composes
```

### Memory tests

```text
test_no_full_fc1_expansion_allocation
test_cache_respects_budget
test_cpu_offload_roundtrip
test_double_buffer_stream_order
test_no_cache_tensor_survives_request_end
```

### Compile tests

```text
test_custom_op_fake_contract
test_fixed_index_shape_avoids_recompile
test_shared_graph_reused_across_layers
test_cache_state_not_specialized_into_graph
```

## 22. Generation validation

Use fixed seeds and retain a fully dense control for every run.

Test classes:

1. Static camera, low-motion subject.
2. Static background with moving face and hands.
3. Camera movement through a mostly static scene.
4. High-motion action.
5. Multiple subjects and occlusion.
6. Hard lighting or texture refinement.
7. Audio-heavy speech.
8. Non-speech sound.
9. FL2VA.
10. Ref2VA.
11. N+1 continuation.
12. Long-form stitched generation.

Measure:

* Final latent relative error.
* Per-step velocity error.
* Block-output error.
* Decoded frame similarity.
* Temporal flicker.
* Edge and texture stability.
* Identity consistency.
* Audio spectrogram difference.
* Audible clipping or discontinuity.
* End-to-end seconds per iteration.
* MLP-only milliseconds.
* Cache transfer time.
* Selected density.
* Dense fallback rate.
* Peak VRAM.
* Pinned and total system RAM.

Diagnostic runs should periodically execute both dense and sparse versions of selected blocks without feeding the dense result forward. This provides ground-truth approximation error during a real generation.

## 23. Performance model

Ignoring cache I/O, H3’s dense MLP matrix-multiply work is approximately:

[
2HF + HF = 3HF
]

where:

* fc1 costs (2HF)
* fc2 costs (HF)

With active logical-feature fraction (p):

### One sparse delta fc2 GEMM

[
T_{\text{sparse compute}}\approx pT_{\text{dense}}
]

### Two INT8 selected fc2 GEMMs

[
T_{\text{sparse compute}}
\approx
\frac{2pHF+2pHF}{3HF}
=====================

\frac{4p}{3}
]

At (p=0.25):

[
\frac{4p}{3}\approx0.33
]

At (p=0.30):

[
\frac{4p}{3}\approx0.40
]

The group-mean selector computes fc1 for approximately (1/C) as many token rows. At (C=128), its raw matrix-multiply work is approximately:

[
\frac{2}{3C}\approx0.52%
]

of the dense MLP.

With one dense refresh every (R) calls, the idealized average compute fraction is:

[
r_{\text{avg}}
==============

\frac{1}{R}
+
\frac{R-1}{R}r_{\text{sparse}}
]

For (p=0.25), two selected fc2 GEMMs, and (R=8):

[
r_{\text{avg}}
\approx
0.125+0.875(0.333)
\approx0.42
]

That corresponds to an idealized MLP compute reduction of roughly 2.4× before accounting for cache traffic, selector overhead, sparse-kernel efficiency, and dense fallbacks.

End-to-end iteration time should be modeled as:

[
T_{\text{new}}
==============

T_{\text{attention}}
+
T_{\text{QKV}}
+
r_{\text{MLP}}T_{\text{MLP}}
+
T_{\text{other}}
+
T_{\text{cache}}
+
T_{\text{selector}}
]

The value of (T_{\text{MLP}}) in the current 51-second iteration is still unknown, so an end-to-end speed prediction would be premature.

## 24. Milestones and acceptance gates

### M0 — H3 sparsity probe

Deliver:

* Individual-neuron and 256-group delta distributions.
* Selector accuracy.
* Drift-window simulations.
* Layer/step/modality heatmaps.

Decision:

```text
ConvRot groups viable
fine-column kernel required
or MLP delta not worthwhile
```

### M1 — reference implementation

Acceptance:

* 100% active path reproduces its defined dense reference.
* Cache update behavior passes all synthetic tests.
* No output change when disabled.
* Fixed-scale saturation handling works.

### M2 — one-block kernel

Acceptance:

* Complete sparse path beats current dense ConvRot MLP below a measured active-density threshold.
* Selector and cache update are included in the timing.
* No full `[rows, 28,672]` expansion is allocated.
* Approximation error matches the reference implementation.

### M3 — GPU-resident subset

Enable 2–4 layers.

Acceptance:

* Real end-to-end generation is faster than the current 20% Sparse Sage baseline.
* No reproducible quality or audio regression under conservative settings.
* Cache state remains stable over complete generations.

### M4 — all-layer CPU-backed cache

Acceptance:

* Cache transfers substantially overlap computation.
* PCIe traffic does not erase the kernel gain.
* System RAM and pinned-memory use remain bounded.
* The dense-fallback rate remains low enough to produce a real speedup.

### M5 — adaptive scheduler

Acceptance:

* Adaptive refresh performs at least as well as the best fixed schedule.
* Sentinel detection catches sudden feature changes.
* High-motion generations automatically use more dense work.

### M6 — compiled integration

Acceptance:

* One shared graph remains reusable across all 50 block weight bindings.
* Dynamic masks do not create recompilations.
* Cache lifecycle remains outside graph specialization.
* Compiled execution improves rather than reduces end-to-end performance.

## 25. Recommended implementation order

The critical path should be:

1. **Build the measurement-only probe.**
2. **Compare individual-neuron sparsity with ConvRot-compatible 256-group sparsity.**
3. **Benchmark a single real H3 block against the current INT8 ConvRot baseline.**
4. **Implement GPU-resident fixed-window group deltas on a small layer subset.**
5. **Validate frozen-scale INT8 accumulation and saturation refreshes.**
6. **Add target-video-only execution across all blocks.**
7. **Implement slab-streamed pinned-CPU cache backing.**
8. **Add adaptive refresh and sentinels.**
9. **Only pursue a non-ConvRot fine-column fc2 layout if 256-group quality is inadequate.**
10. **Reintegrate the finished operation into the shared compiled-block path.**

The central uncertainty is not whether the delta equation works. It does. The open questions are whether H3’s MLP changes remain concentrated at **ConvRot-compatible 256-feature granularity**, and whether the resulting compute saving exceeds the cache traffic and the speed of H3’s existing INT8 dense MLP. Those two questions should be resolved before committing to the full kernel and offload system.

[1]: https://arxiv.org/abs/2506.03275 "https://arxiv.org/abs/2506.03275"
