# H3 Chipmunk MLP

Experimental training-free MLP delta acceleration for MiniMax H3.

## Production-node invariant

The Comfy model node is **CUDA-only for Chipmunk state**. It does not materialize CUDA diagnostics on the host and it does not support synchronous CPU-backed reference-delta caches.

Production execution/reporting must not use `.item()`, `.cpu()`, `.tolist()`, or device-to-host tensor copies. The shared H3 runtime is switched into `forbid_device_sync` mode when Chipmunk is installed; sampler evaluations are counted from the explicit sampler boundary instead of reading sigma values from CUDA.

If a future diagnostic requires host materialization, it belongs in a standalone benchmark/offline analysis tool, not in this model node.

## Modes

- `measure`: output-exact dense smoke mode. It performs the normal dense H3 MLP and does **not** collect CUDA-valued selector diagnostics.
- `reference_delta`: approximate GPU-resident Chipmunk execution. Dense refreshes establish GPU caches; intermediate evaluations recompute selected complete ConvRot-256 SwiGLU groups and apply their old/new `fc2` contribution as a delta to the cached raw MLP output.

`shadow_validate` was removed from the production node after live testing showed that diagnostic observation itself was an unacceptable workload for a real generation path.

## H3 geometry

H3 uses a bias-free 14,336-wide SwiGLU FFN with TensorWise INT8 ConvRot-256 weights. Chipmunk therefore selects complete 256-neuron logical SwiGLU groups. A selected group recomputes both paired `fc1` gate/value rows and the corresponding `fc2` columns.

The exact dense runner already prepackages H3 into two equal 7,168-feature ConvRot tiles. The production sparse selector therefore keeps an equal number of groups from each half. This gives fixed rectangular CUDA shapes and lets sparse fc1/fc2 reuse those already-held tiles without reacquiring or restaging model weights.

At `top_fraction=0.30`, each 28-group half keeps 9 groups, so the actual active width is 18/56 = **32.14%**.

The selector evaluates `fc1` + SwiGLU on token-group means, compares the current summary to the previous dense refresh summary, and ranks each 256-neuron group by RMS cross-step feature delta. Selection/top-k, activation caches, output caches, and delta updates remain on CUDA.

## Next real CUDA test

The first measurement showed the strongest concentration in the earliest transformer blocks, so the next test limits approximation to layers 0-9:

```text
mode = reference_delta
top_fraction = 0.30
refresh_every = 6
first_dense_steps = 2
last_dense_steps = 2
first_dense_layers = 0
layer_start = 0
layer_stop = 10
chunk_rows = 2048
token_group_rows = 128
scope = target_video
cache_location = gpu
cache_budget_gb = 24
random_groups = 0
strict = true
save_report = false
```

Layers 10-49 remain dense. This is intentionally conservative: it tests a real approximate video while keeping persistent GPU cache state modest. If quality and runtime behavior are acceptable, the next expansion is `layer_stop=15`.

Use ordinary H3 sampling for the first run. Keep Activation Memory, shared-block compile, FirstBlockCache, Vector Accel, and timing/profiling diagnostics off so the only new variable is the GPU-resident MLP delta path.

## Patch ordering

```text
MODEL
  -> attention / Hybrid Sparse Attention
  -> H3 Chipmunk MLP
  -> sampler
```

Use Chipmunk instead of Activation Memory/shared-block compilation on that model clone. Compatible H3 patches share one runtime session; installing Chipmunk upgrades that session to no-device-sync step tracking.

## Cache policy

Caches are isolated by request, CFG branch, layer, and MLP chunk. `reference_delta` requires `cache_location=gpu`. The previous pageable synchronous CPU cache path is rejected by configuration rather than silently stalling generation.

Before allocating reference-delta state, the executor estimates cache usage from dynamic-token count, enabled layer range, hidden width, and the balanced selected feature width. It raises if the estimate exceeds `cache_budget_gb`.

Dense execution is forced for the configured first/last evaluations, configured always-dense early layers, every `refresh_every` evaluations, and chunks outside the selected scope.

## Current performance boundary

This remains a research implementation using existing Comfy/Comfy-Kitchen ConvRot INT8 operations. It reuses the already-held two-slice ConvRot weights, but selected rows/columns are still gathered with `index_select(...).contiguous()` before calling the existing kernels. There are not yet bespoke fused sparse kernels. Therefore a successful quality test does not imply a wall-clock speedup yet.

The next optimization stage, if this CUDA-only quality run is acceptable, is fused group-indexed ConvRot execution that avoids those weight gathers and fuses the old/new selected `fc2` contribution update.
