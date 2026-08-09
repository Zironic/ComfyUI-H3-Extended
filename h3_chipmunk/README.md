# H3 Chipmunk MLP

Experimental training-free MLP delta acceleration for MiniMax H3.

## What it changes

H3's exact attention/QKV path is left alone. The patch replaces only the main-block MLP execution after attention. It is designed to compose with the H3 attention patch, including Hybrid Sparse Sage.

H3 uses a bias-free 14,336-wide SwiGLU FFN and TensorWise INT8 ConvRot-256 weights. The implementation therefore selects complete 256-neuron logical SwiGLU groups. A selected group recomputes both its `fc1` gate and value rows. The cached MLP output is updated by subtracting the previous selected `fc2` contribution and adding the current selected contribution.

The selector applies `fc1` + SwiGLU to token-group mean inputs, keeps the previous per-feature summary, and ranks each 256-neuron group by the RMS of its cross-step feature delta. This avoids signed cancellation within a ConvRot group.

## Modes

- `measure`: output-exact. Every MLP is still executed densely. The node measures group dynamics and selected-activation deltas, but never feeds an approximate output into H3. Diagnostic activation history is kept on CPU and no full MLP output cache is retained.
- `reference_delta`: approximate. Dense refresh evaluations establish a dense MLP output cache. Between refreshes only selected 256-neuron groups are recomputed and their contribution is applied as a delta to the cached output.

`reference_delta` is deliberately marked experimental. The selected subset is evaluated with its own dynamic INT8 activation scale, so its contribution is not bit-identical to slicing the original full-width quantized `fc2` operation. Dense refreshes bound this drift.

## Recommended first run

Use:

```text
mode = measure
top_fraction = 0.25
refresh_every = 6
first_dense_steps = 2
last_dense_steps = 2
first_dense_layers = 2
chunk_rows = 128
token_group_rows = 128
scope = target_video
cache_location = cpu
strict = true
save_report = true
```

Reports are written under `output/h3_chipmunk/`.

Only move to `reference_delta` after inspecting the measurement report and running matched-seed dense controls.

## Patch ordering

The intended order is:

```text
MODEL
  -> attention / Hybrid Sparse Attention
  -> H3 Chipmunk MLP
  -> sampler
```

The Chipmunk node includes its own bounded ConvRot two-slice dense MLP path and should be used **instead of** `MiniMax H3 Activation Memory (Zi)` for that model clone. If Activation Memory was already applied, Chipmunk replaces that owned block-forward patch while preserving the underlying original H3 forward for fallback. Foreign block-forward patches and shared-block compilation are rejected.

## Cache policy

Caches are isolated by:

```text
request id
CFG branch
layer
MLP chunk
```

A new request or layout signature clears the cache. Text/reference/audio chunks remain dense by default; only target-video chunks are candidates when `scope=target_video`. `scope=all_dynamic` additionally permits target-audio chunks.

`cache_location=cpu` stores the selected activation cache and raw MLP output cache in ordinary host memory and synchronously copies the current chunk to/from the GPU. This makes the all-layer algorithm testable without keeping tens of GiB of cache in VRAM, but it is not the intended final performance path.

`cache_location=gpu` avoids those transfers. Before allocating persistent reference-delta state, the executor estimates its total cache footprint from the packed dynamic-token count, enabled layer range, hidden width, and selected feature count. It rejects a configuration whose estimate exceeds `cache_budget_gb`.

Dense execution is forced for:

- the first configured diffusion evaluations;
- the last configured evaluations;
- the first configured transformer layers;
- every `refresh_every` evaluations;
- chunks outside the selected scope;
- any unsupported/invalid state when `strict=false`.

H3's async weight leases are kept sequential: selected `fc1` work is completed and its lease released before selected `fc2` is acquired. This avoids aliasing Comfy's reusable weight-staging buffers.

## Real-weight microbenchmark

The branch includes a real-checkpoint benchmark that compares the current dense ConvRot MLP with the existing-primitives 256-group delta path at several active fractions:

```powershell
& .\python_embeded\python.exe custom_nodes\ComfyUI-H3-Extended\benchmarks\benchmark_h3_chipmunk_mlp.py `
  --checkpoint hf_minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --block-index 0 `
  --rows 2048 `
  --fractions 0.10,0.20,0.25,0.30,0.40,0.50 `
  --json h3-chipmunk-mlp.json
```

The benchmark reports median dense/delta latency, speedup, and relative L2 error against the next dense MLP output. Its selected groups are deliberately fixed contiguous groups so it measures execution geometry rather than selector quality.

## Current implementation boundary

This branch implements the complete H3-specific **research path** using existing Comfy/Comfy-Kitchen ConvRot INT8 operations, including exact measurement mode, reference delta accumulation, request/CFG cache isolation, dense refresh scheduling, CPU/GPU cache residency, reporting, and a real-weight microbenchmark.

It does **not** yet contain the bespoke fused CUDA/Triton kernels or asynchronous pinned-CPU double-buffer pipeline required for the final performance target. The current CPU cache is synchronous, and the selected path currently gathers weight rows/columns before calling the existing ConvRot primitives. Those operations make correctness and quality experiments possible but may erase the theoretical compute saving.

The next performance stage, if the measurement data supports 256-feature sparsity, is to fuse selected `fc1` gather + SwiGLU and old/new selected `fc2` contributions into H3-specific kernels, then replace synchronous CPU cache copies with slab-streamed pinned-memory prefetch/writeback. The measurement and real-weight benchmark paths are intended to determine whether that kernel work is justified before committing to it.
