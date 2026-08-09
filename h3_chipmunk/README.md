# H3 Chipmunk MLP

Experimental training-free MLP delta acceleration for MiniMax H3.

## What it changes

H3's exact attention/QKV path is left alone. The patch replaces only the main-block MLP execution after attention. It is designed to compose with the H3 attention patch, including Hybrid Sparse Sage.

H3 uses a bias-free 14,336-wide SwiGLU FFN and TensorWise INT8 ConvRot-256 weights. The implementation therefore selects complete 256-neuron logical SwiGLU groups. A selected group recomputes both its `fc1` gate and value rows. The cached MLP output is updated by subtracting the previous selected `fc2` contribution and adding the current selected contribution.

## Modes

- `measure`: output-exact. Every MLP is still executed densely. The node measures group dynamics and prepares selector/cache state, but never feeds an approximate output into H3.
- `reference_delta`: approximate. Dense refresh evaluations establish an exact dense MLP output cache. Between refreshes only selected 256-neuron groups are recomputed and their contribution is applied as a delta to the cached output.

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

A new request or layout signature clears the cache. Text/reference/audio chunks remain dense by default; only target-video chunks are candidates when `scope=target_video`.

Dense execution is forced for:

- the first configured diffusion evaluations;
- the last configured evaluations;
- the first configured transformer layers;
- every `refresh_every` evaluations;
- chunks outside the selected scope;
- any unsupported/invalid state when `strict=false`.

## Current implementation boundary

This branch implements the H3-specific group-delta algorithm with existing Comfy/Comfy-Kitchen ConvRot INT8 operations. It does not yet contain a bespoke fused CUDA kernel or CPU-backed activation-cache pipeline. Consequently `reference_delta` is a functional research path, not yet expected to achieve the final performance target described in the implementation plan.

The next performance stage is to fuse selected `fc1` gather + SwiGLU and old/new selected `fc2` contributions into H3-specific kernels and then add slab-streamed cache offload. The measurement mode is intended to determine whether the 256-feature ConvRot granularity is worth that kernel work before committing to it.
