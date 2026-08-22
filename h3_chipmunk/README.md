# H3 Chipmunk MLP

Experimental training-free MLP delta acceleration for MiniMax H3.

## Production invariant

All Chipmunk math runs on CUDA. Persistent cache tensors are **storage-only** pinned host buffers used for asynchronous DMA; they are never processed on CPU.

The model thread does not materialize CUDA values on the host and does not call device synchronization APIs. H2D and D2H transfers use dedicated CUDA streams plus events. If a transfer is not ready, that chunk runs dense instead of blocking the CPU.

The shared H3 runtime is switched into `forbid_device_sync` mode when Chipmunk is installed, so sampler evaluations are counted from the explicit sampler boundary instead of reading sigma values from CUDA.

## Why the cache is offloaded

A real H3 Chipmunk cache cannot reasonably be persistent in VRAM. For long video sequences the previous raw MLP output plus selected activation state is tens of GiB across the transformer stack.

The production design therefore uses:

```text
pinned host backing (persistent)
        ^          |
        | D2H      | H2D
        |          v
2 bounded CUDA staging slots
        |
        v
selected MLP CUDA kernels / dense fallback
```

With the default H3 geometry, `chunk_rows=2048`, and `depth_safe_v1`, the two staging slots occupy about **0.108 GiB** of VRAM. `cache_budget_gb=1.0` is a hard staging cap, not a request to reserve 1 GiB.

Pinned host buffers are allocated by a background thread as eligible layer/chunk shapes are discovered. The first dense evaluation gives that allocator time to work. If a particular buffer has not finished allocating by the next evaluation, that chunk simply stays dense. Allocated buffers remain warm across later requests with the same geometry.

## Modes

- `measure`: output-exact dense smoke mode. No CUDA-valued diagnostics are collected.
- `reference_delta`: actual approximate Chipmunk execution. Dense refreshes write state asynchronously to pinned backing; intermediate evaluations JIT-prefetch the state into bounded CUDA slots and apply selected old/new MLP contributions.

## H3 geometry

H3 uses a bias-free 14,336-wide SwiGLU FFN with TensorWise INT8 ConvRot-256 weights. Chipmunk selects complete 256-neuron logical groups, recomputing both paired `fc1` gate/value rows and the matching `fc2` columns.

The exact dense runner already prepackages H3 into two equal 7,168-feature ConvRot tiles. The sparse selector keeps an equal number of groups from each half, which gives fixed rectangular CUDA shapes and lets selected fc1/fc2 reuse the exact runner's already-held weights without reacquiring or restaging them.

## Depth-safe production profile

The measurement run showed that the earliest blocks have strong delta concentration, but those blocks are also the most destructive place to inject approximation error because every downstream block sees the perturbation. The production profile therefore protects the front of the transformer instead of blindly sparsifying where the selector looks strongest:

```text
layers  0-10: dense
layers 11-19: 40% requested density  -> 24/56 groups = 42.86% actual
layers 20-29: 50% requested density  -> 28/56 groups = 50.00% actual
layers 30-49: 60% requested density  -> 34/56 groups = 60.71% actual
```

`top_fraction` is used only when `density_profile=uniform`.

## First runnable test

```text
mode = reference_delta
density_profile = depth_safe_v1
refresh_every = 6
first_dense_steps = 2
last_dense_steps = 2
first_dense_layers = 0
layer_start = 0
layer_stop = 50
chunk_rows = 2048
token_group_rows = 128
scope = target_video
cache_location = async_pinned
cache_budget_gb = 1.0
random_groups = 0
strict = true
save_report = false
```

Use fixed 20% Hybrid Sparse attention for the first matched test. Keep Activation Memory, shared-block compile, FirstBlockCache, Vector Accel, and profiling diagnostics off so the only new approximation is the MLP delta path.

This is not a ten-layer smoke test. It exercises the complete depth policy while keeping persistent VRAM bounded.

## Transfer scheduling

Each block queues cache prefetch roughly two MLP chunks ahead. Two staging slots alternate:

```text
slot A: current chunk compute -> async D2H store
slot B: next chunk H2D prefetch
```

Ordering uses `torch.cuda.Event`, `Stream.wait_event`, and `Stream.wait_stream`. There is no `Event.synchronize`, `Stream.synchronize`, `.item()`, `.cpu()`, `.tolist()`, or blocking CUDA-to-host value materialization in the production Chipmunk modules.

A slot miss, unfinished pinned allocation, or unavailable DMA state never causes the model thread to wait for a CPU operation. The chunk falls back dense and can establish a fresh cache if backing storage is ready.

## Patch ordering

```text
MODEL
  -> attention / Hybrid Sparse Attention
  -> H3 Chipmunk MLP
  -> sampler
```

Use Chipmunk instead of Activation Memory/shared-block compilation on that model clone. Compatible H3 patches share one runtime session; installing Chipmunk upgrades the shared session to no-device-sync step tracking.

## Current performance boundary

The cache/storage architecture is now representative of something that can actually run under a small VRAM budget. The sparse math is still research-grade: selected ConvRot rows/columns are gathered with `index_select(...).contiguous()` before calling the existing Comfy-Kitchen INT8 kernels.

If the full depth-safe quality test is acceptable, the next optimization is a fused group-indexed ConvRot fc1/fc2 implementation that reads selected groups directly from the held weights and fuses the old/new `fc2` contribution update. That optimization can be done without changing the bounded async cache ABI.
