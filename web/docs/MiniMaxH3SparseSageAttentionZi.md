# MiniMax H3 Sparse Sage Attention

Applies Sparse Sage only to MiniMax H3. Other model families pass through unchanged.

The main **Video KV budget** controls the aggregate pure target-video KV-tile budget. Text, reference images/video, audio, mixed boundary tiles, and non-video query tiles remain dense.

The requested value is rounded up to a whole KV-tile count. Effective density therefore depends on the current video geometry. A budget of `1.0` preserves the full pure-video route but still executes through Sparse Sage; it does not switch to the ordinary dense Sage backend.

## Advanced routing controls

- **Routing policy**
  - `fixed`: every pure-video head/query row retains the same quantized K.
  - `adaptive_budget`: rows retain different K values while the total retained block count remains exactly equal to the corresponding fixed route.
- **Minimum video density**: lower per-row rail for adaptive routing.
- **Maximum video density**: upper per-row rail for adaptive routing.
- **Adaptive temperature**: temperature applied to pooled Q/K scores before estimating block demand.
- **Adaptive target mass**: coarse cumulative video-attention mass used to estimate unconstrained row demand before exact global budget balancing.

For adaptive routing, the main Video KV budget must lie between the minimum and maximum densities. Setting both rails equal to the budget deliberately removes all redistribution. Setting only the maximum equal to the budget prevents any row from receiving more blocks than the fixed route.

## Advanced validation and diagnostics

- **Strict packed-layout validation**: requires authoritative H3 token-layout metadata. Disabling it is intended for diagnostics; Sparse Sage still cannot safely route an unknown layout.
- **Write sparse report**: writes per-request JSON and text reports under the ComfyUI output directory. Reports include effective density, routing distributions, provider selection, and per-layer/per-step records.
- **Include deferred CUDA timing**: adds request-scoped CUDA event timing to the report. Timing stages overlap and must not be summed.
- **Report run tag**: prefix for report directories. It accepts 1-64 ASCII letters, digits, underscores, or hyphens.

Enabling deferred CUDA timing automatically enables report generation.

## Advanced shared compilation

**Shared block compilation = inductor** requests one reusable CUDA tensor program for all 50 H3 blocks. It currently requires:

- fixed routing;
- an H3 Sage Memory Optimizer on the same model branch;
- fused Sparse Sage QKV resolving successfully;
- the established ConvRot two-slice MLP;
- no stock `TorchCompileModel` patch.

The two node orders remain supported. When Sparse Sage appears before the Memory Optimizer, compilation remains pending. Applying the compatible Memory Optimizer later resolves and installs the shared program. An incompatible completed plan fails with a clear preflight error.

## Composition

The Sparse Sage and Memory Optimizer nodes are order-independent:

```text
Memory Optimizer -> Sparse Sage
Sparse Sage -> Memory Optimizer
```

The Sparse node owns routing, layout validation, diagnostics, and optional sparse shared compilation. The Memory node owns dense attention selection, fused-QKV policy, MLP execution, chunk size, and held-weight policy.

## Disabled behavior

Disabling this node applies no new sparse request. It does not remove an upstream sparse or memory patch already present on the same model branch.
