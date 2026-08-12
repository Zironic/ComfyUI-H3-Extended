# H3 Sage Optimizations refactor

This package is the extraction boundary for two composable model-patch nodes:

- **MiniMax H3 Sage Memory Optimizer** owns prepared dense Sage selection,
  format-aware fused QKV selection, and chunked/tiled MLP execution.
- **MiniMax H3 Sparse Sage Attention** owns target-video routing, Sparse Sage
  execution, packed-layout policy, sparse diagnostics, and optional shared
  fixed-route compilation.

Unknown models are exact pass-throughs. H3-specific model inspection happens
before cloning, CUDA probing, weight inspection, or patch installation.

## Format-aware execution

The package preserves the checkpoint's existing linear layouts.

- A validated ConvRot-256 TensorWise-INT8 H3 can use the current specialized
  dense or sparse fused-QKV provider.
- Unsupported QKV formats use the standard H3 projection when
  `fused_qkv=auto`; the internal `required` mode fails during preflight.
- Compatible ConvRot MLP weights use the established two-slice feature-tiled
  provider when `mlp_memory=auto`.
- BF16, FP8, NVFP4, MXFP8, and other Comfy-supported layouts use generic token
  chunking and continue through the model's own quantized `F.linear` dispatch.

The attention carrier format is independent from the checkpoint weight format:
dense Sage consumes per-thread INT8 Q/K carriers, while Sparse Sage consumes
architecture-specific block carriers plus routing summaries.

## Sparse routing and diagnostics

The production Sparse node keeps `video_budget` as its primary control and
places every other meaningful former sparse setting under Advanced:

- fixed or adaptive-budget routing;
- minimum/maximum per-row video density;
- adaptive temperature and target mass;
- strict packed-layout validation;
- structural reports, deferred CUDA timing, and report run tags;
- shared Inductor block compilation.

Adaptive routing preserves the fixed route's exact aggregate block count while
redistributing K between head/query rows. Non-video context and mixed boundary
tiles remain dense. The production adaptive maximum defaults to `1.0`, avoiding
the old `video_budget=0.50` plus `max_video_density=0.50` configuration that left
no room for upward redistribution.

Reports are opt-in on the production node. The deprecated combined adapter
preserves its historical behavior of always writing structural reports while
using the old `timing` toggle only to include or omit deferred CUDA events.

Shared compilation remains fixed-route-only and requires the completed two-node
plan to resolve fused Sparse QKV plus the established ConvRot two-slice MLP. A
Sparse-first compile request remains pending until a compatible Memory Optimizer
is applied, preserving node-order independence.

## MLP epilogue prototype

`mlp_memory=epilogue_prototype` is an explicit CUDA/Triton experiment for
homogeneous ConvRot-256 TensorWise-INT8 H3 MLP weights. It retains the existing
two feature slices but changes the temporary tensor boundaries:

1. The fc1 GEMM applies SwiGLU before storing, so each feature slice writes only
   the activated half-width carrier rather than the gate/up pair.
2. The fc2 GEMM multiplies by the AdaLN gate and accumulates directly into the
   block residual. It does not allocate a hidden-width fc2 output tensor.
3. The two fc2 slice contributions are applied sequentially to the residual,
   which is algebraically equivalent to gating their sum but may differ by BF16
   rounding order.

The prototype is not selected by `auto`. It does not support shared H3
compilation, non-ConvRot weight layouts, FP16 activations, or training. CUDA
numerical parity and end-to-end VRAM/latency remain to be measured before it can
replace the established two-slice provider.

## Composition

Both nodes attach immutable requests to the incoming `ModelPatcher`. Every node
re-resolves the complete request and reconciles one package-owned attention
forward, so `Memory -> Sparse` and `Sparse -> Memory` converge on the same
configuration. Conflicting duplicate instances fail rather than silently making
the last node win.

The apply path does not delegate to `h3_memory_optimizer.patch.apply`. It
installs only the selected attention patch, selected MLP patch, sparse runtime,
optional report listener, and optional shared compiler required by the plan.

Sol, AdaLN precompute, FirstBlockCache, adaptive compilation, and stock
`TorchCompileModel` composition remain outside the production nodes.
