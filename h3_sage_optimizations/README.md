# H3 Sage Optimizations refactor

This package is the extraction boundary for two composable model-patch nodes:

- **MiniMax H3 Sage Memory Optimizer** owns prepared dense Sage selection,
  format-aware QKV execution, explicit/automatic MLP execution, and MLP
  activation-memory policy.
- **MiniMax H3 Sparse Sage Attention** owns target-video routing, Sparse Sage
  execution, packed-layout policy, sparse diagnostics, and optional shared
  fixed-route compilation.

Unknown models are exact pass-throughs. H3-specific model inspection happens
before cloning, CUDA probing, weight inspection, or patch installation.

## Kernel policy

Optimization candidates are divided into two buckets. See
[`KERNEL_POLICY.md`](KERNEL_POLICY.md) for the complete registry.

### Bucket 1: existing optimized kernels

Production `auto` may select only paths that retain existing optimized Comfy,
Comfy Kitchen, SageAttention, or SpargeAttention kernels:

- standard QKV through the checkpoint's native quantized linear dispatch;
- generic token-chunked MLP through quantized `F.linear`;
- ConvRot two-slice MLP through `ck.int8_linear`, including
  `input_act="swiglu"`;
- prepared dense Sage and Sparse Sage through their compiled kernels.

These paths can be adopted from normal end-to-end A/B measurements because the
experiment does not replace the optimized GEMM mainloop.

### Bucket 2: new optimized kernel required

The current fused-QKV and MLP-epilogue Triton implementations replace
established GEMMs with custom `tl.dot` mainloops. Other theoretical candidates,
such as gated-residual GEMM epilogues, direct FP8 V preparation, and zero-copy
packed-weight offsets, require new compiled kernel ABIs.

These paths are not selected by production `auto`. Existing prototypes are
available only for explicit kernel work with:

```text
H3_SAGE_ENABLE_RESEARCH_KERNELS=1
```

The gate is for development and characterization; it does not make a Bucket 2
path production-supported.

## Format-aware execution

The package preserves the checkpoint's existing linear layouts.

- `fused_qkv=auto` preserves the checkpoint's existing optimized QKV GEMM. The
  current fused-QKV prototypes remain Bucket 2.
- Compatible ConvRot MLP weights use the established Kitchen-backed two-slice
  provider when `mlp_memory=auto`.
- BF16, FP8, NVFP4, MXFP8, and other Comfy-supported layouts use generic token
  chunking and continue through the model's own quantized `F.linear` dispatch.
- Advanced MLP execution overrides preserve the former explicit BF16, native,
  and required ConvRot two-slice modes; all remain Bucket 1.
- Strict mode controls fallback inside the selected production MLP path. It does
  not promote QKV auto to a research implementation.

The attention carrier format remains independent from the checkpoint weight
format: dense Sage consumes per-thread INT8 Q/K carriers, while Sparse Sage
consumes architecture-specific block carriers plus routing summaries.

## Sparse routing and diagnostics

Sparse Sage retains fixed or adaptive-budget routing, minimum/maximum per-row
video density, adaptive temperature and target mass, strict packed-layout
validation, structural reports, deferred CUDA timing, and report run tags.

Adaptive routing preserves the fixed route's exact aggregate block count while
redistributing K between head/query rows. Non-video context and mixed boundary
tiles remain dense. The maximum defaults to `1.0`, leaving room for upward
redistribution.

Reports remain opt-in. The monolithic experimental Hybrid Sparse node remains in
H3-Extended for development workflows.

The current shared Inductor graph depends on the Bucket 2 fused-QKV prototype,
so it is also research-gated. A Sparse-first research compile request remains
pending until a compatible Memory Optimizer is applied, preserving node-order
independence.

## Composition

Both nodes attach immutable requests to the incoming `ModelPatcher`. Every node
re-resolves the complete request and reconciles one package-owned attention
forward, so `Memory -> Sparse` and `Sparse -> Memory` converge on the same
configuration. Conflicting duplicate instances fail rather than silently making
the last node win.

The apply path does not delegate to `h3_memory_optimizer.patch.apply`. It
installs only the selected attention patch, selected MLP patch, sparse runtime,
optional report listener, and optional research compiler required by the plan.

Sol, AdaLN precompute, FirstBlockCache, adaptive compilation, and stock
`TorchCompileModel` composition remain outside the production nodes.
