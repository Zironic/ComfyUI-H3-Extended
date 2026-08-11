# H3 Sage Optimizations refactor

This package is the extraction boundary for two composable model-patch nodes:

- **MiniMax H3 Sage Memory Optimizer** owns prepared dense Sage selection,
  format-aware fused QKV selection, and chunked/tiled MLP execution.
- **MiniMax H3 Sparse Sage Attention** owns only fixed-density target-video
  routing and Sparse Sage execution.

Unknown models are exact pass-throughs. H3-specific model inspection happens
before cloning, CUDA probing, weight inspection, or patch installation.

## Format-aware execution

The package preserves the checkpoint's existing linear layouts.

- A validated ConvRot-256 TensorWise-INT8 H3 can use the current specialized
  dense or sparse fused-QKV provider.
- Unsupported QKV formats use the standard H3 projection when
  `fused_qkv=auto`; the internal `required` mode fails during preflight.
- Compatible ConvRot MLP weights use the two-slice feature-tiled provider.
- BF16, FP8, NVFP4, MXFP8, and other Comfy-supported layouts use generic token
  chunking and continue through the model's own quantized `F.linear` dispatch.

The attention carrier format is independent from the checkpoint weight format:
dense Sage consumes per-thread INT8 Q/K carriers, while Sparse Sage consumes
128Q x 64KV block carriers plus routing summaries.

## Composition

Both nodes attach immutable requests to the incoming `ModelPatcher`. Every node
re-resolves the complete request and reconciles one package-owned attention
forward, so `Memory -> Sparse` and `Sparse -> Memory` converge on the same
configuration. Conflicting duplicate instances fail rather than silently making
the last node win.

The new apply path no longer delegates to `h3_memory_optimizer.patch.apply`.
It installs only the selected attention patch, the selected MLP patch, and the
runtime layout context required by Sparse Sage.

Shared Inductor compilation, adaptive-density controls, timing, run tags, Sol,
AdaLN precompute, and FirstBlockCache are outside the two production schemas.
