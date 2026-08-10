# H3 Sage Optimizations refactor

This package is the extraction boundary for two composable model-patch nodes:

- **MiniMax H3 Sage Memory Optimizer** owns prepared dense Sage selection,
  backend-native fused QKV projection, and chunked/tiled MLP execution.
- **MiniMax H3 Sparse Sage Attention** owns only fixed-density target-video
  routing and Sparse Sage execution.

Both nodes attach immutable requests to the incoming `ModelPatcher`. Every node
re-resolves the complete request and reconciles one package-owned attention
forward, so `Memory -> Sparse` and `Sparse -> Memory` converge on the same
configuration.

Fused QKV is format-negotiated rather than universally represented:

- dense SM89 Sage receives SageAttention's per-thread INT8 Q/K scale ABI;
- Sparse Sage receives the established 128Q x 64KV block-scale carrier plus
  routing summaries.

The old combined Hybrid Sparse node remains unchanged while this path is
validated. Shared Inductor compilation and adaptive-density controls are not
part of the two production node schemas.
