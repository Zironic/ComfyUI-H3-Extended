# MiniMax H3 Hybrid Sparse Attention — Deprecated Compatibility

This node remains registered so saved workflows continue to load. New workflows should use:

```text
MiniMax H3 Sage Memory Optimizer
MiniMax H3 Sparse Sage Attention
```

The compatibility node translates every meaningful former control into the new immutable plan:

- `mode` becomes the fused-QKV request owned by the Memory Optimizer;
- `activation` and `chunk_rows` become the MLP request;
- `video_budget`, fixed/adaptive routing, minimum/maximum densities, temperature, and target mass become the Sparse Sage routing request;
- `strict` preserves required specialized paths and strict packed-layout handling;
- `run_tag` and `timing` preserve structural report generation and optional deferred CUDA timing;
- `compile_backend=inductor` preserves the shared-block compile request and its original fixed/fused-QKV/ConvRot constraints.

The old widget IDs, order, defaults, and serialized node ID are retained. The adapter does not contain a second attention or MLP implementation; it calls the same apply path as the two production nodes.

Adaptive routing remains eager-only. Shared Inductor compilation requires fixed routing, `sage128_fused_qkv`, and `mlp_chunked_convrot_2slice`, matching the former node's validation.
