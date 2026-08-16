# MiniMax H3 Hybrid Sparse Attention — Deprecated Compatibility

This node exists only so older saved workflows can load while migrating to:

1. **MiniMax H3 Sage Memory Optimizer**
2. **MiniMax H3 Sparse Sage Attention**

It translates the former combined fixed-density controls into the new immutable Memory and Sparse requests. Eager `adaptive_budget` workflows are routed through the preserved Hybrid Sparse implementation so saved workflows continue to run while migrating.

## Supported compatibility behavior

- `auto` is the new-node default. It prefers fused QKV only when the checkpoint
  format, GPU, Triton, and selected Sparse Sage ABI are compatible, and otherwise
  falls back to standard H3 QKV.
- `sage128` maps to standard H3 QKV into Sparse Sage.
- `sage128_fused_qkv` maps to fused QKV when compatible.
- legacy BF16 and native chunked MLP modes are preserved through internal compatibility requests;
- legacy ConvRot two-slice mode is required when `strict` is enabled and may use the production auto fallback when `strict` is disabled;
- `video_budget` maps to the new fixed-density Video KV budget.
- `adaptive_budget` keeps its legacy min/max density rails, score temperature,
  target mass, timing, and run tag when `compile_backend` is `off`.

## Unsupported legacy features

The compatibility adapter intentionally does not reproduce:

- shared Inductor block compilation (the production boundary remains
  fixed-density-only).

Adaptive routing is available in eager mode only. Fixed-density requests continue
to use the production plan path; their legacy `timing` and `run_tag` values are
accepted for workflow loading but ignored, with a visible status warning.

## Migration

Replace this one node with the two production nodes. Set:

- **QKV projection optimization** according to the former `mode` (`auto` is the
  production default);
- **MLP memory optimization** according to the former `activation`;
- **Video KV budget** to the former `video_budget`.

The two replacement nodes may appear in either order.
