# MiniMax H3 Hybrid Sparse Attention — Deprecated Compatibility

This node exists only so older saved workflows can load while migrating to:

1. **MiniMax H3 Sage Memory Optimizer**
2. **MiniMax H3 Sparse Sage Attention**

It translates the former combined fixed-density controls into the new immutable Memory and Sparse requests. The execution implementation is shared with the new nodes.

## Supported compatibility behavior

- `sage128` maps to standard H3 QKV into Sparse Sage.
- `sage128_fused_qkv` maps to fused QKV when compatible.
- legacy BF16 and native chunked MLP modes are preserved through internal compatibility requests;
- legacy ConvRot two-slice mode is required when `strict` is enabled and may use the production auto fallback when `strict` is disabled;
- `video_budget` maps to the new fixed-density Video KV budget.

## Unsupported legacy features

The compatibility adapter intentionally does not reproduce:

- `adaptive_budget` routing;
- shared Inductor block compilation;
- legacy timing report directories and run tags.

Adaptive routing and shared compilation require manual migration or the old experimental implementation. Legacy `timing` and `run_tag` values are accepted for workflow loading but ignored, with a visible status warning.

## Migration

Replace this one node with the two production nodes. Set:

- **QKV projection optimization** according to the former `mode`;
- **MLP memory optimization** according to the former `activation`;
- **Video KV budget** to the former `video_budget`.

The two replacement nodes may appear in either order.
