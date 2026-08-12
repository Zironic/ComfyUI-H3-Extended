# MiniMax H3 Hybrid Sparse Attention — Experimental

This is the full monolithic H3 optimization experiment. It intentionally remains in **ComfyUI-H3-Extended** after the two production nodes move into the dedicated H3 Sage Optimizations repository.

Use this node when testing combinations that the production nodes deliberately do not expose:

- fixed or adaptive Sparse Sage routing;
- standard or fused QKV projection;
- explicit BF16, native, or ConvRot two-slice MLP execution;
- timing reports and named run directories;
- shared Inductor compilation of the 50 main H3 blocks.

## Do not combine with the production nodes

This node owns the entire experimental attention-and-MLP transaction. Do not place **MiniMax H3 Sage Memory Optimizer** or **MiniMax H3 Sparse Sage Attention** on the same model branch. The node reports a clear error when it detects the production optimization plan.

## Density allocation

### `fixed`

Every pure target-video head/query row retains the requested `video_budget` fraction of its target-video KV blocks. Non-video context and mixed boundary tiles remain dense.

### `adaptive_budget`

The same aggregate target-video block budget is redistributed between head/query rows according to omitted coarse attention mass. `min_video_density`, `max_video_density`, `adaptive_temperature`, and `adaptive_target_mass` control that allocation.

Adaptive routing currently runs in eager mode. Shared Inductor compilation supports fixed density only.

## Shared Inductor compilation

`compile_backend=inductor` requires all of the following:

- `density_mode=fixed`;
- `mode=sage128_fused_qkv`;
- `activation=mlp_chunked_convrot_2slice`;
- compatible ConvRot-256 TensorWise-INT8 H3 weights.

Do not combine the shared compiler with ComfyUI's `TorchCompileModel` node.

## Reports

When `timing` is enabled, reports are written under:

```text
<Comfy output>/h3_hybrid_sparse/<run_tag>/
```

The run tag is part of the experimental report identity, so use distinct tags for materially different configurations.

## Production alternative

For ordinary use, prefer the two production nodes:

1. **MiniMax H3 Sage Memory Optimizer**
2. **MiniMax H3 Sparse Sage Attention**

Those nodes intentionally omit the monolithic experiment's combined execution surface.
