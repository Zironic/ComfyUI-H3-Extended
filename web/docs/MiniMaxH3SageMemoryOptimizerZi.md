# MiniMax H3 Sage Memory Optimizer

Applies memory and execution optimizations only to MiniMax H3. Other model families pass through unchanged.

The node inspects the actual QKV, `fc1`, and `fc2` weight layouts and selects a validated provider. It does not convert an FP8, BF16, NVFP4, or other checkpoint into ConvRot INT8 merely because a specialized kernel exists.

## Main controls

- **QKV projection optimization**
  - `auto`: use a fused QKV provider only when the checkpoint format, GPU, and resolved attention backend are compatible; otherwise use standard H3 QKV.
  - `off`: always use standard H3 QKV.
- **MLP memory optimization**
  - `auto`: use ConvRot two-slice execution when compatible; otherwise use generic token chunking through the checkpoint's existing Comfy linear format.
  - `epilogue_prototype`: opt into the experimental ConvRot `fc1+SwiGLU` and `fc2+gated-residual` kernels.
  - `off`: leave the H3 MLP forward unchanged.

The status text shown after execution reports the selected attention backend, QKV provider, MLP provider, compile state, and fallback reason.

## Advanced controls

- **Dense attention when Sparse is absent**
  - `auto`: select the prepared dense Sage backend supported by the current GPU.
  - `existing`: preserve the incoming dense attention implementation.
  - This setting is ignored when the Sparse Sage node is also present.
- **Explicit MLP execution override**
  - `auto`: follow the main MLP memory selector.
  - `chunked_bf16`: preserve the former explicit BF16 SwiGLU chunking path.
  - `chunked_native`: preserve the former native Comfy activation/linear path.
  - `convrot_two_slice`: require the established ConvRot-256 two-slice provider and fail during preflight when the loaded MLP format is incompatible.
  - An explicit override takes precedence over the main MLP selector.
- **Error instead of specialized fallback**
  - With `QKV projection optimization=auto`, convert the request into required fused QKV so incompatible formats or hardware fail during preflight instead of silently using standard QKV.
  - QKV explicitly set to `off` remains off.
  - Automatic, BF16, and native MLP selection use internal strict execution modes that preserve the selected arithmetic but raise on held-weight/provider failures rather than switching to ordinary module calls.
  - Explicit ConvRot two-slice and epilogue requests are already fail-closed.
- **MLP chunk rows**: maximum token rows in one MLP chunk. Larger chunks may be faster but require more activation memory.
- **Hold weights across chunks**: acquire `fc1` and `fc2` once for all chunks when the effective Comfy weight handles are safe to retain. Unsafe reusable cast-buffer combinations fall back automatically unless strict mode is enabled.

These controls preserve the meaningful `mode`, `activation`, `strict`, and `chunk_rows` behavior of the former combined experiment without putting QKV or MLP implementation choices on the Sparse node.

## Composition

The Memory Optimizer and Sparse Sage nodes are order-independent:

```text
Memory Optimizer -> Sparse Sage
Sparse Sage -> Memory Optimizer
```

Both orders resolve the same immutable optimization plan.

Four useful configurations are supported:

```text
Memory off, Sparse off:
    ordinary incoming H3 execution

Memory on, Sparse off:
    dense Sage when selected
    optional fused QKV
    optional chunked/tiled MLP

Memory off, Sparse on:
    standard H3 QKV into Sparse Sage

Memory on, Sparse on:
    optional fused QKV into Sparse Sage
    optional chunked/tiled MLP
```

## Format and carrier distinction

The checkpoint's weight format and Sage's post-projection Q/K carrier are separate concerns.

- Dense Sage uses its per-thread INT8 Q/K carrier layout.
- Sparse Sage uses blockwise Q/K scales plus routing summaries.
- The QKV weight itself may remain BF16, FP8, ConvRot INT8, or another Comfy-supported format when no specialized fused provider exists.

## Disabled behavior

Disabling this node applies no new request. It does not remove patches already applied upstream on the same model branch.
