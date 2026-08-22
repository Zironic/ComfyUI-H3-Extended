# MiniMax H3 Sage Memory Optimizer

Applies memory and execution optimizations only to MiniMax H3. Other model families pass through unchanged.

The node is a compatibility adapter over `H3-Optimizations`. It inspects the
actual QKV, `fc1`, and `fc2` weight layouts and selects a validated provider.
ConvRot INT8 keeps its specialized paths, FP8 uses held FP8 execution, ordinary
BF16/FP16 may use accelerated FP8 conversion, and unsupported quantized formats
preserve upstream Comfy execution.

## Main controls

- **QKV projection optimization**
  - `auto` (default): use a chunked QKV provider only when the checkpoint format and resolved backend's complete producer/consumer contract are compatible; otherwise use standard H3 QKV.
  - `off`: always use standard H3 QKV.
- **MLP memory optimization**
  - `auto`: use ConvRot two-slice, held FP8, converted FP8, or bounded floating execution according to the validated checkpoint and runtime capabilities.
  - `epilogue_prototype`: accepted for saved workflows and migrated to the production ConvRot two-slice path.
  - `off`: leave the H3 MLP forward unchanged.

The status text reports the selected attention backend, V-layout shim, QKV and
MLP providers, chunk sizes, and fallback reasons.

## Advanced controls

- **Dense attention when Sparse is absent**
  - `auto`: select ComfyUI's public `comfy_kitchen_int8` backend when available.
  - `existing`: preserve the incoming dense attention implementation.
  - This setting is ignored when the Sparse Sage node is also present.
- **MLP chunk rows**: maximum token rows in one MLP chunk. Larger chunks may be faster but require more activation memory.
- **Hold weights across chunks**: acquire `fc1` and `fc2` once for all chunks when the effective Comfy weight handles are safe to retain. Unsafe reusable cast-buffer combinations fall back automatically.

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
