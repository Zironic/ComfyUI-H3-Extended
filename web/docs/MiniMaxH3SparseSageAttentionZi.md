# MiniMax H3 Sparse Sage Attention

Applies dependency-owned fixed-density sparse attention only to MiniMax H3.
Other model families pass through unchanged. If Sparse Sage is unavailable, a
supported NVIDIA runtime tries INT8 Triton sparse attention, then FP8
FlexAttention, before preserving the resolved dense H3 path.

## Video KV budget

**Video KV budget** is the requested fraction of pure target-video KV tiles retained for each attention head and pure-video query tile.

The request is rounded up to a whole KV-tile count, so the effective density depends on the packed video geometry. For example, a requested value of `0.50` may produce a slightly higher actual density when the number of pure-video KV tiles is odd.

## Denser Early/Late steps

Enable **Denser Early/Late steps** to add 30 percentage points to the Video KV budget for the first two and last two sampling steps. The adjusted budget is capped at 1.0; all other steps use the configured Video KV budget.

The following content remains dense regardless of the budget:

- text tokens;
- image and reference-conditioning tokens;
- target audio tokens;
- non-video query tiles;
- mixed boundary tiles containing both video and non-video tokens.

`1.0` preserves the complete pure-video route but still executes through the Sparse Sage path. It is mainly useful for comparisons and validation; it is not the same implementation as the ordinary dense Sage backend.

Lower values compute fewer pure-video attention blocks and are more approximate.

## QKV interaction

Sparse Sage works with either:

- standard H3 QKV projection; or
- a fused QKV provider selected by the Memory Optimizer when the checkpoint format and Sparse Sage ABI are compatible.

The Sparse node intentionally has no QKV or MLP controls. Those belong to the Memory Optimizer.

## Node order

The two production nodes are order-independent:

```text
Memory Optimizer -> Sparse Sage
Sparse Sage -> Memory Optimizer
```

The status text reports the requested budget, actual fallback backend, selected
QKV provider, and any upstream MLP optimization.

## Disabled behavior

Disabling this node applies no new sparse request. It does not remove patches already applied upstream on the same model branch.
