# MiniMax H3 Sage Memory Optimizer

Applies memory and execution optimizations only to MiniMax H3. Other model families pass through unchanged.

The node uses two kernel-confidence buckets:

1. **Existing optimized kernels.** These paths retain Comfy, Comfy Kitchen, SageAttention, or SpargeAttention GEMMs and can be adopted after normal end-to-end A/B tests.
2. **New optimized kernel required.** These dataflows currently depend on custom GEMM prototypes or kernel ABIs that have not matched the established optimized path. They are never selected by production `auto`.

## Main controls

- **QKV projection optimization**
  - `auto`: preserve the checkpoint's existing optimized QKV GEMM. The current fused-QKV Triton implementation is bucket 2 and is not selected by production `auto`.
  - `off`: explicitly use standard H3 QKV.
- **MLP memory optimization**
  - `auto`: use the existing Comfy Kitchen ConvRot two-slice path when compatible; otherwise use generic token chunking through the checkpoint's native quantized `F.linear` dispatch.
  - `epilogue_prototype`: request the bucket-2 `fc1+SwiGLU` and `fc2+gated-residual` Triton prototype. It is blocked unless `H3_SAGE_ENABLE_RESEARCH_KERNELS=1` is set for explicit kernel development.
  - `off`: leave the H3 MLP forward unchanged.

The provider resolver carries a candidate ID, while `KERNEL_POLICY.md` defines each candidate's bucket and kernel basis.

## Advanced controls

- **Dense attention when Sparse is absent**
  - `auto`: select the prepared dense Sage backend supported by the current GPU.
  - `existing`: preserve incoming dense attention.
- **Explicit MLP execution override**
  - `auto`: follow the main MLP selector.
  - `chunked_bf16`: bounded BF16 SwiGLU while retaining the existing GEMMs.
  - `chunked_native`: native Comfy chunked execution.
  - `convrot_two_slice`: require the established Comfy Kitchen two-slice provider.
- **Error instead of specialized fallback**
  - Makes the selected bucket-1 MLP path fail closed instead of falling back at runtime.
  - It does not turn QKV `auto` into the bucket-2 fused-QKV prototype and does not enable research kernels.
- **MLP chunk rows**: maximum token rows in one MLP chunk.
- **Hold weights across chunks**: acquire `fc1` and `fc2` once when their effective Comfy handles are safe to retain.

## Research gate

Existing bucket-2 prototypes can be enabled only for kernel development:

```text
H3_SAGE_ENABLE_RESEARCH_KERNELS=1
```

This gate does not make those paths production supported. In particular, the current fused-QKV and MLP-epilogue implementations replace optimized GEMMs with custom Triton `tl.dot` mainloops. They need new Comfy Kitchen/CUTLASS-quality kernels before their algorithmic A/B results are meaningful.

## Composition

The Memory Optimizer and Sparse Sage nodes remain order-independent:

```text
Memory Optimizer -> Sparse Sage
Sparse Sage -> Memory Optimizer
```

Both orders resolve the same immutable plan and kernel policy.

## Disabled behavior

Disabling this node applies no new request. It does not remove patches already applied upstream on the same model branch.
