# H3 Sage optimization kernel policy

Optimization candidates are split into two buckets according to the kernel that
executes their large matrix multiplications.

## Bucket 1: existing optimized kernels

A Bucket 1 candidate preserves the established Comfy, Comfy Kitchen,
SageAttention, or SpargeAttention GEMM/attention kernel. These candidates can be
properly evaluated with ordinary end-to-end A/B tests because the comparison is
about the optimization itself rather than the quality of a replacement GEMM.

Current Bucket 1 paths are:

- standard QKV through the checkpoint's native Comfy quantized linear dispatch;
- generic token-chunked MLP through quantized `F.linear` dispatch;
- ConvRot two-slice MLP through `ck.int8_linear`, including
  `input_act="swiglu"`;
- prepared dense Sage through the compiled SageAttention kernel;
- fixed or adaptive Sparse Sage routing through the compiled SpargeAttention
  kernel.

Production `auto` may select only Bucket 1. A candidate is adopted after its A/B
matrix shows acceptable output parity and a useful end-to-end latency or peak
VRAM improvement on the supported hardware.

A future composition such as RMSNorm/AdaLN followed by the existing Kitchen
quantizer and GEMM also belongs in Bucket 1. It still needs an A/B test because a
separate transform kernel can cost more than the allocation it saves.

## Bucket 2: a new optimized kernel is required

A Bucket 2 candidate has an attractive theoretical dataflow, but it either
replaces an established GEMM with an unvalidated custom mainloop or needs a
compiled kernel ABI that does not yet exist.

Current Bucket 2 work includes:

- fused QKV projection plus RMSNorm, RoPE, and Sage-carrier emission;
- `fc1+SwiGLU` activation-carrier emission;
- `fc2+gated-residual` and `out_proj+gated-residual` GEMM epilogues;
- direct HND-to-FP8 V preparation;
- zero-copy packed-weight slicing through GEMM row/column offsets;
- the current shared Inductor graph, because it depends on the custom fused-QKV
  Triton GEMM.

A mathematically correct Triton prototype does not make a Bucket 2 candidate a
valid production A/B comparison. It remains research-only until a Comfy
Kitchen/CUTLASS-quality implementation is demonstrated to match the established
kernel.

Existing prototypes are gated by:

```text
H3_SAGE_ENABLE_RESEARCH_KERNELS=1
```

The gate exists for kernel development and characterization. It does not make a
research path production-supported and does not allow production `auto` to
select it.
