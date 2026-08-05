# MiniMax H3 Memory Optimizer

`MiniMax H3 Memory Optimizer (Zi)` combines the prepared-QKV attention patch and
bounded activation-memory patch behind one capability-driven node.

## Attention support

With `attention=auto`, the node mirrors SageAttention 2.2's public NVIDIA
architecture dispatch:

| capability | prepared path |
| --- | --- |
| SM80 | per-thread INT8 Q/K, FP16 V, CUDA FP32 accumulation |
| SM86 | per-block INT8 Q/K, FP16 V, Triton attention |
| SM89 | per-thread INT8 Q/K, FP8 V, CUDA FP32+FP16 accumulation |
| SM90 | per-thread INT8 Q/K, FP8 V, Hopper CUDA FP32+FP32 accumulation |
| SM120/121 | per-warp INT8 Q/K, FP8 V, SM89-family CUDA kernel |

Only exact capabilities supported by upstream SageAttention are selected. Other
NVIDIA capabilities, non-CUDA devices, absent SageAttention/Triton, unsupported
package versions, or missing internal exports select `existing` attention when
`attention_fallback=allow`.

Fallback happens before model mutation. A runtime kernel failure remains an
explicit error because a CUDA fault may poison the context.

## Activation memory

`mlp_chunked_bf16` is independent of the attention adapter and remains active
when attention falls back. It bounds the full-sequence MLP expansion with
token slabs. `mlp_chunked_native` additionally requests Comfy's native fused
SwiGLU path when available.

The original standalone attention and activation-memory nodes remain available
for isolated testing.
