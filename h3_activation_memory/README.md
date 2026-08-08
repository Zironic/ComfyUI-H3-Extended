# H3 activation memory

Experimental MiniMax H3 block-forward patch that keeps the persistent residual
stream in BF16 and evaluates the tokenwise MLP in bounded row slabs.

## Node

Add **MiniMax H3 Activation Memory (Zi)** after loading/configuring the model and
before the sampler.

- `mlp_chunked_bf16`: bounded BF16 SwiGLU slabs.
- `mlp_chunked_native`: default; uses Comfy's fused TensorWise-INT8 SwiGLU path
  when the active `fc2` layout supports it.
- `chunk_rows`: default 2048, selected from the real-weight MLP sweep.
- `prefer_held_weights`: enabled by default; acquires `fc1` and `fc2` once per
  block when distinct async cast buffers make that safe. A same-buffer case
  uses ordinary per-slab module calls instead of risking weight corruption.
- `strict`: raises on core drift, unsupported acquisition, or `torch.compile`.

The patch owns `diffusion_model.blocks.N.forward`; the attention work owns
`diffusion_model.blocks.N.attn.forward`, so both patches compose.

## Validation

From the ComfyUI root:

```bash
python custom_nodes/ComfyUI-H3-Extended/tests/test_activation_memory.py
```

Synthetic slab-size sweep:

```bash
python custom_nodes/ComfyUI-H3-Extended/benchmarks/benchmark_h3_activation_memory.py \
  --device cuda --dtype bf16 --seq 45990 --hidden 5376 --ffn 14336 \
  --chunks 1024,2048,4096,8192
```

The synthetic benchmark does not load the real checkpoint or its quantized
weights. To measure one real block's MLP weights, pass a registered diffusion
model name (or an absolute `.safetensors` path):

```bash
python custom_nodes/ComfyUI-H3-Extended/benchmarks/benchmark_h3_activation_memory.py \
  --checkpoint hf_minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --block-index 0 --device cuda --dtype bf16 --seq 63448 \
  --chunks 2048,4096,8192,16384 \
  --swiglu-modes bf16,native --held-modes off,on --json h3-mlp.json
```

Actual-checkpoint mode lazily loads only the selected block's `fc1` and `fc2`
MLP tensors (including their quantization metadata); it is an MLP benchmark,
not a full model load or generation run.

See [`PLAN.md`](PLAN.md) for the staged attention-side follow-ups. Prequantized
QKV production and query-slab attention remain capability-gated on
`h3_attention`; they are not silently approximated here.
