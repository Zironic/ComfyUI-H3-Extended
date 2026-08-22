# H3 activation memory

Experimental MiniMax H3 block-forward patch that keeps the persistent residual
stream in BF16 and evaluates the tokenwise MLP in bounded row slabs.

## Node

Add **MiniMax H3 Activation Memory (Zi)** after loading/configuring the model and
before the sampler.

- `mlp_chunked_bf16`: bounded BF16 SwiGLU slabs.
- `mlp_chunked_native`: default; uses Comfy's fused TensorWise-INT8 SwiGLU path
  when the active `fc2` layout supports it.
- `mlp_chunked_convrot_2slice`: strict BF16-only H3 ConvRot mode. It pre-packs
  exactly two equal feature slices (7,168 each for the 14,336-wide H3 FFN),
  runs ConvRot INT8 fc1 and fused SwiGLU/fc2 per slice, and accumulates the
  partial hidden outputs without materializing the full fc1 expansion. Both
  weights must be non-transposed TensorWise-INT8 ConvRot-256 tensors with
  per-output-channel scales and no bias; unsupported layouts fail explicitly.
- `chunk_rows`: default 2048, selected from the real-weight MLP sweep.
- `prefer_held_weights`: enabled by default; acquires `fc1` and `fc2` once per
  block when distinct async cast buffers make that safe. A same-buffer case
  uses ordinary per-slab module calls instead of risking weight corruption.
- `strict`: raises on core drift or unsupported weight acquisition.

The ConvRot kernels expose fake-tensor contracts for `torch.compile`; weight
tile preparation and both fused linears remain explicit operations in the
compiled graph.

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
  --swiglu-modes bf16,native,tiled_convrot --held-modes off,on \
  --feature-tile-width 3584 --json h3-mlp.json
```

Actual-checkpoint mode lazily loads only the selected block's `fc1` and `fc2`
MLP tensors (including their quantization metadata); it is an MLP benchmark,
not a full model load or generation run. `tiled_convrot` is an opt-in baseline:
it requires BF16 compute, pre-packs feature-major INT8 ConvRot tiles before warmup/timing, bounds each
fc1 expansion by `chunk_rows × (2 * feature_tile_width)`, and reports error
against the raw current ConvRot path after peak capture. The
pre-packed tiles are inherently held, so tiled cases are emitted once per
chunk regardless of `--held-modes`; production should replace the original
layout or consume strided tiles rather than retain both copies.

Use `full` in `--chunks` to include the unchunked sequence. For the native
TensorWise-INT8 ConvRot path, `--profile-native-stages` records the exact fused
SwiGLU/activation-quantizer and CUTLASS INT8 GEMM/dequant CUDA calls separately.
The trace fails if Kitchen selects a different path instead of relabeling it:

```powershell
python custom_nodes/ComfyUI-H3-Extended/benchmarks/benchmark_h3_activation_memory.py `
  --checkpoint hf_minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --device cuda --dtype bf16 --seq 63448 `
  --chunks 256,512,768,1024,1536,2048,3072,4096,8192,full `
  --swiglu-modes native --held-modes on --profile-native-stages
```

To compare the existing fused ConvRot-INT8 down-projection against Transformer
Engine delayed-scaling FP8 using the same real `fc2` weight and BF16 reference:

```powershell
& .\custom_nodes\ComfyUI-H3-Extended\benchmarks\run_te_fp8_mlp.ps1 `
  --checkpoint /models/diffusion_models/hf_minimax_h3/minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --block-index 0 --rows 2048,8192 --recipes delayed_e4m3
```

The runner builds a pinned local benchmark image when needed. The comparison
fails unless the selected checkpoint weight is TensorWise-INT8 ConvRot; it does
not silently substitute a synthetic weight or another Comfy execution path.

Option C can be tested without changing the checkpoint weights to FP8. This
benchmark compares the full ConvRot MLP with a 256-aligned feature-tiled path
that consumes each bounded BF16 fc1 tile through fused SwiGLU+ConvRot fc2 and
accumulates the partial outputs:

```powershell
& .\custom_nodes\ComfyUI-H3-Extended\benchmarks\run_convrot_mlp_c.ps1 `
  --checkpoint D:\AI\ComfyUI\Models\diffusion_models\hf_minimax_h3\minimax_h3_fl2va_pruned_int8_convrot.safetensors `
  --block-index 0 --rows '2048,8192' --feature-tiles '7168,3584'
```

The local runner uses Comfy's `.venv\Scripts\python.exe` after the required
idle-GPU preflight; it does not require Docker or Transformer Engine. The
tiled benchmark pre-packs feature-major INT8 weight tiles so weight layout
copies are outside the activation peak and timed region. Production code must
replace the original layout or consume strided tiles rather than retain both
weight copies. A profiler run fails if it observes a full BF16 fc1 expansion
allocation in the tiled path.

See [`PLAN.md`](PLAN.md) for the staged attention-side follow-ups. Prequantized
QKV production and query-slab attention remain capability-gated on
`h3_attention`; they are not silently approximated here.
