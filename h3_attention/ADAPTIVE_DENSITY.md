# Adaptive video-attention density

`MiniMax H3 Hybrid Sparse Attention (Zi)` supports two pure target-video routing
policies at the existing global Sparse-Sage geometry (`128` query tokens by
`64` KV tokens):

- `fixed`: every pure-video head/query row retains the same quantized number of
  video KV tiles. This is the established path and remains the default.
- `adaptive_budget`: every row may retain a different number of video KV tiles,
  while the sum of all row counts is exactly the same as the corresponding
  fixed route.

Text, reference, audio, mixed-boundary KV tiles, and non-video query tiles stay
dense in both modes.

## Adaptive policy

For each attention head and pure-video 128-token query tile, the router:

1. scores the mean-pooled 64-token video KV summaries;
2. converts the scores to a coarse probability distribution using
   `adaptive_temperature`;
3. finds the number of top blocks required to reach `adaptive_target_mass`;
4. clamps that unconstrained request to `min_video_density` and
   `max_video_density`;
5. applies a fixed-iteration scalar shift so the continuous mean equals the
   fixed route's quantized target count;
6. uses largest-remainder integerization to make the final global count exact;
7. retains each row's own top-K block indices and writes its K to
   `valid_block_num`.

The integerization sorts one remainder per head/query row, rather than sorting
all optional KV blocks globally. The LUT remains a fixed-shape tensor; only its
contents and `valid_block_num` vary.

## Controls

- `video_budget`: fixed per-row density in `fixed`; target mean density in
  `adaptive_budget`.
- `min_video_density`: lower per-row rail for adaptive routing.
- `max_video_density`: upper per-row rail for adaptive routing.
- `adaptive_temperature`: temperature used when interpreting pooled QK scores.
- `adaptive_target_mass`: cumulative coarse video-attention mass used to estimate
  unconstrained row demand.

The realized mean density uses the same tile quantization as fixed routing:
`ceil(video_budget * pure_video_kv_tiles) / pure_video_kv_tiles`.

## Current execution boundary

Adaptive routing is enabled for the eager `sage128` and
`sage128_fused_qkv` paths. The shared Inductor block currently keeps its original
fixed-density router, so the node rejects `adaptive_budget + inductor` rather
than silently executing a different policy.

## Validation

Run the CPU contracts:

```bash
python tests/test_adaptive_hybrid_router.py
```

Run the production-router microbenchmark:

```bash
python benchmarks/bench_adaptive_router.py \
  --budget 0.20 --min-density 0.05 --max-density 0.50 \
  --temperature 1.0 --target-mass 0.80
```

The installed Sparse-Sage CUDA kernel still needs a dedicated numerical and
latency check with deliberately different `valid_block_num` values in the same
launch. Until that check is run, adaptive mode should remain experimental.
