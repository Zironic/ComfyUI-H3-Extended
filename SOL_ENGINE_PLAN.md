# Sol-Engine integration status and validation plan

This branch implements the non-conflicting substrate for testing Sol-Engine's
MiniMax-H3 ideas.  The dense memory optimizer remains the production reference.

## Implemented

- Shared request/step/packed-layout runtime context.
- Optional explicit Sol-Attn prepared backend.
- Released H3 sparse policy, exact prefix KV sink and dense prefix-query overwrite.
- Real-QKV all-exact correctness gate and route-density diagnostics.
- Run-scoped exact AdaLN trajectory tables without destructive weight removal.
- Signal-driven FirstBlockCache composed outside activation-memory block forwards.
- Architecture-safe preflight, BF16 SDPA warmup fallback, and existing-attention fallback.
- Standalone attention, table-geometry, cache-policy and output-comparison tools.

## Intentionally not combined

- Sol-Attn and prepared Sage cannot execute the same attention call.  Sol needs
  contiguous BF16 Q/K/V; prepared Sage consumes Q/K into INT8 and V into FP16/FP8.
- FirstBlockCache does not reuse EasyCache or masked-computation state.
- The dense `auto` mode never selects an approximate feature.

## Validation order

1. Run all CPU coordinator tests.
2. Run `bench_sol_attention.py` at C=39, 73, 90, 124 and 141.
3. Reject any shape that fails the real-QKV correctness gate.
4. Require Sol to reduce attention time by at least 10% or complete denoising by
   at least 5%, and record the physical-VRAM cost of contiguous BF16 Q/K/V.
5. Run `bench_adaln_precompute.py` on every checkpoint variant.  Enable the
   runtime provider only when the table is materially smaller than its projection
   weights and step-1 residency remains safe.
6. Sweep FirstBlockCache thresholds 0.02, 0.04, 0.06, 0.08 and 0.10.  Record
   computed/skipped tails and compare against the dense memory-optimizer output.
7. Compare video and generated audio separately.  PSNR and waveform MSE are
   regression gates only; manually review identity, motion, dialogue and sync.
8. Measure standalone and marginal gains.  Cache-skipped blocks never call
   Sol-Attn, so their advertised speedups must not be added arithmetically.

## Future work gated by measurements

- Accept strided H3 BTHD inputs in the Triton Sol backend to remove three full
  contiguous BF16 copies.
- Capture selected block-0 residuals directly for offline threshold sweeps.
- Add perceptual video, speaker-embedding and lip-sync metrics when their optional
  dependencies are available.
- Consider freeing full-width AdaLN weights only in an explicit serving-only mode;
  the Comfy model patch must remain reusable across schedules and workflows.
