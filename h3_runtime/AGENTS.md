# H3 runtime and compilation guidance

- Prove the smallest real Torch/Comfy/AIMDO contract before introducing a
  registry, patch slot, prefetch scheduler, generalized lease, or other runtime
  abstraction. Diagnose allocator lifecycle before attributing a failure to
  Dynamo or Inductor.
- Keep H3 shared compilation tensor-only. `BlockTopology` owns static shape and
  layout facts; `BlockCarriers` owns the stable tensor order. Do not put modules,
  layer ids, registry tokens, runtime options, collectors, timings, or resource
  handles in the compiled carrier contract.
- Keep AIMDO acquisition/release, layer identity, outer block iteration,
  runtime metadata, timing, and statistics eager. Restore allocator watermarks
  and release resources through their owning lifecycle even when compilation or
  a custom kernel raises.
- Shared compilation requires the established fused-QKV, Sparse Sage, and
  two-slice ConvRot contracts. Do not weaken topology/layout validation or add a
  silent eager fallback that makes a compiled benchmark label ambiguous.
- Do not combine H3 `compile_backend=inductor` with `TorchCompileModel`.
  Compilation warmup and cache behavior are benchmark evidence, not functional
  correctness; validate tensor parity and the eager ownership path separately.
- Run focused CPU signature/dispatch/compatibility tests first. A compiled CUDA
  block, allocator probe, or full model path requires explicit GPU permission
  and the repository-root preflight.
