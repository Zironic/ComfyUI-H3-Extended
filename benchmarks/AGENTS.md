# H3 benchmark guidance

- Benchmarking, model loading, CUDA compilation, and warmup require explicit
  user permission plus the root `nvidia-smi` preflight. Offline report analyzers
  may run CPU-only only after inspection confirms they do not import or
  initialize a GPU path.
- Verify that each label reaches its intended kernel. Backend selection,
  compilation success, or the absence of an exception is not proof when a
  lower layer can fall back.
- Keep compile warmup separate from steady-state samples. Record shapes,
  checkpoint/quantization layout, dtype, backend, tile count, warmup, iteration
  count, and peak-memory method so runs are comparable.
- Separate projection microbenchmarks, routed attention/MLP sections,
  sequential composites, per-block measurements, and full request or model
  evidence. Deferred stage timings overlap and must not be added together.
- `native` MLP is TensorWise-INT8, not FP8. Fused QKV is opt-in and approximate.
  State these contracts in comparisons and apply numerical/parity gates that
  match the path actually exercised.
- Write temporary outputs, profiler traces, Triton/Inductor caches, and local
  JSON/CSV results under `.agent\tmp` or another explicitly agent-owned ignored
  directory. Do not overwrite tracked evidence or add generated artifacts
  without an explicit request.
- A timeout, OOM, compile-only pass, or synthetic result is bounded evidence;
  report it as such and do not promote it to a production recommendation.
