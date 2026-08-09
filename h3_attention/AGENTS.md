# H3 attention guidance

- Preserve `sage128` as the established Hybrid Sparse default unless a request
  explicitly changes it. `sage128_fused_qkv` is opt-in and approximate: it
  changes projection representation and disables K smoothing.
- Preserve globally aligned 128-query and 64-KV routing geometry. Non-video
  context, mixed boundary tiles, and non-video query tiles remain dense. Do not
  turn logical selections into executable metrics without the evaluated
  hardware tile range.
- Selecting a backend must either execute that backend or fail clearly. Check
  runtime logs and numerical evidence for hidden lower-level fallbacks before
  accepting a benchmark label.
- Keep QKV preparation, routing, Sparse Sage, and statistics contracts
  separate. Reuse existing layout, router, sparse-kernel, and report helpers
  instead of creating parallel active-set or geometry implementations.
- CUDA timing uses request-scoped deferred events and synchronizes once at the
  request boundary. Stage events overlap; do not sum them as an end-to-end
  total. Shared compiled blocks intentionally omit per-stage events and retain
  `total_dit_block` around the invocation.
- Shared compilation requires the fused-QKV and two-slice ConvRot contracts.
  Keep modules, layer ids, AIMDO acquisition/release, collectors, and runtime
  options outside the tensor carrier tuple and Dynamo graph.
- Run focused CPU routing/configuration tests first. Real Sparse Sage numerical
  tests, checkpoint projection comparisons, and compile warmups are permission-
  gated GPU work and require the root preflight.
