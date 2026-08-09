# ComfyUI-H3-Extended agent guidance

## Scope and instruction discovery

- This directory is a nested Git repository owned separately from the parent
  ComfyUI checkout. Run Git commands here, or use
  `git -C custom_nodes\ComfyUI-H3-Extended ...` from the ComfyUI root.
- Codex builds its instruction chain from the active project root down to its
  working directory; it does not scan every descendant. Launch Codex with
  `codex --cd custom_nodes\ComfyUI-H3-Extended` when this repository is the
  target. More specific `AGENTS.md` files apply when the working directory is
  at or below their directory.
- Read `README.md` and the closest module README or plan before changing a
  subsystem. Plans record design intent and evidence, but current code and
  tests define implemented behavior.
- Preserve parent ComfyUI instructions. A nested Git root can become Codex's
  active project root, so do not assume `..\..\AGENTS.md` was loaded; read it
  when present before core-coupled work. This file does not authorize changes
  in core ComfyUI or another custom-node pack.

## Ownership and compatibility

- Keep changes in this repository unless the user explicitly expands scope.
  Ask before changing dependencies, installing or updating packages, or
  modifying the parent ComfyUI checkout.
- Preserve registered node ids, category names, socket order, widget order,
  defaults, return shapes, and serialized workflow compatibility unless the
  requested change explicitly replaces that contract.
- Apply model behavior through ComfyUI's model patcher and transformer options;
  nodes must not mutate model internals directly. Keep model/runtime state
  request-scoped and avoid persistent tensor caches.
- Verify the active installed copy, current ComfyUI APIs, and live node schema
  when behavior depends on runtime registration. Do not infer live behavior
  from constants, README text, or a serialized workflow alone.
- Code lives in this checkout on `C:`. Runtime inputs, models, logs, and outputs
  may live under `D:\AI\ComfyUI`; discover current paths instead of hardcoding
  machine snapshots into code.

## GPU, model, and network safety

- Do not run GPU, model loading, inference, benchmark, compilation warmup, or
  other VRAM-allocating work without explicit user permission for that run.
- Before an authorized compute run, use `nvidia-smi` to verify the device and
  current activity. Allocated VRAM alone does not prove the server is busy or
  idle. Report the exact command, model/shape, and evidence boundary.
- Do not download models or dependencies, access remote services, or start,
  stop, interrupt, or unload the Comfy server without explicit authorization.
- A CPU test, stub, synthetic tensor, compile-only check, timeout, or successful
  import is not evidence that a CUDA kernel or live workflow passed.

## Worktree and artifacts

- Inspect `git status --short` and `git status -sb` in this nested repository
  before and after work. Preserve unrelated staged, unstaged, and untracked
  changes; never clean or reset them to make the tree look tidy.
- `.agent/tmp/` is ignored scratch space. Other `.agent/` files include both
  tracked evidence and generated local results, so check `git ls-files` before
  assuming ownership. Do not add new benchmark results, profiles, compiler
  caches, or generated media unless the user explicitly requests them.
- Keep scratch files under `.agent\tmp`, not tracked source paths. Do not use
  Python `tempfile` for cross-process test directories on Windows; follow
  `tests/AGENTS.md`.
- Do not commit, push, rewrite history, or change remotes unless explicitly
  asked. When publishing, operate only on this repository, exclude generated
  local artifacts, push the configured branch, and verify local and remote
  state afterward.

## Implementation contracts

- Prefer the existing ComfyUI/Comfy Kitchen optimized operation and preserve
  dtype, device, layout, offload, patch, and output contracts. Fail clearly
  when a selected backend or model format is unavailable; silent fallback makes
  performance and quality evidence invalid.
- Keep attention backend selection opaque to callers. Do not branch on function
  identity, name, or module. Preserve non-video context densely in sparse
  routing and keep requested, evaluated, and execution geometry distinct.
- `native` MLP means Comfy's TensorWise-INT8 path, not FP8. The two-slice
  ConvRot path has stricter BF16, layout, group-size, orientation, and bias
  requirements; do not relabel one path as another in code or measurements.
- Shared H3 compilation must keep layer identity, AIMDO resource lifetime,
  runtime metadata, timing, and statistics eager while compiling only the
  tensor body. `compile_backend=inductor` must not be combined with stock
  `TorchCompileModel`.
- Deferred CUDA stage timings overlap and are not additive. Treat
  `total_dit_block`, individual stages, sequential composites, and end-to-end
  request time as different measurements.
- For long-form and continuation behavior, follow
  `chunked_ref2v/AGENTS.md`; video defines overlap and audio history is derived
  independently.

## Validation

- Prefer the narrowest direct CPU-safe test from the ComfyUI root with its
  managed interpreter, for example:

  ```powershell
  .venv\Scripts\python.exe custom_nodes\ComfyUI-H3-Extended\tests\test_hybrid_router.py
  ```

- Inspect the test before running it. Some scripts conditionally execute real
  CUDA kernels or construct CUDA tensors even when most assertions are mocked.
  `CUDA_VISIBLE_DEVICES=-1`, not an empty string, is the reliable Windows mask.
- Direct scripts importing ComfyUI need the ComfyUI root and this pack on
  `sys.path`; N+1 scripts may also need `chunked_ref2v` and
  `chunked_ref2v\longform`. Keep that setup in the test rather than relying on
  an agent-specific shell state.
- After changes, run `git diff --check`, the focused test, and any relevant
  syntax check such as `python -m py_compile` or `node --check`. Report skipped
  GPU/live checks as unverified, not passed.
