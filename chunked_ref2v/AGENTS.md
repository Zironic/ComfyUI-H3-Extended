# Chunked Ref2V and long-form guidance

## Authoritative behavior

- Read `README.md`, `PLAN.md`, and the relevant `longform/` implementation
  before editing. Preserve the existing node ids, socket/widget order, plan
  precedence, artifact layout, and resumability behavior.
- Video defines the overlap. Resolve video history against legal H3 video
  geometry first; never enlarge or shrink it merely to fit the audio grid.
- Size video and audio reference histories independently and slice both from
  their ends when `temporal_alignment` is `end`. Ordinary reference spans remain
  start-aligned unless the N+1 path explicitly opts into end alignment.
- Derive conservative audio carry with floor arithmetic. The invariant is
  `audio_latents * fps <= video_frames * 40`, and the chosen count is maximal
  only when adding one audio latent would violate it. Persist
  `audio_carry_policy="video_floor_v1"` wherever the current contract requires
  it.
- Keep model-native packed latent handling in the model/layout owner. Do not
  move packing, overlap alignment, audio conditioning, muxing, or stitching
  into convenience nodes.
- Keep async TAEH3 preview work bounded and request-scoped: one latest-only
  worker, producer events and a dedicated CUDA stream, stale-result suppression,
  sampler-thread UI publication, and idempotent close. Reuse the existing worker
  bridge instead of adding a second preview pipeline; keep synchronous fallback
  behavior intact.

## Identity and resume safety

- Treat a connected plan as authoritative over legacy scalar sockets. Keep
  legacy inputs only where current workflow compatibility requires them.
- Resume reuse is causal. Validate schema/geometry, effective sampling and
  prompt identities, seeds, parent-chunk identity, and video/audio digests;
  reuse only the valid prefix and invalidate the suffix.
- Current version constants are owned by their modules. When changing a
  persisted semantic contract, update the relevant plan/manifest/chunk schema
  intentionally and add invalidation tests. Never edit version numbers only to
  make stale artifacts load.
- Use atomic writes and explicit run-directory ownership. Do not delete or
  overwrite an existing run to recover from an identity mismatch without user
  approval.

## Validation

- Run the smallest direct tests for the changed owner, such as
  `test_nplusone_resume.py`, `test_av_continuation.py`,
  `test_audio_grid_alignment.py`, or the relevant `test_longform_*.py` file.
- Tests that need directory APIs must use `tests/h3_test_tempfile.py`. Keep CPU
  argument parsing before the first Comfy import and do not claim mux, media,
  model, or GPU behavior from synthetic contract tests.
