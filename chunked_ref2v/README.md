# MiniMax H3 Ref2V experiment harness

**Status: implemented, not yet run on the GPU.** Every CPU-testable claim below
is covered by `tests/test_chunked_ref2v.py`; nothing here has been validated
against real weights.

Two planning documents sit alongside this code. `PLAN.md` is the arbitrary-length
production design. The harness plan is the experiment that decides *which* carry
mechanism that production node should use — and this package implements it.

This is not the production node. It generates Chunk A once and evaluates one or
more Chunk B strategies against it under controlled conditions.

## The question

At the 73/22/51 profile, Chunk B has to be told what Chunk A already generated
over their shared 22 frames. Seven ways to do that are plausible and the model's
response to all of them is unknown, because no shipped H3 workflow combines
keyframe and reference conditioning. So they get measured rather than argued
about.

| experiment | what Chunk B receives |
| --- | --- |
| `baseline_none` | nothing — the control |
| `frame_reencode_corrected` | one decoded frame, VAE round-tripped, at target position 0 |
| `frame_direct_corrected` | the same frame's latent, no round trip |
| `frame_direct_stock_position` | the same latent at the *uncorrected* pre-reference time |
| `frame_direct_prompted` | the same, plus keyframe-completion prompt text |
| `aligned_overlap_direct` | the full 7-position generated overlap |
| `aligned_overlap_stock_position` | the same, uncorrected placement |
| `aligned_overlap_prompted` | the same, plus continuation prompt text |
| `generated_overlap_video2` | the overlap as an ordinary `<Video 2>` reference |
| `composite_source` | one `<Video 1>` whose opening is the generated overlap |
| `target_overlap_clamped` | *(not implemented — see below)* |

Suites: `minimal`, `aligned`, `prompt`, `reference`, `clamp`, `all`, `custom`.

## Why the latent overlap is sliceable at all

H3's video latent positions cover unequal numbers of pixel frames, on the
repeating pattern `1, 4, 4, 4, 4`. At `T=22`:

```
positions  0-14  ->  51 pixel frames   (the stride)
positions 15-21  ->  22 pixel frames   (the overlap)
```

So `chunk_a_latent[:, :, 15:22]` *is* the generated overlap, exactly, with no
resampling. That is a property of the 73/22/51 profile, not of the model, so
`geometry.py` computes it and asserts it rather than hard-coding it. A profile
whose stride does not land on a latent boundary raises `UnalignedProfileError`
instead of silently picking the nearest position — the failure mode that would
otherwise produce a plausible-looking but meaningless result.

C=90/O=22 aligns too, at `20:27`. C=73/O=23 does not, and says so.

## The two things that had to be fixed in the model path

Neither is visible from the plan alone, and both are load-bearing.

**Core keeps only one set of condition latents.** `MiniMaxH3.extra_conds`
assigns `payload["cond_video_latents"]` from keyframes and then *overwrites* it
from references (`comfy/model_base.py:2094`, `2098`), so a run carrying both
keeps only the refs. The conditions have to be concatenated ahead of the refs, in
the row order `all_video_rows[~img_update] = cond_video_rows` consumes them.

**A keyframe lands at the wrong temporal address when references are present.**
`PackedLayout` pins a first-frame keyframe to `cond_t = float(text_len)`
(`comfy/ldm/minimax/model.py:319`), but with references it resets the cursor and
walks it past every reference block before laying down the target video grid
(`model.py:335-388`). The keyframe then shares a temporal origin with the *first
reference*, not with target frame 0.

`layout_ops.insert_target_conditions` sidesteps the arithmetic entirely: the
`copy_target` policy copies the exact `(t, h, w)` rows the target position
already has. The `stock` policy reproduces the uncorrected placement so the two
can be compared with everything else held identical — that is what
`frame_direct_stock_position` is for.

Both changes are applied through `add_object_patch`, never by forking
`comfy/ldm/minimax/`. `ModelPatcher.clone()` shares `self.model`
(`comfy/model_patcher.py:451`), so assigning `model.model.extra_conds = ...`
would escape the clone and leak into every other graph using that model.

### One more trap

`_forward` rebuilds the layout from scratch when
`layout.signature != (text_len, latent_t, lat_h, lat_w, audio_t)`
(`model.py:520`). A transformed layout that reported its own inflated row count
would be silently discarded on the first sampling step, and the run would look
like "the model ignored the condition". `TransformedLayout` carries the base
signature through unchanged, and the test asserts it.

## Phases

Model residency is the binding constraint on a 12 GB card, so the run is split so
each of the three stages — 14.6 GB text encoder, 19.5 GB DiT, 4.9 GB VAE — is
loaded at most once per run rather than once per experiment.

```
A  common VAE preprocessing    canvas pinned, both source chunks cut, static refs encoded
B  common Qwen preprocessing   the unmodified prompt, both chunk presentations
C  generate Chunk A            sampled once, ever; decoded; carry assets derived
D  dynamic carry preprocessing only what the selected strategies declared
E  Chunk B experiment runs     one payload per arm, same seed and sigma schedule
```

Phase D reads the *union* of the selected strategies' declared dependencies, so
five arms that all want Qwen cost one Qwen residency. A suite of nothing but
direct-latent arms skips Phase D entirely, which is the common case.

Arms run lower-memory-first, and each result is written before the next arm
starts, so an OOM at experiment five cannot destroy experiments one through four.

## The canvas is pinned, deliberately

Core's `adapt_canvas` sizes a reference video from *its own* dimensions
(`comfy_extras/nodes_minimax_h3.py:241-244`), so a 1080p source fed to a 0.8 MP
target gets a 1344×768 reference against an 800-row target. Here that is a
correctness problem, not just a 13% sequence-length tax: the whole harness rests
on Chunk A and Chunk B latents being sliceable against each other.
`ref_builder.encode_video_ref` encodes on the target's canvas.

## Chunk A is cached on its own identity

Chunk A is the expensive asset — a full sampling run plus a decode — and every
arm shares it. It is keyed on the things that actually determine it: source
frames used, prompt, static reference pixels, canvas, geometry, Chunk A seed,
sampler, sigmas, checkpoint. Chunk B experiment settings deliberately do *not*
enter that key, so adding a ninth arm six weeks from now reuses it.

Assets are safetensors written through `.tmp` + `os.replace`. A corrupt asset
returns `None` and is regenerated rather than raising — the harness can always
make it again, and a run that dies loading its own cache is strictly worse than
one that spends the time.

## Metrics are diagnostic, not a score

```
pixel_overlap_mae        Chunk A frames 51-72 vs Chunk B frames 0-21
latent_overlap_mae       Chunk A positions 15-21 vs Chunk B positions 0-6
motion_delta_mae         frame-to-frame deltas over the same window
motion_energy_ratio      output motion vs source motion
```

The last one is not decoration. **A strategy that freezes the frame scores a
perfect overlap MAE**, so the ranking prints motion next to it and says so. Same
reason a cancelled arm reports `null` rather than zeros — a VRAM-guard
cancellation is a resource result, not a model-quality result, and putting it on
the same axis as a model result is how a resource problem gets read as a
conclusion.

## What is not implemented

`target_overlap_clamped` refuses to run. Sigma-correct clamping needs the known
overlap re-noised to the current sigma with one fixed noise tensor at every step
— a sampler intervention, not condition rows. The obvious shortcut, inserting
clean latents during high-noise steps, produces a latent the model was never
trained to see, so it is refused rather than approximated. The dependency flag
(`needs_sampler_intervention`) and the `clamp` suite exist; the intervention does
not. It is only worth building if `aligned_overlap_direct` turns out to
under-constrain.

## Node

`MiniMax H3 Ref2V Experiment Harness (Zi)`, id
`MiniMaxH3Ref2VExperimentHarnessZi`, category `model/video/minimax/testing`.

Feed it a model that has already been through `MiniMaxH3SigmaShiftZi` — that node
carries the sigma shifts, the attention backend and the VRAM guard.

Outputs are `comparison_video`, `selected_preview`, `report` and
`artifact_path`; there is no output socket per experiment, because the
experiment count varies. Everything complete lands in
`<output>/h3_ref2v_harness/<run_id>/`.

Two comparison views, because they answer different questions. The overlap
comparison puts source / Chunk A / baseline / experiment in columns, which shows
whether the carried state arrived. The **boundary playback** — Chunk A running
into Chunk B as continuous playback — is what actually settles seam quality;
motion discontinuity is obvious in time and nearly invisible in a side-by-side
still.

## Tests

```bash
cd /path/to/ComfyUI
python custom_nodes/ComfyUI-H3-Extended/tests/test_chunked_ref2v.py
```

CPU only, no checkpoint, and **safe to run while a generation is in flight** —
the one other test in this repo with that property is `test_cond_cache.py`.
It forces `--cpu` before the first comfy import, because
`comfy.model_management` initializes a CUDA context at import time and on a
12 GB card that is not free.

It builds a real `PackedLayout` and checks the transform against it: that the
condition rows lead the image-row order, that reference and target rows shift,
that `img_pos`/`img_update` stay aligned, that audio indices shift, that the
original layout is not mutated, and that the patchified condition and reference
latents fill exactly the non-target image rows. That last one is the invariant
that does not raise when it is wrong — it silently pairs each condition with the
wrong rows.

## First GPU run

Keep the scope small:

```
suite:   minimal
profile: 73/22/51
prompt:  unchanged
```

Use a source edit whose effect is **unmistakable** — a strong recolour or an
obvious graphic change — rather than judging subtle identity preservation on the
first run.

If both single-frame arms are ignored, test `frame_direct_prompted` before
building anything else. If either works, run `aligned_overlap_direct` before
investing in the Qwen-visible or sampler-clamped alternatives.
