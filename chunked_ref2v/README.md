# MiniMax H3 Ref2V experiment harness

**Status: the `minimal` suite has run.** Every CPU-testable claim below is
covered by `tests/test_chunked_ref2v.py`; the machinery is validated against
real weights, at 0.2 MP — see [Stage 0 result](#stage-0-result-2026-08-05).
Nothing has yet run at a production canvas.

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

Feed it a model that has already been through `MiniMaxH3SigmaShiftZi` for the
sigma shifts and the VRAM guard. **Leave that node's `attention_backend` on
`comfy`** — attention is armed here instead, through the capability resolver.

## Attention is armed by the harness, once

The harness samples Chunk A plus one chunk per arm, so attention and activation
memory decide whether a suite finishes at all on a 12 GB card. Rather than
inheriting the sigma-shift node's name-based selector (`"sage"`/`"pytorch"`), the
harness calls `h3_memory_optimizer.resolve_attention` itself and installs an
architecture-matched prepared-QKV Sage backend.

Two properties matter more here than in an ordinary graph.

**Arm once, share everywhere.** Resolution and patch installation happen before
Phase A, so Chunk A and every Chunk B arm run the *same* backend and the
peak-VRAM and runtime columns stay comparable across arms. The 50 block forwards
are patched once, not once per experiment. `patch_target_conditions` clones from
the armed model and patches a different key (`extra_conds`), so the attention
patches ride through untouched.

**Say which backend produced the numbers.** A capability fallback is silent by
design — it preserves the incoming attention rather than failing a long
unattended run. But a resource figure attributed to the wrong backend is worse
than no figure, and it is exactly the number someone later quotes into a
headroom budget. So the resolved decision is recorded in `report["runtime"]`,
copied into every arm's `resources` next to its peak VRAM, and the text report
opens with an explicit warning when the run did *not* get optimized Sage:

```
WARNING: optimized attention was requested but fell back to the existing
backend (...). Arm-to-arm comparison is still valid; the absolute resource
figures below are NOT attributable to optimized Sage.
```

A deliberate A/B (`attention=existing`, `activation=off`) reads as a `NOTE:`
rather than a warning — that distinction is the difference between a choice and
a surprise.

Widgets: `attention` (default `auto`), `attention_fallback` (`allow` keeps the
run alive, `error` refuses to start), `activation` (default
`mlp_chunked_bf16`), plus the optional `cuda_async_soft_gc` /
`cuda_async_release_threshold_gib` pool policy, which is worth considering for
an unattended multi-hour suite under cudaMallocAsync.

A MODEL that already carries `MiniMaxH3MemoryOptimizerZi`'s status is
**inherited, not re-armed**: re-applying the same backend is a no-op and a
different one raises inside the installer, and neither is worth risking once a
run has started. The report records `armed_by` so an inherited configuration is
never mistaken for one the harness chose.

The attention selection is part of the Chunk A cache key. Sage quantizes Q/K to
INT8, so a Chunk A generated on one backend and reused against a Chunk B sampled
on another would put a backend difference into the overlap metrics and label it
a carry-strategy result.

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

## Stage 0 result (2026-08-05)

`minimal` suite, C=73/O=22/S=51, **608x352 (0.2 MP)**, seed 1, `res_multistep` /
`simple` / 20 steps, `efficient_sage_sm89` + `mlp_chunked_native`. Ten minutes
for five chunks. Artifacts in
`Output/h3_ref2v_harness/20260805_195850_ee6c74a8/`.

All ten implemented arms, as ratios against the control (lower is better):

```
experiment                       pixel   latent   motion-delta   1st frame
aligned_overlap_direct           0.396    0.276      0.442         0.229
aligned_overlap_prompted         0.397    0.279      0.442         0.228
frame_direct_prompted            0.386    0.424      0.596         0.270
frame_direct_corrected           0.387    0.428      0.601         0.268
frame_reencode_corrected         0.439    0.580      0.611         0.318
frame_direct_stock_position      0.830    0.863      0.986         0.838
baseline_none                    1.000    1.000      1.000         1.000
aligned_overlap_stock_position   1.004    1.073      1.106         0.993
generated_overlap_video2         1.071    1.040      0.743         1.083
composite_source                 1.112    1.244      1.022         1.127
```

**The full overlap is the production mechanism.** Pixel MAE ties with the single
frame, but the overlap wins where it matters: latent agreement 0.276 vs 0.428
and motion-delta 0.442 vs 0.601. That is PLAN.md 16's prediction confirmed - "the
carried frame fixes position, not velocity" - and motion-delta is exactly the
metric that separates them. One frame cannot encode a trajectory; seven
positions can.

**Hybrid keyframe + reference conditioning works.** PLAN.md listed "Ref2VA
checkpoint obeys a keyframe latent Qwen never saw" as the Stage-0 unknown. It
obeys, without being told.

**The MM-RoPE placement correction is the whole mechanism, not an improvement.**
`aligned_overlap_stock_position` scores 1.004 - indistinguishable from carrying
nothing at all, and slightly worse on latent and motion. Placed on the wrong
timeline, seven positions of condition are seven positions of noise. PLAN.md 1.1
called this blocking from a source reading alone; that call was right, and
without the fix the whole technique reads as "the checkpoint ignores keyframes."

**Prompt text adds nothing.** `frame_direct_prompted` vs `frame_direct_corrected`
is 0.386 vs 0.387; the overlap pair is 0.397 vs 0.396. Both differences are far
inside run-to-run noise. The `keyframe_completion` language is unnecessary - the
model responds to the condition rows whether or not Qwen was told they exist.

**Both Qwen-visible strategies fail.** `generated_overlap_video2` (1.071) and
`composite_source` (1.112) are *worse than the control*. Composite is worst on
latent agreement at 1.244, consistent with its known property that the original
source geometry is unavailable inside the overlap. See the caveat below before
treating these two as settled.

**The VAE round trip costs real fidelity.** Direct beats decode-and-re-encode on
every metric; first-position latent MAE 0.0448 vs 0.1086.

Motion-energy ratios span 0.978-1.038, so no arm bought its score by freezing.

### Caveat

0.2 MP is far outside the model's trained range. This validates the machinery
and the *relative ordering* of the arms; it says nothing about absolute quality,
and the run should be repeated at a production canvas before the numbers are
quoted as anything but a ranking.

**The two Qwen-visible arms carry an extra confound and should not be treated as
settled.** `generated_overlap_video2` and `composite_source` are the only
strategies whose mechanism depends on Qwen *reading* the carried footage, and
Qwen sees a reference video at 2 fps on a 608x352 canvas here. Their failure may
be "the model cannot use a visible overlap" or merely "there was nothing legible
to see at this size" - this run cannot distinguish those. Every other arm feeds
the DiT through condition rows, which are canvas-independent in mechanism, so
their ordering is on much firmer ground. Re-test these two at a production
canvas before concluding anything about them.

### Memory, measured

torch peak reserved **1536 MB**, peak allocated 1245-1266 MB, against a
driver-level total that sat at ~10.7-11.0 GB. The difference is AIMDO's
reclaimable weight-page cache filling otherwise-idle VRAM, plus a ~2.3 GB
desktop floor. The predicted transient at 116 KB/token was 1.27 GB against
1.25 GB measured, so the cost model holds at this canvas.

This run also exposed a real bug in `vram_guard.py`: it compared against raw
`mem_get_info`, which counts AIMDO's reclaimable pages as used, and cancelled a
healthy run at "free physical 341 MB" while torch held 320 MB. Core corrects for
this at `comfy/model_patcher.py:421-425`; the guard now does too.

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

## Long-form result (2026-08-06) — carry is required

First multi-boundary test. Three carry arms over the same window, C=90/O=22/S=68,
**7 chunks = 480 frames (20 s)**, 608x320 (0.2 MP), seed 1, `res_multistep` /
`simple` / **15 steps**, models loaded once for all three arms. 40.2 min total.
Artifacts in `Output/h3_longform/20260806_070129_3arm_c90/<arm>/output/final.mp4`.

| arm | carried into chunk i | wall | verdict |
|---|---|---|---|
| `direct_latent_overlap` | 7 latent positions (`20:27`) | 14.6 min | fully coherent |
| `direct_latent_frame` | 1 latent position (`20`) | 12.9 min | fully coherent |
| `none` | nothing | 12.7 min | **fails** — reads as unrelated clips spliced together |

**The source video reference alone does not hold a long-form edit together.**
Every chunk in `none` still received its own source-chunk reference and the same
static reference images, and the chunks still overlapped by 22 source frames —
`none` removes only the *generated* state. That is enough to destroy continuity
across six boundaries.

Two consequences:

- Carry is not an optimization, it is load-bearing. Chunked Ref2V without it is
  not a viable technique at any length beyond a single boundary.
- **Chunks cannot be generated in parallel.** `none` was the only arm whose
  chunk *i* did not depend on chunk *i-1*, so the parallel-generation idea dies
  with it.

This overturns the Stage 0 reading in which `baseline_none` scored best. That
run had **one** boundary and a known prompt-corruption; a single boundary does
not exercise the question. Treat the old `none` number as void.

`frame` and `overlap` are not yet separated — both look coherent, and `frame`
carries one seventh as much state for 1.7 min less. Separating them needs a
harder case than this clip, not a longer one.

### O=4 (2026-08-06) — the overlap tax was almost entirely waste

`C=90 O=4 S=86`, overlap latent `[26:27]`, 6 chunks, 480 frames, same clip /
prompt / refs / seed / 15 steps as above. **12.3 min** against 14.6 min for the
O=22 arm at identical output length. **Fully coherent.**

**O in frames is not O in latents.** H3 compresses 4 frames per latent token, so
a 4-frame overlap contains exactly *one* latent position. `direct_latent_overlap`
at O=4 therefore already carries everything the overlap holds - nothing is
discarded, and carrying "more" requires widening O itself:

| O frames | latent positions | asymptotic overhead `O/(C-O)` |
|---|---|---|
| 4 | 1 | **4.7%** |
| 5 | 2 | 5.9% |
| 9 | 3 | 11.1% |
| 13 | 4 | 16.9% |
| 22 | 7 | 32.4% |

Legal O values are quantized by the 17-frame grid: 4, 5, 9, 13, 17, 21, 22, 26...
O=4 is the floor; O=1 does not exist.

So **one latent position at the chunk boundary is sufficient**, and chunked Ref2V
costs ~5% over monolithic rather than ~32%. The frame-vs-overlap question at O=22
was the wrong axis: carry width does not change compute at all, only O does.

### Reference images are sized wrong for low canvases

`ref_image_size="match"` scales references to the *output canvas* pixel budget.
At 608x320 that encoded a 3000x1462 identity reference down to **640x320**, and
the two references together were 395 of 12479 sequence rows - 3.2%. Outfit and
tattoo likeness were visibly poor because the model could not resolve them.

A reference is conditioning: the resolution it needs depends on the detail it
must carry, not on the output size. Coupling the two is pathological at test
canvases. Cost of decoupling is small:

| short edge | jinx encoded | added rows | seq cost |
|---|---|---|---|
| match | 640x320 | - | - |
| 512 | 1056x512 | +421 | **+3%** |
| 768 | 1568x768 | +1429 | +11% |
| max (2048) | 3008x1472 | +5483 | +44% |

`max` clamps at native resolution and is poor value. A configurable short edge
defaulting near 512-768 is the fix; `ref_builder` currently exposes only the two
modes.

### Settled configuration (2026-08-06)

`C=90 O=4 S=86`, `carry=direct_latent_overlap`, `ref_image_size=native`. Fully
coherent over 6 chunks with good outfit and tattoo likeness. 14.0 min for 480
frames at 0.2 MP / 15 steps. These are now the node defaults.

`native` reference sizing reproduces what the production workflows already did by
hand: resize to the model's own canvas first, then `max` - which is inert
afterwards, because `2048 / 768` clamps to 1.0. The resize was doing all the
work, and `match` had been undoing it.

**Reference resolution has a second-order cost through the text encoder.** Going
`match` -> `native` added 1243 reference rows as predicted, but total sequence
went 12479 -> 15155 (+21%), because `text` grew 1524 -> 2767: Qwen emits more
tokens once it can actually resolve the images it is describing. Wall time went
12.3 -> 14.0 min (+14%). A rows-only estimate understates the true cost by about
half.
