# ComfyUI-H3-Extended

Private fork of ComfyUI's built-in MiniMax H3 nodes (`comfy_extras/nodes_minimax_h3.py`),
so changes survive ComfyUI updates — plus an attention probe used to design a
block-sparse attention mask for H3 from measurement rather than guesswork.

Forked at ComfyUI v0.30.1, including the local `raw_latent_t` addition on
`MiniMaxH3ImageToVideo`.

## Nodes

Node ids and display names carry a `Zi` suffix so both these and the stock nodes
can be loaded at the same time:

| node id | display name |
| --- | --- |
| `EmptyMiniMaxH3LatentAVZi` | Empty MiniMax H3 AV Latent (Zi) |
| `MiniMaxH3ImageToVideoZi` | MiniMax H3 Image to Video (Zi) |
| `MiniMaxH3ReferenceToVideoZi` | MiniMax H3 Reference to Video (Zi) |
| `MiniMaxH3SigmaShiftZi` | MiniMax H3 Sigma Shift (Zi) |
| `MiniMaxH3AttentionProbeZi` | MiniMax H3 Attention Probe (Zi) |
| `MiniMaxH3Ref2VExperimentHarnessZi` | MiniMax H3 Ref2V Experiment Harness (Zi) |
| `MiniMaxH3MaskedRef2VCacheZi` | MiniMax H3 Masked Ref2V Cache (Zi) |

Existing workflows still point at the stock ids; re-add the `(Zi)` nodes to use
this copy.

---

# Chunked ref2v

[`chunked_ref2v/`](chunked_ref2v/) holds the arbitrary-length Ref2V work: the
production design in [`PLAN.md`](chunked_ref2v/PLAN.md), and the two-chunk
experiment harness that decides which carry mechanism that production node should
use — implemented, CPU-tested, **not yet run on the GPU**. See
[`chunked_ref2v/README.md`](chunked_ref2v/README.md).

The harness is the first thing in this repo that needs *model-side* behaviour
changed. Core keeps only one set of condition latents when keyframes and
references are both present, and places a keyframe at the wrong temporal address
when references are present; both are corrected through `add_object_patch`,
so the "not forked here" boundary below still holds and the change reverses on
unpatch.

---

# Conditioning cache

Both conditioning nodes carry a `cond_cache` widget (`auto` / `off` / `refresh`,
default `auto`) that reuses the Qwen3-VL-32B pass across runs.

The encoder's output depends on the token stream and the TE weights, and on
nothing else — not width, height, length, sampler settings or seed. But
ComfyUI's execution cache is keyed on the *whole node*, so nudging `length` by
one frame re-runs a 14.6 GB text encoder over pixels that did not change, and a
server restart discards the result entirely.

So the key is a blake2b digest of what the encoder actually consumes, taken
after tokenization: every text token id and weight, plus the shape, dtype and
full contents of every vision block, plus a fingerprint of the text encoder
itself. Entries are safetensors files in `<user-directory>/h3_cond_cache/`.

A hit means the encoder is never staged onto the GPU at all — which on a 12 GB
card is most of the point, not just a time saving.

| mode | behaviour |
| --- | --- |
| `auto` | read and write |
| `off` | bypass entirely, neither read nor write |
| `refresh` | ignore any stored entry, re-encode, overwrite |

## Identifying the text encoder without hashing 14.6 GB

Core's loaders record how to rebuild a patcher in `cached_patcher_init`, which
for a CLIP is `(load_clip_model_patcher, (ckpt_paths, embedding_directory,
clip_type, model_options))` and survives `clone()`. The fingerprint is the
checkpoint files' basename/size/mtime plus `clip_type` and `model_options`, so
swapping the TE file or its dtype/quantization misses, and `patches_uuid` —
which is regenerated per process and would miss on every restart — is not used.

## Keeping it bounded

`sweep()` runs once on first use and again after every store attempt — after
failures too, since a failed store is exactly when debris is left behind. It
applies three limits in order:

| limit | default | env |
| --- | --- | --- |
| orphaned temp files | older than 1 hour | — |
| unused entries | 30 days since last **use** | `H3_COND_CACHE_MAX_AGE_DAYS` (0 disables) |
| total size | 20 GB, oldest-used first | `H3_COND_CACHE_GB` |

A store writes `<digest>.safetensors.<pid>.tmp` and then `os.replace`s it into
position. A process killed in between leaves an orphan nothing would ever read
again — and on this box that is not hypothetical, since an OOM cascade takes the
prompt worker with it. Those are collected by age; a failed store also deletes
its own temp file immediately rather than waiting.

Age is measured from last use, not creation, because a hit refreshes the entry's
mtime — which is also what makes the size eviction a real LRU. An entry you use
weekly never expires.

`cond_cache.purge()` clears everything by hand and returns
`(files_removed, bytes_freed)`.

## Owning the folder, not vetting the files

The sweep deletes things, so the question "is this mine?" has to be answered
before it runs — and the answer is a property of the *folder*, decided once,
rather than something re-derived per file at delete time.

On first use the cache claims its directory by writing a `.h3_cond_cache`
marker. It will claim a folder it created, one that is empty, or one already
carrying the marker; a folder holding only cache-shaped files is adopted too,
which covers upgrades. **Anything else is refused outright** — the cache logs an
error and disables itself for the session rather than operating in a directory
it cannot account for. Refusing to use a folder is a far better failure than
sweeping one.

That matters because `H3_COND_CACHE_DIR` can point anywhere, and the obvious
mistake points it at a models folder. The test aims it at a directory holding a
fake `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`, sets the cap to zero and the
age limit to a day, then calls both `sweep()` and `purge()` and asserts the file
is still there and no marker was dropped.

Inside an owned folder, a file matching `32 hex digits + .safetensors` is one of
ours and gets deleted on the normal age/size rules — corrupt ones included.
Checking each file's metadata instead would strand exactly the corrupt entries
that most need collecting, since those are the ones that no longer parse.

`purge()` leaves the marker in place so the folder stays claimed.

## What it refuses to do

A wrong hit is much worse than a miss, so anything the key cannot see falls
through to a plain encode: a LoRA-patched text encoder or an active hook
schedule (both change the output without changing the tokens), unknown
checkpoint provenance, an unrecognised token or conditioning payload, and any
read error. Store failures are caught and logged — the conditioning is returned
either way.

Log lines are prefixed `[H3 Extended] cond cache` and name the digest, so a hit
and the store that produced it can be matched up in
`D:\AI\ComfyUI\User\comfyui.log`.

## Two traps worth recording

**Slice views.** The tokenizer emits video as `frames[i:i+2]` pairs, which are
already contiguous, so `.contiguous()` is a no-op and the underlying storage
still spans the whole video. Hashing at storage level would read identical bytes
for every pair in a clip and collide them; the hash goes through numpy's view,
which carries the correct shape and strides.

**safetensors mmap on Windows.** Loading returns mmap-backed tensors, and a
mapped file cannot be deleted or replaced while those views are alive. Holding
them for the length of a sampling run would block both LRU eviction and a
`refresh` of that same entry, so a hit copies out and drops the mapping.

Env overrides, for when the widget is not reachable: `H3_COND_CACHE_DISABLE=1`,
`H3_COND_CACHE_DIR=<path>`, `H3_COND_CACHE_GB=<float>`,
`H3_COND_CACHE_MAX_AGE_DAYS=<float>`.

---

# Attention backend

`MiniMaxH3SigmaShiftZi` also selects the attention backend for the H3 DiT, via
an `attention_backend` widget:

| option | meaning |
| --- | --- |
| `sage` (default) | SageAttention, error if unavailable |
| `comfy` | whatever the global default is |
| `pytorch` | pinned dense baseline |

Note that `comfy` is **not** a reliable baseline: launching with
`--use-sage-attention` sets `optimized_attention = attention_sage` globally, so
`comfy` then means Sage too and an A/B against it compares Sage with Sage. Use
`pytorch` for a baseline that holds regardless of launch flags.

That node already clones the MODEL and writes into
`model_options["transformer_options"]`, which makes it the right seam: core's
`wrap_attn` consults `optimized_attention_override` in the transformer_options
it is handed, so the override reaches only models whose forward pass carries
these options — the H3 DiT — and leaves every other model on the global default.
No core modification, no second attention implementation to maintain.

The override calls the backend's undecorated `__wrapped__` function, so it does
not re-enter itself. Core's `wrap_attn` also injects an `_inside_attn_wrapper`
guard that rides along into the impl, so Sage's own internal fallback to
PyTorch does not re-trigger the override either.

## Measured

Synthetic benchmark at H3's real attention shape (56 heads x 128 dim,
`skip_reshape=True`), RTX 4070, torch 2.10.0+cu130:

| packed seq | comfy | sage | speedup | rel. error |
| --- | --- | --- | --- | --- |
| 9,394 (22f, as probed) | 44.5 ms | 25.0 ms | **1.78x** | 0.0105 |
| 13,966 (22f hi-res) | 98.1 ms | 54.5 ms | **1.80x** | 0.0106 |
| 37,000 (~5s / 124f) | 700.9 ms | 386.0 ms | **1.82x** | 0.0105 |

Times are one attention call; a forward pass runs 50 of them. The ~1% relative
error is expected — SageAttention quantizes Q/K to INT8.

## Installing SageAttention

`sageattention` is deliberately **not** in a `requirements.txt`. The suitable
build depends on Python, Torch, CUDA and GPU architecture, and a wrong one is
worse than none.

Install a build compatible with the Python/Torch/CUDA environment running
ComfyUI, then restart ComfyUI. On Windows, PyPI's `sageattention` (1.0.6) is
Triton-based and needs a matching `triton-windows`:

```bash
pip install triton-windows==3.6.0.post26   # for torch 2.10
pip install --no-deps sageattention==1.0.6
```

`--no-deps` keeps pip from touching your torch install. Verify with
`tests/test_attention_backend.py`, which asserts Sage is actually selected and
did *not* silently fall back.

Selecting `sage` when it is unavailable **raises** rather than falling back
silently — a silent fallback makes benchmarks untrustworthy. Note that core's
`attention_sage` has its own *runtime* fallback if the kernel itself throws, so
benchmark logs should still be checked for:

```text
Error running sage attention ... using pytorch attention instead
```

The attention probe composes with this: it records Q/K and then delegates to the
original attention function with the same `transformer_options`, so the
delegated call goes through the selected backend. Both are covered by tests.

---

# VRAM guard

Armed from either of two places, whichever is in the graph:

- `MiniMaxH3SigmaShiftZi`, via its `vram_guard_mb` widget — patches the model, so
  it covers any sampler;
- `SamplerCustomAdvancedMiniMaxPreview` in `comfyui-minimax-preview`, via the same
  widget — covers the whole sampling run including the preview decode.

Default `800` MB in both, `0` disables. **The model-patch route only arms if the
`(Zi)` shift node is actually in the workflow** — a graph on the stock
`MiniMaxH3SigmaShift` gets no guard, and the giveaway is that none of this
extension's `[H3 Extended]` log lines appear at all. The sampler route exists
because that node is in every H3 workflow here regardless.

Both can be armed at once: the second install is skipped rather than stacking a
duplicate check.

12 GB is tight for H3 — the stages hold ~20 GB of weights with dynamic VRAM
loading — and an OOM raised from inside the DiT forward tends to cascade through
`model_management`'s recovery path and take the `prompt_worker` thread with it,
which needs a full server restart rather than a re-queue.

So rather than waiting for the driver to refuse an allocation, the guard checks
free **physical** VRAM at each point where the next allocation is about to be a
big one — before every DiT forward, and, when armed from the preview sampler,
either side of the preview decode (that decoder is a resident 2.26 GiB fp8 model,
invisible to the pre-forward check) plus once before sampling starts:

- above the threshold: nothing happens (one cheap `torch.cuda.mem_get_info` call);
- below it: logs the memory picture at `WARNING`, releases cached blocks
  (`soft_empty_cache(force=True)`) and re-measures;
- still below after that: logs the full picture at `ERROR` and raises
  `InterruptProcessingException` — the same exception the Cancel button raises, so
  the executor unwinds cleanly, the prompt is marked cancelled, and the worker
  survives.

The check runs *before* the forward, so the allocation that would have OOM'd
never happens.

## What the cancel log says

By the time the guard fires, the numbers worth knowing — requested frame length,
the source resolution of every reference image and video, the canvas each was
resized to — have already been consumed and discarded by the conditioning nodes.
So those nodes deposit their inputs in `run_context.py` as they run, and the
cancel log prints them next to the memory picture:

```text
[H3 Extended] VRAM guard cancelling run at DiT forward (sigma 0.7413)
  memory: free physical 612 MB / 12282 MB total; torch reserved 9942 MB, allocated 8801 MB (threshold 800 MB, still under it after releasing cached blocks)
  sampling: video latent [2, 24, 12, 48, 84], audio latent [2, 32, 2, 207], cond_or_uncond [1, 0]
  packed tokens: seq_len=13834 text=300 ref_img=1024 audio=414(t=207) video=12096(t=12,24x42)
  node inputs for this run:
    MiniMax H3 Reference to Video (Zi) [node 14]:
      canvas: 1344x768
      length: 124 requested -> 124 frames (5.17s at 24 fps)
      ref_image_size: max
      ref_image_1: 3024x4032 source -> 1536x2048 encoded (latent 96x128)
      ref_video_1: 1920x1080 x240 frames source -> 1344x768 canvas x226 frames used (latent t=67, 84x48)
      video latent: [1, 24, 12, 48, 84]
      audio latent: [1, 32, 2, 207]
  Cancelling now rather than letting the next allocation raise a CUDA OOM, ...
```

Two different provenances, deliberately: the `sampling` and `packed tokens` lines
are measured from the live forward pass (the latter resolved through the probe's
`resolve_layout` against core's real `PackedLayout`, since `seq_len` is what
actually drives attention memory), while the node inputs are remembered. Records
are keyed by graph node id and overwritten on re-execution, so a workflow that
runs H3 twice reports the second run's numbers. Each record also stores the
latent it produced; a record whose latent does not match the tensor being
denoised — ignoring batch, which the sampler doubles for cond+uncond — is printed
flagged `[stale: ...]` rather than passed off as current.

Building the description is lazy (it only runs on the cancel path) and every part
of it is best-effort: an unresolvable layout is skipped, an unreadable input
prints as `unreadable`, and a description that throws is logged and stepped over.
The cancellation itself always happens.

This deliberately does not use `comfy.model_management.get_free_memory`, which
adds torch's reserved-but-unused bytes back into the total. The question the guard
asks is what the driver can still hand out — and under `cudaMallocAsync` a full
pool is exactly the condition worth trimming, which is what the release-and-recheck
second chance is for. Log lines report both readings (`free physical` plus torch
`reserved`/`allocated`) so the two can be compared after the fact.

Log lines are prefixed `[H3 Extended] VRAM guard` and land in
`D:\AI\ComfyUI\User\comfyui.log`.

The guard covers sampling and the preview decode. The final VAE decode and the
text encoder run in other nodes and are not wrapped.

The preview sampler finds this module by matching `__file__` across `sys.modules`,
rather than importing it by path — a path import would build a *second*
`run_context`, and the conditioning nodes' recorded inputs would never reach the
cancel log. It resolves lazily at sample time, so custom-node load order does not
matter, and warns once and samples unguarded if this extension is missing.

It matches on the file rather than the module name because directory custom nodes
are **not** registered under their folder name. `nodes.load_custom_node` uses the
*full path* as the module name (`sys_module_name = module_path.replace(".",
"_x_")`), so the real key here is
`c:/...\custom_nodes\ComfyUI-H3-Extended.vram_guard`. The directory itself comes
from `nodes.LOADED_MODULE_DIRS`, so an install under a different custom-node root
still resolves.

---

# The attention probe

**Status: stage 1 of the sparse-attention work.** The probe measures what a mask
*could* drop. It does not implement a mask, and no sparse kernel exists yet.

## The question it answers

> For each H3 query block, which text, reference, audio, spatial and temporal KV
> blocks can be omitted while preserving nearly all dense-attention output?

Answering this first is the point. Committing to an invented pattern and then
tuning it is the expensive way to arrive at the same place.

## Design

### 1. Token-layout metadata

Core builds `PackedLayout` per sampling run, but exposes only a flat
`(start, stop, kind)` segment table. Without named ranges a captured Q/K tensor
is an undifferentiated token sequence.

[`h3_probe/layout.py`](h3_probe/layout.py) turns that table into:

```python
{
    "text_range": (start, end),
    "reference_ranges": [(kind, start, end), ...],   # cond / ref_img / ref_audio
    "audio_range": (start, end),
    "video_range": (start, end),
    "video_shape": (latent_t, patch_h, patch_w),
}
```

It is published on `transformer_options` under `minimax_h3_token_layout`
(the object) and `minimax_h3_token_ranges` (the plain dict). Core reads neither
key, so publishing it cannot alter inference.

The DiT patches video 1x2x2, so a target frame occupies
`(latent_h // 2) * (latent_w // 2)` rows and the video segment is exactly
`latent_t` such frames back to back. Target audio is `2 * audio_t` rows,
channel-major.

### 2. Selective instrumentation

[`h3_probe/capture.py`](h3_probe/capture.py) swaps the module-global
`optimized_attention` name inside `comfy.ldm.minimax.model`. That binding is
H3-only, so nothing else in ComfyUI is touched, and with no probe armed the
replacement is a bit-identical delegation.

Two things are resolved without patching any core class:

* **which layer** — the token refiner shares the attention path but runs on the
  text span alone, so a call is a DiT layer exactly when
  `q.shape[2] == layout.seq_len`; layers are then counted in order.
* **which step** — `transformer_options["sigmas"]` located in
  `transformer_options["sample_sigmas"]`.

A `ProbeSession` brackets each sampling run via an `OUTER_SAMPLE` wrapper, so
re-queues never accumulate into one trace.

**Full `N x N` attention is never materialized.** For a selected query block the
probe computes exact dense attention against all keys one head chunk at a time,
reduces each softmax row set to its mean distribution, and keeps only:

| aggregate | shape | granularity |
| --- | --- | --- |
| `block_mass` | `[heads, n_blocks]` | per 128-token KV block |
| `cat_mass` | `[heads, 6]` | per segment kind, exact |
| `frame_mass` | `[heads, latent_t]` | per target latent frame, exact |
| `spatial_mass` | `[frame_rows]` | per spatial patch, summed over frames |

Selection defaults to 3 layers (early/middle/late) x 3 steps
(early/middle/late) x a handful of query blocks — 4 latent-frame positions, 2
spatial positions each, plus one target-audio query block.

### 3. Decision-oriented metrics

[`h3_probe/metrics.py`](h3_probe/metrics.py) reports, per query block: mass to
text / references / audio / target video; mass by temporal distance; same
spatial region versus elsewhere; and coverage of the candidate fixed mask
(mandatory context + own frame + `+/-1` frame) before and after adding Top-k
dynamically selected distant blocks.

The report looks like this — the numbers below are *illustrative placeholders*,
not measurements; no probe run against real H3 weights has been done yet:

```text
Layer 24, step 7, video query block t=8 (rows 318-350)
  text/reference mandatory context:  11.2%
    text:                             4.1%
    references/keyframes:             5.9%
    target audio:                     1.2%
  current frame:                     38.4%
  adjacent frames:                   31.7%
  other frames:                      18.7%

  by temporal distance:             0:38.4%  +/-1:31.7%  +/-2:9.1%  +/-3..5:6.8%  > +/-5:2.8%
  same spatial region (r=4):         46.8%  (elsewhere  41.9%)

  local mask retained:               81.3%   (exact tokens)
  local blocks retained:             82.6%   (14/25 blocks)
  local + top-4  distant:            94.1%
  local + top-8  distant:            96.8%
  local + top-16 distant:            99.1%
  local + top-32 distant:            99.6%
```

Two granularities are reported deliberately. *Exact* masses come from segment
and frame slices and say what the model genuinely attends to; *block* masses
come from the KV block grid a real kernel would work on, where a block
straddling the mask boundary is retained whole. The block figures are what a
kernel can actually deliver.

The summary block reports **worst case across every probed query block**, not
the mean. A mask is only safe if its worst query block retains enough mass.

## Usage

Insert the probe between the model loader and the sampler:

```
Load Checkpoint -> MiniMax H3 Sigma Shift (Zi) -> MiniMax H3 Attention Probe (Zi) -> KSampler
```

Output lands in `output/h3_probe/<run_tag>_<timestamp>/`:

| file | contents |
| --- | --- |
| `report.txt` | the human-readable report above |
| `summary.json` | the same metrics, machine-readable |
| `trace.npz` | raw per-head aggregates for offline analysis |

Per-head arrays are kept in `trace.npz` even though the report averages over
heads — whether some heads are globally attentive while others are strictly
local is a question the mask design will need answered, and re-running the probe
to get it back is expensive.

Cost: only probed steps are slowed. At 1344x768 / 124 frames the packed sequence
is roughly 38k tokens; transient probe memory stays near 80 MB with the default
head chunk of 4.

## Initial test matrix

Fixed seeds, identical settings, one `run_tag` per entry. The point is not model
evaluation — it is whether the attention structure is stable enough to
generalize.

| tag | clip |
| --- | --- |
| `t22` | 22-frame normal short video |
| `t39` | 39-frame two-block video |
| `long` | one longer clip at reduced resolution, if memory allows |
| `rot` | camera rotation — continuous motion |
| `transform` | the finite clothing/bag transformation |
| `translate` | substantial subject translation |

## What follows

Once the probe establishes a candidate pattern:

1. Implement a **dense masked emulator** — the proposed block mask applied to an
   ordinary correctness-oriented attention calculation.
2. Compare against the dense baseline at identical seeds.
3. Adjust mandatory context and Top-k budget.
4. **Only then** implement a Triton/CUDA block-sparse backend.

The instrumentation stays explicitly H3-focused and disposable. There is no
reason to build a generic Comfy attention profiler before the first measurements
say what H3 needs.

---

# Masked Ref2V computation

`h3_masked_cache/` — `H3 Masked Computation Plan.md` is the full design. The idea
is that in a Ref2V *edit*, most of the target video is supposed to come out the
same as the source video, and the tokens carrying those unchanged regions could
be dropped from the 50 DiT blocks entirely — the whole source stream is still
present as reference rows — and then clamped back to the source with an exact
forced velocity.

**Only Stage 0, measurement, is implemented.** The node observes and reports; it
does not change what the model computes. `blocks.py` and `plan.py` from the
plan's structure do not exist yet, and selecting `fixed` or `dynamic` raises
rather than quietly measuring under a name that promises otherwise.

## The question Stage 0 answers

> Does an early predicted-clean source-difference map reliably identify the
> region that ends up edited?

Everything downstream rests on that. If the edited region is not identifiable
early, is not materially smaller than the whole target, or grows unpredictably
late in the schedule, then no amount of kernel work makes token pruning worth
having, and it is much cheaper to find that out from a report than from a
half-finished sparse attention path.

## What it measures

Per model call, on the conditional branch only:

1. `x0 = model_sampling.calculate_denoised(sigma, model_output, x)` — using the
   sampling object as configured *at this point in the graph*, so an upstream
   sigma-shift node is honoured rather than approximated by a re-derived flow
   formula.
2. A per-latent-cell relative difference against the source reference:
   `rms_channels(x0 - source) / (rms_channels(source) + floor)`.
3. Max-pooled over each `1x2x2` DiT patch, giving exactly one score per
   target-video sequence row.
4. Thresholded, quantized to token tiles, dilated by a spatial and a temporal
   halo.

Every reduction is a `max`, never a mean. The mask decides what may be *dropped*,
so one changed cell has to keep its whole patch; averaging would let a small
bright edit disappear into a large unchanged patch.

## Output

`output/h3_masked_cache/<run_tag>_<timestamp>/`:

| file | contents |
| --- | --- |
| `report.txt` | the readable version of everything below |
| `summary.json` | config, layout, resolved source, per-step rows, aggregates |
| `steps.jsonl` | one line per observed forward |
| `mask.npz` | token score maps (fp16), per-step masks, the run's union mask |

Rewritten in full after every observed forward, so a cancelled or OOM-killed run
still leaves what it had. No pickles — the score maps are the evidence a
threshold gets chosen from and have to outlive this code.

The three numbers the gate turns on:

* **active fraction** after tiles and halo — how much there is to gain;
* **J(prev)** — Jaccard between consecutive steps' masks, i.e. stability;
* **escaped(union)** — the share of a step's active tokens that no earlier step
  covered. This is the direct measure of what freezing an early mask would miss,
  and it is the one that decides whether a warm-up of two steps is enough.

A threshold sweep is reported at every step and averaged over the run, at both
ends of the chain: a threshold that looks selective at token resolution can be
worthless once a 4×4 tile and a halo have been applied to it. **No default
threshold is committed** — the schema's `0.1` is a placeholder, and picking one
is Commit 5's job, from these curves.

## Fail-closed

The whole mask rests on `x0` and `source` describing the same pixel, so the
source reference must match the target latent exactly in channels, latent
length, height and width. Nothing is resized, interpolated, cropped or warped;
the Ref2V node re-canvases reference videos independently of the requested
generation size, so a mismatch is common and is always an error rather than a
best effort.

With `strict=True` (the default) any of these stops the run *before* the forward:
no video reference, an out-of-range `source_video_ref`, a geometry mismatch, a
missing packed layout, a sigma at or below the stability floor. With
`strict=False` the run samples dense, measurement disables itself for the rest of
the run, and the report says `MEASUREMENT DISABLED` in the header — a fallback
cannot be mistaken for a measurement.

A measurement run that quietly measured nothing is worse than one that stopped,
because its output still looks like evidence.

## Placement and cost

```text
Load H3 model
  -> MiniMax H3 Sigma Shift (Zi)      # schedule, attention backend, VRAM guard
  -> MiniMax H3 Masked Ref2V Cache (Zi)
  -> sampler
```

`source_video_ref` is one-based over **video** references only — reference images
and standalone audio do not count — because that is how the reference widgets
read in the graph.

Cost is one extra float32 copy of the video latent per conditional step plus a
few element-wise passes over it; the source latent is moved to the device once
per run and released at the end. Nothing is added to the 50 blocks. `measure`
mode returns the model's own output object unmodified, which the self-test
asserts on directly.

## Test matrix

The probe's matrix plus the cases that specifically stress an edit mask:

| tag | clip |
| --- | --- |
| `t22`, `t39`, `long`, `rot`, `transform`, `translate` | as for the probe |
| `subject_static_camera` | subject replacement, camera locked — the best case |
| `subject_moving_camera` | the same edit with camera motion |
| `global_style_change` | expected to activate nearly everything, and to say so |

`global_style_change` is not a failure case. It is the one that has to make the
report obviously unsuitable for compaction rather than quietly produce a 95%
active mask.

## Tests

No model or checkpoint required — it builds a real `PackedLayout` and drives the
statistics with synthetic Q/K whose attention target is known in advance:

```bash
cd /path/to/ComfyUI
python custom_nodes/ComfyUI-H3-Extended/tests/test_probe.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_attention_backend.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_vram_guard.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_cond_cache.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_chunked_ref2v.py
python custom_nodes/ComfyUI-H3-Extended/tests/test_masked_cache.py
```

`test_cond_cache.py`, `test_chunked_ref2v.py` and `test_masked_cache.py` are safe
to run while a generation is in flight — the first masks the GPU out entirely,
the other two force `--cpu` before the first comfy import so `model_management`
never initializes a CUDA context.

`test_attention_backend.py` **runs real kernels on the card** when CUDA and
SageAttention are present. `test_probe.py` and `test_vram_guard.py` stub the
driver queries but still construct CUDA tensors, so they need a device to exist
and will fail with `No CUDA GPUs are available` if one is masked — they are
cheap, but they are not free.

Note that `CUDA_VISIBLE_DEVICES=""` does *not* mask the device on Windows; it is
silently ignored and torch still sees the GPU. Only `CUDA_VISIBLE_DEVICES=-1`
actually masks it.

The backend test verifies routing with a registered stand-in (so it runs
anywhere) and, when SageAttention and CUDA are present, additionally checks the
real kernel's accuracy and that it did not silently fall back.

The VRAM-guard test stubs the driver query and the cache release — neither can be
exercised honestly without pushing a real GPU to the edge — and covers the
decision logic around them: the threshold comparison, the release-and-recheck
second chance, that a breach raises `InterruptProcessingException` before the
forward runs, and that an already-installed unet wrapper is chained rather than
replaced. It also asserts on the contents of the cancel log, and resolves a real
`PackedLayout` to check the `packed tokens` line.

The masked-cache test plants an edit of known extent in a synthetic source and
predicted-clean pair and checks that the score chain recovers exactly its tokens
and no neighbours, that tiles and halos produce hand-written answers on grids
small enough to verify by eye, that a mask inferred at one sigma does not become
active until the next, and that every validation failure either stops the run or
returns the dense output untouched. It drives the real diffusion-model wrapper
with a fake executor and a `CONST` flow sampling object, so the
`calculate_denoised` relation is exercised rather than re-derived, and it asserts
that `measure` mode hands back the model's own output object.

The cond-cache test stubs the text encoder — loading Qwen3-VL-32B to test a
cache that exists to avoid loading it would defeat the point — and covers what
surrounds it: that a hit is bit-identical, that every input which genuinely
changes the embeddings also changes the key (including a different video frame
pair of the same shape, which is the slice-view trap), that each bypass fires,
and that a hit leaves the entry deletable rather than mmap-locked. The janitor
gets the same treatment: that eviction drops the least recently *used* rather
than the oldest stored, that orphaned and failed temp files are collected, that
age expiry keys off last use, and that a folder holding anything the cache did
not write is refused rather than swept.

## Notes on the fork

The nodes only produce conditioning + latents — all model-side behaviour
(`comfy/ldm/minimax/`, the minimax CLIP/tokenizer, the VAEs) still lives in core
and is *not* forked here. If a core update changes those conditioning keys
(`minimax_keyframes`, `minimax_refs`, `minimax_h3_sigma_shift_*`), the packed
layout, or the attention call site, this fork needs the matching update.

The conditioning cache additionally leans on two core shapes: `cached_patcher_init`
on the CLIP's patcher (for TE provenance) and `encode_from_tokens_scheduled`
returning `[[cond, dict]]`. Both are checked at runtime and fall back to a plain
encode if they change, so a core update degrades performance rather than
correctness.
