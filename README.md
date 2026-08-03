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

Existing workflows still point at the stock ids; re-add the `(Zi)` nodes to use
this copy.

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

## Tests

No model or checkpoint required — it builds a real `PackedLayout` and drives the
statistics with synthetic Q/K whose attention target is known in advance:

```bash
cd /path/to/ComfyUI
python custom_nodes/ComfyUI-H3-Extended/tests/test_probe.py
```

## Notes on the fork

The nodes only produce conditioning + latents — all model-side behaviour
(`comfy/ldm/minimax/`, the minimax CLIP/tokenizer, the VAEs) still lives in core
and is *not* forked here. If a core update changes those conditioning keys
(`minimax_keyframes`, `minimax_refs`, `minimax_h3_sigma_shift_*`), the packed
layout, or the attention call site, this fork needs the matching update.
