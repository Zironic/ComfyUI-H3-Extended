# MiniMax H3 Ref2V Masked Computation — Implementation Plan

**Repository:** `Zironic/ComfyUI-H3-Extended`
**Status:** Stage 0 (§10, measurement only) implemented in `h3_masked_cache/`;
Stages 1-5 pre-implementation. The node refuses any mode that would change
inference, so nothing below §10's Stage 0 is live yet.
**Initial objective:** determine whether unchanged target-video regions can be removed from H3’s 50-block DiT computation after one or more dense warm-up steps, while retaining the complete source/reference stream as context.

## 0. Decision

Implement **masked target-token compaction before block-wise caching**.

The first optimized version will:

1. Run a configurable number of dense warm-up evaluations.
2. Compare H3’s predicted clean target latent against the source-video latent.
3. Build a conservative spatiotemporal active-edit mask.
4. Retain:

   * all text rows;
   * all keyframe/reference rows;
   * all target-audio rows;
   * only active target-video rows.
5. Run all 50 DiT blocks on that compact sequence.
6. Restore the full sequence before H3’s final output layer.
7. Replace inactive target-video predictions with the exact flow velocity whose denoised result is the source latent.

This is not initially a per-token hidden-state cache. It is **source-clamped target-token pruning**. Block-group caching can be added later, after the sequence has already been shortened.

The fork already provides the needed foundations:

* H3-specific model patch nodes and transformer options;
* an H3-only attention interception mechanism;
* named packed-token layout metadata;
* sampling-run lifecycle wrappers;
* synthetic tests built around core’s real `PackedLayout`;
* H3’s existing `patches_replace["dit"][("double_block", i)]` seam.

The probe currently measures attention but deliberately implements no sparse mask or kernel.

---

# 1. Scope

## 1.1 Included in the first implementation

* Ref2V runs containing one explicitly selected source-video reference.
* Source and target video latents with exactly matching:

  * temporal latent length;
  * latent height;
  * latent width;
  * channel count.
* Fixed external masks for correctness experiments.
* Dense warm-up mask inference.
* Conservative tile expansion and spatial/temporal halos.
* Compact execution through H3’s existing block-replacement interface.
* Exact source-denoised output outside the active mask.
* Conditional and unconditional passes using the same mask.
* Automatic dense fallback.
* Per-run reports covering mask size, sequence reduction, timings, and fallback reasons.
* Compatibility with the fork’s `sage`, `pytorch`, and `comfy` attention selections.

## 1.2 Explicitly excluded initially

* Custom Triton or CUDA sparse-attention kernels.
* Per-head or per-layer attention masks.
* Query-only pruning while retaining inactive target K/V rows.
* Reference-stream K/V caching across diffusion steps.
* Independent block-wise residual caching.
* Automatic optical-flow warping.
* Source and target canvases with different latent geometry.
* Multiple source videos contributing to one target-aligned edit mask.
* Masks that shrink after activation.
* Simultaneous use with EasyCache.
* Training, distillation, or LoRA adaptation.
* Changes to ComfyUI core files.

---

# 2. Relevant Current Architecture

## 2.1 Packed H3 sequence

H3 constructs one packed sequence containing text, condition/reference rows, target audio, and target video. Target audio and target video are the final two segments. The fork’s `TokenLayout` already exposes:

```python
text_range
reference_ranges
audio_range
video_range
video_shape
segments
```

Target-video rows are ordered as complete spatial patch grids for latent frame 0, followed by frame 1, and so on. The DiT spatial patch is `1 × 2 × 2`, so one target-video token represents a `2 × 2` area of the VAE latent.

## 2.2 Current dense block execution

H3’s 50 DiT blocks call attention with the entire packed hidden-state tensor. Core attention currently receives `mask=None`. Each block applies:

```text
AdaLN
full-sequence attention
gated residual
AdaLN
full-sequence MLP
gated residual
```

The core loop already permits replacing each DiT block through:

```python
patches_replace["dit"][("double_block", layer_index)]
```

The replacement receives the current hidden state, time embedding, modulation segments, RoPE table, transformer options, and a callable for the original block.

## 2.3 Fork patching conventions

`MiniMaxH3SigmaShiftZi` already:

* clones the incoming `MODEL`;
* copies `transformer_options`;
* writes H3-specific options;
* selects the attention implementation;
* installs the VRAM guard.

The masked-computation node should follow the same conventions and remain a separate model patch node for clean A/B testing.

## 2.4 Source latent availability

`MiniMaxH3ReferenceToVideoZi` stores each encoded video reference in a `minimax_refs` block:

```python
{
    "kind": "video" | "video_audio",
    "latent_t": ...,
    "latent_h": ...,
    "latent_w": ...,
    "latent": source_latent,
    ...
}
```

Core copies those blocks into `minimax_payload["refs"]`, which reaches every diffusion-model call.

---

# 3. Core Invariants

The implementation must preserve these invariants.

## 3.1 No behavior change when disabled

With the node disabled:

* no wrappers are installed;
* no block replacements are registered;
* no global module binding is modified;
* output must be identical to the unpatched model.

## 3.2 Dense fallback is always available

Any uncertainty must result in a dense forward, not an approximate sparse forward.

Fallback conditions include:

* no selected source-video reference;
* source index out of range;
* source/target latent geometry mismatch;
* unsupported model type;
* missing H3 packed layout;
* unexpected block order;
* another replacement already occupying an H3 block;
* invalid or empty mask;
* active fraction above the configured fallback limit;
* sigma too close to zero for stable forced-velocity calculation;
* mask state disagreement between conditions at the same sigma;
* unexpected sequence shape during a compact pass.

## 3.3 The source and target coordinate systems must match

The first version accepts only:

```python
source.shape == target_video_x.shape
```

after batch normalization.

No resizing, temporal interpolation, cropping, or latent warping occurs inside the masked-computation node.

## 3.4 The mask is shared across all 50 blocks

The active token set cannot change between individual blocks within one model evaluation.

## 3.5 Both CFG branches use the same mask

A mask inferred from the conditional branch is staged as pending and becomes active only at the next distinct sigma. This prevents one CFG branch from running dense and the other compact at the same sampling step.

## 3.6 Active masks only grow during one run

After the initial warm-up:

```python
new_mask = old_mask | newly_detected_mask
```

No token becomes inactive later in the same run. This avoids removing regions whose edit becomes clearer at later denoising steps.

---

# 4. Proposed Repository Structure

```text
ComfyUI-H3-Extended/
├── h3_masked_cache/
│   ├── __init__.py
│   ├── config.py
│   ├── source.py
│   ├── mask.py
│   ├── plan.py
│   ├── blocks.py
│   ├── session.py
│   ├── wrappers.py
│   ├── report.py
│   └── nodes.py
├── tools/
│   └── analyze_h3_edit_masks.py
├── tests/
│   ├── test_masked_cache.py
│   └── test_masked_cache_gpu.py
├── __init__.py
└── README.md
```

## 4.1 `config.py`

Define an immutable configuration dataclass:

```python
@dataclass(frozen=True)
class MaskedCacheConfig:
    mode: str
    source_video_ref: int
    warmup_steps: int
    refresh_interval: int
    score_threshold: float
    score_absolute_floor: float
    tile_h: int
    tile_w: int
    spatial_halo: int
    temporal_halo: int
    dense_fallback_fraction: float
    strict: bool
    run_tag: str
```

Supported modes:

```text
measure
fixed
dynamic
```

Do not include block caching in this configuration yet.

## 4.2 `source.py`

Responsibilities:

* enumerate `video` and `video_audio` blocks from `minimax_payload["refs"]`;
* resolve the configured one-based `source_video_ref`;
* validate tensor rank, channels, temporal length, and spatial dimensions;
* move one cached copy to the current target device;
* broadcast batch 1 when required;
* return structured failure reasons rather than silently selecting another reference.

Suggested type:

```python
@dataclass
class SourceResolution:
    latent: torch.Tensor | None
    ref_ordinal: int | None
    valid: bool
    reason: str | None
```

## 4.3 `mask.py`

Responsibilities:

* calculate source-difference score maps;
* convert latent-cell scores to DiT-token scores;
* threshold;
* expand to fixed tiles;
* dilate spatially and temporally;
* import externally supplied masks;
* expand token masks back to latent-cell masks;
* calculate mask statistics.

## 4.4 `plan.py`

Represent one compact forward:

```python
@dataclass
class CompactPlan:
    enabled: bool
    reason: str
    dense_seq_len: int
    compact_seq_len: int
    active_fraction: float

    active_video_mask: torch.Tensor       # [T, patch_h, patch_w], bool
    active_video_flat: torch.Tensor       # [video_rows], bool
    active_sequence_indices: torch.Tensor # [compact_seq_len], long

    compact_rope: torch.Tensor | None
    compact_mod_segments: list
```

The plan is immutable during one diffusion-model evaluation.

## 4.5 `blocks.py`

Implement all 50 replacements.

Do not gather and scatter on every block. Instead:

1. Block 0 gathers the compact sequence once.
2. Blocks 1–48 keep passing the compact hidden state.
3. Block 49 scatters active rows back into the original full hidden-state buffer.
4. Core’s final layer receives the expected full sequence.

This avoids 50 repeated compaction operations.

## 4.6 `session.py`

Own state for one sampling run:

```python
class MaskedCacheSession:
    run: MaskedCacheRun | None
    current_forward: ForwardState | None
    pending_mask: torch.Tensor | None
    active_mask: torch.Tensor | None
    last_sigma: float | None
```

It must not retain tensors between sampling runs.

## 4.7 `wrappers.py`

Provide:

* `OUTER_SAMPLE` wrapper for run lifecycle;
* `DIFFUSION_MODEL` wrapper for:

  * source resolution;
  * step and sigma tracking;
  * mask inference;
  * compact-plan construction;
  * forced inactive output;
  * timing and reporting.

## 4.8 `nodes.py`

Register a separate experimental model-patch node:

```text
MiniMaxH3MaskedRef2VCacheZi
MiniMax H3 Masked Ref2V Cache (Zi)
```

## 4.9 `report.py`

Write:

```text
output/h3_masked_cache/<run_tag>_<timestamp>/
├── summary.json
├── steps.jsonl
├── mask.npz
└── report.txt
```

No Python pickle files.

---

# 5. Mask Definition

## 5.1 Predicted clean latent

The diffusion-model wrapper receives:

* the current H3 video input (x_\sigma);
* raw H3 video model output (v);
* current sigma from `transformer_options["sigmas"]`;
* the model sampling object captured when the node is installed.

Calculate the predicted clean latent using the actual configured sampling object:

```python
x0 = model_sampling.calculate_denoised(
    sigma,
    model_output_video.float(),
    video_x.float(),
)
```

Do not duplicate the flow-denoising formula in the mask inference path.

## 5.2 Per-latent-cell difference score

For each temporal and spatial latent position:

```python
error = rms_channels(x0 - source)
scale = rms_channels(source)
score = error / (scale + absolute_floor)
```

Shapes:

```text
x0/source: [B, 24, T, H, W]
score:      [T, H, W]
```

Use the conditional branch only for mask observation.

## 5.3 Convert to DiT-token resolution

H3’s DiT patch is `1 × 2 × 2`, so conservatively max-pool each `2 × 2` latent area:

```python
token_score = max_pool2d(
    score.reshape(T, 1, H, W),
    kernel_size=2,
    stride=2,
).reshape(T, H // 2, W // 2)
```

This produces exactly one score per target-video sequence row.

## 5.4 Thresholding

Initial implementation:

```python
core_mask = token_score >= score_threshold
```

The first default threshold must be selected from measurement runs. It should not be guessed and committed before Stage 0 analysis.

`measure` mode must save score distributions and threshold sweeps.

## 5.5 Tile expansion

To avoid isolated token decisions and prepare for possible future block-sparse kernels:

1. divide each latent frame into `tile_h × tile_w` token tiles;
2. mark the complete tile active when any token in it is active;
3. pad edge tiles conservatively.

Initial calibration candidates:

```text
1×1 tokens
2×2 tokens
4×4 tokens
```

Do not hardcode one as correct before measurement.

## 5.6 Spatial halo

Dilate active tiles by a configurable number of tile cells.

This protects:

* subject boundaries;
* hair and fine structures;
* shadows;
* reflections;
* occlusion transitions;
* nearby objects affected by replacement.

## 5.7 Temporal halo

Dilate over latent-frame indices:

```python
active[t] |= active[t-k:t+k+1]
```

H3’s temporal latent spans are irregular in pixel-frame terms, but each target token still has a stable latent-frame index. The first implementation should operate in latent indices and record this limitation.

## 5.8 Dense fallback fraction

After expansion:

```python
active_fraction = active_tokens / total_target_video_tokens
```

When the active fraction exceeds the configured limit, run dense. A compact pass with almost every target token retained adds complexity without useful savings.

---

# 6. Compact Sequence Construction

Given:

```text
video_range = [video_start, video_stop)
```

retain every packed row before target video, then append only active target-video rows:

```python
prefix = torch.arange(0, video_start, device=device)
active_target = video_start + torch.nonzero(
    active_video_flat,
    as_tuple=False,
).flatten()

active_sequence_indices = torch.cat([prefix, active_target])
```

This retains:

* complete text;
* complete source/reference conditioning;
* complete target audio;
* active target video.

Target video is the final H3 segment, so the compact order remains valid.

## 6.1 Position encoding

Gather RoPE by original token index:

```python
compact_rope = full_rope[:, active_sequence_indices]
```

The compact tensor order changes, but every retained token keeps its original H3 temporal and spatial coordinates.

No new positions are generated.

## 6.2 Modulation segments

All retained non-target segments remain unchanged.

Target video uses one modality/timestep modulation row. Replace its original full segment with:

```python
(compact_video_start, compact_seq_len, original_video_mod_row)
```

Preserve text tag runs exactly as core generated them.

## 6.3 Hidden-state lifecycle

At layer 0:

```python
full_template = h
compact_h = h.index_select(0, active_sequence_indices)
compact_h = original_block(compact_h, compact metadata)
return compact_h
```

At intermediate layers:

```python
compact_h = original_block(compact_h, compact metadata)
return compact_h
```

At the final layer:

```python
compact_h = original_block(compact_h, compact metadata)
full_template.index_copy_(0, active_sequence_indices, compact_h)
return full_template
```

Inactive target rows in `full_template` retain their initial embedded state. Their final predictions will not be used.

## 6.4 Replacement registration

Copy every nested options dictionary before mutation:

```python
to = m.model_options["transformer_options"] = (
    m.model_options.get("transformer_options", {}).copy()
)

patches_replace = to["patches_replace"] = (
    to.get("patches_replace", {}).copy()
)

dit = patches_replace["dit"] = (
    patches_replace.get("dit", {}).copy()
)
```

Register one replacement for every H3 block.

If any `("double_block", i)` key already exists, the first version should refuse to arm. Arbitrarily composing model-specific block replacements is unsafe.

---

# 7. Exact Inactive-Region Output

Core’s flow sampling calculates:

[
x_0 = x_\sigma - \sigma v
]

To force the inactive region’s denoised prediction to equal the source latent:

[
v_{\text{inactive}}
===================

\frac{x_\sigma-x_{\text{source}}}{\sigma}
]

Implementation:

```python
forced = (video_x.float() - source.float()) / sigma

model_output_video = torch.where(
    inactive_latent_mask,
    forced.to(model_output_video.dtype),
    model_output_video,
)
```

The token mask must be expanded from `[T, H/2, W/2]` to latent-cell resolution `[T, H, W]` by repeating each spatial token over its `2 × 2` latent patch.

This preserves H3’s normal sampler integration. If the inactive region is already on the source-noise flow trajectory, the forced velocity keeps it on that trajectory.

At very small sigma, run dense rather than divide by an unstable value.

Audio output is untouched.

---

# 8. Sampling-Run State Machine

## 8.1 Run start

The `OUTER_SAMPLE` wrapper:

* creates a fresh run object;
* clears active and pending masks;
* resets timing;
* resets expected block order;
* creates the output directory lazily;
* records model and configuration metadata.

## 8.2 Per-sigma promotion

At the first model call for a new sigma:

1. promote `pending_mask` to `active_mask`;
2. clear the pending mask;
3. decide whether the new sigma is:

   * dense warm-up;
   * scheduled dense refresh;
   * compact;
   * dense fallback.

This ensures all conditions evaluated at the previous sigma used the same plan.

## 8.3 Warm-up phase

For the first `warmup_steps` distinct sigmas:

* all blocks run dense;
* the conditional branch generates a candidate mask;
* candidate masks are accumulated conservatively.

Recommended initial policy:

```python
pending_mask = candidate_mask
active_mask = active_mask | pending_mask
```

The first compact step occurs only after the configured number of dense sigmas has completed.

## 8.4 Compact phase

For each compact evaluation:

* resolve source and layout;
* build one immutable `CompactPlan`;
* install it as the current forward state;
* run compact blocks;
* restore the full sequence;
* force inactive output to source velocity;
* clear the current forward state in `finally`.

## 8.5 Dense refresh

Initial dynamic implementation should default to no refresh.

Once the frozen-mask version is validated, support:

```text
refresh_interval = N distinct sigmas
```

A refresh step runs completely dense and may only add active tokens.

## 8.6 Run end

Write the report and release:

* source latent device copy;
* mask tensors;
* active indices;
* full-sequence template;
* compact RoPE;
* timing events.

---

# 9. Node Design

## 9.1 Placement

Recommended workflow:

```text
Load H3 model
  -> MiniMax H3 Sigma Shift (Zi)
  -> MiniMax H3 Masked Ref2V Cache (Zi)
  -> sampler
```

The Sigma Shift node should remain responsible for:

* video/audio shift;
* attention backend;
* VRAM guard.

The new node should be responsible only for masked Ref2V behavior.

## 9.2 Initial schema

```text
model
enabled
mode
source_video_ref
warmup_steps
refresh_interval
score_threshold
tile_size
spatial_halo
temporal_halo
dense_fallback_fraction
strict
run_tag
fixed_mask (optional)
```

Suggested mode behavior:

| Mode      | Behavior                                            |
| --------- | --------------------------------------------------- |
| `measure` | Dense inference only; calculate and save score maps |
| `fixed`   | Use supplied external mask; no dynamic inference    |
| `dynamic` | Infer after dense warm-up, then compact             |

## 9.3 Safe defaults during development

Until calibration is complete:

```text
mode: measure
strict: true
refresh_interval: 0
```

The optimization should not silently become the default path before its mask threshold has been measured.

## 9.4 Logging

Prefix all logs with:

```text
[H3 Extended] masked cache
```

Per run, log:

```text
source video ref: 1
source latent: [1,24,27,48,84]
target latent: [1,24,27,48,84]
dense sequence: 45218 rows
active target: 4210 / 21773 rows (19.3%)
compact sequence: 27655 rows
mode: dynamic
warm-up: 2 sigmas
attention backend: sage
```

Fallback logs must name the exact reason.

---

# 10. Implementation Stages

## Stage 0 — Measurement Only

### Goal

Determine whether early predicted-clean differences provide a stable, conservative edit mask.

### Changes

* Add package skeleton.
* Add source-reference resolver.
* Add sampling lifecycle.
* Add predicted-clean score calculation.
* Add report writer.
* Add `measure` mode node.
* Do not register block replacements yet.

### Output

For every observed sigma:

```text
step
sigma
score quantiles
threshold sweep
active fraction before halo
active fraction after halo
Jaccard against previous observed mask
```

Save token score maps in compressed NumPy format.

### Test matrix

Reuse and extend the probe matrix:

```text
t22
t39
long
rot
transform
translate
subject_static_camera
subject_moving_camera
global_style_change
```

### Gate

Proceed only when at least some realistic subject-replacement clips show:

* a materially smaller active region than the full target;
* high overlap between masks from consecutive early steps;
* few changed tokens appearing outside a conservatively dilated early mask.

Exact thresholds should be selected from results, not specified in advance.

---

## Stage 1 — Dense Clamp-Only Validation

### Goal

Validate source alignment and forced inactive-region velocity before changing transformer computation.

### Behavior

* Run all 50 blocks densely.
* Use a fixed external mask.
* Override inactive target output with source velocity.
* Compare against:

  * normal dense output;
  * source video outside the mask;
  * identical run with all tokens active.

### Required assertions

For random synthetic tensors:

```python
denoised = model_sampling.calculate_denoised(
    sigma,
    forced_output,
    x,
)

assert denoised[inactive] == source[inactive]
```

For actual H3:

* inactive decoded areas should follow the source;
* active areas should remain generated;
* no shape, normalization, or temporal alignment error should appear.

### Gate

Do not proceed to compaction until clamp-only behavior is correct.

---

## Stage 2 — Fixed-Mask Compact Block Emulator

### Goal

Test the central model assumption:

> Active edit tokens can remain faithful when inactive target duplicates are absent from all 50 blocks, provided complete source/reference rows remain available.

### Changes

* Register all block replacements.
* Implement gather-once, compact-through-50, scatter-once execution.
* Use only fixed externally supplied masks.
* Retain full reference, text, and audio sequences.
* Use the source-velocity override.

### Correctness controls

1. **Compact-all control**

Force the compact machinery to retain every sequence row. Under the `pytorch` attention backend, compare against dense baseline at identical seed and settings.

2. **Empty edit mask**

Should trigger dense fallback or a documented source-copy mode. Do not execute a zero-target compact sequence accidentally.

3. **One-tile mask**

Verify indexing, RoPE gathering, modulation remapping, and final scatter.

4. **Discontiguous mask**

Verify original target-token ordering is retained.

### Gate

Proceed when:

* compact-all agrees with dense within numerical tolerance;
* fixed-mask output retains edit quality;
* no changes occur outside the mask beyond the intended source clamp;
* measured runtime improves enough to justify the extra machinery;
* peak VRAM does not regress.

---

## Stage 3 — Dynamic Mask Activation

### Goal

Replace the external mask with the mask inferred during dense warm-up.

### Changes

* pending-mask promotion at sigma boundaries;
* conditional-branch-only observation;
* configurable warm-up;
* conservative union of observed masks;
* active-fraction fallback;
* mask hash and per-step reporting.

### Initial policy

```text
2 dense warm-up sigmas
freeze mask
no refresh
mask may only grow before freeze
```

### Comparisons

For every clip and seed:

```text
dense
fixed oracle mask
dynamic inferred mask
dynamic inferred mask with larger halo
```

### Gate

Dynamic output should remain close to the fixed-oracle result. Failures must be classifiable as:

* early false-negative mask;
* insufficient spatial halo;
* insufficient temporal halo;
* inactive target K/V required by active tokens;
* source/target misalignment;
* threshold instability.

---

## Stage 4 — Periodic Refresh and Three-Tier Masks

Only after the frozen binary mask works.

Introduce:

```text
core edit tokens
margin tokens
context tokens
```

Potential policy:

* core: computed every step;
* margin: computed on refresh steps or every second compact step;
* context: source-clamped and removed.

This is the point where the implementation begins to resemble a full masked cache rather than binary token pruning.

Do not combine this stage with initial correctness work.

---

## Stage 5 — Block-Group Cache on the Compact Sequence

Block-group caching becomes more practical after compaction because cached hidden tensors scale with the reduced sequence.

Suggested groups:

```text
0–4
5–9
...
45–49
```

This remains a separate experiment with separate reports.

Do not store 50 full-sequence residual tensors.

Possible later strategies:

* cache only selected middle groups;
* FP8 cache storage;
* one contiguous cached range;
* mandatory upstream refreshes;
* separate thresholds per group;
* disable block cache on mask-refresh steps.

---

# 11. Unit Tests

Create `tests/test_masked_cache.py`, runnable without an H3 checkpoint.

## 11.1 Source resolution

Test:

* one image then one video;
* multiple video references;
* one-based indexing;
* invalid index;
* `video_audio` selection;
* exact shape match;
* temporal mismatch;
* spatial mismatch;
* source batch broadcasting.

## 11.2 Score calculation

Plant a known changed cuboid in synthetic source and predicted-clean tensors.

Assert:

* score is concentrated in the planted region;
* channel RMS is correct;
* `2 × 2` pooling maps to the correct DiT tokens;
* no neighboring token is activated before halo expansion.

## 11.3 Tile expansion

Test:

* edge padding;
* one active token activates the complete configured tile;
* no token is lost at odd grid sizes.

## 11.4 Spatial and temporal halo

Test exact expected masks for small synthetic grids.

## 11.5 Compact indices

Using a real core `PackedLayout`, assert:

* every row before target video is retained;
* active target rows are appended in original order;
* no inactive target row is retained;
* compact length is correct;
* all indices are unique and increasing.

The existing probe tests already establish the pattern of building a real `PackedLayout` without loading a checkpoint.

## 11.6 Modulation remapping

Create synthetic modulation segments containing multiple text tag runs.

Assert:

* prefix runs retain their lengths and modulation rows;
* video becomes one shortened final run;
* compact segments tile the sequence with no gaps.

## 11.7 Block lifecycle

Use fake blocks that add a known layer-dependent value.

Assert:

* layer 0 gathers once;
* intermediate layers receive compact tensors;
* final layer scatters once;
* active rows receive all layer transformations;
* inactive rows remain at their pre-block values;
* unexpected layer order raises and falls back.

## 11.8 Forced source output

For random tensors and multiple sigma values:

```python
forced = (x - source) / sigma
x0 = sampling.calculate_denoised(sigma, forced, x)
```

Assert exact inactive equality to source.

## 11.9 Sigma-boundary mask promotion

Simulate:

```text
sigma A conditional
sigma A unconditional
sigma B conditional
```

Assert the candidate from sigma A is not used until sigma B.

## 11.10 Fail-closed paths

Every validation failure must either:

* return the unmodified dense model output; or
* raise before model execution when `strict=True`.

---

# 12. GPU Integration Tests

Create `tests/test_masked_cache_gpu.py`.

These require an H3 checkpoint and must not run as part of ordinary lightweight tests.

## 12.1 Dense identity

Configuration:

```text
mode=fixed
force_compact_all=true
attention_backend=pytorch
```

Compare raw model outputs layer-complete against baseline.

## 12.2 Backend composition

Run compact mode with:

```text
pytorch
sage
comfy
```

Verify:

* selected backend remains selected;
* Sage does not silently fall back;
* compact sequence lengths are reported correctly.

## 12.3 Probe composition

Initial policy should reject simultaneous active attention probing and compact execution, or explicitly report that the probe observes only dense warm-up calls.

Do not let the current probe silently label a compact sequence with the full packed layout.

## 12.4 EasyCache conflict

Detect:

```python
"easycache" in transformer_options
```

and refuse active compact mode initially.

Whole-step skipping can be composed later, but it would otherwise interfere with warm-up and refresh accounting.

## 12.5 VRAM guard

Run compact execution with the existing VRAM guard armed.

Extend guard diagnostics to include, when available:

```text
masked cache: active target 4210/21773
compact seq_len=27655 from dense seq_len=45218
```

---

# 13. Benchmark Harness

Add a repeatable benchmark workflow and machine-readable output.

## 13.1 Variants

For each clip and seed:

```text
dense-pytorch
dense-sage
clamp-only-sage
fixed-mask-compact-sage
dynamic-mask-compact-sage
```

## 13.2 Measurements

Record:

* total sampling time;
* per-model-call CUDA time;
* dense warm-up time;
* compact model-call time;
* number of dense evaluations;
* number of compact evaluations;
* dense sequence length;
* compact sequence length;
* active target fraction;
* peak allocated VRAM;
* peak reserved VRAM;
* fallback count and reasons.

## 13.3 Quality outputs

Save:

* final latent;
* decoded video;
* source video;
* active mask;
* difference outside active mask;
* difference inside active mask.

Recommended comparisons:

* latent RMS difference against dense;
* decoded PSNR/SSIM outside the edit mask;
* perceptual comparison inside the edit mask;
* frame-to-frame difference outside the mask;
* seam behavior around the halo boundary.

Do not accept mean-only metrics. Record worst-frame and high-percentile errors.

---

# 14. Performance Expectations and Limits

For a typical Ref2V sequence, approximately half the packed rows may be source/reference video and approximately half target video.

If only 20% of target-video rows remain active:

```text
dense:
    reference/context + 100% target

compact:
    reference/context + 20% target
```

This may reduce the 50-block sequence to roughly 60–65% of its dense length. Actual acceleration is unknown until measured because H3 on this hardware has shown near-linear runtime scaling rather than ideal quadratic attention scaling.

The first compact implementation still performs full work for:

* source/reference rows;
* text;
* target audio;
* video patch projection before the blocks;
* the full-size hidden-state template;
* final-layer projection before inactive output replacement.

The dominant intended saving is inside the 50 DiT blocks.

If gather-once compact execution does not produce useful speedup, do not proceed immediately to a custom sparse kernel. First profile whether the remaining cost is:

* reference-side DiT computation;
* MLP;
* attention;
* model-weight streaming;
* final projection;
* warm-up overhead.

---

# 15. Principal Risks

## 15.1 Inactive target rows may carry useful hidden context

Although source content remains in reference rows, H3 may have learned to route information through the duplicated target lattice.

Mitigations:

* conservative halo;
* fixed-mask oracle experiments;
* retain selected inactive target summary tiles;
* periodic dense refresh;
* later query-only pruning rather than removing K/V;
* abandon pruning if active-region quality fails consistently.

## 15.2 Early masks may miss late-emerging edits

Mitigations:

* two or more dense warm-up steps;
* mask union rather than intersection;
* periodic dense refresh;
* mask only grows;
* large temporal halo;
* dense fallback when mask stability is poor.

## 15.3 Global camera or lighting changes may activate most tokens

This is expected.

Use dense fallback rather than forcing a sparse mode onto an unsuitable task.

## 15.4 Source latent may not be target aligned

The current Ref2V node adapts reference-video canvases independently from the requested target dimensions. Exact-match validation is therefore mandatory.

A future change may add a dedicated edit-source path that pins the source reference to the target canvas, but this should not be mixed into the first masked-computation implementation.

## 15.5 Compact hidden state may interact with model prefetching

H3’s model prefetch queue operates at block granularity and should remain valid because the same blocks execute in the same order. This still requires GPU testing.

## 15.6 Core updates may change the replacement seam

The fork intentionally leaves model-side H3 behavior in core. It already documents that changes to packed layout or attention call sites require matching fork updates. The new code must validate:

* presence of `blocks`;
* expected replacement keys;
* expected block callback arguments;
* target video as final packed segment.

If these checks fail, disable compact mode.

---

# 16. Commit Sequence

## Commit 1 — Mask measurement infrastructure

```text
add h3_masked_cache package
add source resolver
add score map and report
add measure-only node
add unit tests
register extension
document usage
```

No inference modification.

## Commit 2 — Dense source clamp

```text
add fixed mask input
add token-to-latent mask expansion
add forced source velocity
add synthetic flow tests
add GPU clamp-only workflow
```

No block compaction.

## Commit 3 — Compact block emulator

```text
add compact plan
add block replacements
add gather-once/scatter-once lifecycle
add modulation and RoPE remapping
add all-active correctness control
add GPU correctness tests
```

Fixed masks only.

## Commit 4 — Dynamic warm-up masks

```text
add pending mask promotion
add warm-up state machine
add conditional-only observation
add active-fraction fallback
add dynamic reports
```

## Commit 5 — Benchmark and calibration

```text
run test matrix
choose threshold defaults
choose tile and halo defaults
document supported and unsupported tasks
```

## Commit 6 — Optional refresh

Only after frozen-mask results justify it.

---

# 17. Acceptance Criteria for the First Release

The feature remains experimental until all of these hold:

1. Disabled mode is identical to the current fork.
2. Measure mode never changes model output.
3. Source mismatch always fails closed.
4. Forced inactive output produces the source latent under H3’s actual sampling object.
5. Compact-all agrees with dense under the PyTorch baseline.
6. The same fixed mask is used for all conditions at one sigma.
7. At least one representative Ref2V subject-replacement task shows a meaningful measured speedup.
8. No evaluated clip shows unexplained changes outside the active mask.
9. Global-edit tasks reliably fall back to dense.
10. The feature composes with the fork’s attention backend and VRAM guard.
11. All lightweight tests run without loading H3 weights.
12. Reports make it impossible to mistake a dense fallback for a successful compact run.

---

# 18. Recommended First PR Boundary

The first implementation PR should stop after **Stage 0: measurement only**.

That PR answers the most important unknown without introducing approximate inference:

> Does an early predicted-clean source-difference map reliably identify the final edited region across the actual H3 Ref2V workloads used with this fork?

The second PR should add clamp-only validation. Compact block execution should begin only after both source alignment and mask predictability have been measured.
