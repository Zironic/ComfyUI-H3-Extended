# MiniMax H3 Chunked Ref2V — Multi-Strategy Test Harness Plan

## 1. Purpose

Build a reusable two-chunk experiment harness for determining how MiniMax H3 should carry generated state between overlapping Ref2V chunks.

The harness must support testing:

1. No carried state.
2. One decoded and re-encoded target-aligned frame.
3. One directly reused target-aligned latent position.
4. A complete directly reused target-aligned latent overlap.
5. A generated overlap supplied as an additional reference video.
6. A composite source reference whose opening is replaced by the generated overlap.
7. Optional keyframe-completion prompt text.
8. Eventually, sigma-correct clamping of the target overlap.

The harness is not the production arbitrary-length node. It generates Chunk A once, derives reusable assets from it, and evaluates one or more Chunk B strategies under controlled conditions.

---

## 2. Default geometry

Use the current memory-safe default:

```text
Chunk length C: 73 frames
Overlap O:      22 frames
Stride S:       51 frames
Target T:       22 latent positions
FPS:            24
```

Source intervals:

```text
Chunk A: global frames   0–72
Chunk B: global frames  51–123
Overlap: global frames  51–72
```

The source must contain at least 124 frames.

### 2.1 Latent mapping

H3’s video-latent positions use the repeating pixel-frame span pattern:

```text
1, 4, 4, 4, 4
```

For `T=22`:

```text
latent positions  0–14: 51 pixel frames
latent positions 15–21: 22 pixel frames
```

Therefore:

```python
overlap_latent = chunk_a_video_latent[:, :, 15:22]
```

contains the full generated overlap.

The harness must calculate this mapping rather than hard-code it.

```python
def latent_frame_spans(latent_t: int) -> list[int]:
    pattern = (1, 4, 4, 4, 4)
    return [pattern[i % len(pattern)] for i in range(latent_t)]

def find_exact_overlap_slice(
    latent_t: int,
    stride_frames: int,
    overlap_frames: int,
) -> tuple[int, int]:
    """
    Return (latent_start, latent_count) when the stride and overlap
    coincide exactly with latent-position boundaries.
    """
```

For the default profile:

```text
latent_start = 15
latent_count = 7
```

Required assertions:

```python
assert sum(spans[:latent_start]) == 51
assert sum(spans[latent_start:latent_start + latent_count]) == 22
assert latent_start + latent_count == 22
```

If a future profile does not align exactly, direct latent-overlap tests must fail explicitly rather than silently selecting approximate positions.

---

## 3. Core concept: experiments, not hard-coded arms

Every Chunk B run is represented by an `ExperimentSpec`.

```python
@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    display_name: str

    carry_strategy: str
    prompt_policy: str
    position_policy: str
    source_reference_policy: str
    target_policy: str

    enabled: bool = True
    notes: str = ""
```

Example:

```python
ExperimentSpec(
    experiment_id="aligned_overlap_direct",
    display_name="Direct target-aligned overlap",
    carry_strategy="direct_latent_overlap",
    prompt_policy="original",
    position_policy="copy_target",
    source_reference_policy="original",
    target_policy="sample_all",
)
```

The runner must not encode assumptions such as “an experiment always has one keyframe.” It asks the selected strategy to prepare whatever assets and conditioning changes it requires.

---

## 4. Experiment catalog

## 4.1 Baseline

### `baseline_none`

```text
Carry strategy:           none
Prompt:                   original
Source reference:         original Chunk B
Target-aligned condition: none
Target sampling:          normal
```

Purpose:

* Establish normal Chunk B behavior.
* Provide a control for anchor and overlap-adherence measurements.

---

## 4.2 Single-frame experiments

### `frame_reencode_corrected`

```text
Carry source:        decoded Chunk A frame 51
Conversion:          VAE encode as a one-position latent
Target position:     Chunk B latent position 0
Position policy:     exact target position
Prompt:              original
```

This reproduces the original Stage-0 concept.

### `frame_direct_corrected`

```text
Carry source:        Chunk A output latent position 15
Conversion:          none
Target position:     Chunk B latent position 0
Position policy:     exact target position
Prompt:              original
```

This tests whether avoiding the decode/re-encode round trip matters.

### `frame_direct_stock_position`

```text
Carry source:        Chunk A output latent position 15
Target position:     stock pre-reference keyframe time
Prompt:              original
```

Purpose:

* Diagnose the MM-RoPE placement correction.
* Keep the latent source identical to `frame_direct_corrected`.

### `frame_direct_prompted`

```text
Carry source:        Chunk A output latent position 15
Target position:     exact target position
Prompt:              keyframe-completion variant
```

This experiment is deferred until the unmodified prompt has been tested.

---

## 4.3 Full target-aligned overlap

### `aligned_overlap_direct`

```text
Carry source:        Chunk A output latent positions 15–21
Condition length:    T=7
Target positions:    Chunk B positions 0–6
Prompt:              original
Source reference:    original Chunk B
```

This is the primary candidate for the eventual production mechanism.

The condition clip is losslessly copied from the sampler output:

```python
condition = (
    chunk_a_video_latent[:, :, overlap_start:overlap_end]
    .detach()
    .to("cpu", copy=True)
)
```

The target remains fully sampled. The overlap condition is a separate fixed condition stream and does not guarantee identical Chunk B output.

### `aligned_overlap_stock_position`

Optional diagnostic:

```text
Carry source:        same T=7 latent clip
Position policy:     pre-reference timeline
```

This is useful only if the corrected full-overlap arm behaves unexpectedly.

### `aligned_overlap_prompted`

Optional follow-up:

```text
Carry source:        target-aligned T=7 clip
Prompt:              explicit continuation/keyframe text
```

---

## 4.4 Generated overlap as an additional reference

### `generated_overlap_video2`

Construct:

```text
<Video 1>: original Chunk B source video
<Video 2>: Chunk A generated frames 51–72
```

The generated overlap is:

* Decoded from the complete Chunk A output.
* VAE-encoded as a normal 22-frame reference video.
* Presented visually to Qwen as `<Video 2>`.
* Added to the DiT as another ordinary reference block.

This experiment requires a prompt variant defining `<Video 2>` as the previous generated target state.

It does not use target-aligned condition rows.

---

## 4.5 Composite source reference

### `composite_source`

Construct one 73-frame source reference:

```text
frames  0–21: Chunk A generated frames 51–72
frames 22–72: original Chunk B source frames 22–72
```

Then present it as the only source video:

```text
<Video 1>: composite reference
```

Properties:

* No additional transformer reference rows.
* Qwen sees one continuous reference video.
* The original source geometry is unavailable in the overlap.
* The generated-to-original transition occurs inside the reference sequence.

The prompt may remain unchanged because `<Video 1>` is still the sole edit source. Record that its contents are composite in the report.

---

## 4.6 Target clamping

### `target_overlap_clamped`

Deferred until the basic harness works.

Conceptually:

```text
target positions 0–6: constrained to previous overlap trajectory
target positions 7–21: sampled normally
```

The known overlap must be re-noised to the current sigma using one fixed noise tensor. Clean latents must not simply be inserted during high-noise steps.

This experiment requires a sampler intervention and is not implemented through ordinary condition rows.

---

## 5. Carry strategy interface

Implement strategies as classes.

```python
class CarryStrategy(Protocol):
    strategy_id: str

    def dependencies(self) -> "StrategyDependencies":
        ...

    def prepare(
        self,
        context: "HarnessContext",
    ) -> "PreparedStrategy":
        ...
```

### 5.1 Dependencies

```python
@dataclass(frozen=True)
class StrategyDependencies:
    needs_chunk_a_pixels: bool = False
    needs_chunk_a_latent: bool = False
    needs_anchor_reencode: bool = False

    needs_dynamic_video_vae: bool = False
    needs_dynamic_qwen: bool = False
    needs_sampler_intervention: bool = False
```

Examples:

```text
direct_latent_overlap:
    Chunk A latent: yes
    Chunk A pixels: no
    Dynamic VAE: no
    Dynamic Qwen: no

generated_overlap_video2:
    Chunk A latent: no
    Chunk A pixels: yes
    Dynamic VAE: yes
    Dynamic Qwen: yes

composite_source:
    Chunk A pixels: yes
    Dynamic VAE: yes
    Dynamic Qwen: yes
```

### 5.2 Prepared strategy

```python
@dataclass
class PreparedStrategy:
    prompt: str

    qwen_ref_items: list[dict]
    dit_ref_blocks: list[dict]

    target_conditions: list["TargetAlignedCondition"]
    target_initializer: object | None
    sampler_intervention: object | None

    metadata: dict
```

A strategy may modify any of these independently.

---

## 6. Generic target-aligned condition model

Do not represent overlap clips as a collection of special keyframes.

Use:

```python
@dataclass
class TargetAlignedCondition:
    latent: torch.Tensor
    target_latent_start: int
    label: str
    position_policy: str = "copy_target"
```

Examples:

```python
# Single directly carried frame
TargetAlignedCondition(
    latent=chunk_a_latent[:, :, 15:16],
    target_latent_start=0,
    label="previous frame",
)

# Full overlap
TargetAlignedCondition(
    latent=chunk_a_latent[:, :, 15:22],
    target_latent_start=0,
    label="previous overlap",
)
```

Validation:

```python
condition_t = condition.latent.shape[2]

assert condition.target_latent_start >= 0
assert condition.target_latent_start + condition_t <= target_latent_t
assert condition.latent.shape[-2:] == target_latent.shape[-2:]
```

This abstraction supports:

* One frame.
* Full overlap.
* Future interior condition clips.
* First- and last-frame conditions.

---

## 7. Experimental layout construction

Build the ordinary Ref2V layout first, with no keyframes or target-aligned conditions:

```text
[text | references | target audio | target video]
```

Then transform it into:

```text
[text | target-aligned conditions | references | target audio | target video]
```

This avoids duplicating the full core `PackedLayout` implementation.

### 7.1 Layout transformation

```python
def insert_target_conditions(
    base_layout,
    conditions: list[TargetAlignedCondition],
    *,
    position_policy: str,
):
    ...
```

Steps:

1. Locate the text segment.
2. Locate the target video segment.
3. Determine rows per target latent position.
4. Copy exact target position rows for every condition.
5. Insert condition rows physically after the text segment.
6. Shift all existing row indices after the insertion.
7. Rebuild:

   * `position_ids`
   * `segments`
   * `img_pos`
   * `img_update`
   * `audio_pos`
   * `audio_update`
   * `seq_len`

### 7.2 Exact target-position policy

For each condition:

```python
target_row_start = (
    target_video_start
    + condition.target_latent_start * frame_rows
)

condition_rows = condition.latent.shape[2] * frame_rows

condition_position_ids = base_layout.position_ids[
    target_row_start:target_row_start + condition_rows
].clone()
```

This copies the exact `(t,h,w)` positions from the target rather than recalculating them.

### 7.3 Stock-position policy

For diagnostic experiments only:

```text
Condition timeline begins at text_len.
Condition temporal spans advance normally from that point.
```

This reproduces the incorrect pre-reference placement while keeping all other mechanics identical.

### 7.4 Conditioning row order

The payload must match the transformed layout:

```python
payload["cond_video_latents"] = [
    *[condition.latent for condition in target_conditions],
    *[ref["latent"] for ref in refs if "latent" in ref],
]
```

The `False` entries of `img_update` must correspond to those tensors in exactly that order.

### 7.5 Assertions

```python
assert int((~layout.img_update).sum()) == total_condition_video_rows
assert layout.position_ids.shape[0] == layout.seq_len
assert layout.img_pos.shape[0] == layout.img_update.shape[0]
assert layout.audio_pos.shape[0] == layout.audio_update.shape[0]
```

For exact alignment:

```python
assert torch.equal(
    condition_position_ids,
    shifted_target_position_ids[target_slice],
)
```

---

## 8. Harness context and reusable assets

```python
@dataclass
class HarnessContext:
    geometry: "HarnessGeometry"

    source_chunk_a_pixels: torch.Tensor
    source_chunk_b_pixels: torch.Tensor

    source_chunk_a_ref: object
    source_chunk_b_ref: object

    static_refs: object

    qwen_chunk_a: object
    qwen_chunk_b_base: object

    chunk_a_output_latent: torch.Tensor
    chunk_a_output_pixels: torch.Tensor

    overlap_latent: torch.Tensor
    overlap_pixels: torch.Tensor
    direct_frame_latent: torch.Tensor
    reencoded_frame_latent: torch.Tensor | None

    dynamic_assets: dict[str, object]
```

All assets should be CPU-resident between model stages unless they are actively used.

---

## 9. Execution pipeline

The harness has five phases.

## Phase A — common VAE preprocessing

With the video VAE resident:

1. Resolve and pin one canvas.
2. Slice source Chunk A and Chunk B.
3. Encode static image references once.
4. Encode original source Chunk A.
5. Encode original source Chunk B.
6. Store resulting latents on CPU or disk.

No generated carry asset exists yet.

---

## Phase B — common Qwen preprocessing

With Qwen resident:

1. Encode the original prompt with Chunk A’s presentation.
2. Encode the original prompt with Chunk B’s presentation.
3. Store both conditioning outputs.

These are reused by:

* Baseline.
* Direct frame conditions.
* Direct overlap conditions.
* Any unmodified-prompt target-aligned strategy.

---

## Phase C — generate Chunk A

With the DiT resident:

1. Sample Chunk A once.
2. Store the complete output video latent losslessly.
3. Decode Chunk A.
4. Store the complete decoded pixel batch losslessly.
5. Derive:

   * Direct frame latent.
   * Direct overlap latent.
   * Pixel frame anchor.
   * Pixel overlap clip.

Chunk A is common to every Chunk B experiment.

It must never be regenerated merely because a new Chunk B experiment is added.

---

## Phase D — dynamic carry preprocessing

Inspect the selected experiment dependency graph.

If any selected strategy needs pixel-derived VAE assets:

1. Load the VAE once.
2. Encode the re-encoded one-frame anchor.
3. Encode generated overlap as a 22-frame reference video.
4. Encode the composite source reference.
5. Store all needed dynamic latents.

If any selected strategy needs dynamic Qwen presentation:

1. Load Qwen once.
2. Encode the prompted single-frame variant, if requested.
3. Encode the generated `<Video 2>` presentation.
4. Encode the composite-source presentation.
5. Store all outputs.

Do not reload Qwen separately for every dynamic experiment.

---

## Phase E — Chunk B experiment runs

With the DiT resident:

For each selected experiment:

1. Build its independent conditioning payload.
2. Build or transform its layout.
3. Create a fresh empty Chunk B target latent.
4. Use the same Chunk B noise seed.
5. Use the same sampler and sigma schedule.
6. Sample.
7. Save the output latent immediately.
8. Decode and save output pixels.
9. Compute metrics.
10. Release arm-specific GPU resources.

Experiment order should run lower-memory arms first:

```text
baseline
single-frame conditions
full target-aligned overlap
composite source
generated Video 2
clamped target
```

A later OOM must not destroy earlier artifacts.

---

## 10. Artifact cache and restartability

The harness should persist intermediate assets so new tests can be added without regenerating Chunk A.

Directory:

```text
<output>/h3_ref2v_harness/<run_id>/
```

Contents:

```text
manifest.json
settings.json
prompt.txt

common/
├── source_a.safetensors
├── source_b.safetensors
├── qwen_a.safetensors
├── qwen_b_base.safetensors
├── chunk_a_output.safetensors
├── chunk_a_frames/
├── overlap_latent.safetensors
└── overlap_frames/

dynamic/
├── frame_reencoded.safetensors
├── video2_ref.safetensors
├── composite_ref.safetensors
├── qwen_prompted.safetensors
├── qwen_video2.safetensors
└── qwen_composite.safetensors

experiments/
├── baseline_none/
├── frame_direct_corrected/
├── aligned_overlap_direct/
└── ...
```

### 10.1 Asset identity

The run manifest must hash:

* Source-video frames used.
* Prompt.
* Static reference pixels.
* Canvas.
* C/O/S geometry.
* Chunk A seed.
* Sampler and sigma schedule.
* Model/checkpoint identity where available.

Changing one of these invalidates Chunk A reuse.

Chunk B experiment settings do not invalidate Chunk A unless they alter Chunk A itself.

---

## 11. Experiment suites

Expose named suites.

### `minimal`

```text
baseline_none
frame_reencode_corrected
frame_direct_corrected
frame_direct_stock_position
```

Purpose:

* Validate hybrid target conditioning.
* Compare re-encoded versus direct latent frame.
* Diagnose temporal positioning.

### `aligned`

```text
baseline_none
frame_direct_corrected
aligned_overlap_direct
```

Purpose:

* Determine whether carrying the full overlap materially improves continuity.

### `prompt`

```text
frame_direct_corrected
frame_direct_prompted
aligned_overlap_direct
aligned_overlap_prompted
```

Purpose:

* Determine whether Qwen needs explicit keyframe-completion language.

### `reference`

```text
baseline_none
aligned_overlap_direct
generated_overlap_video2
composite_source
```

Purpose:

* Compare target-aligned, independent-reference, and zero-extra-row approaches.

### `clamp`

```text
aligned_overlap_direct
target_overlap_clamped
```

Available only after sampler intervention is implemented.

### `all`

Runs every implemented experiment. This is expensive and should not be the default.

---

## 12. Node interface

### Node

```text
Node ID:      MiniMaxH3Ref2VExperimentHarnessZi
Display name: MiniMax H3 Ref2V Experiment Harness (Zi)
Category:     model/video/minimax/testing
Experimental: true
```

### Required inputs

```text
model
clip
video_vae
audio_vae
source_video
prompt
sampler
sigmas
seed
```

The model should already include:

* H3 sigma shifts.
* Selected attention backend.
* VRAM guard.

### Reference inputs

```text
ref_images
source_audio
```

Extra reference videos and standalone audio can be added after the visual carry experiments work.

### Controls

```text
experiment_suite:
    minimal
    aligned
    prompt
    reference
    clamp
    all
    custom

custom_experiments:
    comma-separated experiment IDs

reuse_run:
    optional prior run ID

ref_image_size:
    match
    max

cond_cache:
    auto
    off
    refresh

save_latents:
    true

save_frames:
    true

continue_after_failure:
    true
```

### Outputs

Because experiment count varies, do not expose one Comfy output socket per experiment.

Return:

```text
comparison_video   IMAGE
selected_preview   IMAGE
report             STRING
artifact_path      STRING
```

All complete outputs remain in the artifact directory.

---

## 13. Seed controls

Keep independent seeds:

```text
chunk_a_noise_seed
chunk_b_noise_seed
conditioning_augmentation_seed
clamp_noise_seed
```

Derive deterministic defaults from the node seed using SplitMix64.

Every Chunk B arm uses the same:

```text
chunk_b_noise_seed
conditioning_augmentation_seed
```

The clamped-target experiment uses one fixed overlap-noise tensor for the entire sigma trajectory.

---

## 14. Metrics

Metrics are diagnostic rather than automatic quality judgments.

## 14.1 Pixel overlap metrics

For each Chunk B output:

```python
pixel_overlap_mae = mean(
    abs(chunk_a_pixels[51:73] - chunk_b_pixels[0:22])
)
```

Also record per-frame values.

## 14.2 Latent overlap metrics

For aligned profiles:

```python
latent_overlap_mae = mean(
    abs(chunk_a_latent[:, :, 15:22] - chunk_b_latent[:, :, 0:7])
)
```

This directly measures whether the new generated overlap follows the carried latent state.

## 14.3 First-frame metrics

```text
Pixel error at B frame 0.
Latent error at B position 0.
```

## 14.4 Motion metrics

```python
motion_a = chunk_a_pixels[52:73] - chunk_a_pixels[51:72]
motion_b = chunk_b_pixels[1:22]  - chunk_b_pixels[0:21]

motion_delta_mae = mean(abs(motion_a - motion_b))
```

## 14.5 Source adherence

Compare simple frame-difference structure between:

```text
Chunk B output
Chunk B original source
```

This does not establish semantic fidelity, but can detect strategies that freeze or ignore the source motion.

## 14.6 Resource metrics

Per experiment:

```text
Packed sequence length.
Number of target-condition rows.
Number of reference rows.
Runtime.
Peak allocated VRAM.
Peak reserved VRAM.
Free physical VRAM before and after.
Qwen re-encode required: yes/no.
Dynamic VAE encode required: yes/no.
```

---

## 15. Comparison outputs

Create two comparison videos.

### 15.1 Overlap comparison

Each frame contains columns for:

```text
Original source
Chunk A
Baseline Chunk B
Current experiment Chunk B
```

Use global overlap frames 51–72.

### 15.2 Boundary playback

Construct:

```text
Chunk A frames 39–50
then selected seam
then Chunk B frames 0–11
```

This is more useful for judging visible motion discontinuity than viewing corresponding overlap frames side by side.

Generate one boundary playback per experiment on disk. The Comfy output previews the currently selected experiment.

---

## 16. Failure handling

Each experiment runs in an isolated block.

```python
try:
    result = run_experiment(...)
except InterruptProcessingException:
    record_cancelled(...)
    if not continue_after_failure:
        raise
except torch.cuda.OutOfMemoryError:
    record_oom(...)
    recover_if_possible()
    if not continue_after_failure:
        raise
except Exception:
    record_failed(...)
    if not continue_after_failure:
        raise
```

A VRAM-guard cancellation is a resource result, not a model-quality result.

An assertion failure in layout construction is an implementation failure and must stop that experiment before sampling.

---

## 17. Report schema

```json
{
  "schema_version": 2,
  "run_id": "",
  "profile": {
    "chunk_frames": 73,
    "overlap_frames": 22,
    "stride_frames": 51,
    "target_latent_t": 22,
    "overlap_latent_start": 15,
    "overlap_latent_t": 7,
    "fps": 24
  },
  "common_assets": {
    "chunk_a_reused": false,
    "base_qwen_b_reused": true
  },
  "experiments": {
    "aligned_overlap_direct": {
      "status": "completed",
      "strategy": {},
      "dependencies": {},
      "layout": {},
      "metrics": {},
      "resources": {},
      "artifacts": {}
    }
  }
}
```

Use `null` for unknown values.

---

## 18. CPU test coverage

### Temporal mapping

Test:

```text
C=73, S=51, O=22 → latent slice 15:22
C=90, S=68, O=22 → latent slice 20:27
unaligned profile → explicit failure
```

### Layout transformation

Test:

* No conditions leaves the base layout unchanged.
* One condition inserts one spatial grid.
* Seven-position condition inserts seven grids.
* Condition positions equal the corresponding target positions.
* Reference and target row positions shift correctly.
* `img_pos` and `img_update` remain aligned.
* Audio row indices shift correctly.
* Original layout is not mutated.

### Strategy dependencies

Test that:

* Direct overlap does not request Qwen or VAE dynamic preprocessing.
* Video 2 requests both.
* Composite source requests both.
* Prompted frame requests Qwen but not a dynamic video VAE.
* Clamping requests a sampler intervention.

### Conditioning isolation

Test that one experiment cannot mutate another experiment’s:

* Qwen conditioning.
* Reference blocks.
* Target conditions.
* Layout.
* Seeds.

### Artifact reuse

Test that:

* Adding a new Chunk B experiment reuses Chunk A.
* Changing Chunk A seed invalidates Chunk A.
* Changing only prompt policy invalidates only relevant Qwen assets.
* Corrupt assets are rejected rather than partially loaded.

---

## 19. Implementation milestones

### Milestone 1 — common engine

Implement:

```text
Geometry and latent mapping.
Artifact manifest.
Chunk A generation.
Baseline Chunk B generation.
Common metrics.
```

### Milestone 2 — generic target conditions

Implement:

```text
TargetAlignedCondition.
Layout insertion transform.
Direct single-frame condition.
Re-encoded single-frame condition.
Stock-position diagnostic.
```

Run the `minimal` suite.

### Milestone 3 — full overlap

Implement:

```text
Direct T=7 overlap condition.
Latent-overlap metrics.
Aligned suite.
```

### Milestone 4 — Qwen-dependent strategies

Implement:

```text
Prompt policies.
Generated Video 2.
Composite source.
Grouped dynamic VAE and Qwen preprocessing.
```

Run the `reference` and `prompt` suites.

### Milestone 5 — sampler interventions

Implement:

```text
Sigma-correct overlap clamping.
Clamp-specific fixed noise.
Clamp suite.
```

### Milestone 6 — production decision

Use the results to choose the production carry mode and then simplify the production node around that mode. The production node does not need to expose every experimental strategy.

---

## 20. First GPU run

The first actual run should remain small in experimental scope:

```text
Suite: minimal
Profile: 73/22/51
Prompt: unchanged
Experiments:
    baseline_none
    frame_reencode_corrected
    frame_direct_corrected
    frame_direct_stock_position
```

If direct and re-encoded single-frame conditions are both ignored, test the prompted variant before implementing full-overlap conditioning.

If either single-frame condition works, implement and run:

```text
aligned_overlap_direct
```

before investing in Qwen-visible or sampler-clamped alternatives.
