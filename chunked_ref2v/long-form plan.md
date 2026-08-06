# MiniMax H3 Long-Form Chunked Ref2V — Streaming I/O Implementation Plan

## 0. Status and scope

**Status:** design, pre-implementation.

**Purpose:** extend the existing MiniMax H3 chunked Ref2V work from controlled two-chunk experiments to videos lasting several minutes or longer without:

* decoding the complete source into one Comfy `IMAGE` batch;
* retaining every source chunk, conditioning tensor, sampled latent, or decoded frame in RAM;
* switching VAE, Qwen, and DiT residency once per chunk;
* concatenating thousands of images inside the Comfy graph;
* losing a multi-hour run because final video assembly failed;
* requiring a large graph of loader, slicing, batching, stitching, and video-combine nodes.

The current harness accepts the entire source as an `IMAGE` input and Phase A slices its two source chunks from that already-materialized tensor.   That remains appropriate for the two-chunk strategy laboratory, but it is not the production input model.

The production system remains one high-level workflow node. Internally it uses multiple explicit phases and persistent run artifacts.

---

## 1. Primary design decision

The long-form implementation is **not** a fork of VideoHelperSuite and is **not** a generic video loader followed by a generic video-combine node.

It owns two narrow FFmpeg-backed components:

1. `FFmpegFrameSource`: sequential, canvas-pinned source decoding with bounded buffering.
2. `SegmentWriter`: finalized-frame encoding into resumable video segments.

The complete execution path is:

```text
source file
    ↓
Pass A: sequential source decode + VAE reference preprocessing
    ↓
Pass B: disk-backed Qwen conditioning preprocessing
    ↓
Pass C: sequential DiT-only chunk sampling
    ↓
Pass D: sequential VAE-only decode + seam resolution + segment writing
    ↓
Pass E: segment concatenation + source-audio mux
```

No phase requires memory proportional to the total video duration.

### Critical optimization

For the direct latent-frame and direct latent-overlap carry strategies, the next chunk depends on the previous chunk’s **sampled latent**, not its decoded pixels.

Therefore sampling and decoding should be separated:

```text
Pass C:
sample chunk 0
→ save latent
→ derive latent carry
→ sample chunk 1
→ save latent
→ repeat

Pass D:
load one saved latent
→ decode
→ stitch
→ encode finalized frames
→ free decoded frames
→ repeat
```

This avoids alternating DiT and VAE residency hundreds of times. Pixel-derived carry strategies remain supported by the experimental harness but are not part of the first long-form shipping path.

---

## 2. Goals and non-goals

### Goals

1. Process source videos of at least 30 minutes without loading the complete source into RAM.
2. Keep source-frame buffering bounded to approximately one model chunk plus modest read-ahead.
3. Keep output-frame buffering bounded to one decoded chunk plus one overlap tail.
4. Keep only one chunk’s source latent and conditioning resident during sampling.
5. Persist every expensive phase result atomically.
6. Resume after interruption without regenerating completed chunks.
7. Reassemble or re-encode a completed sampling run without running the DiT again.
8. Preserve exact frame accounting at 24 fps.
9. Preserve or remux source audio independently from generated video audio.
10. Return only bounded previews and file paths to Comfy.

### Non-goals for the first implementation

* Generated-audio overlap stitching.
* Dynamic Qwen presentation based on each generated overlap.
* Composite-source carry requiring VAE and Qwen work between every sampled chunk.
* Arbitrary frame-rate model execution; H3 continues to operate at 24 fps.
* HDR preservation. HDR inputs should initially require an explicit tone-mapping policy or fail preflight.
* Frame-perfect random access into every possible variable-frame-rate source.
* Generic replacement of VHS loaders for other Comfy workflows.

---

## 3. Production package layout

```text
chunked_ref2v/
├── geometry.py                 existing frame/latent mapping
├── layout_ops.py               existing target-condition insertion
├── model_patch.py              existing reversible extra_conds patch
├── strategies.py               shared carry preparation
├── harness.py                  existing two-chunk experiment engine
├── nodes.py                    existing experiment node
│
└── longform/
    ├── __init__.py
    ├── config.py               immutable long-form run configuration
    ├── metadata.py             ffprobe parsing and source identity
    ├── frame_source.py         FFmpegFrameSource and TensorFrameSource
    ├── audio_source.py         sequential PCM windows and frame/sample mapping
    ├── chunk_stream.py         bounded overlapping source-chunk iterator
    ├── precompute.py           Pass A VAE preprocessing
    ├── conditioning.py         Pass B Qwen preprocessing and disk loading
    ├── sampling.py             Pass C DiT-only sequential sampling
    ├── decode.py               sequential latent decode to bounded CPU buffers
    ├── seam.py                 seam scoring and exact frame finalization
    ├── writer.py               segment writer backends
    ├── mux.py                  segment concat and source-audio mux
    ├── manifest.py             immutable config, mutable state, event journal
    ├── runner.py               phase orchestration and resume
    ├── nodes.py                user-facing production and recovery nodes
    └── cli.py                  I/O-only and recovery testing outside Comfy
```

The existing harness remains the place where carry strategies are compared. Shared geometry, target-aligned condition types, and model patches should be refactored only enough for both engines to use them.

---

## 4. Node interface

### 4.1 Main node

```text
MiniMax H3 Long-Form Ref2V (Zi)
```

#### Required inputs

```text
model
clip
video_vae
audio_vae
video_path
prompt
sampler
sigmas
seed
```

#### Source controls

```text
start_seconds=0
duration_seconds=0          # 0 means through EOF
ffmpeg_location=""          # PATH when blank
source_audio=true
hdr_policy=error
```

#### Chunk controls

```text
chunk_frames=141 | 124 | 73 | auto
overlap_frames=22
carry_strategy=direct_latent_overlap
seed_mode=fixed | hashed
final_padding=repeat_last
```

#### Canvas controls

```text
width=0
height=0
canvas_policy=source
```

`0,0` derives one canvas from source metadata and pins every source chunk, target latent, and static reference to that canvas. The existing reference builder already treats canvas pinning as necessary for latent compatibility and to avoid an unnecessary sequence-length increase.

#### Execution controls

```text
operation=full | precompute | sample | assemble | resume
run_directory=""
conditioning_cache=auto
preview_interval_chunks=0
keep_artifacts=all | samples | final
continue_after_decode_failure=false
```

#### Output controls

```text
container=mkv | mp4
video_codec
codec_options
audio_mode=preserve_source | none
segment_backend=per_chunk | persistent
```

#### Outputs

```text
preview                 bounded IMAGE clip
final_video_path        STRING
run_directory           STRING
report                   STRING
```

The node must not return the complete decoded video as an `IMAGE` batch. The existing comparison code already caps preview dimensions because Comfy `IMAGE` outputs are float32 and can consume gigabytes.

### 4.2 Recovery node

```text
MiniMax H3 Assemble Long-Form Run (Zi)
```

Inputs:

```text
video_vae
run_directory
output settings
```

This node reads saved sampled latents, performs Pass D and Pass E, and never loads the DiT or Qwen. It allows seam parameters or encoding settings to be changed without resampling.

---

## 5. Source metadata and identity

### 5.1 FFmpeg resolution

Resolve executables in this order:

1. explicit `ffmpeg_location`;
2. `shutil.which("ffmpeg")` and `shutil.which("ffprobe")`;
3. clear preflight failure.

Use subprocess argument arrays, never shell-formatted command strings.

### 5.2 Metadata

`ffprobe` should collect:

```text
video codec
coded width and height
display rotation
nominal and average frame rates
duration estimate
pixel format
color primaries, transfer and matrix
audio stream index, codec, channel count and sample rate
```

The duration and frame count reported by the container are estimates. The authoritative normalized frame count is the number emitted by `FFmpegFrameSource`.

### 5.3 Source identity

The current harness hashes the supplied frame tensor and notes that class-name-level model provenance is insufficient for safe automatic reuse.

The long-form identity should include:

```text
source file content hash
selected source stream
start and duration
complete FFmpeg normalization arguments
normalized fps
canvas
C/O/S profile
prompt hash
static-reference hashes
sampler and sigma hashes
seed policy
checkpoint/model fingerprint when available
VAE fingerprint when available
attention and activation-memory patch configuration
carry strategy
```

Resume with incomplete model provenance should require an explicit override rather than silently accepting a possible mismatch.

---

## 6. Bounded source decoding

### 6.1 FFmpegFrameSource

```python
@dataclass(frozen=True)
class VideoMetadata:
    source_width: int
    source_height: int
    canvas_width: int
    canvas_height: int
    normalized_fps: int
    estimated_frames: int | None
    has_audio: bool
    source_identity: str


class FFmpegFrameSource:
    def open(self) -> VideoMetadata: ...
    def read_frames(self, count: int) -> torch.Tensor: ...
    def close(self) -> None: ...
```

`read_frames` returns:

```text
CPU uint8
shape [frames, height, width, 3]
RGB
already resized/cropped to the pinned target canvas
```

Frames remain uint8 until the active chunk is handed to the VAE or Qwen preprocessing code.

### 6.2 Decoder process

Conceptual decoder pipeline:

```text
source stream
→ optional time trim
→ autorotation
→ deterministic 24 fps normalization
→ pinned canvas scale/crop
→ SDR conversion when explicitly enabled
→ rgb24 rawvideo
→ stdout
```

Implementation requirements:

* `-nostdin`;
* restricted FFmpeg logging;
* stderr drained continuously so the process cannot block on a full pipe;
* exact byte count checked for every frame;
* truncated partial frames treated as decoder failure;
* process termination in `finally`;
* bounded read-ahead of no more than one stride by default.

### 6.3 Overlapping chunk iterator

```python
@dataclass(frozen=True)
class SourceChunk:
    index: int
    global_start: int
    model_frames: int
    actual_frames: int
    frames_u8: torch.Tensor
    is_final: bool
```

Algorithm:

```text
read C frames
yield chunk 0

for each later chunk:
    retain the final O source frames
    read S = C - O new frames
    concatenate retained O + new S
    yield next chunk
```

At EOF:

* record the number of real frames;
* repeat the final real frame to reach `C`;
* mark `actual_frames`;
* discard generated padding later.

Only `C + optional_read_ahead` source frames are resident.

### 6.4 TensorFrameSource

Retain a small `TensorFrameSource` implementation for:

* unit tests;
* current two-chunk harness compatibility;
* synthetic test clips;
* short workflows already represented as `IMAGE`.

The production node should not convert a path source into `TensorFrameSource`.

---

## 7. Audio input

### 7.1 Conditioning audio

A separate FFmpeg process emits PCM at the audio VAE’s required sample rate.

Chunk boundaries use frame-derived sample positions:

```python
start_sample = round(global_start * sample_rate / 24)
stop_sample = round((global_start + model_frames) * sample_rate / 24)
```

This avoids accumulating a fractional-sample error across chunks.

`PcmSource` uses the same overlap principle as the video source and retains only the samples required by the next chunk.

### 7.2 Final audio

Generated audio is discarded in the first production implementation.

After video segments are concatenated, Pass E extracts the selected source-audio interval, trims it to:

```text
final_video_frames / 24
```

and muxes it into the final container.

Video should be stream-copied during this final mux. Audio may be re-encoded once when exact trimming or container compatibility requires it.

---

## 8. Pass A — source VAE preprocessing

### Purpose

Decode the source once while the video and audio VAEs are resident. Persist everything later phases need without keeping the full source in RAM.

### Per-chunk work

For every `SourceChunk`:

1. Convert only that chunk from uint8 to the VAE’s expected float representation.
2. Encode the source reference video latent.
3. Encode the corresponding source audio latent when enabled.
4. Select and save the low-rate Qwen presentation frames.
5. Save metadata and checksums.
6. Release float frames and GPU tensors.
7. Retain only the uint8 overlap required by the source iterator.

Qwen currently sees reference video at 2 fps, so only that sampled presentation needs to survive for Pass B rather than the complete decoded source chunk.

### Per-chunk artifact

```text
precompute/chunk_000123.safetensors
    source_video_latent
    source_audio_latent              optional

precompute/chunk_000123_qwen.safetensors
    sampled_rgb_frames               uint8 where supported
    timestamps

precompute/chunk_000123.json
    global_start
    actual_frames
    model_frames
    padding_frames
    checksums
```

Static image references are encoded once and stored separately.

### Memory invariant

After each chunk completes, Pass A retains only:

```text
source overlap ring
FFmpeg process state
static-reference state
one active VAE chunk
small metadata
```

---

## 9. Pass B — Qwen conditioning preprocessing

### Purpose

Load Qwen once, encode every chunk presentation, and store conditionings on disk.

### Operation

For each precomputed chunk:

1. Load its low-rate Qwen frames and metadata.
2. Reconstruct its reference-item presentation.
3. Apply the chunk prompt policy.
4. Use the existing conditioning cache.
5. Save or index the resulting conditioning.
6. Release it before loading the next chunk.

Conditionings should not be accumulated in a Python list. Multi-minute videos can produce enough conditioning data for disk storage to be preferable.

### Artifact

```text
conditioning/chunk_000123.safetensors
```

or a validated pointer into the existing conditioning cache.

### Prompt policy

The MVP uses the same prompt for every chunk. Timeline-aware prompt segmentation belongs after the basic long-form path is reliable.

---

## 10. Pass C — sequential DiT-only sampling

### Purpose

Generate every target chunk while keeping the DiT resident and avoiding VAE or Qwen model swaps.

### Supported carry strategies

Initial production support:

```text
direct_latent_frame
direct_latent_overlap
```

Recommended default:

```text
direct_latent_overlap
```

The geometry validator must reject any profile where the chosen overlap does not map exactly to complete latent positions.

### Per-chunk loop

```text
load source reference latent
load conditioning
load previous carry latent, except chunk 0
build target-aligned conditions
sample target latent
save sampled latent atomically
derive and retain next latent carry
release all other per-chunk tensors
update manifest
```

No output pixels are decoded in the ordinary sampling pass.

### Sample artifact

```text
samples/chunk_000123.safetensors
    video_latent
    audio_latent           optional diagnostic only
```

Store the sampler’s native output dtype unless a separately validated storage conversion is selected.

### Resume

To resume sampling:

1. Validate run identity.
2. Find the highest contiguous completed sample.
3. Load its saved output latent.
4. derive the carry slice;
5. continue from the next chunk.

A corrupt or missing middle chunk invalidates that chunk and every later sampled chunk because carry state is sequential.

### Optional preview checkpoints

`preview_interval_chunks > 0` may pause sampling after every N chunks to decode a small diagnostic output. It is disabled by default because it reintroduces model-residency switching.

---

## 11. Pass D — decode, stitch, and segment writing

### Purpose

Unload the DiT, load the video VAE once, and process saved target latents sequentially.

### Decode lifecycle

For each sampled chunk:

1. Load one sampled latent.
2. Decode only that chunk.
3. Convert decoded pixels to CPU uint8 as early as practical.
4. Free the float/GPU decode result.
5. resolve the seam with the previous pending tail;
6. send finalized uint8 frames to `SegmentWriter`;
7. retain only the new pending tail;
8. update the assembly journal.

The implementation must never call the VAE with hundreds of chunk latents concatenated together.

### Future decode optimization

The first implementation may decode one complete model chunk at a time. A later optimization may temporally slab VAE decoding if one 124- or 141-frame decode produces excessive CPU or GPU peak memory. That optimization should not alter the stitcher or writer interfaces.

---

## 12. Seam controller

```python
class SeamController:
    def push_chunk(
        self,
        chunk_index: int,
        frames_u8: torch.Tensor,
        *,
        actual_frames: int,
        is_final: bool,
    ) -> FinalizedFrames:
        ...
```

State:

```text
pending_tail_u8
pending_chunk_index
frames_finalized
last_seam
```

### Non-final transition

For previous tail `A` and current chunk `B`:

```text
A = previous chunk frames [S:C]
B = current chunk frames [0:O]
```

Choose seam `k` and finalize:

```text
A[0:k]
+
B[k:S]
```

Retain:

```text
B[S:C]
```

The transition emits exactly `S` new global frames.

### Final chunk

Use `actual_frames` to discard model padding and emit every remaining real frame exactly once.

### Seam scoring

Default:

```text
appearance disagreement
+ edge disagreement
+ motion disagreement
```

Scoring runs on downscaled CPU frames.

The existing source-conditioning analysis indicates that the previous chunk’s tail may contain padding-contaminated source latent positions while the next chunk’s opening is fully real, so the search should retain its existing early-overlap bias.

No blend by default. Optional blending is gated on low correspondence error.

### Resume state

After every committed segment, save:

```text
pending tail
next sample index
frames finalized
seam history
segment history
```

The pending tail must be lossless.

---

## 13. SegmentWriter

### 13.1 Interface

```python
class SegmentWriter:
    def open(self, metadata, start_segment: int = 0) -> None: ...
    def write_segment(self, frames_u8, segment_index: int) -> SegmentRecord: ...
    def close(self) -> None: ...
```

The seam controller emits one finalized block per chunk transition. This naturally permits one independently committed segment per transition.

### 13.2 Correctness-first backend

The first backend launches FFmpeg once per finalized segment:

```text
raw RGB24 frames on stdin
→ configured final video codec
→ temporary segment file
→ close encoder
→ ffprobe validation
→ atomic rename
→ journal commit
```

Segment process startup is expected to be insignificant relative to H3 sampling, while this design provides straightforward crash recovery.

Each segment must have identical:

```text
codec
codec profile
pixel format
dimensions
frame rate
time base
color metadata
```

### 13.3 Persistent backend

After correctness is established, add a persistent FFmpeg segment-muxer backend. It retains one encoder process and creates numbered segments automatically.

This is an internal writer substitution. The seam controller and manifest format should not change.

### 13.4 Codec policy

The writer must support:

* a lossless intermediate codec;
* a near-lossless final codec;
* user-specified FFmpeg codec arguments.

The initial default should be selected after measuring disk throughput, segment size, decode cost, and concatenation reliability. It should not be hard-coded into the stitching logic.

### 13.5 No second video encode

Pass E concatenates validated segments with stream copy. The final audio mux also stream-copies video.

---

## 14. Pass E — concatenate and mux

1. Verify that committed segments cover exactly the expected number of video frames.
2. Generate an FFmpeg concat manifest.
3. Concatenate video segments with no video re-encode.
4. Extract and trim source audio.
5. Mux video and audio into a temporary final file.
6. Run `ffprobe` validation:

   * expected width and height;
   * 24 fps;
   * expected video frame count where available;
   * expected duration tolerance;
   * audio stream presence when requested.
7. Atomically rename to the final output path.
8. Mark the run complete.

A failed final mux does not invalidate sampled latents or video segments.

---

## 15. Run-directory format

```text
h3_long_ref2v/<run_id>/
├── manifest.json              immutable run identity and configuration
├── state.json                 atomic current phase and resume cursor
├── events.jsonl               append-only event journal
├── source/
│   ├── metadata.json
│   └── static_refs.safetensors
├── precompute/
│   ├── chunk_000000.safetensors
│   ├── chunk_000000_qwen.safetensors
│   └── ...
├── conditioning/
│   ├── chunk_000000.safetensors
│   └── ...
├── samples/
│   ├── chunk_000000.safetensors
│   └── ...
├── assembly/
│   ├── pending_tail.safetensors
│   ├── seams.jsonl
│   └── segments.txt
├── segments/
│   ├── segment_000000.mkv
│   └── ...
├── debug/
│   ├── thumbnails/
│   └── boundary_previews/
└── output/
    └── final_video.ext
```

### Persistence rules

* Tensor artifacts use temporary files followed by `os.replace`.
* JSON state uses atomic replacement.
* Events use append-only JSONL records.
* Segments use `.partial` names until validated.
* Every artifact has a checksum and schema version.
* A run-directory lock prevents concurrent writers.
* Stale locks require explicit recovery.

The existing `RunStore` already provides atomic text writes and atomic safetensors replacement; the long-form manifest should reuse those mechanics while separating immutable configuration, mutable state, and the event journal.

---

## 16. Artifact retention

```text
keep_artifacts=all
    Preserve precompute, conditioning, samples, segments and debug assets.

keep_artifacts=samples
    Delete source precompute and conditionings after successful sampling.
    Preserve samples and segments so assembly can be repeated.

keep_artifacts=final
    After final validation, preserve manifest, reports, final video and selected
    diagnostics. Delete recoverable intermediate assets.
```

For the initial multi-minute stress tests, use `all`.

A disk-space preflight should estimate:

```text
source reference latents
+ conditioning cache
+ sampled output latents
+ encoded segments
+ safety margin
```

and refuse to begin when the configured minimum free-space reserve would be violated.

---

## 17. Progress, diagnostics, and cancellation

### Per-phase metrics

Record:

```text
source decode frames/s
VAE source-encode time/chunk
Qwen time/chunk
DiT sample time/chunk
sample save time/chunk
VAE decode time/chunk
seam time/chunk
segment encode time/chunk
frames committed
disk bytes written
CPU RSS
GPU allocated
GPU reserved
physical GPU free
```

### Per-chunk record

```text
chunk index
global source start
actual/model frames
padding
seed
source artifact checksum
conditioning checksum
sample checksum
carry slice
sample duration
peak VRAM
decode duration
seam index and score
output segment
status
```

### Diagnostic previews

Return and save only:

* first boundary;
* periodic boundaries;
* worst-scoring boundaries;
* first versus current identity checkpoints;
* final boundary.

All previews remain dimension-capped.

### Cancellation

On cancellation or exception:

1. stop active FFmpeg processes;
2. close pipes;
3. preserve completed artifacts;
4. atomically write the current state;
5. remove incomplete temporary files;
6. release the run lock;
7. propagate the interruption.

---

## 18. Implementation stages

### Stage 0 — I/O-only proof

Implement:

```text
FFmpegFrameSource
overlapping chunk iterator
no-op seam controller
SegmentWriter
concat/mux
manifest
CLI passthrough test
```

Run a 30-minute source through:

```text
decode → chunk → immediately finalize original frames → encode → concat
```

No H3 models are loaded.

Gate:

* exact normalized frame count;
* bounded RAM;
* no accumulating process handles;
* output duration matches frame count;
* cancellation and resume work;
* source decoded once.

### Stage 1 — disk-backed source preprocessing

Implement Pass A with real video/audio VAEs.

Gate:

* process several minutes;
* source frame memory remains bounded;
* every chunk has correct global indexing and final padding;
* precompute resumes after forced interruption;
* Qwen presentation frames correspond to the correct source intervals.

### Stage 2 — disk-backed Qwen pass

Implement Pass B and integrate the existing conditioning cache.

Gate:

* Qwen loads once;
* conditionings are not accumulated in RAM;
* resume skips completed conditioning artifacts;
* prompt/reference identity changes invalidate affected cache entries.

### Stage 3 — DiT-only long-form sampling

Implement Pass C with `direct_latent_overlap`.

Gate:

* 10 or more chunks complete with one DiT residency phase;
* no VAE decode occurs during ordinary sampling;
* resume derives the correct carry from the last completed sample;
* GPU memory plateaus after allocator warmup;
* output latent count equals planned chunk count.

### Stage 4 — sequential assembly

Implement Pass D with one-latent-at-a-time VAE decode, seam selection, and correctness-first segment writing.

Gate:

* output contains exactly the requested number of frames;
* decoded-frame RAM does not grow with chunk count;
* forced interruption resumes from the last committed segment;
* assembly can be rerun with different seam settings without DiT work.

### Stage 5 — final audio and production node

Implement Pass E, the production Comfy node, bounded previews, reports, and cleanup policies.

Gate:

* source audio remains synchronized;
* final video passes ffprobe validation;
* complete run can be reproduced from its manifest;
* final mux failure is recoverable without generation.

### Stage 6 — performance optimization

After correctness:

* persistent segment-muxer writer;
* source VAE clip reuse where exact;
* temporal VAE decode slabs;
* limited source read-ahead;
* conditioning-store compression or lower-precision validation;
* automatic chunk-size selection using the active memory optimizer;
* periodic quality checkpoints;
* timeline-segmented prompts.

---

## 19. Test plan

### Unit tests

* FFmpeg path resolution.
* Metadata parsing.
* Exact RGB frame-byte parsing.
* C/O/S overlap iterator.
* EOF padding and trimming.
* Global-to-local frame mapping.
* 24 fps frame-to-audio-sample mapping.
* Source identity invalidation.
* Atomic manifest updates.
* Event-journal replay.
* Carry-slice recovery.
* Seam output-length invariants.
* Segment validation.
* Concat-list generation.
* Cleanup policies.

### Synthetic media tests

Generate videos with:

* an embedded frame number;
* 23.976 fps;
* 24 fps;
* 25 fps;
* 30 fps;
* variable frame timing;
* rotation metadata;
* mono and stereo audio;
* no audio;
* truncated containers.

Normalize each to 24 fps and verify that resumed output matches uninterrupted output.

### Fault-injection tests

Terminate the process:

* during source decode;
* during VAE encode;
* while writing conditioning;
* during sampling;
* after sample save but before state update;
* during VAE decode;
* during segment encoding;
* during final concat;
* during audio mux.

Every case must either resume safely or clearly identify the artifact that requires regeneration.

### Long-run dry test

Run at least 50,000 synthetic frames through the I/O-only path while recording:

```text
CPU RSS
open handles
temporary-file count
frames read
frames written
segment count
```

Memory and handle count should stabilize rather than scale with duration.

### GPU tests

* Existing two-chunk baseline.
* 10-chunk continuity run.
* 30–100 chunk run.
* Multi-minute stress run.
* Resume after a completed sample chunk.
* Reassembly with different seam settings.
* Memory plateau under the active H3 memory optimizer.

---

## 20. Acceptance criteria

1. The source is never represented as one complete Comfy `IMAGE` batch.
2. The final video is never represented as one complete Comfy `IMAGE` batch.
3. Input decoding memory is bounded by one source chunk plus read-ahead.
4. Sampling memory is bounded by one source reference, one conditioning, one target latent, and carry state.
5. Assembly memory is bounded by one decoded chunk and one overlap tail.
6. Sampling does not load or invoke the video VAE for latent-only carry strategies.
7. Source preprocessing decodes the input video once during an uninterrupted run.
8. Output contains exactly the normalized real-frame count requested.
9. Every expensive completed chunk survives interruption.
10. A completed sampling pass can be assembled without loading the DiT or Qwen.
11. Final video assembly performs no second lossy video encode.
12. Peak RAM and VRAM do not increase with total source duration after warmup.
13. Long-run logs identify the first chunk associated with continuity, memory, timing, or seam degradation.
14. The full production workflow remains one high-level node, with an optional recovery/assembly node rather than a graph containing hundreds of manual chunk operations.

---

## 21. First implementation target

The narrowest useful production configuration is:

```text
input:             local video path
fps:               normalized 24
carry:             direct latent overlap
chunk profile:     explicit C/O, exact latent alignment required
audio:             source conditioning + final source-audio preservation
Pass A:            streaming FFmpeg decode, disk-backed VAE preprocessing
Pass B:            disk-backed Qwen preprocessing
Pass C:            DiT-only sequential sampling
Pass D:            VAE-only sequential decoding and per-transition segments
Pass E:            stream-copy concat and source-audio mux
resume:            enabled after every completed chunk or segment
Comfy outputs:     bounded preview, report, run path, final video path
```

This is sufficient for a multi-minute no-cut stress test without introducing duration-proportional input RAM, output RAM, graph size, or model-residency churn.
