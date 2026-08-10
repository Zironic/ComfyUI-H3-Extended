# H3 Adaptive Temporal RES V5 Plan

## Outcome

Build an experimental, video-driven RES controller that answers two separate
questions from genuine H3 evaluations:

1. Has a coarse, coherent timeline been established across the whole video?
2. After it has been established, how large can the next RES interval be before
   the predicted temporal organization changes unexpectedly?

V5 must not treat numerical smoothness, low motion, or spatial sharpness as
proof that a timeline exists. It must not encode a Beta schedule, a dense tail,
or a content-dependent motion penalty. It may use a fixed three-anchor bootstrap
because three genuine anchors are the minimum history needed by the measured
transition and interval controller.

The first deliverable is an offline descriptor study. Sampling behavior must
not change until the descriptor separates the known successful and failed
cases.

## Verified starting point

- The complete 20-step latent capture shows the first prediction as a blob,
  broad scene layout at the second prediction, and recognizable coarse geometry
  across sampled positions at the third prediction. Later predictions mostly
  refine that geometry.
- The current embedded controller takes only one fixed interval before it can
  grow. The failed aggressive run therefore jumped before the third genuine
  anchor.
- A sharp endpoint can still belong to a temporally incoherent trajectory.
  Whole-video RMS and the current embedded RES correction are not sufficient
  quality signals.
- Continuous eta-zero RES and genuine-anchor accounting already exist. V5 must
  preserve those contracts and keep V1-V4 selectable as characterization
  controls.
- Only one complete latent trajectory is currently available for descriptor
  calibration. The failed capture contains no callback tensors. Thresholds and
  presets therefore cannot be selected yet.

## Non-goals

V5 does not initially attempt to:

- predict or synthesize missing H3 outputs;
- use raw motion magnitude as an ordinary step-size penalty;
- declare a visual-quality guarantee from a latent metric;
- control ordinary spacing from audio;
- publish conservative, balanced, or aggressive presets before calibration;
- force accepted output when the evaluation budget cannot complete safely;
- replace or silently change any existing adaptive controller.

The descriptor reports temporal readiness and reconfiguration. Terms such as
`safe` and `corrupt` are reserved for results established by matched output
review, not inferred from the descriptor alone.

## Research gates

Implementation is divided by evidence gates rather than by file count.

### Gate A: descriptor validity

The offline descriptor must:

1. distinguish the featureless first prediction from the structured third
   prediction in the complete 20-step capture;
2. show that readiness covers the entire timeline rather than only early
   frames;
3. remain stable while ordinary spatial refinement continues;
4. flag the damaging early jump at or immediately after its endpoint;
5. avoid flagging the clean accelerated trajectory at comparable endpoints;
6. remain finite and meaningful for nearly static timelines.

No online lock thresholds are selected until all six can be evaluated on
captured data.

### Gate B: shadow validity

On an unchanged full-20 RES schedule, the controller must:

- leave every sigma and model call unchanged;
- acquire readiness no earlier than the first timeline-wide structured anchor;
- produce proposed intervals and endpoint decisions only as diagnostics;
- avoid false hard rejections on the dense reference;
- report measurable controller time and memory overhead.

### Gate C: lock-gated spacing

With endpoint rejection still disabled:

- anchors at source coordinates 0, 1, and 2 remain genuine and dense;
- growth cannot occur until readiness and temporal stability pass;
- the known early-jump failure is not reproduced;
- after lock, intervals are continuous in transformed time and do not select
  from interior source-grid coordinates.

### Gate D: enforced rejection

Rollback is enabled only after shadow diagnostics identify a damaging trial
without rejecting the clean controls. The first success case must show one
large trial being rejected, retried at a shorter interval, and completed without
the corresponding temporal failure.

## Data required before online control

Capture matched `x0` trajectories for the same prompt, seed, model, schedule,
conditioning mode, and latent dimensions:

1. full-20 RES reference;
2. clean accelerated RES trajectory;
3. failed jump-after-anchor-1 trajectory;
4. at least one low-motion and one high-motion case.

The failed and clean cases are mandatory for Gate A. A descriptor calibrated
only on the successful full-20 trajectory is exploratory and cannot control
sampling.

Add an offline command:

```text
benchmarks/h3_temporal_descriptor_probe.py
```

It reads latent-capture manifests and safetensors without loading H3. It writes
deterministic JSON/CSV plus optional plots and descriptor matrices. Optional
TAEH3 strips may help interpret the measurements, but production control uses
only latent features.

## Target-video extraction

Use the existing packed AV layout metadata to extract the video `x0` stream as:

```text
[B, C, T, H, W]
```

Do not alter packed AV state, RES integration, or audio layout. Audio remains in
the packed solver state and is recorded separately for diagnostics.

## Descriptor design

The descriptor has two independent parts. Timeline lock requires both.

### 1. Spatial readiness and coverage

This answers whether recognizable coarse structure exists throughout the
timeline. It must not subtract away structure shared by every frame.

Spatially pool every latent frame to a small grid, initially `8 x 8`, while
retaining all latent channels and all temporal positions:

```text
pooled_frames: [T, C, 8, 8]
```

The offline probe should compare scale-normalized candidates such as:

- within-frame coarse spatial variance;
- coarse spatial gradient energy;
- low-frequency spatial energy ratio;
- feature dispersion across the pooled grid;
- anchor-to-anchor change of those measurements.

Record each value per temporal position. Summarize at least:

```text
p10, p50, p90, minimum, coverage_fraction
```

The lock signal uses a low percentile or explicit coverage fraction. A strong
front half must not hide an unformed back half.

The exact readiness statistic and threshold are outputs of Gate A, not design
constants guessed in advance.

### 2. Temporal organization and stability

From the pooled frames, retain centered and uncentered forms. Compute:

```text
first temporal differences
second temporal differences
frame self-similarity matrix
motion self-similarity matrix
cross-anchor frame correspondence
cross-anchor motion correspondence
per-temporal-position anchor change
```

Compare consecutive genuine anchors using:

- frame-SSM change;
- motion-SSM change;
- same-position correspondence;
- distant-position competitor score;
- alignment margin;
- temporal warp or displacement;
- p50, p90, p95, and maximum localized temporal change.

Whole-video means are diagnostic only. Control uses a robust upper quantile so
a localized timeline break cannot disappear into the other frames.

### Low-motion observability

Cosine motion correspondence is undefined or noisy when temporal-difference
energy approaches zero. Every correspondence value therefore carries an
observability flag.

- Mask temporal edges below a calibrated energy floor.
- Do not include unobservable edges in motion alignment or warp.
- Fall back to uncentered frame-feature correspondence when motion is
  unobservable.
- Permit a static structured timeline to lock through high spatial readiness
  plus low frame change.
- Do not permit a spatially featureless static blob to lock.

No descriptor path may create NaNs or turn missing observability into a passing
score.

## Timeline state machine

Use:

```text
BOOTSTRAP -> UNLOCKED -> PROVISIONAL -> LOCKED
                               ^          |
                               |          v
                             RECOVERY <---+
```

### Bootstrap

Always evaluate genuine anchors at source coordinates 0, 1, and 2. These are
three anchors but only two protected intervals:

```text
0 -> 1    base interval
1 -> 2    base interval
2 -> ?    first interval eligible for growth
```

This minimum is explicit and versioned. It is not described as a learned phase
boundary.

### Unlocked

Readiness or stability is not yet established. Use the base interval and forbid
growth. Measurements continue at every genuine anchor.

### Provisional

Spatial readiness passes across the timeline and one observable anchor
transition is stable. Permit at most modest growth, initially `1.5x` relative
to the previous accepted interval.

### Locked

Readiness passes and two consecutive observable transitions are stable. Permit
the interval solver to grow, hold, or contract within versioned limits.

The earliest possible lock is at anchor 2 because anchors 0, 1, and 2 provide
two measured transitions.

### Recovery

A large endpoint reconfiguration after lock enters recovery. Retry policy is
handled transactionally. After a conservative accepted interval, return through
`PROVISIONAL`; do not jump directly to `LOCKED`.

## Interval selection after lock

Use transformed time:

```text
t = -log(sigma)
t_next = t + h
sigma_next = exp(-t_next)
```

After bootstrap, proposals must not depend on:

- the containing source-grid interval;
- a source-schedule index;
- the nearest source sigma;
- fixed tail anchors.

The source schedule remains relevant only for the initial base interval, the
final positive sigma, and reporting scheduler-equivalent coordinates.

### Temporally structured embedded defect

Retain the current eta-zero RES correction basis, but measure video `x0`
differences in three domains:

```text
raw video x0
first temporal differences
second temporal differences
```

For each domain, compute the h-independent basis once at the current anchor.
Use a scalar monotonic RES coefficient during interval solving. Localized
temporal bases use a robust upper quantile rather than global RMS.

The controlling normalized defect is the maximum of the enabled domains. The
diagnostics record every component, the selected component, the tolerance
solution, and every active clamp.

This is a proposal signal. It is not an endpoint-quality verdict.

### Interval limits

Keep distinct concepts distinct:

- `growth_limit`: maximum ratio to the previous accepted `h`;
- `absolute_max_h`: fixed maximum transformed-time interval;
- `remaining_h`: distance to the final positive sigma;
- `base_h`: initial dense interval used while unlocked.

Do not derive `absolute_max_h` as `initial_h * 4`. The initial H3 interval is
far smaller than accepted late intervals in the characterized schedules. Gate A
and shadow data must select a versioned `absolute_max_h` directly in transformed
time. It is not exposed as a normal UI control until its useful range is known.

## Shadow endpoint validation

At each candidate endpoint, compare its descriptor with accepted history.
Initially this is diagnostic only:

```text
current accepted anchor
    -> compute candidate x with RES
    -> evaluate H3 once at candidate sigma
    -> extract candidate descriptor
    -> record accept/reject recommendation
    -> accept unconditionally in shadow mode
```

Candidate measurements include:

- readiness coverage and its drop;
- frame- and motion-SSM change;
- observable alignment margin and warp;
- p95 first- and second-difference change;
- optional compact-descriptor prediction residual.

A recommendation requires either one calibrated hard condition or two
calibrated moderate conditions. Non-finite packed state or H3 output is always
a hard failure.

The validator reports `large temporal reconfiguration`, not `corruption`, until
matched media review establishes the relationship.

## Transactional RES and enforced retry

Only Gate D requires transactional solver changes.

Extend `IncrementalRES` with:

```python
candidate(x, sigma, denoised, sigma_next) -> x_next
commit(sigma, denoised, sigma_next) -> None
snapshot() -> RESState
restore(state) -> None
```

`candidate()` must not mutate solver history. Preserve the existing `step()`
API as `candidate()` followed by `commit()` so V1-V4 retain their call contract
and parity.

The enforced loop is:

```text
current = evaluated accepted anchor

while current.sigma > final_positive_sigma:
    proposed_h = controller.propose(current)
    candidate_x = stepper.candidate(current, proposed_h)
    candidate_x0 = H3(candidate_x, candidate_sigma)
    true_nfe += 1

    if validation accepts:
        stepper.commit(current interval)
        current = already-evaluated candidate
    else:
        leave current and all accepted histories unchanged
        retry with a shorter h
```

Rejected candidates never enter:

- RES history;
- accepted descriptor history;
- accepted sigma sequence;
- ordinary preview callbacks.

Accepted candidates are carried forward without another H3 evaluation.

The first enforcement version permits one retry at `0.5h`. Two retries and
error-scaled retry sizing are later characterization changes, not initial
requirements.

If a temporal recommendation still fails at the base interval, accept the
finite dense endpoint into `RECOVERY` unless the state is non-finite. The dense
trajectory is the fallback reference; a hard numerical failure aborts.

## NFE budget and finite completion

`true_nfe` counts every H3 call:

```text
initial anchor
accepted trial
rejected trial
retry
final positive anchor
```

Accepted anchors are never reported as NFE when rejected calls occurred.

The characterization controller uses a hard call budget only as a safety
abort. It must reserve the final positive evaluation. If the remaining budget
cannot reach terminal without violating the current safety contract, stop with
a clear error and complete diagnostics. Do not force an enormous final jump to
manufacture an output under the limit.

The initial enforced experiment may allow more than 20 total calls because a
rejected trial is paid evidence. A quality comparison must report both accepted
anchor count and true NFE.

## Terminal and audio behavior

Clip the last positive proposal to `source_sigmas[-2]`, evaluate H3 there once
if necessary, then apply the existing terminal-zero rule. This is numerical
terminal handling, not a dense tail.

Video controls ordinary spacing. Audio remains part of packed RES and records:

- raw embedded defect;
- x0 change;
- non-finite status;
- extreme discontinuity diagnostics.

Only non-finite or invalid packed audio may veto ordinary execution initially.
No audio maximum is combined into the normal video controller.

## Diagnostics and fingerprints

Version and fingerprint:

```text
controller definition
descriptor definition
pool size
readiness statistic and thresholds
stability thresholds
observability floor
interval defect definitions and tolerances
growth and absolute h limits
retry policy
budget behavior
source sigma hash
terminal positive sigma
```

Per accepted anchor, record:

```text
accepted anchor index and total model-call index
sigma, transformed t, accepted h, previous h
timeline state and state transition
readiness p10/p50/coverage
motion observability coverage
SSM changes, alignment margin, warp
localized p50/p90/p95/max changes
raw/first/second embedded defects
controlling defect and active clamp
```

Per attempted endpoint, record:

```text
parent accepted anchor
trial and model-call indices
proposed sigma and h
candidate descriptor metrics
shadow/enforced recommendation
reasons, retry h, and final disposition
```

The run summary includes accepted and attempted sigma sequences, accepted
anchors, true NFE, rejected calls, lock coordinate, lock losses, largest
accepted h, model time, controller time, and wall time.

## User-facing node

Do not add a node during the offline descriptor phase.

After Gate B, add a separate node only if online shadow or enforcement is ready:

```text
MiniMax H3 Adaptive Temporal RES Sampler (Zi)
```

Initial visible controls should be limited to controls that alter execution:

```text
experiment mode:
    full_20_shadow
    adaptive_shadow
    adaptive_enforce
diagnostics: summary | full
```

`full_20_shadow` preserves the supplied full schedule while recording lock,
interval, and endpoint recommendations. `adaptive_shadow` changes intervals
after lock but never rejects an evaluated endpoint. `adaptive_enforce` also
permits calibrated rejection and retry. This single mode avoids exposing an
endpoint-validation control that is inactive on the full-20 path.

Research-only limits may appear under advanced inputs once calibrated:

```text
absolute maximum h
maximum growth ratio
maximum total NFE
maximum retries
```

Do not expose quality presets until multiple conditioning modes and motion
classes establish meaningful threshold sets. The `full_20_shadow` description
must state clearly that it preserves the full schedule and changes diagnostics
only.

## Implementation ownership

Use the narrowest owners:

```text
benchmarks/h3_temporal_descriptor_probe.py
    offline capture analysis and reports

h3_vector_accel/temporal_descriptor.py
    descriptor extraction and comparisons

h3_vector_accel/adaptive_temporal_res.py
    lock state, interval decisions, endpoint validation

h3_vector_accel/adaptive_res.py
    transactional IncrementalRES extension while preserving step()

h3_vector_accel/sampler.py
    dedicated V5 shadow/enforced loop and callback boundary

h3_vector_accel/diagnostics.py
    existing run-owned persistence extended only as needed
```

Do not add a separate diagnostics subsystem unless the existing run diagnostics
cannot represent attempted and accepted records cleanly.

## CPU tests

### Descriptor tests

- Spatially constant featureless frames: readiness fails; no lock.
- Structured identical frames: readiness passes, motion is unobservable, the
  static fallback can lock without NaNs.
- Coherent translation: temporal correspondence remains aligned.
- Temporal permutation with the same individual frames: warp or SSM change
  rises despite similar global RMS.
- Localized temporal break: p95 detects it when the mean remains small.
- Front-only structure: timeline coverage fails.
- Scale multiplication: normalized descriptor decisions remain unchanged
  within tolerance.

### State tests

- Exactly anchors 0, 1, and 2 are required before growth.
- Featureless stability never locks.
- One passing transition is provisional; two are locked.
- Unobservable motion uses the static frame path.
- Severe post-lock reconfiguration enters recovery.

### RES transaction tests

- `step()` retains full and irregular stock RES parity.
- `candidate()` does not mutate history.
- `candidate()` plus `commit()` equals `step()`.
- A rejected trial leaves x, sigma, denoised, RES history, descriptor history,
  and accepted schedule unchanged.
- An accepted trial endpoint is reused without a duplicate H3 call.

### Sampler tests

- Shadow mode produces the exact full-20 schedule and call count.
- Enforced accepted calls count once.
- Rejected calls and retries each increment true NFE.
- Rejected trials do not emit ordinary callbacks.
- The final positive sigma and terminal zero are included once.
- Budget exhaustion aborts instead of forcing an unsafe terminal jump.
- Changing unused source-grid interior points does not change post-bootstrap
  proposals for identical controller inputs.

## Runtime characterization

GPU/model runs require explicit authorization and report:

- descriptor/controller time outside model-call time;
- peak and retained VRAM attributable to descriptor history;
- accepted anchors and true NFE;
- exact attempted and accepted schedules;
- conditioning mode, dimensions, duration, prompt, seed, and model identity.

Evaluate T2VA, FL2VA, and Ref2VA separately. Measure motion magnitude and motion
complexity independently from the full-20 control output. They are study
variables, not controller inputs.

Quality review prioritizes timeline coherence, identity continuity, broad motion
and camera trajectory, reference adherence, and temporal discontinuities.
Same-seed PSNR is supporting evidence, not the sole criterion.

## Milestones

### Milestone 1: offline probe

Implement descriptor candidates and reports. Analyze the complete full-20
capture. Collect the missing clean and failed trajectories. Select no online
threshold until Gate A passes.

### Milestone 2: descriptor library

Move only successful measurements into `temporal_descriptor.py`. Add the
synthetic descriptor and observability tests. Measure overhead independently.

### Milestone 3: full-20 shadow

Add the lock state and interval proposal without changing the schedule. Record
shadow endpoint recommendations. Gate B must pass.

### Milestone 4: three-anchor lock-gated spacing

Enable continuous proposals after lock, preserve unconditional endpoints, and
keep validation shadow-only. Gate C must pass on the known difficult case.

### Milestone 5: transactional RES

Add pure candidate and explicit commit operations while preserving `step()`.
Prove accepted-only parity and retry-state restoration on CPU.

### Milestone 6: one-retry enforcement

Enable one `0.5h` retry after a calibrated endpoint recommendation. Demonstrate
Gate D and exact NFE accounting.

### Milestone 7: broader characterization

Compare full-20 RES, fixed Beta at comparable NFE, clean V4, V5 shadow, and V5
enforced across motion and conditioning classes. Determine whether temporal
complexity predicts NFE better than raw motion magnitude.

### Milestone 8: presets

Only after reproducible characterization, publish versioned quality presets and
their tested scope. Until then, V5 remains explicitly experimental.

## Success criteria

V5 is technically successful when:

1. its readiness signal rejects a stable featureless timeline and passes a
   structured timeline across all temporal regions;
2. the known damaging jump is forbidden before anchor 2 without protecting a
   larger fixed prefix;
3. post-bootstrap coordinates are continuous and independent of interior source
   grid points;
4. clean and difficult generations can acquire lock at different anchors;
5. an evidenced damaging trial can be rejected and retried without mutating
   accepted solver state;
6. accepted endpoint evaluations are never duplicated;
7. true NFE includes rejected trials;
8. budget exhaustion fails clearly instead of silently forcing an output;
9. existing V1-V4 behavior and RES parity remain unchanged;
10. controller overhead is measured and small relative to H3 model time.

The central rule is:

> Acceleration begins only when coarse structure exists across the timeline and
> that temporal organization has stabilized. Larger intervals are proposed
> numerically, validated against temporal reconfiguration, and retried only
> after the validation signal has proved useful in shadow data.
