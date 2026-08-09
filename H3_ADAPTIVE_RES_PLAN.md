# H3 History-Controlled Adaptive RES Plan

## Outcome

Add a reproducible `res_multistep + adaptive_history_v1` experiment that uses
only genuine H3 evaluations. It must protect the known-sensitive prefix and
tail, choose intermediate sigma coordinates causally from trajectory changes
observed at paid anchors, preserve RES multistep state across those choices,
and report the resulting schedule and true NFE.

This tests whether H3 can select its own evaluation density. It does not use
velocity forecasts, synthetic denoised values, hidden error probes, rejected
steps, or rollback.

The named `adaptive_embedded_res_v1` profile is a separate video-controlled
experiment: after one bootstrap interval it uses the eta-zero IncrementalRES
embedded correction defect and bounded scalar bisection to select transformed
time. Audio disagreement remains diagnostic only, and the final positive
source sigma is mandatory before the existing Euler-to-zero transition.

## Evidence behind the change

- `res_multistep + late_aggressive_13` is perceptually equivalent to the tested
  20-NFE outputs, while an equal-NFE early-aggressive placement corrupted video.
- The current fixed sparse RES path already proves that stock RES accepts a
  supplied nonuniform schedule.
- Stock `sample_res_multistep` owns its complete schedule and keeps multistep
  history in local variables. Calling it once per adaptive interval would reset
  that history and is therefore not an adaptive RES implementation.

The result is evidence for evaluation placement on the tested generation, not
yet a general H3 quality claim.

## Version-one contract

### User-visible selection

- Keep the existing solver choices and widget order.
- Add `adaptive_history_v1` to the actual-evaluation schedule choices.
- Permit it only with `res_multistep`; incompatible combinations fail prompt
  validation with a clear error.
- Expose no uncalibrated controller knobs. The controller constants and version
  are fingerprinted so later tuning produces a new named version.

### RES integration

Extract the deterministic eta-zero RES update into a run-scoped incremental
stepper. Its state is the previous denoised prediction, previous sigma-down,
and previous evaluated sigma. The first interval and the terminal-zero interval
use the same Euler branches as stock RES; all other intervals use the same
log-sigma, `c2`, `phi1`, `phi2`, `b1`, and `b2` equations.

For any predetermined descending schedule, the incremental stepper must match
stock `sample_res_multistep` before it is used by the adaptive controller.

### Controller

- Use RES time `t = -log(sigma)` for interval arithmetic.
- Evaluate source indices 0 through 5 exactly.
- Protect source indices 17, 18, and 19 and the terminal-zero transition.
- Between those regions, propose continuous off-grid `t` coordinates using the
  local spacing of the original 20-step scheduler as the baseline.
- Maintain a dimensionless step scale from 1 to the configured maximum, which
  defaults to 3. Each low-change observation grows it by 1.5, each high-change
  or audio-emergency observation shrinks it by 0.7, and a moderate observation
  keeps it unchanged.
- Establish the reference trajectory-change rate from the last three
  protected-prefix intervals. Low and high bands compare later per-unit-t
  rates against that fixed, versioned reference so widening source intervals
  are not mistaken for increasing trajectory curvature.
- Calculate per-modality relative velocity change, relative x0 change,
  velocity-direction cosine, and their per-unit-t rates. Video drives normal
  decisions; audio is an emergency shrink signal.
- Clamp every proposal to strict descent, the protected-tail boundary, and a
  maximum of 20 genuine evaluations. Every proposed anchor is accepted and
  counted; there are no unreported probes.

### Observability

Every callback and full diagnostic anchor records the actual sigma, true NFE,
step scale, local base interval, proposed interval, video/audio trajectory
metrics, decision reason, and whether the anchor belongs to the protected
prefix or tail. V2 console and full diagnostics also expose separate video
velocity/x0 rates, their combined rate, the fixed reference, and their ratio.
Run metadata records the source schedule, selected effective schedule,
controller version/constants, and hashes of both schedules.

The configuration fingerprint includes the adaptive controller identity and
constants. The final effective-schedule hash is a run result because it is not
known when the sampler object is constructed.

## Version-two contract

`adaptive_history_v2` removes the hand-authored protected head while preserving
the same RES integration and controller limits. It forces only source anchors
0-2. The first measured interval establishes the fixed reference, so the second
interval at anchor 2 can count as the first low-change observation rather than
starting a separate calibration phase. After that, no head coordinate is
forced. Two consecutive low-video-change observations are required before
widening; moderate or high change clears the streak, and high change or an
audio emergency shrinks immediately. Only tail anchors 18/19 and terminal zero
remain protected.

The v2 identity fingerprints the three-anchor bootstrap, one-interval reference,
zero protected-prefix length, two-observation growth gate, and shortened tail
separately from v1.

## Acceptance evidence

CPU tests must establish:

1. Incremental RES is equal to stock RES on full and irregular predetermined
   schedules, with one model call and callback per nonterminal anchor.
2. Adaptive schedules are finite, strictly descending, terminate at zero,
   contain the version's exact bootstrap/protected anchors, and never exceed
   20 NFE.
3. Controller decisions use only current and earlier genuine anchors; no
   synthetic anchor is added and no extra model call is made for error control.
4. Low change grows spacing, high video change shrinks it, and an audio
   emergency can shrink it without making audio the normal controller.
5. Diagnostics and callback metadata expose the selected schedule, decisions,
   trajectory metrics, and honest NFE.
6. Fixed Euler, fixed RES, and forecast modes retain their existing parity,
   masks, node input ordering, and legacy-name normalization.

No CPU test establishes H3 output quality. After restart, the next authorized
live experiment is `res_multistep + adaptive_history_v3` with full diagnostics.
Compare its selected coordinates, NFE, residual actions, media quality,
sampler/model time, and fingerprint against V2's characterized 11-NFE schedule,
`res_multistep + late_aggressive_13`, and the established 20-NFE RES reference.
The research success criterion is comparable quality at variable NFE across
held-out prompts and seeds, not exact reproduction of a fixed coordinate set.

`adaptive_history_v3` is a separate local predictive-error controller using
linear log-sigma secants and symmetric modality residual bands; predictions are
diagnostic only and never enter RES state.

V3 establishes per-run video and audio reference errors from the first valid
prediction and holds that calibration interval at 1x. It grows below a 0.40
video-error ratio, holds below 0.70, shrinks
below 1.00, resets below 1.30, and performs critical recovery at or above 1.30.
Video uses the maximum derivative/x0 residual; audio errors and ratios are
diagnostic only and never control spacing. At minimum 1x, high or nonfinite
video residuals emit `minimum_step_hold` while observations continue. Critical
recovery after an accelerated interval adds one additional 1x interval.
Every decision retains reference values, ratios, scale, delta-t, and modality
residuals.

V3 deliberately has no protected tail. After source anchors 0-2 and the first
baseline prediction interval, its only hard schedule constraints are strict
descent, terminal zero, the configured maximum scale, the 20-NFE ceiling, and
critical recovery. Late contractions or accelerations are therefore experimental
outputs rather than encoded source anchors.

## V2 10x characterization result

Diagnostics run `a1bdd76dca44483c86d2e2ea17164e7f` completed at 11 NFE and
673.0 seconds with all invariants passing. Its effective sigmas were
`1.0, .995633185, .990825653, .985507309, .976647437, .961862445,
.933706164, .869447351, .713067532, .571428597, .387096792, 0`.
The user reported no visible artifacts. The largest realized adaptive interval
used a 7.59375x requested scale; the subsequent 10x proposal was clipped to
protected source anchor 18, so this run does not establish a realized 10x
interval as safe.

The analyzer reconstructs the old run's completed interval lengths and previous
scales. Raw video velocity/x0 relative changes across the 7.59375x interval were
11.70% and 14.54%; across the following tail-clipped interval they were 10.00%
and 11.16%. These raw numerators show that the falling V2 per-unit-t rate partly
reflects the widening observation window rather than a proportionate collapse
in endpoint movement.
