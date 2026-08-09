# MiniMax H3 Vector Acceleration Sampler (Zi) — Revised Implementation Plan

## 1. Objective

Implement a custom ComfyUI sampler that reduces MiniMax H3 transformer evaluations by forecasting selected rectified-flow derivatives while retaining the existing dense sigma schedule as the logical integration grid.

The experiment should test this specific hypothesis:

> Early integration errors may be large locally but have low final importance because many subsequent H3 evaluations can repair them. Late integration errors may be smaller locally but survive because few correction opportunities remain.

The sampler therefore must distinguish three different properties:

1. **Predictability:** how accurately previous H3 evaluations predict the current derivative.
2. **Local integration error:** how much state error a forecast interval probably introduces.
3. **Repairability:** how much of that state error survives after the remaining genuine H3 evaluations.

The eventual adaptive policy should forecast based on estimated **final surviving error**, not merely on velocity smoothness.

Whether H3 actually exhibits this repairability pattern is currently unknown. The implementation must measure it before embedding it into an adaptive policy.

---

## 2. Architectural boundary

This remains a **custom sampler node**, not a custom scheduler.

The existing scheduler continues to provide the complete nominal sigma sequence:

[
\sigma_0,\sigma_1,\ldots,\sigma_N.
]

The sampler decides which derivative evaluations are:

* `A`: actual H3 transformer forwards.
* `F`: forecast derivatives with no H3 forward.

A scheduler cannot retain derivative history, forecast missing derivatives, inspect prediction errors, count real NFEs, or dynamically choose whether the next nominal point receives a model call. The custom sampler owns that logic. This preserves the central architectural decision from the original plan. 

The intended workflow remains:

```text
H3 model
    ↓
MiniMax H3 Sigma Shift (Zi), model patches, guider
    ↓
existing dense scheduler → SIGMAS
MiniMax H3 Vector Accel Sampler (Zi) → SAMPLER
    ↓
existing advanced sampling path
```

The current H3 harness already accepts arbitrary `sampler` and `sigmas` objects and passes them to `CFGGuider.sample`, so long-form and experiment nodes should not need to understand the internal acceleration method.

---

## 3. Core design invariants

The first implementation must preserve these invariants:

1. **Native parity first.**
   `method=native` must reproduce stock deterministic Euler for identical inputs.

2. **Deterministic flow integration only.**
   No churn, ancestral noise, stochastic corrections, or Heun stages in V1.

3. **One packed audiovisual state.**
   The sampler advances H3’s packed video/audio state with one packed derivative.

4. **Separate audiovisual diagnostics.**
   Integration remains packed, but prediction errors and repairability are measured independently for video and audio.

5. **Only genuine H3 evaluations become predictor anchors.**
   Forecast derivatives are never inserted into anchor history.

6. **Every unsafe forecast becomes an actual evaluation.**
   Missing history, duplicate sigmas, non-finite values, extrapolation bounds, or policy uncertainty must trigger a real H3 call.

7. **The dense sigma sequence is a candidate-decision grid, not a guarantee of numerical fidelity.**

8. **Only one approximation system at a time during characterization.**
   Existing DiT caching, prediction caches, spectral forecasting, or other step-skipping systems must be disabled while evaluating Vector Accel.

9. **Logical steps and true NFEs are reported separately.**

10. **Adaptive behavior remains disabled until a repairability profile has been measured.**

---

## 4. Important numerical correction

The original benchmark separated:

* sparse Euler over actual-anchor sigmas; and
* dense-grid Euler holding the last actual derivative across forecast points.

Those are the same integration method.

Suppose one actual derivative (d_a) is held across multiple nominal intervals:

[
x_1=x_0+d_a(\sigma_1-\sigma_0),
]

[
x_2=x_1+d_a(\sigma_2-\sigma_1).
]

Then:

[
x_2=x_0+d_a(\sigma_2-\sigma_0).
]

That is exactly the sparse Euler update from (\sigma_0) directly to (\sigma_2), aside from floating-point accumulation order and intermediate callbacks.

Therefore:

> `dense hold` and `sparse Euler using the same anchors` are an equivalence test, not two meaningful quality arms.

The dense schedule remains useful because it provides:

* candidate locations for actual evaluations;
* callbacks and progress previews;
* consistent comparison between different evaluation masks;
* future adaptive decision points.

It does not preserve additional numerical information when the derivative is simply held constant.

The same telescoping property applies to analytically integrated linear velocity while the same slope remains active between two genuine anchors. Intermediate nominal points matter only when they can become new actual anchors or policy decision points.

---

## 5. Repository structure

Add:

```text
h3_vector_accel/
    __init__.py
    config.py
    nodes.py
    sampler.py
    predictor.py
    policy.py
    diagnostics.py
    repairability.py
    fingerprint.py

h3_vector_accel/profiles/
    README.md
    # Generated repairability profiles go here later.

benchmarks/
    h3_vector_accel_sweep.py
    h3_vector_repairability.py

tests/
    test_h3_vector_sampler.py
    test_h3_vector_predictors.py
    test_h3_vector_policies.py
    test_h3_vector_diagnostics.py
    test_h3_vector_fingerprint.py
```

Responsibilities:

* `config.py`: immutable sampler configuration dataclasses and validation.
* `nodes.py`: Comfy node schema and `SAMPLER` construction.
* `sampler.py`: nominal sigma loop, actual/forecast dispatch, callbacks, fallback.
* `predictor.py`: hold, linear velocity, and later VDE predictors.
* `policy.py`: native, fixed-mask, and later adaptive-risk decisions.
* `diagnostics.py`: per-step and per-modality scalar measurements.
* `repairability.py`: offline branch/replay analysis and survival-profile generation.
* `fingerprint.py`: stable configuration and sigma-sequence identity.

Register the extension in the root extension tuple and add its category to `NODE_CATEGORIES`, following the repository’s existing extension pattern.

Initial category:

```text
H3-Extender/Experiments
```

---

## 6. Sampler-producing node

Expose:

## `MiniMax H3 Vector Accel Sampler (Zi)`

Output:

```text
SAMPLER
```

Initial visible inputs:

```text
method:
    native
    hold
    linear_velocity

evaluation_profile:
    native_20
    conservative_12
    early_aggressive_13
    uniform_13
    late_aggressive_13

diagnostics:
    off
    summary
    full
```

Initial advanced inputs:

```text
fallback_on_guard: true
max_extrapolation_ratio: conservative default
```

Do not initially expose arbitrary thresholds, custom masks, error-controller gains, separate video/audio tolerances, or consecutive-forecast limits. Fixed named profiles make the first comparisons reproducible.

Later versions can add:

```text
policy:
    fixed
    adaptive_repair

quality_preset:
    conservative
    balanced
    aggressive
```

The adaptive options should remain hidden or unavailable until a matching repairability profile exists.

---

## 7. Model compatibility

The sampler should verify that it is operating on H3-compatible flow sampling.

Current ComfyUI constructs H3 as `FLOW_AV`, combining `ModelSamplingAV` with `CONST`. `CONST.calculate_denoised` returns:

[
\text{denoised}=x-\sigma v.
]

Comfy’s `to_d` then gives:

[
d=\frac{x-\text{denoised}}{\sigma}=v.
]

So the derivative retained by the sampler after `to_d` is the sampler-space flow derivative.

H3 also transforms the audio output so that it represents the derivative of the packed audio variable with respect to the shared video-sigma coordinate. The custom sampler must therefore integrate the returned packed derivative directly rather than reproducing the audio-shift mathematics itself.

Implement:

```python
resolve_h3_sampling(model) -> H3SamplingContext
```

The context should expose:

```text
is_h3_flow_av
latent_shapes
audio_scale, when available
model fingerprint, when available
```

Failure behavior:

```text
method=native:
    may optionally permit generic CONST flow models for unit testing

hold / linear_velocity / adaptive:
    reject unsupported model sampling with a clear error
```

Do not hard-code one fragile chain of `.inner_model.inner_model...` attributes. Centralize wrapper traversal in the compatibility helper and test it against the guider/sampler wrappers used by ComfyUI.

---

## 8. Native Euler parity

Start by structurally matching deterministic Comfy Euler:

```python
denoised = model(x, sigma * s_in, **extra_args)
d = to_d(x, sigma, denoised)
h = sigma_next - sigma
x_next = x + h * d
```

The current stock Euler sampler follows this structure.

`method=native` must:

* perform one H3 call for every nonterminal sigma;
* produce the same callback order;
* use the same `denoised`;
* advance with the same dtype behavior;
* return the same final state within numerical roundoff;
* report true NFE equal to nominal step count.

This parity milestone must be completed before implementing any forecast branch.

---

## 9. Predictor interface

Define a shared predictor protocol:

```python
class VectorPredictor:
    def reset(self) -> None: ...

    def ready(self) -> bool: ...

    def predict(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
    ) -> Prediction: ...

    def observe_actual(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        derivative: torch.Tensor,
    ) -> None: ...

    def integrate(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prediction: Prediction,
    ) -> torch.Tensor: ...
```

`Prediction` should contain:

```text
derivative
slope, optional
valid
failure_reason
diagnostic_scalars
```

Predictor history contains only:

```text
(actual sigma, actual derivative)
```

The state may also be retained later for VDE, but forecast outputs must never become actual anchors.

Anchor derivative and slope arithmetic should initially use FP32 even when the model runs in BF16 or FP8.

---

## 10. Prediction methods

### 10.1 `native`

Every step executes H3.

This mode is for parity and callback validation only.

### 10.2 `hold`

At forecast sigma (\sigma_i), retain the most recent genuine derivative:

[
\hat d_i=d_a.
]

Advance with:

[
x_{i+1}=x_i+h_i\hat d_i,
\qquad
h_i=\sigma_{i+1}-\sigma_i.
]

This is the zero-order baseline.

It is also the exact numerical equivalent of removing forecast sigma points and taking sparse Euler steps between the same actual anchors.

### 10.3 `linear_velocity`

Retain the last two genuine anchors:

[
(\sigma_b,d_b),\qquad(\sigma_a,d_a),
]

with (\sigma_a) the most recent.

Estimate derivative slope with respect to sigma:

[
m=
\frac{d_a-d_b}
{\sigma_a-\sigma_b}.
]

Predict at the current sigma:

[
\hat d(\sigma_i)
================

d_a+(\sigma_i-\sigma_a)m.
]

Analytically integrate the linear derivative approximation over the interval:

[
x_{i+1}
=======

x_i
+
h_i\hat d(\sigma_i)
+
\frac12h_i^2m.
]

This should be called `linear_velocity`, not `AB2`, because:

* the history contains only genuine, potentially nonconsecutive model evaluations;
* forecast outputs are deliberately excluded;
* the method is linear extrapolation between actual anchors rather than textbook consecutive-step Adams–Bashforth.

Actual H3 steps continue to use ordinary Euler. The analytic correction is used only on forecast intervals.

---

## 11. Extrapolation guards

A forecast is valid only when all of the following hold:

* the predictor has enough genuine anchors;
* anchor sigmas are distinct;
* the sigma sequence is finite and monotonically decreasing;
* predicted derivative and slope are finite;
* the proposed state is finite;
* the extrapolated correction remains within configured bounds;
* the evaluation policy permits a forecast;
* the step is not part of forced warmup or tail protection.

Initial linear guards should include:

### Derivative-growth guard

[
\frac{\operatorname{RMS}(\hat d)}
{\operatorname{RMS}(d_a)+\epsilon}
\le r_d.
]

### Curvature-correction guard

[
\frac{
\operatorname{RMS}\left(\frac12h^2m\right)
}{
\operatorname{RMS}(h\hat d)+\epsilon
}
\le r_c.
]

### Optional direction guard

Reject the forecast when predicted derivative direction diverges too sharply from the last genuine derivative:

[
\cos(\hat d,d_a)<c_{\min}.
]

Initial thresholds should be conservative implementation constants rather than user-facing knobs. Diagnostics must record which guard caused every fallback.

On guard failure:

```text
run H3 at this nominal point
record actual_fallback
continue normally
```

---

## 12. Fixed evaluation profiles

### 12.1 Conservative functionality profile

For 20 nominal derivative evaluations:

```text
actual indices:
0, 1, 3, 5, 7, 9, 11, 13, 15, 17, 18, 19
```

This gives:

* 20 logical steps;
* 12 true H3 evaluations;
* first two evaluations actual;
* no consecutive forecasts;
* final three evaluations actual.

This profile determines whether simple forecasting works at all without immediately testing extreme skip runs.

### 12.2 Equal-NFE anchor-placement profiles

The central fixed-policy experiment should compare where the 13 actual evaluations occur.

#### Early-aggressive, late-protected

```text
actual:
0, 1, 4, 7, 8, 10, 12, 14, 15, 16, 17, 18, 19
```

This permits most two-step forecast runs early and becomes dense near the end.

#### Uniform

```text
actual:
0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17, 18, 19
```

This distributes forecasts approximately evenly.

#### Early-protected, late-aggressive

```text
actual:
0, 1, 2, 3, 4, 5, 7, 9, 12, 15, 17, 18, 19
```

This is the negative control: it spends extra evaluations early and places more forecast pressure later, while still retaining the final three actual steps.

All three have:

* the same nominal sigmas;
* the same predictor;
* the same 13 true NFEs;
* the same two-step maximum forecast run;
* the same forced tail length.

The hypothesis predicts that early-aggressive/late-protected should outperform the late-aggressive control even when its early derivative-prediction errors are larger.

Profiles for step counts other than 20 should eventually be generated algorithmically from normalized progress weights, but the initial study should use exact masks to keep runs reproducible.

---

## 13. Sampler loop

The sampling loop should follow this structure:

```python
predictor.reset()
policy.reset()
diagnostics.start_run(...)

for i in range(len(sigmas) - 1):
    sigma = sigmas[i]
    sigma_next = sigmas[i + 1]
    h = sigma_next - sigma

    decision = policy.decide(
        step=i,
        sigma=sigma,
        sigma_next=sigma_next,
        predictor_ready=predictor.ready(),
        diagnostics_state=diagnostics.policy_state(),
    )

    counterfactual = None
    if predictor.ready():
        counterfactual = predictor.predict(x, sigma)

    use_forecast = decision.is_forecast and counterfactual.valid

    if use_forecast:
        prediction = counterfactual

        if not forecast_guards_pass(...):
            use_forecast = False
            fallback_reason = ...
        else:
            denoised_for_callback = x - sigma * prediction.derivative
            x_next = predictor.integrate(
                x, sigma, sigma_next, prediction
            )
            true_nfe_increment = 0

    if not use_forecast:
        denoised = model(x, sigma * s_in, **extra_args)
        derivative = to_d(x, sigma, denoised)

        diagnostics.observe_actual_anchor(
            step=i,
            sigma=sigma,
            x=x,
            actual_derivative=derivative,
            counterfactual_prediction=counterfactual,
            previous_actual_sigma=predictor.last_actual_sigma,
            fallback_reason=fallback_reason,
        )

        predictor.observe_actual(x, sigma, derivative)

        denoised_for_callback = denoised
        x_next = x + h * derivative
        true_nfe_increment = 1

    callback({
        "x": x,
        "i": i,
        "sigma": sigma,
        "sigma_hat": sigma,
        "denoised": denoised_for_callback,
        "h3_vector_forecast": use_forecast,
        "h3_vector_true_nfe": running_true_nfe,
        "h3_vector_method": method,
        "h3_vector_profile": profile,
        "h3_vector_fallback_reason": fallback_reason,
    })

    diagnostics.observe_step(...)
    x = x_next

diagnostics.finish_run(...)
return x
```

The callback should be invoked before assigning `x = x_next`, matching normal Euler semantics.

---

## 14. Callback and preview behavior

A forecast step has no genuine model-produced denoised estimate. Reconstruct a sampler-equivalent approximation:

[
\widehat{x_0}=x-\sigma\hat d.
]

Pass this value to the existing callback so that:

* progress still reflects all nominal steps;
* latent preview continues to update;
* TAEH3 preview receives a meaningful estimate;
* ordinary callback consumers continue functioning.

Add explicit metadata:

```text
h3_vector_forecast: bool
h3_vector_true_nfe: int
h3_vector_actual_anchor_index: int
h3_vector_method: str
h3_vector_profile: str
h3_vector_fallback_reason: optional str
```

Existing diagnostics must not silently treat synthetic forecast callbacks as genuine H3 predictions. Update the latent-dynamics tracker so it can:

* label forecast callbacks;
* exclude them from actual-to-actual prediction statistics;
* optionally maintain separate logical-step and actual-anchor streams.

The existing tracker already separates sampler state and prediction updates, so its reduction conventions can be reused rather than creating an incompatible format.

---

## 15. Per-anchor prediction diagnostics

Before each actual model call, calculate what the predictor would have produced at the current sigma using only previous genuine anchors.

After the real derivative (d_i) is available, compare:

[
d_i
\quad\text{and}\quad
\hat d_i.
]

Record these metrics separately for video and audio.

### Derivative relative L2

[
E_{\mathrm{rel}}
================

\frac{
|d_i-\hat d_i|_2
}{
|d_i|_2+\epsilon
}.
]

### Derivative RMS error

[
E_{\mathrm{RMS}}
================

\operatorname{RMS}(d_i-\hat d_i).
]

### Direction cosine

[
C=
\frac{
\langle d_i,\hat d_i\rangle
}{
|d_i|_2|\hat d_i|_2+\epsilon
}.
]

### Integration-error proxy

If the prediction spans from the previous actual anchor (\sigma_a) to the current actual anchor (\sigma_i):

[
E_{\mathrm{int}}
================

|\sigma_i-\sigma_a|
\operatorname{RMS}(d_i-\hat d_i).
]

This is only a local state-error proxy. It is not a measurement of final quality impact.

Record:

```text
step
sigma
previous_actual_sigma
logical_span
method
video metrics
audio metrics
packed metrics, diagnostic only
actual/fallback status
guard status
```

Do not use one packed aggregate as the main policy signal. Video can numerically dominate because of tensor size, while audio can fail perceptually despite contributing little to global packed L2.

For later policy decisions, use:

[
E_{\mathrm{modal}}
==================

\max(
E_{\mathrm{video,norm}},
E_{\mathrm{audio,norm}}
).
]

---

## 16. Repairability experiment

Prediction error alone does not test the central hypothesis. Add a separate offline repairability runner.

### 16.1 Native reference trajectory

Run deterministic native 20-step Euler and retain selected:

```text
x_i
d_i
sigma_i
```

at all candidate perturbation points or at a selected set such as:

```text
2, 5, 8, 11, 14, 16, 18
```

Snapshots should initially be stored in FP32 for low-resolution diagnostic runs. Once restart parity is established, storage can be reduced to FP16 where the resulting snapshot error is negligible relative to the injected perturbation.

### 16.2 Natural omission branches

For each selected step (i):

1. Start from native state (x_i).
2. Restore the previous genuine predictor anchors.
3. Replace the actual derivative at (i) with hold or linear prediction.
4. Advance to (x_{i+1}^{\mathrm{branch}}).
5. Use genuine H3 evaluations for every remaining step.
6. Compare the branch to the native trajectory after every later actual evaluation.

This measures the realistic consequence of omitting one H3 evaluation at each trajectory location.

### 16.3 Normalized perturbation branches

Natural forecast errors vary in size, so they confound:

* how inaccurate the predictor is; and
* how repairable an error is at that point.

Add a normalized experiment:

1. Compute the natural forecast update error:

   [
   \delta x_i^{\mathrm{natural}}
   =============================

   ## x_{i+1}^{\mathrm{forecast}}

   x_{i+1}^{\mathrm{native}}.
   ]

2. Normalize it to a fixed per-modality RMS magnitude.

3. Add the normalized perturbation to the native (x_{i+1}).

4. Resume with genuine H3 evaluations.

Run at least:

```text
joint AV perturbation
video-only perturbation
audio-only perturbation
```

This determines whether equivalent state errors introduced at different trajectory locations have different survival rates.

### 16.4 State divergence curve

For each branch and each later step (j), calculate:

[
D_j^{(m)}
=========

\frac{
\operatorname{RMS}
\left(
x_j^{\mathrm{branch},m}
-----------------------

x_j^{\mathrm{native},m}
\right)
}{
\operatorname{RMS}
\left(
x_j^{\mathrm{native},m}
\right)+\epsilon
},
]

where (m) is video or audio.

A repair event appears as:

[
D_{j+1}<D_j
]

after a genuine H3 evaluation.

### 16.5 Final survival factor

For an error introduced after step (i):

[
S_i^{(m)}
=========

\frac{
D_N^{(m)}
}{
D_{i+1}^{(m)}+\epsilon
}.
]

Interpretation:

```text
S << 1:
    most of the error was repaired

S ≈ 1:
    the error mostly survived

S > 1:
    the remaining trajectory amplified the error
```

Do not clamp values above one. Amplification is important evidence.

### 16.6 Output profile

Generate a repairability profile containing:

```text
normalized sigma/progress bin
video survival quantiles
audio survival quantiles
joint conservative maximum
sample count
scheduler/sigma hash
model/checkpoint identity
video sigma shift
audio sigma shift
nominal step count
predictor method
conditioning mode
```

The adaptive sampler should later use a conservative quantile, such as a measured upper envelope, rather than the mean survival value.

---

## 17. Fixed-policy benchmark sequence

Run these experiments in order.

### Phase A — numerical controls

1. Native custom sampler versus stock Euler.
2. Dense hold versus sparse Euler with identical actual anchors.
3. Constant-velocity fake model.
4. Linear-in-sigma fake velocity model.
5. True model-call counts.

### Phase B — predictor comparison

For one fixed 12- or 13-NFE anchor mask:

```text
native 20
hold
linear_velocity
```

This answers whether linear velocity forecasting is better than simply holding the last derivative.

### Phase C — anchor-placement comparison

Use `linear_velocity` for all arms:

```text
early_aggressive_13
uniform_13
late_aggressive_13
```

This directly tests the hypothesis that early errors are more repairable than late errors.

### Phase D — one-step repairability sweep

Run natural omissions and normalized perturbations at selected trajectory positions.

### Phase E — multi-step forecast runs

Only after the single-step results are understood:

```text
maximum 1 consecutive forecast
maximum 2 consecutive forecasts
maximum 3 consecutive forecasts
```

Each forecast-length comparison must retain equal true NFE where possible.

### Phase F — adaptive policy

Build and test the adaptive controller only after the repairability profile exists.

---

## 18. Adaptive repair-aware policy

A conventional adaptive ODE controller minimizes local error. That may allocate extra evaluations early if the early field is difficult, even when those errors are later repaired.

The H3 policy should estimate final surviving risk.

### 18.1 Online local-error estimate

At actual anchor (i), calculate per modality:

[
e_i^{(m)}
=========

|\sigma_i-\sigma_a|
\operatorname{RMS}
\left(
d_i^{(m)}
---------

\hat d_i^{(m)}
\right).
]

Estimate the next forecast’s local error conservatively:

[
\hat e_{i+1}^{(m)}
==================

k_{\mathrm{safety}}
\max(
e_i^{(m)},
e_{i-1}^{(m)}
).
]

An EMA can be added later, but the initial controller should use the conservative recent maximum.

### 18.2 Offline survival prior

From the measured profile, obtain:

[
q_{i+1}^{(m)}
=============

\text{conservative survival quantile at the next progress position}.
]

### 18.3 Surviving-risk estimate

[
R_{i+1}^{(m)}
=============

q_{i+1}^{(m)}
\hat e_{i+1}^{(m)}.
]

Joint policy risk:

[
R_{i+1}
=======

\max(
R_{i+1}^{\mathrm{video}},
R_{i+1}^{\mathrm{audio}}
).
]

Decision:

```text
forecast next point:
    R <= tolerance
    predictor guards pass
    not in warmup
    not in forced tail
    consecutive-forecast limit not reached

otherwise:
    actual H3 evaluation
```

### 18.4 Recovery behavior

When an anchor produces unexpectedly high prediction error:

```text
force next K nominal points actual
reset consecutive-forecast count
retain genuine anchors
do not clear valid history unless non-finite
```

Initial value:

```text
recovery_actual_steps = 2
```

### 18.5 Profile matching

Adaptive mode must verify that the repairability profile matches:

* model/checkpoint identity;
* sigma sequence or compatible schedule hash;
* video/audio shifts;
* nominal step count;
* predictor method;
* relevant conditioning mode.

On mismatch:

```text
reject adaptive mode
or fall back to a named conservative fixed profile
```

Do not silently apply a 20-step simple-schedule profile to a different scheduler such as beta or linear-quadratic.

---

## 19. VDE phase

Add VDE only after hold and linear velocity have established:

* native parity;
* useful NFE reduction;
* known audio behavior;
* measured repairability.

The predictor API should allow:

```python
HoldPredictor
LinearVelocityPredictor
VDEPredictor
```

without changing the sampler loop.

VDE may depend on the current state as well as previous derivatives, so `predict(x, sigma)` must already accept the current packed state.

VDE should be evaluated against linear velocity with:

* identical anchor masks;
* identical true NFE;
* identical fallback policy;
* separate video/audio errors;
* the same repairability analysis.

Do not combine VDE with adaptive skipping in its first test. Establish fixed-mask behavior first.

---

## 20. Diagnostics output

### Summary mode

Log one compact row per run:

```text
nominal steps
true H3 NFE
forecast count
fallback count
method
evaluation profile
wall time
model-call time
sampler overhead
maximum video prediction error
maximum audio prediction error
configuration fingerprint
```

### Full mode

Write run-scoped JSON:

```text
ComfyUI/output/h3_vector_accel/<run_id>/diagnostics.json
```

Include:

```text
configuration
sigma sequence
actual/forecast mask
every actual-anchor prediction metric
every fallback reason
video/audio metrics
callback metadata
true NFE
timing
model identity
sigma hash
repairability-profile identity, when applicable
```

Avoid mutable process-global “last run” diagnostics. Multiple queued or concurrent runs must not overwrite one another.

Sampler overhead should be timed independently from H3 model time.

---

## 21. Harness and cache identity

The existing harness can continue passing the sampler and sigmas unchanged.

Add a stable fingerprint for Vector Accel configurations:

```text
method
evaluation profile
actual-mask version
predictor version
guard constants
adaptive profile hash
quality preset
sigma-sequence hash
```

Serialize the configuration in canonical sorted JSON and hash it.

Attach the fingerprint to the custom `KSAMPLER` configuration and incorporate it into experiment/cache identities. This prevents results from:

```text
hold + conservative_12
linear + conservative_12
linear + early_aggressive_13
adaptive + profile A
```

being treated as equivalent merely because they share the same Python sampler class.

---

## 22. Unit tests

### Native parity

Same fake flow model, initial state, sigmas, and callback:

```text
stock Euler == custom native mode
```

Check final tensor and every callback value.

### Constant velocity

For:

[
d(x,\sigma)=c,
]

all of these should agree:

```text
native
hold with arbitrary skipped points
linear_velocity
sparse Euler using the same actual anchors
```

### Hold/sparse equivalence

For an arbitrary deterministic model and predetermined actual anchors:

```text
dense logical grid + held derivative
==
sparse Euler across actual-anchor sigmas
```

Test multiple consecutive forecast lengths and irregular sigmas.

### Linear velocity

Construct a velocity independent of state and linear in sigma:

[
d(\sigma)=a+b\sigma.
]

The analytic `linear_velocity` integration should reproduce the expected integral to tolerance once two genuine anchors exist.

Do not define the exactness test with a generic field linear in both (x) and (\sigma); that is not generally integrated exactly by this predictor.

### Actual-only history

Verify that forecast derivatives never become anchors.

### Model-call count

Every profile must produce the exact expected number of `model()` calls unless a guard fallback occurs.

### Guard fallback

Test:

```text
NaN slope
Inf derivative
duplicate anchor sigma
excessive extrapolation
non-monotonic sigma sequence
insufficient history
```

Each must trigger an actual call rather than propagate the forecast.

### Callback semantics

Verify:

* one callback per nominal step;
* synthetic `denoised` only on forecast steps;
* correct forecast flag;
* correct true-NFE counter;
* pre-update `x` semantics.

### Audiovisual diagnostics

Use a fake packed AV state with deliberately different video/audio errors and verify that:

* metrics are separated;
* packed element count does not hide audio error;
* joint policy uses the conservative modal maximum.

### Fingerprint

Different methods, masks, thresholds, or sigma sequences must produce different identities.

### Determinism

Repeated runs with the same inputs and fixed profile must match exactly within expected floating-point tolerance.

---

## 23. GPU evaluation suite

Begin at a modest resolution and duration so many controlled runs are practical.

Use the same prompt, seed, conditioning, references, scheduler, model precision, and backend within each comparison.

The suite should contain at least:

1. Low-motion speech with clear audio.
2. High-motion human action.
3. Camera translation or tracking motion.
4. Fine texture and small facial details.
5. FL2VA with a strong final-frame constraint.
6. Ref2VA or N+1 audiovisual continuation.

For each arm, record:

```text
true NFE
wall time
H3-forward time
sampler overhead
final video latent difference from native
final audio latent difference from native
per-step branch divergence
decoded visual comparison
decoded audio comparison
AV synchronization observations
```

Objective audio distances are useful diagnostics but not sufficient. Manual listening remains mandatory because a numerically small latent difference can still cause speech, transient, or synchronization artifacts.

The first production-oriented success criterion should be:

> At equal true NFE, `linear_velocity` and early-aggressive anchor placement must outperform hold/sparse Euler without introducing audible degradation.

---

## 24. Decision gates

### Gate 1 — parity

Proceed only when native mode matches stock Euler.

### Gate 2 — predictor value

Proceed only when `linear_velocity` consistently beats hold at identical anchors and true NFE.

If it does not, retain hold as the baseline and investigate VDE or another predictor before adaptation.

### Gate 3 — hypothesis test

Proceed to repair-aware adaptation only when the equal-NFE placement and perturbation experiments show that early errors generally have lower final survival than comparable late errors.

If early and late survival are similar, the proposed theory is not supported.

If repairability varies primarily by content rather than trajectory position, use a conservative fixed policy or develop an online verification method instead of a static survival prior.

### Gate 4 — audio safety

Any repeated speech corruption, audio transient loss, synchronization drift, or modality imbalance blocks aggressive profiles even when video quality remains acceptable.

### Gate 5 — adaptive value

Adaptive mode is useful only when it reduces median true NFE relative to the best fixed profile while maintaining the same measured quality envelope.

---

## 25. Implementation sequence

### Milestone 1 — infrastructure and parity

* Create module and node.
* Implement model compatibility resolution.
* Clone deterministic Euler behavior.
* Register extension.
* Add native parity tests.
* Add sampler fingerprint.

### Milestone 2 — fixed predictors

* Implement hold.
* Implement linear velocity with analytic integration.
* Add forecast callbacks.
* Add guards and actual fallback.
* Add fixed named masks.
* Add exact call-count tests.

### Milestone 3 — diagnostics and numerical controls

* Add separate video/audio metrics.
* Update latent-dynamics callback handling.
* Add hold/sparse equivalence test.
* Add constant and linear fake-flow tests.
* Add run-scoped JSON output.

### Milestone 4 — fixed-policy GPU study

* Run native, hold, and linear at the same anchors.
* Run early-aggressive, uniform, and late-aggressive masks.
* Characterize audio failures separately.
* Select candidate NFE/profile combinations.

### Milestone 5 — repairability study

* Implement trajectory snapshots and branch continuation.
* Run natural omission sweeps.
* Run normalized video, audio, and joint perturbations.
* Generate survival curves and profile metadata.

### Milestone 6 — adaptive repair-aware policy

* Implement profile loading and compatibility validation.
* Implement surviving-risk decisions.
* Add recovery actual steps.
* Keep maximum one forecast initially.
* Compare against the best fixed policy.

### Milestone 7 — VDE

* Implement as a predictor backend.
* Evaluate under fixed masks.
* Add adaptive support only after fixed-mask results justify it.

---

The resulting system is not merely “skip where the velocity looks smooth.” It is:

> **Estimate the local cost of forecasting, weight that cost by how much H3 is empirically able to repair at this trajectory position, and spend genuine H3 evaluations where errors are most likely to survive.**
