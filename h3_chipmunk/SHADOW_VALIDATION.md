# Chipmunk shadow validation

`shadow_validate` is the next test after the lightweight `measure` run.

It is **output-exact**: every H3 MLP is executed densely and the dense result is the only result fed into the model. On a small sampled subset, a recurrent Chipmunk approximation is computed beside the dense result and compared against it.

## Default test profile

The first real H3 measurement showed substantially stronger 256-feature group concentration in early transformer blocks than late blocks. The validator therefore uses this fixed `depth_v1` profile:

```text
layers  0-14: 30% active
layers 15-24: 40% active
layers 25-29: 50% active
layers 30-49: dense
```

With `shadow_layer_stride=5`, validation is performed at:

```text
layers 0, 5, 10, 15, 20, 25, 29
plus layer 30 at 100% as a numerical control
```

Layer 30 is intentionally shadowed even though the proposed production profile makes it dense. If the 100%-active control has meaningful error, that establishes a requantization/validation-path floor that must be solved before interpreting sparse-layer error.

Every eligible 2048-row target-video slab validates only its centered `shadow_sample_rows` window (128 rows by default). This keeps recurrent shadow state small enough to remain on GPU. `cache_location` is intentionally ignored for shadow state; there are no shadow CPU cache round-trips. Both shadow fc1 and fc2 reuse the ConvRot tiles already held by the exact dense MLP, avoiding additional weight-staging phases.

## Recommended run

```text
mode = shadow_validate
refresh_every = 6
first_dense_steps = 2
last_dense_steps = 2
layer_start = 0
layer_stop = 50
chunk_rows = 2048
token_group_rows = 128
scope = target_video
random_groups = 0.0
strict = true
save_report = true
measure_layer_stride = 5       # ignored by shadow_validate
shadow_layer_stride = 5
shadow_sample_rows = 128
```

`top_fraction` is ignored by `shadow_validate`; the depth profile supplies the requested active fraction. Reports include both the requested fraction and the actual whole-group density after rounding 56 ConvRot groups.

For each sparse shadow update the report records:

- raw MLP-output relative L2 error;
- gated MLP-contribution relative L2 error;
- full post-block relative L2 error against the exact residual stream;
- raw output cosine similarity;
- error RMS, dense-output RMS, and maximum absolute error;
- layer, diffusion step, refresh age, requested/actual active fraction, chunk, and sampled row window.

The summary JSON aggregates those metrics overall and by layer, step, and active fraction.

## Interpretation

Start with the 100%-active layer-30 control. Its `block_relative_l2` is the numerical floor of the shadow recurrence/requantization path. Sparse-layer error should then be judged relative to that floor.

The primary sparse-profile decision should use `block_relative_l2`, because it measures how much the approximation would perturb the actual post-MLP transformer residual after the current AdaLN gate. `raw_relative_l2` is deliberately harsher and remains useful for diagnosing where the MLP approximation itself is weak.

If the profile is clean, the next test can feed the same schedule into the real approximation path. If one depth band is bad, adjust that band's active fraction or make it dense before doing any approximate video generation.
