# H3 vector acceleration profiles

Generated repairability profiles belong in this directory. None are bundled
yet: adaptive skipping stays unavailable until a fixed-policy H3 study produces
a profile matching the model, sigma schedule, shifts, step count, predictor,
and conditioning mode described in `H3_VECTOR_ACCEL_PLAN.md`.

Generate a profile with `benchmarks/h3_vector_repairability.py`. The profile
stores conservative per-progress video/audio survival quantiles, measured
quality-preset tolerances, and an exact compatibility record. The sampler
rejects mismatched profiles. A VDE profile must also explicitly list `vde` in
`adaptive_methods`; fixed-mask VDE does not require a profile.
