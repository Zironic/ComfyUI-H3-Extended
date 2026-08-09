# H3 probe guidance

- Preserve existing logical probe semantics and reports. Add executable,
  hardware-coarsened geometry alongside logical metrics rather than replacing
  or silently reinterpreting them.
- Reuse the shared packed layout, activity maps, morphology, tiling, dilation,
  halo, and offline-analysis helpers. Do not create a parallel active-set or
  sparse-geometry implementation for a new experiment.
- The causal lifecycle begins with callback 0 as the baseline, callback 1 as
  the first activity update, and callback 2 as the first valid comparison of a
  previous mask with a current update. Keep this staging explicit in tests and
  reports.
- Persist raw float32 activity and energy maps for offline threshold/tile/halo
  sweeps. Do not bake one experimental threshold into capture behavior merely
  because it performed well in a prior run.
- Initialize shared layout, anchors, and dynamics before allowing attention
  capture to turn off. Keep requested, evaluated, and execution ranges distinct
  and densely preserve any tile containing non-video context.
- CPU synthetic probes establish geometry and metric contracts only. CUDA
  capture, memory cost, kernel selection, and real-run repairability remain
  permission-gated live evidence.
