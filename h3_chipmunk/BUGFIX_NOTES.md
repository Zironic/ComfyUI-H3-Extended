# Chipmunk integration bug fixes

The first live `measure` run exposed two composition regressions.

1. Chipmunk wrapped its MLP executor in a private `chipmunk_mlp` timing stage. Hybrid Sparse owns the published request timer and intentionally rejects undeclared timing-stage names, so the run failed before the first block completed. The private stage is removed for the research path; Chipmunk work remains covered by `total_dit_block`, while exact dense MLP calls retain the existing `mlp_fc1` and `mlp_swiglu_fc2` stages.
2. If Chipmunk installed the H3 runtime before Hybrid Sparse, the memory optimizer installed a second sampler/diffusion runtime wrapper instead of attaching to the existing session. The memory optimizer now reuses `minimax_h3_runtime_session`, appends its listeners, and upgrades strict-layout handling in place when required.

Activation Memory/shared block execution remains incompatible with the Chipmunk-owned main-block forward and must stay disabled when Chipmunk is active.
