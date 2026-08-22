# MiniMax H3 Sparse Sage Attention Advanced

Uses the same dependency-owned sparse backends and dense-context guarantees as
MiniMax H3 Sparse Sage Attention, with independent sampling-step windows.

## Sparse schedule

- **Video KV budget** controls all middle steps.
- **Early steps** and **Early KV** control the first requested number of steps.
- **Late steps** and **Late KV** control the last requested number of steps.
- If the early and late windows overlap, the denser requested budget wins.

Zero early or late steps disables that window. All budgets are fixed-density
requests rounded up to whole KV tiles at execution. Text, reference, audio,
non-video query, and mixed boundary tiles remain dense.

The node composes with MiniMax H3 Sage Memory Optimizer in either order and
reports the selected Sparse Sage, INT8 Triton, FP8 FlexAttention, or dense
fallback path.
