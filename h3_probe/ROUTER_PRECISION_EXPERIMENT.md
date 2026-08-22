# Router precision dense-teacher experiment

Date: 2026-08-22

## Decision

Keep BF16 routing as the production default. Native FP32 summary construction
resolves BF16 boundary ties, but this experiment did not find a systematic
dense-teacher advantage large enough to justify its measured router latency and
memory cost.

Do not use a post-pooling `.float()` cast as a compromise. If this decision is
revisited, compare BF16 pooling and scoring directly with native FP32 pooling
and scoring.

## Live experiment

Two real MiniMax H3 activation snapshots used the same seed and geometry at
denoising steps 1 and 2. Each snapshot sampled layers 0, 24, and 49 and two
128-token video query regions per layer. The packed sequence length was 22,440,
with 56 heads and fixed 30% routing at the SM89 128Q x 64KV geometry.

Across 12 dense-teacher captures and 45,328 changed sampled row-heads:

| Metric | Result |
| --- | ---: |
| Route rows changed | 54.68% |
| Selected slots substituted | 1.15% |
| Mean route Jaccard | 0.9780 |
| FP32 retained-mass wins | 51.05% |
| FP32 relative-L2 wins | 48.15% |
| Mean retained-mass delta, FP32 minus BF16 | +0.000314 |
| Mean relative-L2 delta, FP32 minus BF16 | -0.000218 |
| Mean p95 relative-L2, BF16 / FP32 | 0.20048 / 0.19975 |
| Worst-head relative-L2, BF16 / FP32 | 0.31736 / 0.32649 |

BF16 produced exact cutoff ties on roughly 50% to 78% of route rows, while
FP32 cutoff ties stayed at or below 0.021%. FP32 therefore changes many route
rows by resolving quantized ties, but the changed selections were effectively a
coin flip against the dense teacher. Small mean improvements were inconsistent
across layers and query regions, and worst-head error was slightly worse.

## Artifacts

- `D:\AI\ComfyUI\Output\h3_probe\router_precision_teacher_20260822-084949\moba3d_summary.json`
- `D:\AI\ComfyUI\Output\h3_probe\router_precision_teacher_step2_20260822-085318\moba3d_summary.json`

Both prompts completed sampling and wrote six attention records before the
generic `SaveLatent` output failed on H3's `NestedTensor`. That post-probe output
failure does not affect the saved measurements.

## Revisit only with stronger evidence

Reopen the production decision if a broader real-activation sweep or a
downstream model-output/latent experiment shows a repeatable FP32 advantage,
especially at longer sequences or for adaptive allocation. Route disagreement
alone is not evidence of better routing.
