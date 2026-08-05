"""Per-sampling-run state for masked Ref2V computation.

One `MaskedCacheSession` lives on the armed model and outlives every run; one
`MaskedCacheRun` is created per `OUTER_SAMPLE` and owns everything that must not
survive it - the device copy of the source latent, the accumulated masks, the
score maps queued for the report.

The split matters for re-queues: a second sampling run through the same model
must not append to the first run's trace or reuse its frozen mask, and a run
that raised must not leave a device-resident source latent behind.

Stage 0 records only. The sigma bookkeeping (`observe_sigma`, the pending/active
pair) is here already because the promotion rule - a mask inferred at one sigma
becomes usable only at the next - is what keeps both CFG branches on the same
plan, and it is far easier to trust when the measurement pass has already been
running it against real schedules.
"""

import os
import time

from . import mask as mask_ops
from .config import SCORE_QUANTILES, THRESHOLD_SWEEP
from .source import SourceCache


class MaskedCacheRun:
    """Everything one sampling run accumulates."""

    def __init__(self, config, tag, out_dir):
        self.config = config
        self.tag = tag
        self.out_dir = out_dir
        self.started = time.time()

        self.layout = None
        self.source = None                 # SourceResolution
        self.notes = {}
        self.steps = []                    # one dict per observed forward
        self.score_maps = []               # (label, tensor[T,ph,pw]) queued for mask.npz
        self.masks = []                    # (label, bool tensor[T,ph,pw])
        self.fallbacks = []                # (reason, count)

        # mask state machine
        self.pending_mask = None           # inferred here, usable from the next sigma
        self.active_mask = None            # promoted, in force for the current sigma
        self.union_mask = None             # every token ever active this run
        self.prev_observed = None          # last sigma's mask, for Jaccard
        self.last_sigma = None
        self.sigma_count = 0

        self.disabled_reason = None        # set once; measurement stops, sampling continues

    # -- lifecycle ---------------------------------------------------------

    def disable(self, reason):
        if self.disabled_reason is None:
            self.disabled_reason = reason
        self.note_fallback(reason)

    def note_fallback(self, reason):
        for i, (r, n) in enumerate(self.fallbacks):
            if r == reason:
                self.fallbacks[i] = (r, n + 1)
                return
        self.fallbacks.append((reason, 1))

    # -- sigma bookkeeping -------------------------------------------------

    def observe_sigma(self, sigma):
        """Register a model call at `sigma`; True when it opens a new sigma.

        Promotion happens here and nowhere else: at the first call of a new
        sigma the pending mask becomes active, so every condition evaluated at
        one sigma sees the same mask no matter which branch ran first.
        """
        new = self.last_sigma is None or sigma != self.last_sigma
        if new:
            if self.pending_mask is not None:
                self.active_mask = self.pending_mask
                self.pending_mask = None
            self.last_sigma = sigma
            self.sigma_count += 1
        return new

    def stage_mask(self, m):
        """Record a mask inferred at the current sigma. Masks only ever grow."""
        self.pending_mask = m if self.pending_mask is None else (self.pending_mask | m)
        self.union_mask = m.clone() if self.union_mask is None else (self.union_mask | m)

    def release(self):
        self.pending_mask = None
        self.active_mask = None
        self.union_mask = None
        self.prev_observed = None
        self.score_maps = []
        self.masks = []


class MaskedCacheSession:
    """Configuration plus the run currently being sampled."""

    def __init__(self, config, base_dir, model_sampling=None):
        self.config = config
        self.base_dir = base_dir
        self.model_sampling = model_sampling
        self.run = None
        self.sources = SourceCache()

    def begin(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s_%s" % (self.config.run_tag or "h3mask", stamp)
        self.run = MaskedCacheRun(self.config, name, os.path.join(self.base_dir, name))
        return self.run

    def end(self):
        run = self.run
        self.run = None
        self.sources.release()
        if run is None:
            return None
        try:
            from .report import write_run
            return write_run(run)
        finally:
            run.release()

    # -- measurement -------------------------------------------------------

    def record_step(self, run, token_scores, step, sigma, cond_or_uncond, dense_wall_s):
        """Score one observed forward and append its row to the trace."""
        cfg = self.config
        core, expanded, _ = mask_ops.build_mask(
            token_scores, cfg.score_threshold, cfg.tile_h, cfg.tile_w,
            cfg.spatial_halo, cfg.temporal_halo)

        row = {
            "step": step,
            "sigma": sigma,
            "cond_or_uncond": cond_or_uncond,
            "sigma_index": run.sigma_count - 1,
            "dense_wall_s": dense_wall_s,
            "score_quantiles": mask_ops.quantiles(token_scores, SCORE_QUANTILES),
            "threshold": cfg.score_threshold,
            "active_core": mask_ops.active_fraction(core),
            "active_expanded": mask_ops.active_fraction(expanded),
            "threshold_sweep": mask_ops.threshold_sweep(
                token_scores, THRESHOLD_SWEEP, cfg.tile_h, cfg.tile_w,
                cfg.spatial_halo, cfg.temporal_halo),
            "jaccard_prev": mask_ops.jaccard(expanded, run.prev_observed),
            "escaped_prev": mask_ops.escaped_fraction(expanded, run.prev_observed),
            "escaped_union": mask_ops.escaped_fraction(expanded, run.union_mask),
        }
        run.steps.append(row)

        label = "s%03d_%s" % (len(run.steps) - 1, "cond" if cond_or_uncond == 0 else "uncond")
        run.score_maps.append((label, token_scores.detach().to("cpu")))
        run.masks.append((label, expanded.detach().to("cpu")))

        run.stage_mask(expanded)
        run.prev_observed = expanded
        return row
