"""Per-run state for masked Ref2V measurement."""

import os
import time

import torch

from . import mask as mask_ops
from .config import SCORE_QUANTILES, THRESHOLD_SWEEP
from .source import SourceCache


class MaskedCacheRun:
    def __init__(self, config, tag, out_dir, latent_shapes=None):
        self.config = config
        self.tag = tag
        self.out_dir = out_dir
        self.started = time.time()
        self.latent_shapes = latent_shapes
        self.sample_sigmas_tensor = None

        self.layout = None
        self.source = None
        self.notes = {}
        self.steps = []
        self.score_maps = []
        self.error_maps = []
        self.source_maps = []
        self.saliency_maps = []
        self.masks = []
        self.fallbacks = []
        self.final = None

        self.pending_mask = None
        self.active_mask = None
        self.union_mask = None
        self.frozen_mask = None
        self.frozen_range = None
        self.prev_observed = None
        self.last_sigma = None
        self.sigma_count = 0
        self.disabled_reason = None

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

    def observe_sigma(self, sigma):
        new = self.last_sigma is None or sigma != self.last_sigma
        if new:
            if self.pending_mask is not None:
                self.active_mask = self.pending_mask
                self.pending_mask = None
            self.last_sigma = sigma
            self.sigma_count += 1
        return new

    def stage_mask(self, m):
        self.pending_mask = m if self.pending_mask is None else (self.pending_mask | m)
        self.union_mask = m.clone() if self.union_mask is None else (self.union_mask | m)

        # Early H3 predictions can have a much broader reconstruction-error
        # distribution than the stable steps that follow. Discard the configured
        # burn-in observations, then freeze exactly the following warmup window.
        start = self.config.freeze_start
        stop = self.config.freeze_stop
        if self.frozen_mask is None and len(self.masks) >= stop:
            selected = [x for _, x in self.masks[start:stop]]
            if selected:
                frozen = selected[0].clone()
                for x in selected[1:]:
                    frozen |= x
                self.frozen_mask = frozen
                self.frozen_range = (start, stop)

    def release(self):
        self.pending_mask = None
        self.active_mask = None
        self.union_mask = None
        self.frozen_mask = None
        self.prev_observed = None
        self.sample_sigmas_tensor = None
        self.score_maps = []
        self.error_maps = []
        self.source_maps = []
        self.saliency_maps = []
        self.masks = []


class MaskedCacheSession:
    def __init__(self, config, base_dir, model_sampling=None):
        self.config = config
        self.base_dir = base_dir
        self.model_sampling = model_sampling
        self.run = None
        self.sources = SourceCache()

    def begin(self, latent_shapes=None):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s_%s" % (self.config.run_tag or "h3mask", stamp)
        self.run = MaskedCacheRun(self.config, name, os.path.join(self.base_dir, name), latent_shapes)
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

    def record_step(self, run, token_scores, error_rms, source_rms, step, sigma,
                    cond_or_uncond=0, dense_wall_s=None, source_kind="guided"):
        cfg = self.config
        core, expanded, _ = mask_ops.build_mask(
            token_scores, cfg.score_threshold, cfg.tile_h, cfg.tile_w,
            cfg.spatial_halo, cfg.temporal_halo)
        frozen_before = run.frozen_mask
        row = {
            "step": int(step), "sigma": float(sigma), "source_kind": source_kind,
            "cond_or_uncond": int(cond_or_uncond), "sigma_index": run.sigma_count - 1,
            "dense_wall_s": dense_wall_s,
            "score_quantiles": mask_ops.quantiles(token_scores, SCORE_QUANTILES),
            "saliency_quantiles": mask_ops.quantiles(mask_ops.spatial_saliency(token_scores), SCORE_QUANTILES),
            "threshold": cfg.score_threshold,
            "active_core": mask_ops.active_fraction(core),
            "active_expanded": mask_ops.active_fraction(expanded),
            "threshold_sweep": mask_ops.threshold_sweep(
                token_scores, THRESHOLD_SWEEP, cfg.tile_h, cfg.tile_w,
                cfg.spatial_halo, cfg.temporal_halo),
            "jaccard_prev": mask_ops.jaccard(expanded, run.prev_observed),
            "escaped_prev": mask_ops.escaped_fraction(expanded, run.prev_observed),
            "escaped_union": mask_ops.escaped_fraction(expanded, run.union_mask),
            "escaped_frozen": mask_ops.escaped_fraction(expanded, frozen_before),
            "coverage_frozen": mask_ops.coverage_fraction(expanded, frozen_before),
            "missed_score_mass_frozen": mask_ops.missed_score_mass(
                token_scores, frozen_before, cfg.score_threshold),
        }
        run.steps.append(row)
        label = "s%03d_%s" % (len(run.steps) - 1, source_kind)
        run.score_maps.append((label, token_scores.detach().cpu().float()))
        run.error_maps.append((label, error_rms.detach().cpu().float()))
        run.source_maps.append((label, source_rms.detach().cpu().float()))
        run.saliency_maps.append((label, mask_ops.spatial_saliency(token_scores).detach().cpu().float()))
        run.masks.append((label, expanded.detach().cpu()))
        run.stage_mask(expanded)
        run.prev_observed = expanded
        return row

    def record_final(self, run, token_scores, error_rms, source_rms):
        cfg = self.config
        core, expanded, _ = mask_ops.build_mask(
            token_scores, cfg.score_threshold, cfg.tile_h, cfg.tile_w,
            cfg.spatial_halo, cfg.temporal_halo)
        run.final = {
            "active_core": mask_ops.active_fraction(core),
            "active_expanded": mask_ops.active_fraction(expanded),
            "coverage_by_frozen": mask_ops.coverage_fraction(expanded, run.frozen_mask),
            "escaped_frozen": mask_ops.escaped_fraction(expanded, run.frozen_mask),
            "missed_score_mass_frozen": mask_ops.missed_score_mass(
                token_scores, run.frozen_mask, cfg.score_threshold),
            "score_quantiles": mask_ops.quantiles(token_scores, SCORE_QUANTILES),
        }
        run.score_maps.append(("final", token_scores.detach().cpu().float()))
        run.error_maps.append(("final", error_rms.detach().cpu().float()))
        run.source_maps.append(("final", source_rms.detach().cpu().float()))
        run.saliency_maps.append(("final", mask_ops.spatial_saliency(token_scores).detach().cpu().float()))
        run.masks.append(("final", expanded.detach().cpu()))
        return run.final
