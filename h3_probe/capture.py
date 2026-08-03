"""Selective attention instrumentation for MiniMax H3.

Interception point is the module-global `optimized_attention` name inside
`comfy.ldm.minimax.model`, so the patch is H3-scoped by construction: no other
architecture resolves attention through that binding. When no probe is armed the
patched function is a straight delegation, so normal inference is unaffected.

For each *selected* (layer, step, query block) the probe computes that block's
exact dense attention against all keys, reduces the softmax rows to their mean
distribution, and keeps only aggregates:

    block_mass    [H, n_blocks]     mass per BLOCK-token KV block
    cat_mass      [H, n_kinds]      mass per packed segment kind (exact)
    frame_mass    [H, latent_t]     mass per target-video latent frame (exact)
    spatial_mass  [frame_rows]      mass per spatial patch, summed over frames

Full S x S attention is never materialized or saved; only [Bq, S] scores for one
head chunk at a time exist transiently.
"""

import logging
import math
import os
import threading

import torch

from . import layout as h3_layout

BLOCK = 128
HEAD_CHUNK = 4

_active = threading.local()
_orig_attention = None

_KINDS = (h3_layout.KIND_TEXT, h3_layout.KIND_COND, h3_layout.KIND_REF_IMG,
          h3_layout.KIND_REF_AUDIO, h3_layout.KIND_AUDIO, h3_layout.KIND_VIDEO)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def resolve_indices(spec, total, fracs=(0.1, 0.5, 0.9)):
    """Turn a spec string into sorted, de-duplicated indices in [0, total).

    "auto" (or empty) picks early/middle/late positions; otherwise the spec is a
    comma-separated list of integers, negatives counting from the end.
    """
    spec = (spec or "").strip()
    if spec in ("", "auto"):
        idxs = [min(total - 1, max(0, int(round(f * (total - 1))))) for f in fracs]
    else:
        idxs = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            i = int(part)
            if i < 0:
                i += total
            if 0 <= i < total:
                idxs.append(i)
    return sorted(set(idxs))


def select_query_blocks(layout, n_time, n_spatial, block=BLOCK, include_audio=True, include_text=False):
    """Choose the query blocks to probe.

    Video blocks are picked semantically (latent frame + spatial offset) rather
    than snapped to the global block grid, so each one is attributable to a
    single frame and reads as "t=8" in the report.
    """
    out = []
    latent_t = layout.latent_t
    frame_rows = layout.frame_rows

    t_idx = resolve_indices("auto", latent_t, fracs=[i / max(1, n_time - 1) for i in range(n_time)]) \
        if n_time > 1 else [latent_t // 2]
    for t in t_idx:
        f0, f1 = layout.video_frame_range(t)
        span = min(block, f1 - f0)
        if n_spatial <= 1:
            offsets = [max(0, (frame_rows - span) // 2)]
        else:
            step = max(1, (frame_rows - span) // (n_spatial - 1)) if frame_rows > span else 0
            offsets = sorted({min(i * step, frame_rows - span) for i in range(n_spatial)})
        for off in offsets:
            qs = f0 + off
            out.append({"kind": "video", "frame": t, "spatial_offset": off,
                        "start": qs, "stop": min(qs + span, f1)})

    if include_audio:
        a0, a1 = layout.audio_range
        span = min(block, a1 - a0)
        if span > 0:
            mid = a0 + max(0, ((a1 - a0) // 2 - span // 2))
            out.append({"kind": "audio", "frame": None, "spatial_offset": None,
                        "start": mid, "stop": min(mid + span, a1)})

    if include_text:
        t0, t1 = layout.text_range
        span = min(block, t1 - t0)
        if span > 0:
            out.append({"kind": "text", "frame": None, "spatial_offset": None,
                        "start": t0, "stop": t0 + span})

    return out


# --------------------------------------------------------------------------
# per-query-block statistics
# --------------------------------------------------------------------------

def _block_stats(q, k, layout, qs, qe, block, head_chunk=HEAD_CHUNK):
    """Exact dense attention for one query block, reduced to aggregates.

    q, k: [1, H, S, D] post-rope, exactly what the real attention call receives.
    """
    _, heads, seq, dim = q.shape
    scale = 1.0 / math.sqrt(dim)
    n_blocks = (seq + block - 1) // block
    pad = n_blocks * block - seq

    v0, v1 = layout.video_range
    latent_t, ph, pw = layout.video_shape
    frame_rows = ph * pw

    blk, cats, frames, spatial = [], [], [], []
    q_sel = q[0, :, qs:qe, :].to(torch.float32)          # [H, Bq, D]
    k_all = k[0]                                          # [H, S, D]

    for h0 in range(0, heads, head_chunk):
        h1 = min(h0 + head_chunk, heads)
        scores = torch.matmul(q_sel[h0:h1], k_all[h0:h1].to(torch.float32).transpose(-1, -2)) * scale
        p = torch.softmax(scores, dim=-1)                 # [hc, Bq, S]
        del scores
        pm = p.mean(dim=1)                                # [hc, S] mean attention distribution
        del p

        padded = torch.nn.functional.pad(pm, (0, pad)) if pad else pm
        blk.append(padded.reshape(pm.shape[0], n_blocks, block).sum(-1))

        cat = torch.zeros(pm.shape[0], len(_KINDS), dtype=torch.float32, device=pm.device)
        for a, b, kind in layout.segments:
            cat[:, _KINDS.index(kind)] += pm[:, a:b].sum(-1)
        cats.append(cat)

        vid = pm[:, v0:v1].reshape(pm.shape[0], latent_t, frame_rows)
        frames.append(vid.sum(-1))
        spatial.append(vid.sum(1))
        del pm, vid

    return {
        "block_mass": torch.cat(blk).cpu(),                        # [H, n_blocks]
        "cat_mass": torch.cat(cats).cpu(),                         # [H, len(_KINDS)]
        "frame_mass": torch.cat(frames).cpu(),                     # [H, latent_t]
        "spatial_mass": torch.cat(spatial).mean(0).cpu(),          # [frame_rows] head-mean
        "n_blocks": n_blocks,
        "block": block,
    }


# --------------------------------------------------------------------------
# per-forward probe state
# --------------------------------------------------------------------------

class ForwardProbe:
    """Live state for one `_forward` call: which layer we are in, what to grab."""

    def __init__(self, run, layout, step, sigma, cond_or_uncond):
        self.run = run
        self.layout = layout
        self.step = step
        self.sigma = sigma
        self.cond_or_uncond = cond_or_uncond
        self.layer = -1
        self.queries = None

    def observe(self, q, k):
        # the token refiner shares this attention path but runs on the text span
        # alone; only full packed-sequence calls are DiT layers
        if q.shape[2] != self.layout.seq_len:
            return
        self.layer += 1
        if self.layer not in self.run.layers:
            return
        if self.queries is None:
            self.queries = select_query_blocks(
                self.layout, self.run.n_time, self.run.n_spatial, self.run.block,
                include_audio=self.run.include_audio, include_text=self.run.include_text)
        for spec in self.queries:
            stats = _block_stats(q, k, self.layout, spec["start"], spec["stop"], self.run.block)
            rec = dict(spec)
            rec.update(stats)
            rec.update({"layer": self.layer, "step": self.step, "sigma": self.sigma,
                        "cond_or_uncond": self.cond_or_uncond})
            self.run.records.append(rec)


def _probed_attention(q, k, v, heads, *args, **kwargs):
    state = getattr(_active, "probe", None)
    if state is not None:
        try:
            with torch.no_grad():
                state.observe(q, k)
        except Exception:
            logging.exception("[H3 probe] capture failed; continuing inference")
    return _orig_attention(q, k, v, heads, *args, **kwargs)


def install():
    """Swap the H3 module's attention binding. Idempotent."""
    global _orig_attention
    import comfy.ldm.minimax.model as mm
    if getattr(mm.optimized_attention, "_h3_probe", False):
        return
    _orig_attention = mm.optimized_attention
    _probed_attention._h3_probe = True
    mm.optimized_attention = _probed_attention
    logging.info("[H3 probe] attention interception installed")


def uninstall():
    global _orig_attention
    import comfy.ldm.minimax.model as mm
    if _orig_attention is not None and getattr(mm.optimized_attention, "_h3_probe", False):
        mm.optimized_attention = _orig_attention
        _orig_attention = None
        logging.info("[H3 probe] attention interception removed")


# --------------------------------------------------------------------------
# run store + diffusion-model wrapper
# --------------------------------------------------------------------------

class ProbeRun:
    """Accumulates records for one sampling run and owns the output files."""

    def __init__(self, session, tag, out_dir):
        self.tag = tag
        self.out_dir = out_dir
        self.layers_spec = session.layers_spec
        self.steps_spec = session.steps_spec
        self.n_time = session.n_time
        self.n_spatial = session.n_spatial
        self.block = session.block
        self.include_audio = session.include_audio
        self.include_text = session.include_text
        self.capture_uncond = session.capture_uncond
        self.layers = set()
        self.steps = set()
        self.records = []
        self.layout = None
        self.notes = {}


class ProbeSession:
    """Probe configuration, plus the run currently being sampled.

    A fresh `ProbeRun` per sampling run keeps re-queues from accumulating into
    one trace and gives every run its own output directory.
    """

    def __init__(self, tag, layers_spec, steps_spec, n_time, n_spatial, block,
                 include_audio, include_text, capture_uncond, base_dir):
        self.tag = tag
        self.layers_spec = layers_spec
        self.steps_spec = steps_spec
        self.n_time = n_time
        self.n_spatial = n_spatial
        self.block = block
        self.include_audio = include_audio
        self.include_text = include_text
        self.capture_uncond = capture_uncond
        self.base_dir = base_dir
        self.run = None

    def begin(self):
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s_%s" % (self.tag, stamp)
        self.run = ProbeRun(self, name, os.path.join(self.base_dir, name))
        return self.run

    def end(self):
        run = self.run
        self.run = None
        if run is None or not run.records:
            logging.warning("[H3 probe] run finished with no captures "
                            "(is this an H3 model, and do the selected steps exist?)")
            return None
        from .report import write_run
        path = write_run(run)
        logging.info("[H3 probe] %d records -> %s", len(run.records), path)
        return path


def _step_index(transformer_options):
    """Locate the current sigma in the run's schedule."""
    sched = transformer_options.get("sample_sigmas")
    sigma = transformer_options.get("sigmas")
    if sigma is None:
        return -1, float("nan")
    s = float(sigma.flatten()[0])
    if sched is None:
        return -1, s
    sched = sched.flatten().to("cpu", torch.float32)
    idx = int(torch.argmin((sched - s).abs()))
    return idx, s


def make_outer_wrapper(session):
    """OUTER_SAMPLE wrapper: one trace per sampling run."""

    def wrapper(executor, *args, **kwargs):
        session.begin()
        try:
            return executor(*args, **kwargs)
        finally:
            try:
                session.end()
            except Exception:
                logging.exception("[H3 probe] final report write failed")

    return wrapper


def make_wrapper(session):
    """DIFFUSION_MODEL wrapper: publish layout metadata, arm the probe."""

    def wrapper(executor, *args, **kwargs):
        run = session.run
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        x = args[0]
        context = args[2]
        payload = kwargs.get("minimax_payload") or {}

        try:
            layout = h3_layout.resolve_layout(x, context, payload)
        except Exception:
            logging.exception("[H3 probe] could not resolve token layout; probe disabled for this pass")
            return executor(*args, **kwargs)

        # metadata travels through transformer_options; nothing in core reads
        # these keys, so inference is unchanged whether or not the probe fires
        transformer_options["minimax_h3_token_layout"] = layout
        transformer_options["minimax_h3_token_ranges"] = layout.as_dict()

        if run is None:
            # metadata is published regardless; capture needs an active run
            return executor(*args, **kwargs)

        if run.layout is None:
            run.layout = layout
            total_steps = 0
            sched = transformer_options.get("sample_sigmas")
            if sched is not None:
                total_steps = max(1, sched.numel() - 1)
            model = getattr(executor, "class_obj", None)
            num_layers = len(getattr(model, "blocks", [])) or 50
            run.layers = set(resolve_indices(run.layers_spec, num_layers))
            run.steps = set(resolve_indices(run.steps_spec, total_steps or 1))
            run.notes["total_steps"] = total_steps
            run.notes["num_layers"] = num_layers
            logging.info("[H3 probe] layout: %s", layout.describe())
            logging.info("[H3 probe] layers=%s steps=%s (of %d layers, %d steps)",
                         sorted(run.layers), sorted(run.steps), num_layers, total_steps)

        step, sigma = _step_index(transformer_options)
        cu = transformer_options.get("cond_or_uncond") or [0]
        cond_or_uncond = int(cu[0])

        armed = step in run.steps and (run.capture_uncond or cond_or_uncond == 0)
        if not armed:
            return executor(*args, **kwargs)

        _active.probe = ForwardProbe(run, layout, step, sigma, cond_or_uncond)
        try:
            return executor(*args, **kwargs)
        finally:
            _active.probe = None
            try:
                from .report import write_run
                write_run(run)
            except Exception:
                logging.exception("[H3 probe] report write failed")

    return wrapper
