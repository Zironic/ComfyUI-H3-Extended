"""Selective capture for the probe-only 3D MoBA-style H3 router."""

from __future__ import annotations

import logging
import os

from . import capture, layout as h3_layout, moba3d

try:
    from ..h3_attention.observer import observing
except ImportError:
    from h3_attention.observer import observing


class MobaProbeRun:
    def __init__(self, session, tag, out_dir):
        self.tag = tag
        self.out_dir = out_dir
        self.layers_spec = session.layers_spec
        self.steps_spec = session.steps_spec
        self.n_time = session.n_time
        self.n_spatial = session.n_spatial
        self.query_block = session.query_block
        self.include_audio = session.include_audio
        self.include_text = session.include_text
        self.capture_uncond = session.capture_uncond
        self.block_t = session.block_t
        self.block_h = session.block_h
        self.block_w = session.block_w
        self.budgets = session.budgets
        self.layers = set()
        self.steps = set()
        self.records = []
        self.layout = None
        self.notes = {}


class MobaProbeSession:
    def __init__(self, tag, layers_spec, steps_spec, n_time, n_spatial, query_block,
                 include_audio, include_text, capture_uncond, block_t, block_h,
                 block_w, budgets, base_dir):
        self.tag = tag
        self.layers_spec = layers_spec
        self.steps_spec = steps_spec
        self.n_time = int(n_time)
        self.n_spatial = int(n_spatial)
        self.query_block = int(query_block)
        self.include_audio = bool(include_audio)
        self.include_text = bool(include_text)
        self.capture_uncond = bool(capture_uncond)
        self.block_t = int(block_t)
        self.block_h = int(block_h)
        self.block_w = int(block_w)
        self.budgets = moba3d.parse_budgets(budgets)
        self.base_dir = base_dir
        self.run = None

    def begin(self):
        import time
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = "%s_%s" % (self.tag, stamp)
        self.run = MobaProbeRun(self, name, os.path.join(self.base_dir, name))
        return self.run

    def end(self):
        run = self.run
        self.run = None
        if run is None or not run.records:
            logging.warning("[H3 MoBA3D probe] run finished with no captures")
            return None
        from .moba_report import write_run
        path = write_run(run)
        logging.info("[H3 MoBA3D probe] %d records -> %s", len(run.records), path)
        return path


class ForwardMobaProbe:
    """Live state for one H3 forward pass."""

    def __init__(self, run, layout, step, sigma, cond_or_uncond):
        self.run = run
        self.layout = layout
        self.step = step
        self.sigma = sigma
        self.cond_or_uncond = cond_or_uncond
        self.layer = -1
        self.queries = None
        self.explicit = False

    def observe(self, q, k, layer_index=None):
        if q.shape[2] != self.layout.seq_len:
            return
        if layer_index is None:
            if self.explicit:
                return
            self.layer += 1
            layer = self.layer
        else:
            self.explicit = True
            layer = int(layer_index)
        if layer not in self.run.layers:
            return

        if self.queries is None:
            self.queries = capture.select_query_blocks(
                self.layout,
                self.run.n_time,
                self.run.n_spatial,
                self.run.query_block,
                include_audio=self.run.include_audio,
                include_text=self.run.include_text,
            )

        for spec in self.queries:
            result = moba3d.analyze_routing(
                q,
                k,
                self.layout,
                spec["start"],
                spec["stop"],
                block_t=self.run.block_t,
                block_h=self.run.block_h,
                block_w=self.run.block_w,
                budgets=self.run.budgets,
            )
            rec = dict(spec)
            rec.update({
                "layer": layer,
                "step": self.step,
                "sigma": self.sigma,
                "cond_or_uncond": self.cond_or_uncond,
                "moba3d": result,
            })
            self.run.records.append(rec)


def make_outer_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        session.begin()
        try:
            return executor(*args, **kwargs)
        finally:
            try:
                session.end()
            except Exception:
                logging.exception("[H3 MoBA3D probe] final report write failed")
    return wrapper


def make_wrapper(session):
    def wrapper(executor, *args, **kwargs):
        run = session.run
        transformer_options = args[3] if len(args) > 3 else kwargs.get("transformer_options", {})
        x = args[0]
        context = args[2]
        payload = kwargs.get("minimax_payload") or {}

        try:
            layout = h3_layout.resolve_layout(x, context, payload)
        except Exception:
            logging.exception("[H3 MoBA3D probe] could not resolve token layout")
            return executor(*args, **kwargs)

        transformer_options["minimax_h3_token_layout"] = layout
        transformer_options["minimax_h3_token_ranges"] = layout.as_dict()

        if run is None:
            return executor(*args, **kwargs)

        if run.layout is None:
            run.layout = layout
            sched = transformer_options.get("sample_sigmas")
            total_steps = max(1, sched.numel() - 1) if sched is not None else 1
            model = getattr(executor, "class_obj", None)
            num_layers = len(getattr(model, "blocks", [])) or 50
            run.layers = set(capture.resolve_indices(run.layers_spec, num_layers))
            run.steps = set(capture.resolve_indices(run.steps_spec, total_steps))
            run.notes.update({
                "total_steps": int(total_steps),
                "num_layers": int(num_layers),
                "mode": "probe-only parameter-free 3D mean-pooled routing",
            })
            logging.info(
                "[H3 MoBA3D probe] layout=%s layers=%s steps=%s block=%dx%dx%d budgets=%s",
                layout.describe(), sorted(run.layers), sorted(run.steps),
                run.block_t, run.block_h, run.block_w, run.budgets,
            )

        step, sigma = capture._step_index(transformer_options)
        cu = transformer_options.get("cond_or_uncond") or [0]
        cond_or_uncond = int(cu[0])
        if step not in run.steps or (not run.capture_uncond and cond_or_uncond != 0):
            return executor(*args, **kwargs)

        probe = ForwardMobaProbe(run, layout, step, sigma, cond_or_uncond)
        try:
            with observing(transformer_options, probe.observe):
                return executor(*args, **kwargs)
        finally:
            try:
                from .moba_report import write_run
                write_run(run)
            except Exception:
                logging.exception("[H3 MoBA3D probe] report write failed")

    return wrapper
