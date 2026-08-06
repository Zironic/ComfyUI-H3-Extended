import logging

from .cache import CacheEntry
from .metrics import residual_summary, tensor_metrics
from .policy import decide_action
from .report import RunReport
from .units import parse_unit_spec, unit_map

LOG_PREFIX = "[H3 Extended] block cache"

class BlockCacheSession:
    def __init__(self, config, output_root=None):
        self.config = config
        self.units = parse_unit_spec(config.unit_spec)
        self.by_block = unit_map(self.units)
        self.output_root = output_root
        self.report = None
        self.step = -1
        self.sigma = None
        self.total_steps = 0
        self.branch = 0
        self.entries = {}
        self.range_starts = {}
        self.range_actions = {}
        self.active = False

    def begin(self):
        self.step = -1
        self.sigma = None
        self.total_steps = 0
        self.branch = 0
        self.entries.clear()
        self.range_starts.clear()
        self.range_actions.clear()
        self.active = True
        if self.output_root:
            self.report = RunReport(self.output_root, self.config.run_tag, self.config, self.units)
        return self

    def end(self):
        self.active = False
        self.range_starts.clear()
        self.range_actions.clear()
        self.entries.clear()
        if self.report:
            return self.report.finish()
        return None

    def prepare_forward(self, transformer_options):
        sigma_t = transformer_options.get("sigmas")
        if sigma_t is None:
            raise RuntimeError("no sigma in transformer_options")
        sigma = float(sigma_t.flatten()[0])
        if self.sigma is None or sigma != self.sigma:
            self.step += 1
            self.sigma = sigma
            self.range_starts.clear()
            self.range_actions.clear()
        sched = transformer_options.get("sample_sigmas")
        if sched is not None:
            self.total_steps = max(0, int(sched.numel()) - 1)
        cu = transformer_options.get("cond_or_uncond") or [0]
        self.branch = int(cu[0])
        # AIMDO remains mandatory. Only H3's static all-block lookahead queue is
        # disabled; executed modules still fault through AIMDO on demand.
        transformer_options["prefetch_dynamic_vbars"] = False

    def _entry(self, unit):
        key = (self.branch, unit.key)
        return self.entries.setdefault(key, CacheEntry())

    def action(self, unit):
        key = (self.branch, unit.key, self.step)
        if key not in self.range_actions:
            self.range_actions[key] = decide_action(
                self.config, self._entry(unit), self.step, self.total_steps)
        return self.range_actions[key]

    def block(self, index, args, original_block):
        unit = self.by_block.get(index)
        if unit is None:
            return original_block(args)["img"]
        x = args["img"]
        entry = self._entry(unit)
        action, reason = self.action(unit)

        if action == "reuse":
            if index == unit.start:
                out = entry.apply(x)
                self._record(unit, index, action, reason, entry, None)
                return out
            self._record(unit, index, "reuse-interior", reason, entry, None)
            return x

        if index == unit.start:
            self.range_starts[(self.branch, unit.key, self.step)] = x.detach().clone()

        out = original_block(args)["img"]

        if index == unit.stop:
            start_key = (self.branch, unit.key, self.step)
            before = self.range_starts.pop(start_key)
            residual = out.detach() - before
            shadow = None
            if self.config.mode == "shadow" and entry.valid:
                shadow = tensor_metrics(before + entry.residual.to(before), out)
            entry.store(residual, self.step)
            self._record(unit, index, "refresh", reason, entry, shadow,
                         residual_summary(residual))
        return out

    def _record(self, unit, index, action, reason, entry, shadow=None, residual=None):
        row = {
            "step": self.step,
            "sigma": self.sigma,
            "branch": self.branch,
            "unit": unit.key,
            "block": index,
            "action": action,
            "reason": reason,
            "reuse_count": entry.reuse_count,
        }
        if shadow is not None:
            row["shadow"] = shadow
        if residual is not None:
            row["residual"] = residual
        if self.report:
            self.report.row(row)
        logging.debug("%s %s", LOG_PREFIX, row)
