"""Evaluation policies for native and fixed-mask vector acceleration."""

from dataclasses import dataclass

from .config import SamplerConfig


@dataclass(frozen=True)
class Decision:
    is_forecast: bool
    reason: str


class NativePolicy:
    def reset(self):
        pass

    def decide(self, step, **kwargs) -> Decision:
        return Decision(False, "native")


class FixedMaskPolicy:
    def __init__(self, mask):
        self.mask = tuple(bool(value) for value in mask)

    def reset(self):
        pass

    def decide(self, step, **kwargs) -> Decision:
        if step < 0 or step >= len(self.mask):
            return Decision(False, "mask_out_of_range")
        return Decision(not self.mask[step], "fixed_mask_forecast" if not self.mask[step] else "fixed_mask_actual")


def make_policy(config: SamplerConfig):
    return NativePolicy() if config.method == "native" else FixedMaskPolicy(config.mask)
