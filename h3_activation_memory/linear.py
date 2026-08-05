"""Bounded MLP execution and safe held-weight sessions."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

import comfy.ops
import comfy.quant_ops
from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout


class UnsafeHeldWeights(RuntimeError):
    """Two acquired weights would alias the same reusable async cast buffer."""


def swiglu_eager(x):
    gate, up = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(gate).mul_(up)


def _stream_from_handle(handle):
    if isinstance(handle, tuple) and handle:
        return handle[0]
    return None


@dataclass
class AcquiredLinear:
    module: object
    weight: object
    bias: object
    handle: object
    released: bool = False

    def linear(self, x):
        return F.linear(x, self.weight, self.bias)

    def release(self):
        if self.released:
            return
        comfy.ops.uncast_bias_weight(
            self.module, self.weight, self.bias, self.handle
        )
        self.released = True


def acquire_linear(module, sample, want_requant=True):
    """Acquire one Comfy linear weight once, preserving quantized dispatch."""
    weight, bias, handle = comfy.ops.cast_bias_weight(
        module,
        sample,
        offloadable=True,
        compute_dtype=sample.dtype,
        want_requant=bool(want_requant),
    )
    return AcquiredLinear(module, weight, bias, handle)


class HeldMLP:
    """Hold fc1/fc2 across every token slab when Comfy's buffers permit it."""

    def __init__(self, mlp, sample):
        self.mlp = mlp
        self.sample = sample
        self.fc1_weight = None
        self.fc2_weight = None

    def __enter__(self):
        try:
            self.fc1_weight = acquire_linear(self.mlp.fc1, self.sample)
            self.fc2_weight = acquire_linear(self.mlp.fc2, self.sample)
            stream1 = _stream_from_handle(self.fc1_weight.handle)
            stream2 = _stream_from_handle(self.fc2_weight.handle)
            if stream1 is not None and stream1 is stream2:
                raise UnsafeHeldWeights(
                    "fc1 and fc2 were acquired from the same async cast stream; "
                    "the second reusable cast buffer can overwrite the first"
                )
            return self
        except Exception:
            self.release()
            raise

    def release(self):
        if self.fc2_weight is not None:
            self.fc2_weight.release()
            self.fc2_weight = None
        if self.fc1_weight is not None:
            self.fc1_weight.release()
            self.fc1_weight = None

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False

    def fc1(self, x):
        # Ordinary Comfy module forwards call this before every operation. The
        # held path bypasses the module wrapper, so preserve cancellation.
        comfy.ops.run_every_op()
        return self.fc1_weight.linear(x)

    def fc2_swiglu(self, expanded, native):
        weight = self.fc2_weight.weight
        bias = self.fc2_weight.bias
        if (
            native
            and isinstance(weight, QuantizedTensor)
            and getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(weight._params, "transposed", False)
        ):
            qdata, scale = TensorWiseINT8Layout.get_plain_tensors(weight)
            out = comfy.quant_ops.ck.int8_linear(
                expanded,
                qdata,
                scale,
                bias,
                expanded.dtype,
                convrot=getattr(weight._params, "convrot", False),
                convrot_groupsize=getattr(
                    weight._params, "convrot_groupsize", 256
                ),
                input_act="swiglu",
            )
            return out, "held_tensorwise_int8_native"

        activated = swiglu_eager(expanded)
        return F.linear(activated, weight, bias), "held_bf16_swiglu"


def module_mlp_chunk(mlp, h, native):
    """Correct fallback using ordinary Comfy module calls for one bounded slab."""
    expanded = mlp.fc1(h)
    if native:
        return (
            comfy.ops.linear_input_act(mlp.fc2, expanded, "swiglu"),
            "module_native",
        )
    return mlp.fc2(swiglu_eager(expanded)), "module_bf16_swiglu"
