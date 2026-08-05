"""Checkpoint inventory and optional CUDA phase-memory tracing."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

try:
    from comfy.quant_ops import QuantizedTensor
except ImportError:
    class QuantizedTensor:
        pass


def _weight_record(module):
    weight = getattr(module, "weight", None)
    if weight is None:
        return {"present": False}
    params = getattr(weight, "_params", None)
    return {
        "present": True,
        "class": type(module).__name__,
        "shape": list(weight.shape),
        "dtype": str(weight.dtype),
        "device": str(weight.device),
        "quantized": isinstance(weight, QuantizedTensor),
        "layout": getattr(weight, "_layout_cls", None),
        "transposed": bool(getattr(params, "transposed", False)),
        "weight_functions": len(getattr(module, "weight_function", ())),
        "bias_functions": len(getattr(module, "bias_function", ())),
        "has_lowvram_weight_function": hasattr(
            module, "weight_lowvram_function"
        ),
        "vbar_backed": hasattr(module, "_v"),
    }


def inventory_blocks(model_patcher):
    blocks = model_patcher.get_model_object("diffusion_model.blocks")
    result = []
    for index, block in enumerate(blocks):
        result.append(
            {
                "index": index,
                "norm1": type(block.norm1).__name__,
                "norm2": type(block.norm2).__name__,
                "qkv_proj": _weight_record(block.attn.qkv_proj),
                "out_proj": _weight_record(block.attn.out_proj),
                "fc1": _weight_record(block.mlp.fc1),
                "fc2": _weight_record(block.mlp.fc2),
            }
        )
    return result


@dataclass
class MemoryEvent:
    event: str
    layer_index: int
    allocated: int
    peak_allocated: int
    payload: dict


class MemoryTrace:
    """Observer callback compatible with ``h3_activation_memory.observing``."""

    def __init__(self, synchronize=False):
        self.synchronize = bool(synchronize)
        self.events = []

    def __call__(self, event, layer_index, payload):
        if torch.cuda.is_available():
            if self.synchronize:
                torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
            peak = torch.cuda.max_memory_allocated()
        else:
            allocated = peak = 0
        self.events.append(
            MemoryEvent(event, int(layer_index), allocated, peak, dict(payload))
        )

    def as_dicts(self):
        return [asdict(event) for event in self.events]

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dicts(), indent=2), encoding="utf-8")
        return path
