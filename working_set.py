"""Conservative H3 working-set signatures and upper bounds."""

from dataclasses import dataclass
import math

import torch
from comfy.ldm.minimax.model import PackedLayout

try:
    from .h3_probe.layout import from_packed_layout
except ImportError:
    from h3_probe.layout import from_packed_layout


CALIBRATED_BYTES_PER_ROW = 128 * 1024
CALIBRATED_SOURCE = "calibrated H3 envelope, rounded from 118750 B/row to 128 KiB/row"
OBSERVED_ALLOWANCE = 1.10


class UnknownWorkingSet(RuntimeError):
    pass


@dataclass(frozen=True)
class ForwardSignature:
    seq_len: int
    segments: tuple
    layout: tuple
    compute_dtype: str
    attention: tuple
    activation: tuple
    cfg_batch: tuple
    compiled: bool
    capability: tuple | None
    runtime: tuple


def resolve_layout(args):
    x = args.get("input")
    c = args.get("c") or {}
    tensors = getattr(x, "tensors", None)
    context = c.get("c_crossattn")
    if not tensors or len(tensors) < 2 or context is None:
        raise UnknownWorkingSet("the packed H3 layout is unavailable before this forward")
    video, audio = tensors[:2]
    if len(video.shape) != 5:
        raise UnknownWorkingSet("the H3 video latent is not rank 5")
    text_len = int(context.shape[1])
    latent_t = int(video.shape[2])
    latent_h = (int(video.shape[3]) + 1) // 2 * 2
    latent_w = (int(video.shape[4]) + 1) // 2 * 2
    audio_t = int(audio.shape[-1])
    signature = (text_len, latent_t, latent_h, latent_w, audio_t)
    payload = c.get("minimax_payload") or {}
    packed = payload.get("layout")
    if packed is None or tuple(getattr(packed, "signature", ())) != signature:
        packed = PackedLayout(
            text_len,
            latent_t,
            latent_h,
            latent_w,
            audio_t,
            keyframes=payload.get("keyframes"),
            refs=payload.get("refs"),
            frame_count=payload.get("frame_count"),
        )
    return from_packed_layout(packed)


def _diffusion_model(model_patcher):
    model = getattr(model_patcher, "model", None)
    return getattr(model, "diffusion_model", model)


def _freeze(value):
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _execution_identity(model_patcher, transformer_options):
    model = _diffusion_model(model_patcher)
    blocks = getattr(model, "blocks", ())
    block = blocks[0] if len(blocks) else None
    block_forward = getattr(block, "forward", None)
    attention_forward = getattr(getattr(block, "attn", None), "forward", None)
    configured = transformer_options.get("minimax_h3_attention_backend", "comfy")
    attention = (
        str(configured),
        getattr(attention_forward, "_h3_backend", None),
        getattr(attention_forward, "_h3_projector", None),
        _freeze(getattr(attention_forward, "_h3_installation_signature", None)),
    )
    activation = tuple(getattr(block_forward, "_h3_activation_config", ()) or ())
    compiled = bool(getattr(block_forward, "_h3_shared_block_compile", False))
    return attention, activation, compiled


def make_signature(args, model_patcher):
    layout = resolve_layout(args)
    c = args.get("c") or {}
    transformer_options = c.get("transformer_options") or {}
    attention, activation, compiled = _execution_identity(model_patcher, transformer_options)
    context = c.get("c_crossattn")
    dtype = str(getattr(context, "dtype", "unknown"))
    tensors = getattr(args.get("input"), "tensors", ())
    batches = tuple(int(t.shape[0]) for t in tensors)
    branch = tuple(int(value) for value in (args.get("cond_or_uncond") or ()))
    device = getattr(tensors[0], "device", None) if tensors else getattr(model_patcher, "load_device", None)
    capability = None
    if getattr(device, "type", None) == "cuda":
        capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    return ForwardSignature(
        int(layout.seq_len),
        tuple((int(a), int(b), str(kind)) for a, b, kind in layout.segments),
        (tuple(int(value) for value in layout.video_shape), int(layout.audio_t)),
        dtype,
        attention,
        activation,
        (branch, batches),
        compiled,
        capability,
        (str(torch.__version__), str(getattr(torch.version, "cuda", None))),
    )


_OBSERVED = {}


def observed(signature):
    return _OBSERVED.get(signature)


def record_observed(signature, peak_increment):
    peak_increment = max(0, int(peak_increment))
    _OBSERVED[signature] = max(peak_increment, _OBSERVED.get(signature, 0))
    return _OBSERVED[signature]


def clear_observed():
    _OBSERVED.clear()


def upper_bound(signature):
    calibrated = int(signature.seq_len) * CALIBRATED_BYTES_PER_ROW
    measured = observed(signature)
    if measured is None:
        return calibrated, CALIBRATED_SOURCE
    measured_bound = int(math.ceil(measured * OBSERVED_ALLOWANCE))
    if measured_bound > calibrated:
        return measured_bound, "observed signature peak + 10% allowance"
    return calibrated, CALIBRATED_SOURCE
