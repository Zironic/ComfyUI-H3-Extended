"""TAEH3: the tiny H3 decoder used as the default live-preview backend.

Loading
-------
``comfy.taesd.taehv.TAEHV`` mis-builds these weights. It only raises
``patch_size`` to 2 for 48- and 32-channel latents, but taeh3 is a 24-channel
model with ``patch_size == 2``: its first encoder conv takes ``3 * 2**2 == 12``
channels and its last decoder conv emits 12. Constructed as ``TAEHV(24)`` those
two convs come out 3-channel and a strict load fails. :func:`load_taeh3` derives
the patch size from the weights and rebuilds exactly those two layers, which is
also what makes the spatial arithmetic work: three 2x upsamples times a 2x
pixel-shuffle is 16x, matching ``MiniMaxH3Video.spacial_downscale_ratio``. TAEH3
previews are therefore full resolution, not half.

No latent scaling is applied. TAEHV is trained directly on diffusion latents and
``MiniMaxH3Video.scale_factor`` is 1.0, so ``process_latent_out`` would be the
identity anyway.

The temporal mismatch
---------------------
TAEHV is a fixed **4x** temporal upscaler (two stride-2 ``TGrow`` blocks, read
off the weights), so ``L`` latents give ``4L - 3`` frames. The real H3 VAE is
**17k+5 frames <-> 5k+2 latents**, i.e. 3.4 frames per latent. They coincide
only at ``L == 2``. Left alone a TAEH3 preview therefore runs ~15% slow and
reports more frames than the chunk really has, so :meth:`TAEH3Previewer.frames`
index-resamples its output down to ``geometry.decoded_frame_count(L)``. That is
a display correction, not a quality one; the native count is logged.
"""

from __future__ import annotations

import logging
import os

import torch

import comfy.model_management
import comfy.utils
from comfy.taesd.taehv import (
    TAEHV,
    apply_model_with_memblocks,
    conv as taehv_conv,
)

from ..geometry import decoded_frame_count

LOG = "[H3 Extended] taeh3"

#: MiniMaxH3Video latents. Anything else is a different model and must not load.
LATENT_CHANNELS = 24
WEIGHT_NAMES = ("taeh3.safetensors", "taeh3.pth")
#: Refuse to build a preview larger than this many frames, whatever is asked.
MAX_PREVIEW_FRAMES = 1024


class TAEH3Error(RuntimeError):
    pass


def resolve_taeh3_path(explicit=None):
    """Find the TAEH3 weights, preferring an explicit path."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        try:
            import folder_paths

            found = folder_paths.get_full_path("vae_approx", explicit)
        except Exception:
            found = None
        if found:
            return found
        raise TAEH3Error("taeh3 weights not found at %r" % (explicit,))

    import folder_paths

    for name in WEIGHT_NAMES:
        found = folder_paths.get_full_path("vae_approx", name)
        if found:
            return found
    raise TAEH3Error(
        "no taeh3 weights in models/vae_approx (looked for %s); fetch "
        "safetensors/taeh3.safetensors from github.com/madebyollin/taehv"
        % ", ".join(WEIGHT_NAMES)
    )


def _repatch_patch_size(model, patch_size):
    """Rebuild the two convs whose shape depends on ``patch_size``."""
    pixels = 3 * patch_size ** 2
    encoder_first = model.encoder[0]
    decoder_last = model.decoder[-1]
    model.encoder[0] = taehv_conv(pixels, encoder_first.out_channels)
    model.decoder[-1] = taehv_conv(decoder_last.in_channels, pixels)
    model.patch_size = patch_size
    return model


def load_taeh3(path=None, *, device=None, dtype=None, parallel=False):
    """Load TAEH3 for H3 latents and return ``(model, metadata)``.

    Fails loudly on any missing or unexpected key rather than running a
    partly random decoder.
    """
    path = resolve_taeh3_path(path)
    state = comfy.utils.load_torch_file(path, safe_load=True)

    if "decoder.1.weight" not in state or "encoder.0.weight" not in state:
        raise TAEH3Error("%s is not a TAEHV checkpoint" % path)

    latent_channels = int(state["decoder.1.weight"].shape[1])
    if latent_channels != LATENT_CHANNELS:
        raise TAEH3Error(
            "expected %d-channel H3 latents, weights are %d-channel"
            % (LATENT_CHANNELS, latent_channels)
        )
    # The first encoder conv takes ``3 * patch_size**2`` pixel-shuffled channels.
    pixel_channels = int(state["encoder.0.weight"].shape[1])
    patch_size = round((pixel_channels / 3) ** 0.5)
    if patch_size < 1 or 3 * patch_size ** 2 != pixel_channels:
        raise TAEH3Error(
            "encoder input %d channels is not 3 * patch^2" % pixel_channels
        )

    if device is None:
        device = comfy.model_management.vae_device()
    device = torch.device(device)
    if dtype is None:
        weight_dtype = state["decoder.1.weight"].dtype
        dtype = torch.float32 if device.type == "cpu" else weight_dtype

    model = TAEHV(latent_channels=latent_channels, parallel=parallel)
    if model.patch_size != patch_size:
        _repatch_patch_size(model, patch_size)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise TAEH3Error(
            "TAEH3 weights do not match the architecture: missing=%s unexpected=%s"
            % (sorted(missing)[:10], sorted(unexpected)[:10])
        )

    model.eval()
    model.show_progress_bar = False
    model.to(device=device, dtype=dtype)

    metadata = {
        "path": path,
        "weight_dtype": str(state["decoder.1.weight"].dtype),
        "runtime_dtype": str(dtype),
        "device": str(device),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "latent_channels": latent_channels,
        "patch_size": patch_size,
        "t_downscale": int(model.t_downscale),
        "t_upscale": int(model.t_upscale),
        "frames_to_trim": int(model.frames_to_trim),
        "spatial_upscale": 8 * patch_size,
        "parallel": bool(parallel),
        "missing_keys": sorted(missing),
        "unexpected_keys": sorted(unexpected),
    }
    del state
    return model, metadata


def _resample_indices(have, want):
    """Endpoint-matched index map from ``have`` frames to ``want``."""
    if want >= have or want < 1:
        return None
    if want == 1:
        return [0]
    return [round(i * (have - 1) / (want - 1)) for i in range(want)]


class TAEH3Previewer:
    """One TAEH3 instance, reused for every preview of a long-form run.

    Built once outside the sampler and held for the whole run: the decoder is
    11 M parameters (~23 MB fp16), so keeping it resident costs nothing next to
    reloading it inside a DiT callback.
    """

    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata
        self.device = torch.device(metadata["device"])
        self.dtype = getattr(torch, metadata["runtime_dtype"].split(".")[-1])
        self.t_upscale = int(metadata["t_upscale"])
        self.frames_to_trim = int(metadata["frames_to_trim"])
        self._logged_native = False

    @classmethod
    def load(cls, path=None, *, device=None):
        """Build a previewer, or return ``None`` if TAEH3 is unavailable.

        Never raises: a missing or broken taeh3 must fall back to the latent
        preview, not take a generation down.
        """
        try:
            if device is None:
                device = comfy.model_management.get_torch_device()
            model, metadata = load_taeh3(path, device=device)
        except Exception as exc:
            logging.info(
                "%s unavailable, falling back to the latent preview: %s: %s",
                LOG, type(exc).__name__, exc,
            )
            return None
        logging.info(
            "%s ready: %s | %.2f M params | %s on %s | %d ch, patch %d, "
            "spatial %dx, temporal %dx",
            LOG,
            os.path.basename(metadata["path"]),
            metadata["parameters"] / 1e6,
            metadata["runtime_dtype"],
            metadata["device"],
            metadata["latent_channels"],
            metadata["patch_size"],
            metadata["spatial_upscale"],
            metadata["t_upscale"],
        )
        return cls(model, metadata)

    def latents_for_frames(self, frame_limit):
        """Fewest latent positions that still yield ``frame_limit`` frames.

        Decoding the whole chunk and throwing most of it away would make a
        bounded preview cost the same as an unbounded one.
        """
        if frame_limit <= 0:
            return None
        return max(1, -(-(int(frame_limit) + self.frames_to_trim) // self.t_upscale))

    def _decode_to_cpu(self, latent):
        """``TAEHV.decode`` with the finished frames pinned to host memory.

        The stock method sends its output to ``intermediate_device()``, which is
        the GPU, and takes no argument to change that. A whole-chunk preview is
        ~105 frames at 768x1344; staging that in VRAM inside a sampler callback
        is ~650 MB next to a resident DiT on a 12 GB card. Driving the decoder
        stack directly costs four lines and keeps the accumulation on the host,
        while the per-timestep activations still run on the GPU.

        The movedim mirrors ``TAEHV.decode`` for a ``[B, C, T, H, W]`` input;
        ``process_in`` is the identity because no latent format is attached.
        """
        x = latent.to(self.device, self.dtype)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if int(x.shape[1]) != self.model.latent_channels:
            x = x.movedim(1, 2)
        x = x.movedim(2, 1)  # [B, C, T, H, W] -> [B, T, C, H, W]
        x = apply_model_with_memblocks(
            self.model.decoder,
            x,
            self.model.parallel,
            False,
            output_device=torch.device("cpu"),
            patch_size=self.model.patch_size,
            decode=True,
        )
        return x[:, self.model.frames_to_trim:].movedim(2, 1)

    @torch.inference_mode()
    def frames(self, latent, *, limit=0):
        """Decode an H3 video latent to a uint8 NHWC batch on the CPU.

        ``limit`` of 0 means every frame the chunk covers. The result is
        resampled to the frame count the real VAE would produce, so the preview
        plays at the run's true frame rate.
        """
        if not torch.is_tensor(latent) or latent.ndim != 5:
            raise TAEH3Error(
                "expected a [B,C,T,H,W] video latent, got %r"
                % (tuple(latent.shape) if torch.is_tensor(latent) else type(latent),)
            )
        latent = latent[:1].detach()

        needed = self.latents_for_frames(limit)
        if needed is not None:
            latent = latent[:, :, :needed]
        latent_t = int(latent.shape[2])

        # What the real VAE would have produced from these latents.
        target = decoded_frame_count(latent_t)
        if limit > 0:
            target = min(target, int(limit))
        target = min(target, MAX_PREVIEW_FRAMES)

        decoded = self._decode_to_cpu(latent)
        if decoded.ndim != 5 or int(decoded.shape[1]) != 3:
            raise TAEH3Error("unexpected TAEH3 output %r" % (tuple(decoded.shape),))
        native = int(decoded.shape[2])
        frames = decoded[0].permute(1, 2, 3, 0).to(torch.float32)
        del decoded

        if not self._logged_native:
            self._logged_native = True
            logging.info(
                "%s decode: %d latents -> %d native frames, shown as %d "
                "(VAE rate); %dx%d",
                LOG, latent_t, native, target,
                int(frames.shape[2]), int(frames.shape[1]),
            )

        indices = _resample_indices(native, target)
        if indices is not None:
            frames = frames[torch.tensor(indices)]
        elif native > target:
            frames = frames[:target]

        return (frames.clamp_(0, 1) * 255.0 + 0.5).to(torch.uint8).contiguous()


__all__ = [
    "TAEH3Error",
    "TAEH3Previewer",
    "load_taeh3",
    "resolve_taeh3_path",
    "MAX_PREVIEW_FRAMES",
]
