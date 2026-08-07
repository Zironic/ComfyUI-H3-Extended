"""Make TAEH3 the previewer for *any* MiniMax H3 sampling, not just long-form.

The long-form nodes drive TAEH3 themselves through their own publisher. Every
other way of sampling H3 - a plain ``SamplerCustomAdvanced``, the core Ref2V
nodes, anything at all - goes through ``latent_preview.get_previewer`` instead
and lands on the 24-factor latent2rgb approximation.

Two separate things stop core from finding taeh3 on its own:

* ``comfy.latent_formats.MiniMaxH3Video`` sets no ``taesd_decoder_name``, and
  ``latent_preview.VIDEO_TAES`` does not list ``taeh3``, so the TAESD branch is
  never reached; and
* even if it were, ``comfy.taesd.taehv.TAEHV`` mis-builds these weights. It only
  raises ``patch_size`` to 2 for 48- and 32-channel latents, and taeh3 is a
  24-channel model with ``patch_size == 2``. A strict load fails. See
  ``chunked_ref2v.longform.taeh3.load_taeh3`` for the repair.

Rather than edit either file - this is a live auto-updating upstream checkout
and both would revert on the next pull - the pack wraps ``get_previewer`` at
import time and answers for H3 latents only. Everything else falls through to
core untouched.

``Auto`` is treated as "use TAEH3 if the weights are there". Core maps Auto to
latent2rgb, which for H3 is the blurry colour-field that made a real decoder
worth building in the first place. An explicit ``--preview-method latent2rgb``
or ``none`` is still honoured.
"""

from __future__ import annotations

import logging

LOG = "[H3 Extended] taeh3 preview"
_PATCHED = False
#: One previewer per device for the life of the process. ``get_previewer`` is
#: called once per sampling run, and reloading 23 MB of weights per run is pure
#: waste.
_CACHE = {}


def _is_h3_latent_format(latent_format) -> bool:
    try:
        from comfy.latent_formats import MiniMaxH3Video

        return isinstance(latent_format, MiniMaxH3Video)
    except Exception:
        return False


class TAEH3LatentPreviewer:
    """Core's previewer protocol, backed by TAEH3.

    The protocol is one still per step, so this decodes a single temporal
    position. That is also the cheapest thing TAEH3 can do: one latent position
    is one frame, no matter how long the clip is.
    """

    def __init__(self, backend):
        self.backend = backend

    def decode_latent_to_preview(self, x0):
        from PIL import Image

        if x0.ndim == 4:  # [B, C, H, W] - a still, not a clip
            x0 = x0.unsqueeze(2)
        frames = self.backend.frames(x0[:1, :, :1], limit=1)
        return Image.fromarray(frames[0].numpy(), "RGB")

    def decode_latent_to_preview_image(self, preview_format, x0):
        import latent_preview

        return (
            "JPEG",
            self.decode_latent_to_preview(x0),
            latent_preview.MAX_PREVIEW_RESOLUTION,
        )


def _backend_for(device):
    """Load TAEH3 once per device; ``None`` means fall through to core."""
    key = str(device)
    if key not in _CACHE:
        from .chunked_ref2v.longform.taeh3 import TAEH3Previewer

        _CACHE[key] = TAEH3Previewer.load(device=device)
        if _CACHE[key] is not None:
            logging.info(
                "%s: H3 previews will use TAEH3 on %s", LOG, key
            )
    return _CACHE[key]


def install() -> None:
    global _PATCHED
    if _PATCHED:
        return
    import latent_preview
    from comfy.cli_args import LatentPreviewMethod, args

    original = latent_preview.get_previewer

    def get_previewer(device, latent_format):
        # Only H3, and only when the user has not asked for something specific.
        if _is_h3_latent_format(latent_format) and args.preview_method in (
            LatentPreviewMethod.Auto,
            LatentPreviewMethod.TAESD,
        ):
            try:
                backend = _backend_for(device)
                if backend is not None:
                    return TAEH3LatentPreviewer(backend)
            except Exception as exc:
                # A preview must never cost a generation.
                logging.warning(
                    "%s unavailable, using core's previewer: %s: %s",
                    LOG, type(exc).__name__, exc,
                )
        return original(device, latent_format)

    latent_preview.get_previewer = get_previewer
    _PATCHED = True


__all__ = ["TAEH3LatentPreviewer", "install"]
