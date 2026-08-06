"""Building Qwen presentation items and DiT reference blocks, canvas-pinned.

This mirrors `MiniMaxH3ReferenceToVideoZi` with one deliberate difference: the
reference video is encoded on the *target's* canvas, not on one derived from its
own dimensions.

Core's `adapt_canvas` sizes a reference video from its own width and height, so
a 1080p source fed to a 0.8 MP target gets a 1344x768 reference against an
800-row target. That is a correctness problem here - the whole harness rests on
Chunk A and Chunk B latents being sliceable against each other - and it is also
a 13% sequence-length tax at a chunk size chosen for its VRAM headroom.
"""

import math

import torch
import torchaudio

import comfy.utils

try:
    from .. import latent_cache
except ImportError:  # the self-tests import this file as a top-level module
    import latent_cache

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24


def adapt_canvas(width, height):
    """768-short-edge canvas with a 768*1344 area cap, per-axis round to 32."""
    ratio = width / height
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nom_w, nom_h = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        s = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * s, nom_h * s
    return (max(CANVAS_MULTIPLE, round(nom_w / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
            max(CANVAS_MULTIPLE, round(nom_h / CANVAS_MULTIPLE) * CANVAS_MULTIPLE))


def resize(image, width, height, crop="disabled"):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def pin_canvas(source_frames, width=0, height=0):
    """One canvas for the whole run, derived from the source unless overridden."""
    if width and height:
        return (max(CANVAS_MULTIPLE, width // CANVAS_MULTIPLE * CANVAS_MULTIPLE),
                max(CANVAS_MULTIPLE, height // CANVAS_MULTIPLE * CANVAS_MULTIPLE))
    return adapt_canvas(source_frames.shape[2], source_frames.shape[1])


def encode_image_ref(vae, image, canvas, ref_image_size="match", cond_cache="auto"):
    """A static image reference, at `match` or `max` sizing."""
    h, w = image.shape[1], image.shape[2]
    width, height = canvas
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
    resized = resize(image[:1], tw, th)
    latent = latent_cache.encode(vae, resized, mode=cond_cache, label='ref image')
    item = {"type": "image", "data": resized}
    block = {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": latent}
    return item, block, "%dx%d source -> %dx%d encoded" % (w, h, tw, th)


def qwen_video_item(frames):
    """Qwen sees a reference video at 2 fps with timestamps."""
    sample_idx = list(range(0, frames.shape[0], FPS // 2))
    return {"type": "video", "data": frames[sample_idx],
            "timestamps": [i / 2.0 for i in range(len(sample_idx))]}


def snap_video_frames(frames):
    """Truncate down to the model's `n %% 17 == 5` reference-video grid."""
    n = frames.shape[0]
    if n < 5:
        raise ValueError("MiniMax H3 reference videos need at least 5 frames")
    while n % 17 != 5:
        n -= 1
    return frames[:n]


def encode_video_ref(vae, frames, canvas, audio=None, audio_vae=None, cond_cache="auto"):
    """A reference video pinned to the run's canvas.

    Returns `(qwen_items, dit_block, note)`. `qwen_items` is a list because a
    soundtrack contributes its own `<Audio j>` item immediately before the
    `<Video k>` it belongs to.
    """
    width, height = canvas
    frames = snap_video_frames(resize(frames, width, height))
    latent = latent_cache.encode(vae, frames, mode=cond_cache, label='ref video')

    items = []
    audio_latent, ref_audio_t = None, 0
    if audio is not None and audio_vae is not None:
        audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio, cond_cache)
        items.append({"type": "audio"})
    items.append(qwen_video_item(frames))

    block = {
        "kind": "video_audio" if ref_audio_t else "video",
        "latent_t": latent.shape[2],
        "latent_h": height // 16,
        "latent_w": width // 16,
        "ref_audio_t": ref_audio_t,
        "latent": latent,
        "audio_latent": audio_latent,
    }
    note = "%d frames -> %dx%d canvas, latent t=%d" % (
        frames.shape[0], width, height, latent.shape[2])
    return items, block, note


def encode_ref_audio(audio_vae, audio, cond_cache="auto"):
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sample_rate != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, vae_sr)
    latent = latent_cache.encode(audio_vae, waveform[:1].movedim(1, -1),
                                 mode=cond_cache, label='ref audio')
    return latent, latent.shape[-1]


def composite_frames(generated_overlap, source_frames, overlap_frames):
    """Source reference whose opening is replaced by the generated overlap.

    `generated_overlap` are Chunk A's frames over the shared window; the rest is
    the original Chunk B source from `overlap_frames` on. The join is a hard cut
    inside the *reference*, not in the output.
    """
    if generated_overlap.shape[0] < overlap_frames:
        raise ValueError("generated overlap is %d frames, need %d"
                         % (generated_overlap.shape[0], overlap_frames))
    head = generated_overlap[:overlap_frames]
    tail = source_frames[overlap_frames:]
    if tail.shape[1:] != head.shape[1:]:
        tail = resize(tail, head.shape[2], head.shape[1])
    return torch.cat([head, tail])
