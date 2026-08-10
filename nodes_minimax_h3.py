"""MiniMax H3 nodes: AV latent creation and task conditioning (t2va / fl2va / ref2va).

Custom-node fork of comfy_extras/nodes_minimax_h3.py so it can evolve
independently of ComfyUI updates. Node ids are suffixed with "Zi" to avoid
clashing with the built-in extension, which still registers its own copies.

The H3 packed-DiT consumes, via conditioning:
- Qwen3-VL-32B hidden states with per-token modality tags (from the minimax CLIP)
- keyframe / reference condition latents, re-injected every step (never denoised)

Latents are NestedTensor pairs (video [B,24,T,H/16,W/16], audio [B,32,2,T40]);
sampling runs on the flat pack with any stock sampler (the model handles the
audio stream's shifted schedule internally).
"""

import logging
import math

import torch
import torchaudio

import nodes
import comfy.model_management
import comfy.model_sampling
import comfy.nested_tensor
import comfy.utils
import node_helpers
from comfy.ldm.modules.attention import REGISTERED_ATTENTION_FUNCTIONS, get_attention_function
from comfy_api.latest import ComfyExtension, io

try:
    from . import run_context
    # bound as names, not as the module: the widget is also called cond_cache
    # and would shadow it inside execute()
    from .cond_cache import MODES as COND_CACHE_MODES, encode as encode_conditioning
    from . import latent_cache
    from .vram_guard import install_unet_guard
except ImportError:  # the self-tests import this file as a top-level module
    import run_context
    from cond_cache import MODES as COND_CACHE_MODES, encode as encode_conditioning
    import latent_cache
    from vram_guard import install_unet_guard

CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048
FPS = 24
AUDIO_LATENT_FPS = 40


def align_frame_count(n):
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def temporal_shape(length):
    frame_count = align_frame_count(max(5, length))
    duration = frame_count / FPS
    return frame_count, video_latent_t(frame_count), round(duration * AUDIO_LATENT_FPS)


def decoded_frame_count_from_latent_t(latent_t):
    """Return the number of pixel frames produced by the H3 VAE.

    With the released VAE defaults:

        latent T:  1  2  3  4  5  6  7  8  9  10 11 12
        frames:    1  5  9 13 17 18 22 26 30 34 35 39

    The irregular fourth increment at each five-token boundary comes from the
    VAE's 17-frame chunks, temporal ratio 4, and three-token tail drop.
    """
    if latent_t < 1:
        raise ValueError("MiniMax H3 temporal latent length must be at least 1")
    if latent_t == 1:
        return 1
    groups, remainder = divmod(latent_t - 2, 5)
    return 5 + groups * 17 + min(remainder * 4, 13)


def adapt_canvas(width, height):
    """768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""
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


def _resize(image, width, height, crop):
    # image [B, H, W, C] -> [B, height, width, 3]
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _empty_av_latent(width, height, length, batch_size=1):
    frame_count, latent_t, audio_t = temporal_shape(length)
    video = torch.zeros([batch_size, 24, latent_t, height // 16, width // 16],
                        device=comfy.model_management.intermediate_device())
    audio = torch.zeros([batch_size, 32, 2, audio_t],
                        device=comfy.model_management.intermediate_device())
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}, frame_count


def _empty_av_latent_raw_t(width, height, latent_t, batch_size=1):
    """Create an H3 target using an exact video temporal-latent length."""
    frame_count = decoded_frame_count_from_latent_t(latent_t)
    duration = frame_count / FPS
    audio_t = round(duration * AUDIO_LATENT_FPS)

    video = torch.zeros(
        [batch_size, 24, latent_t, height // 16, width // 16],
        device=comfy.model_management.intermediate_device(),
    )
    audio = torch.zeros(
        [batch_size, 32, 2, audio_t],
        device=comfy.model_management.intermediate_device(),
    )
    return {
        "samples": comfy.nested_tensor.NestedTensor((video, audio))
    }, frame_count


class EmptyMiniMaxH3LatentAV(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="EmptyMiniMaxH3LatentAVZi",
            display_name="Empty MiniMax H3 AV Latent (Zi)",
            category="model/latent/minimax",
            description="Joint video+audio latent for MiniMax H3. Duration snaps to the model's 17k+5 frame grid at 24 fps.",
            inputs=[
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
            ],
            outputs=[io.Latent.Output()],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, width, height, length) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        video, audio = latent["samples"].tensors
        run_context.record(
            "Empty MiniMax H3 AV Latent (Zi)", run_context.node_id(cls),
            [("canvas", "%dx%d" % (width, height)),
             ("length", "%d requested -> %d frames (%.2fs at %d fps)"
              % (length, frame_count, frame_count / FPS, FPS)),
             ("video latent", list(video.shape)),
             ("audio latent", list(audio.shape))],
            video_latent_shape=video.shape,
        )
        return io.NodeOutput(latent)


class MiniMaxH3ImageToVideo(io.ComfyNode):
    """t2va and fl2va: prompt (+ optional first/last keyframes) -> conditioning + AV latent."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ImageToVideoZi",
            display_name="MiniMax H3 Image to Video (Zi)",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s; trained range is ~124-362, longer is untested)"),
                io.Int.Input(
                    "raw_latent_t",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Experimental exact temporal latent length. "
                        "0 uses normal length handling. "
                        "1 outputs 1 frame; 2 outputs 5; 3 outputs 9; "
                        "4 outputs 13; 5 outputs 17; 6 outputs 18; "
                        "7 outputs 22."
                    ),
                ),
                io.Combo.Input(
                    "cond_cache",
                    options=COND_CACHE_MODES,
                    default="auto",
                    tooltip=(
                        "Reuse the Qwen3-VL pass across runs, keyed on a hash of the token "
                        "stream (prompt text plus reference pixels) and the text encoder "
                        "identity. 'auto' reads and writes the cache; 'off' bypasses it; "
                        "'refresh' re-encodes and overwrites the entry. A hit skips loading "
                        "the 14.6 GB encoder entirely, so changing only length or sampler "
                        "settings costs nothing here."
                    ),
                ),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, length,
                raw_latent_t=0, cond_cache="auto",
                first_frame=None, last_frame=None) -> io.NodeOutput:
        if raw_latent_t > 0:
            latent, frame_count = _empty_av_latent_raw_t(
                width, height, raw_latent_t
            )
        else:
            latent, frame_count = _empty_av_latent(width, height, length)

        video, audio = latent["samples"].tensors
        run_context.record(
            "MiniMax H3 Image to Video (Zi)", run_context.node_id(cls),
            [("canvas", "%dx%d" % (width, height)),
             ("length", ("raw_latent_t %d -> %d frames" % (raw_latent_t, frame_count))
              if raw_latent_t > 0 else
              ("%d requested -> %d frames (%.2fs at %d fps)"
               % (length, frame_count, frame_count / FPS, FPS))),
             ("first_frame", run_context.image_res(first_frame)),
             ("last_frame", run_context.image_res(last_frame)),
             ("cond_cache", cond_cache),
             ("prompt", "%d chars" % len(prompt)),
             ("video latent", list(video.shape)),
             ("audio latent", list(audio.shape))],
            video_latent_shape=video.shape,
        )

        images = []
        keyframes = []
        if first_frame is not None:
            # geometry anchor: plain stretch to canvas
            img = _resize(first_frame[:1], width, height, "disabled")
            images.append(img)
            keyframes.append({"resolved_frame_index": 0, "image": img})
        if last_frame is not None:
            # follower: aspect-preserving cover-crop
            img = _resize(last_frame[:1], width, height, "center")
            images.append(img)
            keyframes.append({"resolved_frame_index": frame_count - 1, "image": img})

        tokens = clip.tokenize(prompt, images=images)
        cond = encode_conditioning(clip, tokens, mode=cond_cache, label=prompt)

        if keyframes:
            for kf in keyframes:
                kf["latent"] = latent_cache.encode(
                    vae, kf.pop("image"), mode=cond_cache, label="keyframe")
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        return io.NodeOutput(cond, latent)


class MiniMaxH3ReferenceToVideo(io.ComfyNode):
    """ref2va: prompt + reference images / videos / audio -> conditioning + AV latent.

    References enter the presentation in fixed order: images, then videos (each
    soundtrack's <Audio j> label right before its <Video k>), then standalone
    audio. Ordinals are 1-based per type, so the prompt refers to them as
    <Picture i> / <Video k> / <Audio j>.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3ReferenceToVideoZi",
            description="<Picture i> / <Video k> / <Audio j> reference conditioning for MiniMax H3. Use the same tags when prompting.",
            display_name="MiniMax H3 Reference to Video (Zi)",
            category="model/conditioning/minimax",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION, step=32),
                io.Int.Input("length", default=124, min=5, max=3600, step=17, tooltip="Frame count at 24 fps, (124 = ~5s, trained range is ~124-362)"),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. 'match' scales each ref (down only, keeping aspect) to the generation's pixel area; 'max' uses the reference pipeline's 2048px short edge for best identity fidelity. Reference tokens ride through every sampling step, so 'max' can be several times slower."),
                io.Combo.Input(
                    "cond_cache",
                    options=COND_CACHE_MODES,
                    default="auto",
                    tooltip=(
                        "Reuse the Qwen3-VL pass across runs, keyed on a hash of the token "
                        "stream (prompt text plus reference pixels) and the text encoder "
                        "identity. 'auto' reads and writes the cache; 'off' bypasses it; "
                        "'refresh' re-encodes and overwrites the entry. A hit skips loading "
                        "the 14.6 GB encoder entirely, so changing only length or sampler "
                        "settings costs nothing here."
                    ),
                ),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (downscaled to 2048 short edge if larger, never upscaled)"),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video", tooltip="Reference video frames at 24 fps (2-15s)"),
                        prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input("ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio", tooltip="Soundtrack of the same-numbered reference video"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Standalone reference audio"),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
            hidden=[io.Hidden.unique_id],
        )

    @staticmethod
    def _encode_ref_audio(audio_vae, audio, cond_cache="auto"):
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = latent_cache.encode(audio_vae, waveform[:1].movedim(1, -1),
                                mode=cond_cache, label='ref audio')  # [1, 32, 2, T]
        return z, z.shape[-1]

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length, ref_image_size="match",
                cond_cache="auto",
                ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # for the tokenizer presentation, in request order
        ref_blocks = []  # for the DiT payload, same order
        recorded = []    # source -> encoded size of every reference, for the VRAM guard

        for name, img in (ref_images or {}).items():
            if img is None:
                continue
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                # aspect-preserving scale (down only) to the generation's pixel area
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img[:1], tw, th, "disabled")
            z = latent_cache.encode(vae, resized, mode=cond_cache, label='ref image')
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})
            recorded.append((name, "%dx%d source -> %dx%d encoded (latent %dx%d)"
                             % (w, h, tw, th, tw // 16, th // 16)))

        ref_video_audios = ref_video_audios or {}
        for name, video_frames in (ref_videos or {}).items():
            if video_frames is None:
                continue
            # index-paired soundtrack: ref_video_audio_N belongs to ref_video_N
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
                ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            frames = _resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = latent_cache.encode(vae, frames, mode=cond_cache, label='ref video')
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, soundtrack, cond_cache)
                # the soundtrack gets its own <Audio j> label, emitted before <Video k>
                ref_items.append({"type": "audio"})
            # Qwen sees the video at 2 fps with timestamps
            sample_idx = list(range(0, frames.shape[0], FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})
            recorded.append((name, "%dx%d x%d frames source -> %dx%d canvas x%d frames used "
                                   "(latent t=%d, %dx%d)"
                             % (vw, vh, video_frames.shape[0], cw, ch, n,
                                z.shape[2], cw // 16, ch // 16)))
            if soundtrack is not None:
                recorded.append((name + " soundtrack",
                                 "%s -> latent t=%d" % (run_context.audio_desc(soundtrack),
                                                        ref_audio_t)))

        for name, audio in (ref_audios or {}).items():
            if audio is None:
                continue
            audio_latent, ref_audio_t = cls._encode_ref_audio(audio_vae, audio, cond_cache)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})
            recorded.append((name, "%s -> latent t=%d"
                             % (run_context.audio_desc(audio), ref_audio_t)))

        video, audio_latent_target = latent["samples"].tensors
        run_context.record(
            "MiniMax H3 Reference to Video (Zi)", run_context.node_id(cls),
            [("canvas", "%dx%d" % (width, height)),
             ("length", "%d requested -> %d frames (%.2fs at %d fps)"
              % (length, frame_count, frame_count / FPS, FPS)),
             ("ref_image_size", ref_image_size),
             ("cond_cache", cond_cache),
             ("prompt", "%d chars" % len(prompt))]
            + recorded
            + [("video latent", list(video.shape)),
               ("audio latent", list(audio_latent_target.shape))],
            video_latent_shape=video.shape,
        )

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = encode_conditioning(clip, tokens, mode=cond_cache, label=prompt)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
        return io.NodeOutput(cond, latent)


def _set_h3_attention_backend(transformer_options, backend):
    """Point H3's DiT attention at a specific backend, H3-scoped.

    Core's `wrap_attn` consults `optimized_attention_override` in the
    transformer_options it is handed, so writing the key here reaches only the
    models whose forward pass carries these options - the H3 DiT - and leaves
    every other model on the global default.
    """
    transformer_options["minimax_h3_attention_backend"] = str(backend)
    if backend == "comfy":
        logging.info("[H3 Extended] Using Comfy default attention")
        return

    attention = get_attention_function(backend, default=None)
    if attention is None:
        raise RuntimeError(
            "Attention backend '%s' is not available in the Python environment "
            "running ComfyUI (registered: %s). Install a compatible build - for "
            "'sage', a sageattention package matching this Python/Torch/CUDA - "
            "or select 'comfy' as the attention backend." % (
                backend, ", ".join(sorted(REGISTERED_ATTENTION_FUNCTIONS)) or "none"))

    # Registered attention functions are themselves wrap_attn-decorated so they
    # can honor an override; call the undecorated function so this override does
    # not re-enter itself.
    attention_impl = getattr(attention, "__wrapped__", attention)

    def attention_override(_original, *args, **kwargs):
        return attention_impl(*args, **kwargs)

    transformer_options["optimized_attention_override"] = attention_override
    logging.info("[H3 Extended] Using '%s' for H3 DiT attention", backend)


class MiniMaxH3SigmaShift(io.ComfyNode):
    """Set the video/audio flow shifts coherently.

    The video shift drives the sampler's sigma schedule; both values are also
    handed to the DiT, which inverts the video schedule to the shared base grid
    and derives the audio schedule from it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3SigmaShiftZi",
            description="Set the video/audio flow shifts.",
            display_name="MiniMax H3 Sigma Shift (Zi)",
            category="model/patch/minimax",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("shift_video", default=12.0, min=0.01, max=100.0, step=0.01),
                io.Float.Input("shift_audio", default=3.0, min=0.01, max=100.0, step=0.01),
                io.Combo.Input(
                    "attention_backend",
                    options=["sage", "comfy", "pytorch"],
                    default="sage",
                    tooltip=(
                        "Attention backend for the H3 DiT. 'sage' is faster dense attention "
                        "when a compatible sageattention package is installed. 'comfy' follows "
                        "the global default, which --use-sage-attention makes Sage - so use "
                        "'pytorch' for a guaranteed dense baseline regardless of launch flags. "
                        "Errors rather than falling back silently, so benchmarks stay honest."
                    ),
                ),
                io.Int.Input(
                    "vram_guard_mb",
                    default=800,
                    min=0,
                    max=24576,
                    step=64,
                    tooltip=(
                        "Safety margin in MB for the H3 capacity proof and the emergency "
                        "low-free monitor. The proof runs once per forward shape and "
                        "cancels before apply_model when physical capacity is insufficient; "
                        "0 disables."
                    ),
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, shift_video, shift_audio, attention_backend="sage",
                vram_guard_mb=800) -> io.NodeOutput:
        model_sampling_av = getattr(comfy.model_sampling, "ModelSamplingAV", None)
        if model_sampling_av is None:
            raise RuntimeError(
                "MiniMax H3 Sigma Shift (Zi) requires ComfyUI v0.31.0+; "
                "update ComfyUI and try again."
            )

        m = model.clone()

        class ModelSamplingAdvanced(model_sampling_av, comfy.model_sampling.CONST):
            pass

        original = m.get_model_object("model_sampling")
        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift_video, audio_shift=shift_audio)
        if hasattr(original, "noise_scale"):
            model_sampling.set_noise_scale(original.noise_scale)
        m.add_object_patch("model_sampling", model_sampling)

        to = m.model_options["transformer_options"] = m.model_options.get("transformer_options", {}).copy()
        to["minimax_h3_sigma_shift_video"] = shift_video
        to["minimax_h3_sigma_shift_audio"] = shift_audio
        _set_h3_attention_backend(to, attention_backend)
        install_unet_guard(m, vram_guard_mb)
        return io.NodeOutput(m)


class MiniMaxH3Extension(ComfyExtension):
    async def get_node_list(self):
        return [
            EmptyMiniMaxH3LatentAV,
            MiniMaxH3ImageToVideo,
            MiniMaxH3ReferenceToVideo,
            MiniMaxH3SigmaShift
            ]


async def comfy_entrypoint() -> MiniMaxH3Extension:
    return MiniMaxH3Extension()
