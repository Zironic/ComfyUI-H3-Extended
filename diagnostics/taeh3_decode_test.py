"""TAEH3 vs. full MiniMax H3 VAE decode smoke test.

One real clip is encoded **once** by the production H3 VAE, and that single
latent is then decoded twice: by the full VAE (the control) and by TAEH3. Both
decoders therefore see byte-identical input, so anything that differs in the
output is the decoder and not the hand-off.

Why this loads TAEH3 explicitly instead of going through Comfy's previewer
--------------------------------------------------------------------------
``latent_preview.get_previewer`` would not produce a TAEH3 at all right now:

* ``comfy.latent_formats.MiniMaxH3Video`` declares no ``taesd_decoder_name``, so
  the TAESD branch is never even reached, and
* ``latent_preview.VIDEO_TAES`` does not list ``taeh3``.

More importantly, ``comfy.taesd.taehv.TAEHV`` **mis-builds** these weights. It
only raises ``patch_size`` to 2 for 48- and 32-channel latents, but taeh3 is a
24-channel model with ``patch_size == 2``: its first encoder conv takes
``3 * 2**2 == 12`` channels and its last decoder conv emits 12. Constructed as
``TAEHV(24)`` the architecture gets ``patch_size == 1``, those two convs come out
3-channel, and a strict load fails. :func:`load_taeh3` repairs exactly those two
layers, which is also what makes the spatial arithmetic work out: three 2x
upsamples times a 2x pixel-shuffle is 16x, matching H3's
``spacial_downscale_ratio == 16``. TAEH3 decodes at *full* resolution, not half.

Temporally the stock defaults are already right - ``t_upscale == 4`` matches
``temporal_downscale_ratio == 4``, and ``frames_to_trim == 3`` turns 5 latent
positions into 17 frames, the same 17 the full VAE emits for one decode group.

No latent scaling is applied anywhere. TAEHV is trained directly on diffusion
latents, and ``MiniMaxH3Video.scale_factor`` is 1.0 regardless.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime

import torch

import comfy.model_management
import comfy.utils
from comfy.taesd.taehv import TAEHV, conv as taehv_conv

from ..chunked_ref2v import ref_builder
from ..chunked_ref2v.longform import runner
from ..chunked_ref2v.longform.writer import FFmpegVideoWriter

LOG = "[H3 Extended] taeh3 decode test"

#: MiniMaxH3Video latents. Anything else is a different model and must not load.
LATENT_CHANNELS = 24
#: One decode group: ``1 + 4 * 4`` pixel frames behind 5 temporal latents.
GROUP_FRAMES = 17
GROUP_LATENTS = 5
WEIGHT_NAMES = ("taeh3.safetensors", "taeh3.pth")


class TAEH3TestError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------


def resolve_taeh3_path(explicit=None):
    """Find the TAEH3 weights, preferring an explicit path."""
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        found = None
        try:
            import folder_paths

            found = folder_paths.get_full_path("vae_approx", explicit)
        except Exception:
            found = None
        if found:
            return found
        raise TAEH3TestError("taeh3 weights not found at %r" % (explicit,))

    import folder_paths

    for name in WEIGHT_NAMES:
        found = folder_paths.get_full_path("vae_approx", name)
        if found:
            return found
    raise TAEH3TestError(
        "no taeh3 weights in models/vae_approx (looked for %s); fetch "
        "safetensors/taeh3.safetensors from github.com/madebyollin/taehv"
        % ", ".join(WEIGHT_NAMES)
    )


def _repatch_patch_size(model, patch_size):
    """Rebuild the two convs that depend on ``patch_size``.

    ``TAEHV.__init__`` hard-codes which latent widths use pixel-shuffle, and 24
    is not on that list. Only the first encoder conv and the last decoder conv
    change shape, so replacing them in place is enough to make a strict load
    succeed - every other layer is already correct.
    """
    pixels = 3 * patch_size ** 2
    encoder_first = model.encoder[0]
    decoder_last = model.decoder[-1]
    model.encoder[0] = taehv_conv(pixels, encoder_first.out_channels)
    model.decoder[-1] = taehv_conv(decoder_last.in_channels, pixels)
    model.patch_size = patch_size
    return model


def load_taeh3(path=None, *, device=None, dtype=None, parallel=False):
    """Load TAEH3 for H3 latents and return ``(model, metadata)``.

    Fails loudly on any missing or unexpected key: a smoke test that silently
    ran on a partly random decoder would answer the wrong question.
    """
    path = resolve_taeh3_path(path)
    state = comfy.utils.load_torch_file(path, safe_load=True)

    if "decoder.1.weight" not in state or "encoder.0.weight" not in state:
        raise TAEH3TestError("%s is not a TAEHV checkpoint" % path)

    latent_channels = int(state["decoder.1.weight"].shape[1])
    if latent_channels != LATENT_CHANNELS:
        raise TAEH3TestError(
            "expected %d-channel H3 latents, weights are %d-channel"
            % (LATENT_CHANNELS, latent_channels)
        )
    # The first encoder conv takes ``3 * patch_size**2`` pixel-shuffled channels,
    # so the patch size is the square root of a third of its input width.
    pixel_channels = int(state["encoder.0.weight"].shape[1])
    patch_size = round((pixel_channels / 3) ** 0.5)
    if patch_size < 1 or 3 * patch_size ** 2 != pixel_channels:
        raise TAEH3TestError(
            "encoder input %d channels is not 3 * patch^2" % pixel_channels
        )

    if device is None:
        device = comfy.model_management.vae_device()
    device = torch.device(device)
    if dtype is None:
        # The checkpoint ships fp16 and the decoder is 11 M params; keeping the
        # native dtype avoids an upcast that buys a preview nothing.
        weight_dtype = state["decoder.1.weight"].dtype
        dtype = torch.float32 if device.type == "cpu" else weight_dtype

    model = TAEHV(latent_channels=latent_channels, parallel=parallel)
    if model.patch_size != patch_size:
        logging.info(
            "%s: rebuilding patch_size %d -> %d for %d-channel weights",
            LOG,
            model.patch_size,
            patch_size,
            latent_channels,
        )
        _repatch_patch_size(model, patch_size)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise TAEH3TestError(
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
    logging.info(
        "%s: loaded %s | %s weights -> %s on %s | %.2f M params | %d ch, patch %d, "
        "t_up %d, trim %d, spatial %dx",
        LOG,
        os.path.basename(path),
        metadata["weight_dtype"],
        metadata["runtime_dtype"],
        metadata["device"],
        metadata["parameters"] / 1e6,
        latent_channels,
        patch_size,
        metadata["t_upscale"],
        metadata["frames_to_trim"],
        metadata["spatial_upscale"],
    )
    del state
    return model, metadata


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------


class _Measure:
    """Wall time plus CUDA peaks around one operation.

    This install runs the ``cudaMallocAsync`` allocator, where the cumulative
    alloc/free counters are meaningless; the peak figures below still are, and
    ``reserved`` is the one to trust.
    """

    def __init__(self, label, device):
        self.label = label
        self.device = torch.device(device) if device is not None else None
        self.cuda = self.device is not None and self.device.type == "cuda"
        self.result = {"label": label}

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.cuda:
            torch.cuda.synchronize(self.device)
        self.result["seconds"] = time.perf_counter() - self._start
        if self.cuda:
            self.result["peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(self.device) / 1024 ** 2
            )
            self.result["peak_reserved_mb"] = (
                torch.cuda.max_memory_reserved(self.device) / 1024 ** 2
            )
        return False


def _tensor_stats(tensor):
    flat = tensor.detach().to(torch.float32)
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


# --------------------------------------------------------------------------
# frames in, video out
# --------------------------------------------------------------------------


def _snap_encode_frames(count, requested):
    """Largest ``1 + 4k`` frame count at or below both bounds.

    The H3 VAE consumes whole ``1 + 4k`` groups; 17 frames is ``1 + 4*4`` and
    encodes to the 5 temporal latents this test is built around.
    """
    limit = min(int(count), int(requested))
    if limit < 5:
        raise TAEH3TestError(
            "need at least 5 frames to exercise temporal decoding, got %d" % limit
        )
    return ((limit - 1) // 4) * 4 + 1


def prepare_frames(frames, *, canvas=None, frame_count=GROUP_FRAMES):
    """Pin a clip to an H3 canvas and to the VAE's temporal grid.

    Spatial resizing goes through ``ref_builder.resize`` / ``adapt_canvas``, the
    same path the reference encoder uses, so these are the pixels the production
    encoder would actually see rather than an arbitrary rescale.
    """
    if not torch.is_tensor(frames) or frames.ndim != 4 or frames.shape[-1] < 3:
        raise TAEH3TestError(
            "expected an NHWC image batch, got %r"
            % (tuple(frames.shape) if torch.is_tensor(frames) else type(frames),)
        )
    frames = frames[..., :3].detach().to("cpu", torch.float32)
    if float(frames.max()) > 1.5:
        raise TAEH3TestError("frames must be 0..1 floats, saw max %.3f" % frames.max())

    source_h, source_w = int(frames.shape[1]), int(frames.shape[2])
    if canvas is None:
        canvas = ref_builder.adapt_canvas(source_w, source_h)
    width, height = int(canvas[0]), int(canvas[1])

    take = _snap_encode_frames(frames.shape[0], frame_count)
    frames = ref_builder.resize(frames[:take], width, height).clamp(0, 1)
    info = {
        "source_frames": int(frames.shape[0]),
        "source_resolution": [source_w, source_h],
        "canvas": [width, height],
        "requested_frames": int(frame_count),
        "encoded_frames": take,
    }
    return frames.contiguous(), info


def _to_nhwc_u8(frames):
    """uint8 NHWC from whatever a decoder handed back."""
    if frames.ndim == 5:
        # [B, C, T, H, W] from TAEHV, or [B, T, H, W, C] from the H3 VAE.
        if int(frames.shape[1]) == 3:
            frames = frames[0].permute(1, 2, 3, 0)
        else:
            frames = frames.reshape(-1, *frames.shape[-3:])
    if frames.ndim != 4:
        raise TAEH3TestError("cannot interpret decoder output %r" % (tuple(frames.shape),))
    if int(frames.shape[1]) == 3 and int(frames.shape[-1]) != 3:
        frames = frames.permute(0, 2, 3, 1)
    frames = frames[..., :3].detach().to("cpu", torch.float32)
    return (frames.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).contiguous()


def _write_video(path, frames_u8, fps, ffmpeg_location=None):
    if int(frames_u8.shape[0]) == 0:
        raise TAEH3TestError("refusing to write an empty video: %s" % path)
    writer = FFmpegVideoWriter(
        path,
        width=int(frames_u8.shape[2]),
        height=int(frames_u8.shape[1]),
        fps=int(fps),
        ffmpeg_location=ffmpeg_location,
        crf=16,
        preset="medium",
    ).open()
    try:
        writer.write(frames_u8)
    finally:
        writer.close(commit=True)
    return path


def _contact_indices(count):
    if count <= 0:
        return []
    last = count - 1
    picks = [0, round(last * 0.25), round(last * 0.5), round(last * 0.75), last]
    return sorted({int(min(max(i, 0), last)) for i in picks})


def _save_contact_sheet(path, frames_u8, label):
    """One labelled strip of first / 25% / mid / 75% / last."""
    from PIL import Image

    indices = _contact_indices(int(frames_u8.shape[0]))
    if not indices:
        return None
    tiles = [Image.fromarray(frames_u8[i].numpy(), "RGB") for i in indices]
    width = sum(t.width for t in tiles)
    height = max(t.height for t in tiles)
    sheet = Image.new("RGB", (width, height), (0, 0, 0))
    offset = 0
    for tile in tiles:
        sheet.paste(tile, (offset, 0))
        offset += tile.width
    sheet.save(path)
    return {"path": path, "label": label, "frame_indices": indices}


def _match_for_display(left_u8, right_u8):
    """Normalize two streams to one geometry and length, for viewing only.

    The native shapes are recorded separately; nothing here feeds a measurement.
    """
    import torch.nn.functional as F

    count = min(int(left_u8.shape[0]), int(right_u8.shape[0]))
    left_u8, right_u8 = left_u8[:count], right_u8[:count]
    height = max(int(left_u8.shape[1]), int(right_u8.shape[1]))

    def fit(frames):
        h, w = int(frames.shape[1]), int(frames.shape[2])
        if h == height:
            return frames
        out_w = max(2, round(w * height / h) // 2 * 2)
        nchw = frames.permute(0, 3, 1, 2).to(torch.float32)
        resized = F.interpolate(
            nchw, size=(height, out_w), mode="bilinear",
            align_corners=False, antialias=True,
        )
        return (
            resized.round_().clamp_(0, 255).to(torch.uint8)
            .permute(0, 2, 3, 1).contiguous()
        )

    left_u8, right_u8 = fit(left_u8), fit(right_u8)
    pair = torch.cat([left_u8, right_u8], dim=2)
    if int(pair.shape[2]) % 2:  # libx264 yuv420p needs even dimensions
        pair = pair[:, :, :-1]
    if int(pair.shape[1]) % 2:
        pair = pair[:, :-1]
    return pair.contiguous(), count


# --------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------


def default_output_dir(tag="taeh3"):
    import folder_paths

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(
        folder_paths.get_output_directory(),
        "h3_taeh3_test",
        "%s_%s" % (tag, stamp),
    )


@torch.inference_mode()
def run_taeh3_decode_test(
    frames,
    h3_vae,
    taeh3_path=None,
    fps=24,
    output_dir=None,
    *,
    canvas=None,
    frame_count=GROUP_FRAMES,
    parallel=False,
    save_latent=True,
    ffmpeg_location=None,
):
    """Encode once with the H3 VAE, decode with both decoders, save everything.

    ``frames`` is an NHWC 0..1 float batch (a Comfy ``IMAGE``). Returns the
    ``results.json`` payload as a dict.
    """
    output_dir = output_dir or default_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    fps = int(fps)

    results = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
        "fps": fps,
        "artifacts": {},
        "notes": [],
    }

    # --- input ------------------------------------------------------------
    pixels, input_info = prepare_frames(
        frames, canvas=canvas, frame_count=frame_count
    )
    results["input"] = input_info
    logging.info(
        "%s: %d frames @ %dx%d -> canvas %dx%d, encoding %d",
        LOG,
        int(frames.shape[0]),
        input_info["source_resolution"][0],
        input_info["source_resolution"][1],
        input_info["canvas"][0],
        input_info["canvas"][1],
        input_info["encoded_frames"],
    )
    source_u8 = _to_nhwc_u8(pixels)
    results["artifacts"]["source"] = _write_video(
        os.path.join(output_dir, "source.mp4"), source_u8, fps, ffmpeg_location
    )

    device = comfy.model_management.get_torch_device()

    # --- encode exactly once ---------------------------------------------
    # Deliberately not latent_cache.encode: a cache hit would return the right
    # tensor with a meaningless encode time.
    with _Measure("h3_vae_encode", device) as encode_m:
        latent = h3_vae.encode(pixels)
    latent = latent.detach()
    results["encode"] = dict(encode_m.result, latent=_tensor_stats(latent))
    logging.info(
        "%s: encoded to %r in %.2f s",
        LOG, tuple(latent.shape), encode_m.result["seconds"],
    )

    latent_t = int(latent.shape[2]) if latent.ndim == 5 else None
    if latent_t is not None and latent_t != GROUP_LATENTS:
        results["notes"].append(
            "latent has %d temporal positions, not the expected %d for %d frames"
            % (latent_t, GROUP_LATENTS, input_info["encoded_frames"])
        )

    if save_latent:
        latent_path = os.path.join(output_dir, "h3_latent.safetensors")
        comfy.utils.save_torch_file(
            {"h3_latent": latent.to("cpu", torch.float32).contiguous()},
            latent_path,
        )
        results["artifacts"]["latent"] = latent_path

    # A copy per decoder, so neither can mutate the other's input.
    control_latent = latent.clone()
    approx_latent = latent.clone()

    # --- control: the full H3 VAE ----------------------------------------
    with _Measure("full_vae_decode", device) as full_m:
        full_pixels = runner.decode_chunk(h3_vae, control_latent)
    full_u8 = _to_nhwc_u8(full_pixels)
    del full_pixels, control_latent
    results["full_vae"] = dict(
        full_m.result,
        frames=int(full_u8.shape[0]),
        resolution=[int(full_u8.shape[2]), int(full_u8.shape[1])],
    )
    results["artifacts"]["full_vae"] = _write_video(
        os.path.join(output_dir, "full_vae.mp4"), full_u8, fps, ffmpeg_location
    )
    logging.info(
        "%s: full VAE -> %d frames @ %dx%d in %.2f s",
        LOG, int(full_u8.shape[0]),
        int(full_u8.shape[2]), int(full_u8.shape[1]),
        full_m.result["seconds"],
    )

    # --- TAEH3 ------------------------------------------------------------
    taeh3, model_meta = load_taeh3(taeh3_path, device=device, parallel=parallel)
    results["taeh3_model"] = model_meta

    runtime_dtype = getattr(torch, model_meta["runtime_dtype"].split(".")[-1])
    with _Measure("taeh3_decode", device) as approx_m:
        approx_pixels = taeh3.decode(approx_latent.to(device, runtime_dtype))
    native_shape = list(approx_pixels.shape)
    approx_u8 = _to_nhwc_u8(approx_pixels)
    del approx_pixels, approx_latent
    results["taeh3"] = dict(
        approx_m.result,
        native_output_shape=native_shape,
        frames=int(approx_u8.shape[0]),
        resolution=[int(approx_u8.shape[2]), int(approx_u8.shape[1])],
    )
    results["artifacts"]["taeh3"] = _write_video(
        os.path.join(output_dir, "taeh3.mp4"), approx_u8, fps, ffmpeg_location
    )
    logging.info(
        "%s: TAEH3 -> %d frames @ %dx%d in %.2f s",
        LOG, int(approx_u8.shape[0]),
        int(approx_u8.shape[2]), int(approx_u8.shape[1]),
        approx_m.result["seconds"],
    )

    if int(approx_u8.shape[0]) != int(full_u8.shape[0]):
        results["notes"].append(
            "frame counts differ: full VAE %d, TAEH3 %d (native %r); trimmed only "
            "for side_by_side.mp4"
            % (int(full_u8.shape[0]), int(approx_u8.shape[0]), native_shape)
        )
    if list(results["taeh3"]["resolution"]) != list(results["full_vae"]["resolution"]):
        results["notes"].append(
            "resolutions differ: full VAE %r, TAEH3 %r; scaled only for "
            "side_by_side.mp4"
            % (results["full_vae"]["resolution"], results["taeh3"]["resolution"])
        )

    # --- comparison artifacts --------------------------------------------
    pair_u8, pair_frames = _match_for_display(full_u8, approx_u8)
    results["artifacts"]["side_by_side"] = _write_video(
        os.path.join(output_dir, "side_by_side.mp4"), pair_u8, fps, ffmpeg_location
    )
    results["side_by_side"] = {
        "layout": "full H3 VAE | TAEH3",
        "frames": pair_frames,
        "resolution": [int(pair_u8.shape[2]), int(pair_u8.shape[1])],
    }
    del pair_u8

    sheets = []
    for name, batch, label in (
        ("source", source_u8, "source"),
        ("full_vae", full_u8, "full H3 VAE"),
        ("taeh3", approx_u8, "TAEH3"),
    ):
        sheet = _save_contact_sheet(
            os.path.join(output_dir, "contact_%s.png" % name), batch, label
        )
        if sheet:
            sheets.append(sheet)
    results["contact_sheets"] = sheets

    if results["full_vae"]["seconds"] > 0:
        results["speedup_vs_full_vae"] = (
            results["full_vae"]["seconds"] / max(results["taeh3"]["seconds"], 1e-9)
        )

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True, default=str)
    results["artifacts"]["results_json"] = results_path

    summary = format_results(results)
    summary_path = os.path.join(output_dir, "results.txt")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(summary)
    results["artifacts"]["results_txt"] = summary_path

    del source_u8, full_u8, approx_u8, taeh3
    comfy.model_management.soft_empty_cache()
    logging.info("%s: wrote %s", LOG, output_dir)
    return results


def format_results(results):
    """Human-readable ``results.txt``."""

    def mem(entry):
        if "peak_reserved_mb" not in entry:
            return ""
        return "  peak alloc %.0f MB / reserved %.0f MB" % (
            entry.get("peak_allocated_mb", 0.0),
            entry["peak_reserved_mb"],
        )

    inp = results["input"]
    enc = results["encode"]
    full = results["full_vae"]
    approx = results["taeh3"]
    model = results["taeh3_model"]
    lines = [
        "TAEH3 vs full MiniMax H3 VAE - decode from one identical latent",
        "=" * 64,
        "created      %s" % results["created"],
        "output       %s" % results["output_dir"],
        "",
        "INPUT",
        "  source     %d frames @ %dx%d" % (
            inp["source_frames"], inp["source_resolution"][0],
            inp["source_resolution"][1]),
        "  canvas     %dx%d, encoded %d frames"
        % (inp["canvas"][0], inp["canvas"][1], inp["encoded_frames"]),
        "",
        "TAEH3 MODEL",
        "  path       %s" % model["path"],
        "  weights    %s -> runtime %s on %s"
        % (model["weight_dtype"], model["runtime_dtype"], model["device"]),
        "  params     %.2f M" % (model["parameters"] / 1e6),
        "  latent     %d ch, patch %d, spatial %dx, temporal %dx, trim %d"
        % (model["latent_channels"], model["patch_size"],
           model["spatial_upscale"], model["t_upscale"], model["frames_to_trim"]),
        "  keys       missing %d, unexpected %d"
        % (len(model["missing_keys"]), len(model["unexpected_keys"])),
        "",
        "ENCODE (H3 VAE, once)",
        "  latent     %r %s, min %.3f max %.3f mean %.3f std %.3f"
        % (tuple(enc["latent"]["shape"]), enc["latent"]["dtype"],
           enc["latent"]["min"], enc["latent"]["max"],
           enc["latent"]["mean"], enc["latent"]["std"]),
        "  time       %.2f s%s" % (enc["seconds"], mem(enc)),
        "",
        "DECODE - full H3 VAE (control)",
        "  output     %d frames @ %dx%d"
        % (full["frames"], full["resolution"][0], full["resolution"][1]),
        "  time       %.2f s%s" % (full["seconds"], mem(full)),
        "",
        "DECODE - TAEH3",
        "  native     %r" % (tuple(approx["native_output_shape"]),),
        "  output     %d frames @ %dx%d"
        % (approx["frames"], approx["resolution"][0], approx["resolution"][1]),
        "  time       %.2f s%s" % (approx["seconds"], mem(approx)),
    ]
    if "speedup_vs_full_vae" in results:
        lines.append("  speedup    %.1fx vs full VAE" % results["speedup_vs_full_vae"])
    lines += ["", "ARTIFACTS"]
    for name, path in sorted(results["artifacts"].items()):
        lines.append("  %-13s %s" % (name, os.path.basename(path)))
    if results["notes"]:
        lines += ["", "NOTES"]
        lines += ["  - %s" % note for note in results["notes"]]
    lines += [
        "",
        "Judge side_by_side.mp4 (full H3 VAE | TAEH3): can you tell who is in",
        "frame, the composition, pose, direction of motion, large object state,",
        "and gross temporal consistency? Fine texture is not the question.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "TAEH3TestError",
    "resolve_taeh3_path",
    "load_taeh3",
    "prepare_frames",
    "run_taeh3_decode_test",
    "format_results",
    "default_output_dir",
]
