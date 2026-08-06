"""Run a long-form Ref2V job outside ComfyUI.

`long-form plan.md` section 3 calls for a CLI so I/O and recovery can be tested
without the server. It earns its place immediately for a second reason: a new
node needs a ComfyUI restart to appear, and restarting a server someone is not
sitting in front of - with no guarantee the desktop wrapper brings it back - is
a worse failure than not running at all.

This loads the models directly, so it needs no restart and no graph. Ask the
running server to release its models first (`--free-server`), or the two
processes contend for the same card.

    python -m chunked_ref2v.longform.cli \
        --video "D:/clip.webm" --start-frame 4320 --seconds 180 \
        --chunk 90 --overlap 22 --carry direct_latent_overlap \
        --prompt-file "D:/prompt.txt" --ref-image a.png --ref-image b.png
"""

import argparse
import json
import logging
import os
import sys
import time


DESKTOP_MODEL_PATHS = os.path.join(
    os.path.expanduser("~"), "AppData", "Roaming", "Comfy Desktop",
    "shared_model_paths.yaml")


def _bootstrap():
    """Put the ComfyUI root and this pack on the path and neutralize argv.

    `comfy.cli_args` parses `sys.argv` the moment a comfy module is imported, so
    our own flags have to be off it by then or ComfyUI's parser rejects them.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    pack = os.path.dirname(os.path.dirname(here))
    root = os.path.dirname(os.path.dirname(pack))
    for path in (pack, root):
        if path not in sys.path:
            sys.path.insert(0, path)
    sys.argv = [sys.argv[0]]
    import comfy.options
    comfy.options.enable_args_parsing()

    # `main.py` line 72 sets this on Windows before torch loads. Without it
    # mimalloc holds freed pages instead of returning them to the OS, and a VAE
    # decode is nothing but churn - 5 temporal clips x 6 spatial tiles, each
    # allocating and dropping activation buffers, plus the clone/blend
    # temporaries. RSS then climbs with the number of allocations rather than
    # the live set, which is why a T=7 decode cost as much as a T=27 one.
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

    # `main.py` imports cuda_malloc at line 100, before torch, and that import
    # sets PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync. The variable is only
    # read when torch first initializes CUDA, so importing it late does nothing.
    # Without it this process runs the default caching allocator while the
    # server runs cudaMallocAsync - a different allocator under the same AIMDO
    # host-backed pages.
    try:
        import cuda_malloc  # noqa: F401
        logging.info("allocator config: %s",
                     os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<default>"))
    except Exception as exc:
        logging.warning("could not apply ComfyUI's allocator config: %s", exc)

    _enable_dynamic_vram()
    return pack, root


def _enable_dynamic_vram():
    """Turn on AIMDO, exactly as `main.py` does.

    Without this every model *fully materializes*: the log reads
    "loaded completely ... full load: True" instead of "prepared for dynamic
    VRAM loading ... Staged", and a 19.5 GB DiT plus a 14.6 GB encoder get
    resident in host RAM at once. That is what ran the first smoke test past
    60 GB - not pinned memory, which is only a ceiling.

    It is two steps in two places (`main.py` lines 59-70 and 251-280) and doing
    only the second silently leaves the allocator in the legacy mode.
    """
    from comfy.cli_args import args, enables_dynamic_vram
    if not enables_dynamic_vram():
        logging.warning("dynamic VRAM is disabled by args; models will fully load")
        return False
    try:
        import comfy_aimdo.control
    except ImportError:
        logging.warning("comfy_aimdo unavailable; models will fully load")
        return False

    headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
    try:
        comfy_aimdo.control.init(simple_vram_headroom=headroom,
                                 nvml_pressure=not args.disable_nvml_pressure)
    except TypeError:
        try:
            comfy_aimdo.control.init(simple_vram_headroom=headroom)
        except TypeError:
            comfy_aimdo.control.init()

    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher
    devices = comfy.model_management.get_all_torch_devices()
    try:
        started = comfy_aimdo.control.init_devices(
            (d.index, int(args.vram_headroom * 1024 ** 3)) for d in devices)
    except TypeError:
        started = comfy_aimdo.control.init_devices(d.index for d in devices)

    if not started:
        logging.warning("comfy_aimdo did not initialize; models will fully load")
        return False
    # main.py picks a level from the console log setting; without that call the
    # library defaults to VERBOSE and emits a line per 32 MB page faulted, which
    # is megabytes of output and a real slowdown over a multi-hour run.
    for level in ("set_log_warning", "set_log_error"):
        setter = getattr(comfy_aimdo.control, level, None)
        if setter is not None:
            setter()
            break
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    logging.info("DynamicVRAM (AIMDO) enabled")
    return True


def _load_model_paths(explicit=None):
    """Teach a standalone process where the models are.

    This install keeps code on C: and models on D:, wired up by Comfy Desktop
    rather than by an `extra_model_paths.yaml` in the repo - so a process that
    is not the desktop-launched server has to load that config itself or it will
    not find a single checkpoint.
    """
    import utils.extra_config
    for candidate in (explicit, DESKTOP_MODEL_PATHS):
        if candidate and os.path.isfile(candidate):
            utils.extra_config.load_extra_path_config(candidate)
            logging.info("loaded model paths from %s", candidate)
            return candidate
    logging.warning("no extra model paths config found; only the repo's own "
                    "models/ tree will be searched")
    return None


def free_server(host="http://127.0.0.1:8188"):
    """Ask a running ComfyUI to drop its models so we are not fighting for VRAM."""
    import json as _json
    import urllib.request
    try:
        request = urllib.request.Request(
            host + "/api/free",
            data=_json.dumps({"unload_models": True, "free_memory": True}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(request, timeout=10)
        # The release is not instantaneous; a reading taken immediately is stale.
        time.sleep(8)
        return True
    except Exception as exc:
        logging.warning("could not free the running server (%s); continuing", exc)
        return False


def load_models(args):
    import torch
    import comfy.sd
    import comfy.utils
    import folder_paths

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", args.unet)
    model = comfy.sd.load_diffusion_model(unet_path)

    clip_path = folder_paths.get_full_path_or_raise("text_encoders", args.clip)
    clip = comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, "MINIMAX"),
        model_options={})

    def load_vae(name):
        path = folder_paths.get_full_path_or_raise("vae", name)
        sd, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
        vae.throw_exception_if_invalid()
        return vae

    return model, clip, load_vae(args.video_vae), load_vae(args.audio_vae)


def load_ref_images(paths):
    import numpy as np
    import torch
    from PIL import Image
    out = {}
    for i, path in enumerate(paths or []):
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        out["ref_image_%d" % i] = torch.from_numpy(arr)[None]
    return out


def main(argv=None):
    # Parse our own flags off argv *before* any comfy import sees it.
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=180.0)
    parser.add_argument("--chunks", type=int, default=0,
                        help="explicit chunk count; overrides --seconds")
    parser.add_argument("--chunk", type=int, default=90)
    parser.add_argument("--overlap", type=int, default=22)
    parser.add_argument("--carry", default="direct_latent_overlap")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--ref-image", action="append", default=[])
    parser.add_argument("--megapixels", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--sampler", default="res_multistep")
    parser.add_argument("--scheduler", default="simple")
    parser.add_argument("--unet", default=r"hf_minimax_h3\minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    parser.add_argument("--clip", default="qwen3vl_32b_minimax_h3_fp8_clean.safetensors")
    parser.add_argument("--video-vae", default=r"hf_minimax_h3\minimax_h3_video_vae_fp16.safetensors")
    parser.add_argument("--audio-vae", default=r"hf_minimax_h3\minimax_h3_audio_vae_fp32.safetensors")
    parser.add_argument("--run-directory", default="")
    parser.add_argument("--vram-guard-mb", type=int, default=800)
    parser.add_argument("--free-server", action="store_true")
    parser.add_argument("--model-paths", default=None,
                        help="extra_model_paths-style yaml; defaults to the desktop config")
    parser.add_argument("--no-frames", action="store_true")
    args = parser.parse_args(argv)

    # Windows consoles default to cp1252, and source filenames routinely carry
    # characters it cannot encode (a fullwidth vertical bar, here). Without this
    # the run dies on a print statement after the models are already loaded.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    if args.free_server:
        free_server()

    _bootstrap()
    _load_model_paths(args.model_paths)

    import math
    import torch
    import comfy.samplers
    import folder_paths

    from chunked_ref2v import memory
    from chunked_ref2v.geometry import HarnessGeometry
    from chunked_ref2v.longform import runner
    from chunked_ref2v.longform.chunk_stream import chunk_count_for, frames_needed_for
    from chunked_ref2v.longform.frame_source import probe
    from vram_guard import install_unet_guard

    geometry = HarnessGeometry(chunk_frames=args.chunk,
                               overlap_frames=args.overlap).validate()
    if args.chunks:
        chunk_count = args.chunks
    else:
        chunk_count = chunk_count_for(int(args.seconds * geometry.fps),
                                      geometry.chunk_frames, geometry.stride_frames)
    needed = frames_needed_for(chunk_count, geometry.chunk_frames,
                               geometry.stride_frames)

    metadata = probe(args.video)
    available = (metadata.estimated_frames or 0) - args.start_frame
    if available < needed:
        raise SystemExit("source has ~%d frames after start_frame %d; %d chunks "
                         "need %d" % (available, args.start_frame, chunk_count, needed))

    # One canvas for the whole run, at the requested pixel budget.
    ratio = metadata.source_width / metadata.source_height
    h = math.sqrt(args.megapixels * 1e6 / ratio)
    canvas = (max(32, round(h * ratio / 32) * 32), max(32, round(h / 32) * 32))

    with open(args.prompt_file, encoding="utf-8") as fh:
        prompt = fh.read()
    subject = next((l.strip() for l in prompt.splitlines() if "<Subject 1> is" in l),
                   "NO <Subject 1> LINE")

    print("=" * 72)
    print("long-form Ref2V")
    print("  source     %s" % args.video)
    print("  window     start_frame %d, %d chunks, %d source frames"
          % (args.start_frame, chunk_count, needed))
    print("  profile    %s" % geometry.describe())
    print("  canvas     %dx%d  (%.2f MP)" % (canvas[0], canvas[1],
                                             canvas[0] * canvas[1] / 1e6))
    print("  carry      %s" % args.carry)
    print("  prompt     %d chars | %s" % (len(prompt), subject))
    print("  output     ~%.1f s" % (((chunk_count - 1) * geometry.stride_frames
                                     + geometry.chunk_frames) / geometry.fps))
    print("=" * 72)

    started = time.time()
    model, clip, video_vae, audio_vae = load_models(args)
    print("models loaded in %.1f s" % (time.time() - started))

    install_unet_guard(model, args.vram_guard_mb)
    model, status = memory.arm(model, attention="auto",
                               activation="mlp_chunked_native")
    print(memory.describe(status))

    sampler = comfy.samplers.sampler_object(args.sampler)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), args.scheduler, args.steps)

    root = args.run_directory or os.path.join(
        folder_paths.get_output_directory(), "h3_longform",
        "%s_%s_c%d" % (time.strftime("%Y%m%d_%H%M%S"),
                       args.carry.replace("direct_latent_", ""), args.chunk))

    summary = runner.run(
        video_path=args.video, start_frame=args.start_frame,
        chunk_frames=args.chunk, overlap_frames=args.overlap,
        chunk_count=chunk_count, model=model, clip=clip, video_vae=video_vae,
        audio_vae=audio_vae, prompt=prompt, sampler=sampler, sigmas=sigmas,
        seed=args.seed, carry=args.carry, canvas=canvas, root=root,
        ref_images=load_ref_images(args.ref_image),
        save_frames=not args.no_frames)

    summary["elapsed_minutes"] = round((time.time() - started) / 60, 1)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
