"""Node entry point for the TAEH3 decode smoke test.

A node rather than a script because the test needs the *production* H3 VAE. Any
standalone runner would have to load a second 4.9 GB copy of it beside whatever
is already resident, which on a 12 GB card is how the OOM cascade starts.
"""

import logging

from comfy_api.latest import ComfyExtension, io

from . import taeh3_decode_test

LOG = taeh3_decode_test.LOG


class MiniMaxH3TAEH3DecodeTest(io.ComfyNode):
    """Decode one identical H3 latent with the full VAE and with TAEH3.

    Answers only the first question in the preview investigation: can TAEH3 read
    valid H3 latent space at all? Sampler-intermediate latents are a separate
    test and are deliberately not covered here.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3TAEH3DecodeTestZi",
            display_name="MiniMax H3 TAEH3 Decode Test (Zi)",
            category="H3-Extender/Diagnostics",
            description=(
                "Encode a clip once with the real H3 VAE, then decode that one "
                "latent with both the full VAE and TAEH3. Writes source/full_vae/"
                "taeh3/side_by_side MP4s, the latent, and timing + VRAM figures to "
                "output/h3_taeh3_test/<tag>_<timestamp>/."
            ),
            inputs=[
                io.Image.Input(
                    "frames",
                    tooltip="Source clip as an IMAGE batch. Resized to an H3 canvas before encoding.",
                ),
                io.Vae.Input(
                    "h3_vae",
                    tooltip="The production MiniMax H3 *video* VAE. Used for both the encode and the control decode.",
                ),
                io.Int.Input(
                    "frame_count",
                    default=17,
                    min=5,
                    max=257,
                    tooltip="Frames to encode. Snapped down to the VAE's 1+4k grid; 17 frames is one 5-latent decode group.",
                ),
                io.Int.Input("fps", default=24, min=1, max=120),
                io.String.Input(
                    "taeh3_path",
                    default="",
                    tooltip="Blank auto-discovers taeh3.safetensors / taeh3.pth in models/vae_approx.",
                ),
                io.String.Input("run_tag", default="taeh3"),
                io.Int.Input(
                    "canvas_width",
                    default=0,
                    min=0,
                    max=4096,
                    step=32,
                    tooltip="0 derives the canvas from the source aspect, exactly as the reference encoder does.",
                ),
                io.Int.Input("canvas_height", default=0, min=0, max=4096, step=32),
                io.Boolean.Input(
                    "parallel",
                    default=False,
                    tooltip="Decode all timesteps at once. Faster, more VRAM. Off matches the low-memory sequential path a preview would use.",
                ),
                io.String.Input(
                    "ffmpeg_location",
                    default="",
                    tooltip="Blank resolves ffmpeg from PATH, then imageio_ffmpeg.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="full_vae"),
                io.Image.Output(display_name="taeh3"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(
        cls,
        frames,
        h3_vae,
        frame_count,
        fps,
        taeh3_path,
        run_tag,
        canvas_width,
        canvas_height,
        parallel,
        ffmpeg_location,
    ) -> io.NodeOutput:
        import torch

        canvas = None
        if canvas_width and canvas_height:
            canvas = (int(canvas_width), int(canvas_height))

        results = taeh3_decode_test.run_taeh3_decode_test(
            frames,
            h3_vae,
            taeh3_path=taeh3_path.strip() or None,
            fps=int(fps),
            output_dir=taeh3_decode_test.default_output_dir(run_tag or "taeh3"),
            canvas=canvas,
            frame_count=int(frame_count),
            parallel=bool(parallel),
            ffmpeg_location=ffmpeg_location.strip() or None,
        )
        report = taeh3_decode_test.format_results(results)
        logging.info("%s report:\n%s", LOG, report)

        # The MP4s on disk are the artifact of record; these outputs just let the
        # graph show the two decodes without reopening the files.
        def _preview(path):
            import numpy as np
            from PIL import Image

            image = Image.open(path).convert("RGB")
            array = np.asarray(image, dtype="float32") / 255.0
            return torch.from_numpy(array).unsqueeze(0)

        sheets = {sheet["label"]: sheet["path"] for sheet in results["contact_sheets"]}
        full = _preview(sheets["full H3 VAE"])
        approx = _preview(sheets["TAEH3"])
        return io.NodeOutput(full, approx, report)


class MiniMaxH3DiagnosticsExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3TAEH3DecodeTest]
