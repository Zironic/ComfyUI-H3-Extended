"""Minimal multimodal regression comparison for H3 acceleration experiments."""

import argparse
import json
import os
import sys

import torch
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from h3_runtime.metrics import audio_mse, video_psnr  # noqa: E402


def frames(path):
    names = sorted(
        name for name in os.listdir(path)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    return [Image.open(os.path.join(path, name)).convert("RGB") for name in names]


def load_audio(path):
    if not path:
        return None
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("audio", "waveform", "samples"):
            if key in value:
                return value[key]
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-frames", required=True)
    parser.add_argument("--candidate-frames", required=True)
    parser.add_argument("--reference-audio")
    parser.add_argument("--candidate-audio")
    args = parser.parse_args()

    result = {
        "video": video_psnr(
            frames(args.reference_frames),
            frames(args.candidate_frames),
        ),
        "audio": audio_mse(
            load_audio(args.reference_audio),
            load_audio(args.candidate_audio),
        ),
        "note": (
            "PSNR/MSE are regression gates, not perceptual quality certificates; "
            "review identity, motion, dialogue and synchronization separately."
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
