"""CPU-only compatibility checks for the Zi MiniMax H3 sigma-shift node."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
_ARGV, sys.argv = list(sys.argv), [sys.argv[0], "--cpu"]

import comfy.model_sampling  # noqa: E402
import nodes_minimax_h3  # noqa: E402


class _OriginalSampling:
    noise_scale = 0.75


class _FakeModel:
    def __init__(self, model_config=None):
        self.model = SimpleNamespace(model_config=model_config or SimpleNamespace(sampling_settings={}))
        self.model_options = {"transformer_options": {"existing": True}}
        self._objects = {"model_sampling": _OriginalSampling()}

    def clone(self):
        clone = _FakeModel(self.model.model_config)
        clone.model_options = {key: dict(value) if isinstance(value, dict) else value
                               for key, value in self.model_options.items()}
        clone._objects = dict(self._objects)
        return clone

    def get_model_object(self, name):
        return self._objects[name]

    def add_object_patch(self, name, obj):
        self._objects[name] = obj


class MiniMaxH3SamplingTests(unittest.TestCase):
    def test_sigma_shift_installs_av_sampling_and_transformer_keys(self):
        model = _FakeModel()
        with mock.patch.object(nodes_minimax_h3, "install_unet_guard"):
            output = nodes_minimax_h3.MiniMaxH3SigmaShift.execute(
                model, 12.0, 3.0, attention_backend="comfy", vram_guard_mb=0
            )

        patched = output.result[0]
        sampling = patched.get_model_object("model_sampling")
        self.assertIsInstance(sampling, comfy.model_sampling.ModelSamplingAV)
        self.assertEqual(sampling.shift, 12.0)
        self.assertEqual(sampling.audio_shift, 3.0)
        self.assertEqual(sampling.audio_scale, 4.0)
        options = patched.model_options["transformer_options"]
        self.assertEqual(options["minimax_h3_sigma_shift_video"], 12.0)
        self.assertEqual(options["minimax_h3_sigma_shift_audio"], 3.0)
        self.assertNotIn("h3_video_shift", options)
        self.assertNotIn("h3_audio_shift", options)

    def test_sigma_shift_explains_old_comfy_version(self):
        model = _FakeModel()
        with mock.patch.object(comfy.model_sampling, "ModelSamplingAV", None):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI v0.31.0+"):
                nodes_minimax_h3.MiniMaxH3SigmaShift.execute(
                    model, 12.0, 3.0, attention_backend="comfy", vram_guard_mb=0
                )


if __name__ == "__main__":
    sys.argv = _ARGV
    unittest.main()
