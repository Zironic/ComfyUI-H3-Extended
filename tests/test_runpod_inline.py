"""CPU contracts for the storage-free RunPod transport."""

import base64
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest

import h3_test_tempfile as tempfile


_ROOT = Path(__file__).resolve().parents[1]
_DEPLOY = _ROOT / "deploy" / "runpod"


def load_module(name, path, stubs=None):
    previous = {}
    for module_name, module in (stubs or {}).items():
        previous[module_name] = sys.modules.get(module_name)
        sys.modules[module_name] = module
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, module in previous.items():
            if module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = module


class RunPodInlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runpod = types.ModuleType("runpod")
        runpod.serverless = types.SimpleNamespace(start=lambda _config: None)
        websocket = types.ModuleType("websocket")
        cls.handler = load_module(
            "h3_runpod_handler_test",
            _DEPLOY / "handler.py",
            {"runpod": runpod, "websocket": websocket},
        )
        cls.prepare = load_module(
            "h3_runpod_prepare_test", _DEPLOY / "prepare_workflow.py"
        )
        cls.client = load_module(
            "h3_runpod_client_test",
            _DEPLOY / "run_h3.py",
            {"prepare_workflow": cls.prepare},
        )
        cls.linker = load_module(
            "h3_runpod_linker_test", _DEPLOY / "link_h3_models.py"
        )

    def test_asset_round_trip_and_workflow_placeholders(self):
        with tempfile.TemporaryDirectory(prefix="runpod-inline-") as root:
            root = Path(root)
            self.handler.INPUT_ROOT = root / "input"
            self.handler.OUTPUT_ROOT = root / "output"
            data = b"small reference file"
            asset = {
                "name": "references/reference.png",
                "base64": base64.b64encode(data).decode("ascii"),
            }
            decoded = self.handler.decode_asset("job-1", asset)
            self.assertEqual(Path(decoded["path"]).read_bytes(), data)
            workflow, prefix = self.handler.prepare_workflow(
                "job-1",
                {
                    "1": {"inputs": {"image": "{{ASSET:references/reference.png}}"}},
                    "2": {"inputs": {"filename_prefix": "{{RUNPOD_OUTPUT_PREFIX}}"}},
                    "3": {"inputs": {"run_tag": "{{RUNPOD_JOB_ID}}"}},
                },
                [decoded],
            )
            self.assertEqual(
                workflow["1"]["inputs"]["image"],
                "runpod/job-1/references/reference.png",
            )
            self.assertEqual(workflow["2"]["inputs"]["filename_prefix"], "runpod/job-1/result")
            self.assertEqual(workflow["3"]["inputs"]["run_tag"], "job-1")
            self.assertEqual(prefix, "runpod/job-1/result")

    def test_invalid_asset_base64_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="runpod-inline-") as root:
            self.handler.INPUT_ROOT = Path(root) / "input"
            with self.assertRaisesRegex(ValueError, "invalid base64"):
                self.handler.decode_asset(
                    "job-2", {"name": "reference.png", "base64": "not base64!"}
                )

    def test_artifact_is_returned_and_saved_inline(self):
        with tempfile.TemporaryDirectory(prefix="runpod-inline-") as root:
            root = Path(root)
            source = root / "result.mp4"
            source.write_bytes(b"video bytes")
            artifact = self.handler.encode_artifact(source)
            result = {"status": "COMPLETED", "output": {"artifacts": [artifact]}}
            saved = self.client.save_artifacts(result, root / "saved")
            self.assertEqual(saved, [root / "saved" / "result.mp4"])
            self.assertEqual(saved[0].read_bytes(), b"video bytes")

    def test_local_input_name_can_differ_from_source_filename(self):
        with tempfile.TemporaryDirectory(prefix="runpod-inline-") as root:
            source = Path(root) / "local-file.png"
            source.write_bytes(b"image")
            name, path = self.client.parse_input(f"reference.png={source}")
            self.assertEqual(name, "reference.png")
            self.assertEqual(path, source)

    def test_workflow_keeps_hybrid_and_removes_local_only_nodes(self):
        workflow = {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": "local.png"},
            },
            "2": {
                "class_type": "ZiroScaleImageToNativeCanvas",
                "inputs": {"image": ["1", 0], "short_edge": 768},
            },
            "3": {
                "class_type": "MiniMaxH3ReferenceToVideoZi",
                "inputs": {"ref_images.ref_image_0": ["2", 0]},
            },
            "4": {
                "class_type": "SamplerCustomAdvancedMiniMaxPreview",
                "inputs": {
                    "noise": ["10", 0],
                    "guider": ["11", 0],
                    "sampler": ["12", 0],
                    "sigmas": ["13", 0],
                    "latent_image": ["3", 1],
                    "enable_preview": True,
                },
            },
            "5": {
                "class_type": "MiniMaxH3HybridSparseAttentionZi",
                "inputs": {
                    "model": ["6", 0],
                    "enabled": True,
                    "mode": "portable_sparse",
                    "run_tag": "local",
                },
            },
            "6": {
                "class_type": "UNETLoader",
                "inputs": {"unet_name": "hf_minimax_h3\\model.safetensors"},
            },
            "7": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": "local-fp8.safetensors"},
            },
            "8": {
                "class_type": "SaveVideo",
                "inputs": {"video": ["3", 0], "filename_prefix": "video/local"},
            },
            "9": {
                "class_type": "LoadImage",
                "inputs": {"image": "unused.png"},
            },
        }
        prepared = self.prepare.prepare_workflow(workflow, "reference.png")
        self.assertNotIn("2", prepared)
        self.assertEqual(prepared["3"]["inputs"]["ref_images.ref_image_0"], ["1", 0])
        self.assertEqual(prepared["1"]["inputs"]["image"], "{{ASSET:reference.png}}")
        self.assertEqual(prepared["9"]["inputs"]["image"], "unused.png")
        self.assertEqual(prepared["4"]["class_type"], "SamplerCustomAdvanced")
        self.assertNotIn("enable_preview", prepared["4"]["inputs"])
        self.assertTrue(prepared["5"]["inputs"]["enabled"])
        self.assertEqual(prepared["5"]["inputs"]["mode"], "portable_sparse")
        self.assertEqual(prepared["5"]["inputs"]["run_tag"], "{{RUNPOD_JOB_ID}}")
        self.assertEqual(prepared["6"]["inputs"]["unet_name"], "model.safetensors")
        self.assertEqual(
            prepared["7"]["inputs"]["clip_name"],
            "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        )
        self.assertEqual(
            prepared["8"]["inputs"]["filename_prefix"],
            "{{RUNPOD_OUTPUT_PREFIX}}",
        )

    def test_saved_connected_inputs_are_resolved_without_input_flags(self):
        with tempfile.TemporaryDirectory(prefix="runpod-workflow-") as root:
            root = Path(root)
            workflow_path = root / "User" / "default" / "workflows" / "h3.json"
            input_root = root / "Input"
            saved_input = input_root / "references" / "person.png"
            workflow_path.parent.mkdir(parents=True)
            saved_input.parent.mkdir(parents=True)
            workflow_path.write_text("{}", encoding="utf-8")
            saved_input.write_bytes(b"image")
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": "references/person.png"},
                },
                "2": {
                    "class_type": "SamplerCustomAdvanced",
                    "inputs": {"latent_image": ["1", 0]},
                },
                "3": {
                    "class_type": "MiniMaxH3HybridSparseAttentionZi",
                    "inputs": {"enabled": True},
                },
                "4": {
                    "class_type": "SaveVideo",
                    "inputs": {"video": ["2", 0]},
                },
            }
            prepared = self.prepare.prepare_workflow(workflow)
            self.assertEqual(
                prepared["1"]["inputs"]["image"],
                "{{ASSET:references/person.png}}",
            )
            names = self.client.workflow_asset_names(prepared)
            self.assertEqual(names, {"references/person.png"})
            resolved_root = self.client.resolve_input_root(workflow_path, None)
            self.assertEqual(resolved_root, input_root.resolve())
            self.assertEqual(
                self.client.resolve_saved_input(resolved_root, "references/person.png"),
                saved_input.resolve(),
            )

    def test_asset_parent_traversal_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid input name"):
            self.client.safe_asset_name("../secret.png")
        with self.assertRaisesRegex(ValueError, "Invalid asset name"):
            self.handler.safe_asset_name("../secret.png")

    def test_required_cached_model_set_rejects_a_partial_snapshot(self):
        with tempfile.TemporaryDirectory(prefix="runpod-models-") as root:
            root = Path(root)
            cache = root / "cache"
            snapshot = (
                cache
                / "models--Comfy-Org--MiniMax-H3"
                / "snapshots"
                / "revision"
            )
            snapshot.mkdir(parents=True)
            missing = "minimax_h3_audio_vae_fp32.safetensors"
            targets = {}
            for name in self.linker.MODEL_TARGETS:
                targets[name] = root / "models"
                if name != missing:
                    (snapshot / name).write_bytes(b"model")
            self.linker.CACHE_ROOT = cache
            self.linker.MODEL_TARGETS = targets
            self.linker.REQUIRED = True
            with self.assertRaisesRegex(SystemExit, missing):
                self.linker.main()


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    unittest.main()
