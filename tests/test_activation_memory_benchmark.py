"""CPU checks for the activation-memory benchmark's checkpoint plumbing."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from benchmarks import benchmark_h3_activation_memory as bench  # noqa: E402
from h3_test_tempfile import TemporaryDirectory  # noqa: E402


def test_case_parsing_matrix():
    cases = bench.iter_cases(
        bench.parse_chunks("2048,4096,8192,16384"),
        bench.parse_modes("bf16,native", bench.DEFAULT_SWIGLU_MODES, "--swiglu-modes"),
        bench.parse_modes("off,on", bench.DEFAULT_HELD_MODES, "--held-modes"),
    )
    assert len(cases) == 16
    assert cases[0] == {"chunk_rows": 2048, "swiglu_mode": "bf16", "held_mode": "off"}
    assert cases[-1] == {"chunk_rows": 16384, "swiglu_mode": "native", "held_mode": "on"}


def test_checkpoint_resolution():
    with TemporaryDirectory() as temp:
        path = Path(temp) / "weights.safetensors"
        path.write_bytes(b"x")
        assert bench.resolve_checkpoint(str(path)) == str(path.resolve())
        with mock.patch("folder_paths.get_full_path_or_raise", return_value=str(path)) as resolve:
            assert bench.resolve_checkpoint("registered.safetensors") == str(path.resolve())
            resolve.assert_called_once_with("diffusion_models", "registered.safetensors")
        try:
            bench.resolve_checkpoint(str(Path(temp) / "wrong.bin"))
        except ValueError:
            pass
        else:
            raise AssertionError("non-safetensors absolute paths must be rejected")


def test_actual_mode_requires_cuda_before_loading_checkpoint():
    with mock.patch.object(bench, "resolve_checkpoint") as resolve:
        try:
            bench.main(["--checkpoint", "weights.safetensors", "--device", "cpu", "--dtype", "bf16"])
        except ValueError as exc:
            assert "--device cuda" in str(exc)
        else:
            raise AssertionError("actual checkpoint mode must reject CPU execution")
        resolve.assert_not_called()


class _FakeSafeHandle:
    def __init__(self):
        self.requested = []
        self.tensors = {
            "model.diffusion_model.blocks.2.mlp.fc1.weight": torch.empty(8, 4),
            "model.diffusion_model.blocks.2.mlp.fc1.comfy_quant": torch.tensor([1], dtype=torch.uint8),
            "model.diffusion_model.blocks.2.mlp.fc2.weight": torch.empty(4, 4),
            "model.diffusion_model.blocks.2.mlp.fc2.weight_scale": torch.empty(1),
            "model.diffusion_model.blocks.1.mlp.fc1.weight": torch.empty(8, 4),
        }

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self.tensors)

    def get_tensor(self, key):
        self.requested.append(key)
        return self.tensors[key]


def test_selective_loader_rebases_one_block():
    handle = _FakeSafeHandle()

    def fake_open(*args, **kwargs):
        assert kwargs == {"framework": "pt", "device": "cpu"}
        return handle

    loaded = bench.load_block_mlp_tensors("fake.safetensors", block_index=2, safe_open_fn=fake_open)
    assert loaded["prefix"] == "model.diffusion_model.blocks.2.mlp."
    assert set(loaded["state_dict"]) == {
        "fc1.weight", "fc1.comfy_quant", "fc2.weight", "fc2.weight_scale"
    }
    assert set(handle.requested) == {
        "model.diffusion_model.blocks.2.mlp.fc1.weight",
        "model.diffusion_model.blocks.2.mlp.fc1.comfy_quant",
        "model.diffusion_model.blocks.2.mlp.fc2.weight",
        "model.diffusion_model.blocks.2.mlp.fc2.weight_scale",
    }


def test_checkpoint_modules_remain_offloaded():
    class FakeLinear:
        def __init__(self, *args, **kwargs):
            self.loaded = None
            self.to_calls = []

        def load_state_dict(self, state, strict):
            assert strict
            self.loaded = state

        def to(self, **kwargs):
            self.to_calls.append(kwargs)
            return self

    modules = []

    class FakeOps:
        def Linear(self, *args, **kwargs):
            module = FakeLinear(*args, **kwargs)
            modules.append(module)
            return module

    def fake_ops_factory(**kwargs):
        assert kwargs == {"compute_dtype": torch.bfloat16}
        return FakeOps()

    loaded = {
        "state_dict": {
            "fc1.weight": torch.empty(8, 4),
            "fc2.weight": torch.empty(4, 4),
        }
    }
    bench.build_checkpoint_mlp(
        loaded,
        torch.bfloat16,
        ops_factory=fake_ops_factory,
    )
    assert len(modules) == 2
    assert all(not module.to_calls for module in modules)


def test_case_dispatch_and_native_rejection():
    fake_linear = SimpleNamespace(weight=torch.nn.Parameter(torch.ones(4, 4)))
    mlp = SimpleNamespace(fc1=fake_linear, fc2=fake_linear)
    fake_module = SimpleNamespace()

    class FakeHeld:
        def __init__(self, module, sample):
            self.module = module

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def fc1(self, x):
            return x

        def fc2_swiglu(self, x, native):
            return x, "held_bf16_swiglu"

    fake_module.HeldMLP = FakeHeld
    fake_module.module_fc1 = lambda module, x: x
    fake_module.module_swiglu_fc2 = lambda module, x, native: (x, "module_bf16_swiglu")
    with mock.patch.dict(sys.modules, {"h3_activation_memory.linear": fake_module}):
        out, path, _, _ = bench.run_actual_case(
            mlp, torch.ones(3, 4), 2, "bf16", "off", torch.device("cpu")
        )
        assert out.shape == (3, 4)
        assert path == "module_bf16_swiglu"
        out, path, _, _ = bench.run_actual_case(
            mlp, torch.ones(3, 4), 2, "bf16", "on", torch.device("cpu")
        )
        assert path == "held_bf16_swiglu"
        try:
            bench.run_actual_case(mlp, torch.ones(3, 4), 2, "native", "off", torch.device("cpu"))
        except RuntimeError as exc:
            assert "native" in str(exc)
        else:
            raise AssertionError("native mode must reject a silent fallback")


def test_report_serialization_and_synthetic_helpers():
    args = bench.build_parser().parse_args(["--seq", "8", "--hidden", "4", "--ffn", "8", "--chunks", "4", "--warmup", "0", "--iterations", "1"])
    payload = bench.run_synthetic(args, torch.device("cpu"), torch.float32)
    text = json.dumps(payload, indent=2, sort_keys=True)
    assert json.loads(text)["mode"] == "synthetic"
    assert bench.sampled_checksum(torch.ones(4, 4)) == 1.0


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_"):
            value()
    print("activation-memory benchmark tests passed")
