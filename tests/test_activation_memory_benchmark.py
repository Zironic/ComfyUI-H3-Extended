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


def test_tiled_cases_are_prepacked_once_per_chunk():
    cases = bench.iter_cases((128,), ("bf16", "tiled_convrot"), ("off", "on"))
    assert cases == (
        {"chunk_rows": 128, "swiglu_mode": "bf16", "held_mode": "off"},
        {"chunk_rows": 128, "swiglu_mode": "bf16", "held_mode": "on"},
        {"chunk_rows": 128, "swiglu_mode": "tiled_convrot", "held_mode": "prepacked"},
    )


def test_epilogue_cases_are_prepacked_once_per_chunk_and_parser_accepts_mode():
    args = bench.build_parser().parse_args(["--swiglu-modes", "convrot_epilogue"])
    assert args.swiglu_modes == "convrot_epilogue"
    cases = bench.iter_cases((128, 256), ("bf16", "convrot_epilogue"), ("off", "on"))
    epilogue_cases = tuple(case for case in cases if case["swiglu_mode"] == "convrot_epilogue")
    assert epilogue_cases == (
        {"chunk_rows": 128, "swiglu_mode": "convrot_epilogue", "held_mode": "prepacked"},
        {"chunk_rows": 256, "swiglu_mode": "convrot_epilogue", "held_mode": "prepacked"},
    )


def test_tiled_mode_rejects_non_bf16_before_building_modules():
    args = bench.build_parser().parse_args(["--swiglu-modes", "tiled_convrot"])
    with mock.patch.object(bench, "build_checkpoint_mlp") as build:
        try:
            bench.run_actual({}, args, torch.device("cpu"), torch.float16)
        except ValueError as exc:
            assert "--dtype bf16" in str(exc)
        else:
            raise AssertionError("tiled ConvRot must reject non-BF16 compute")
        build.assert_not_called()


def test_epilogue_mode_rejects_non_bf16_before_building_modules():
    args = bench.build_parser().parse_args(["--swiglu-modes", "convrot_epilogue"])
    with mock.patch.object(bench, "build_checkpoint_mlp") as build:
        try:
            bench.run_actual({}, args, torch.device("cpu"), torch.float16)
        except ValueError as exc:
            assert "convrot_epilogue" in str(exc)
            assert "--dtype bf16" in str(exc)
        else:
            raise AssertionError("ConvRot epilogue must reject non-BF16 compute")
        build.assert_not_called()


def test_epilogue_mode_requires_checkpoint():
    try:
        bench.main(["--swiglu-modes", "convrot_epilogue"])
    except ValueError as exc:
        assert "--checkpoint" in str(exc)
    else:
        raise AssertionError("ConvRot epilogue must reject synthetic mode")


def test_epilogue_dispatch_uses_residual_clone_and_stage_labels():
    activation = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    residual = torch.ones_like(activation)
    gate = torch.full((4,), 2, dtype=torch.bfloat16)
    original_residual = residual.clone()
    calls = []
    stage_names = []

    class FakeSession:
        def fc1_swiglu_fc2_gated_(self, x, destination, current_gate, stage_factory):
            calls.append((tuple(x.shape), destination.data_ptr()))
            stage_names.extend(("mlp_fc1", "mlp_swiglu_fc2"))
            with stage_factory("mlp_fc1"):
                activated = x + 1
            with stage_factory("mlp_swiglu_fc2"):
                destination.add_(activated * current_gate)
            return "held_convrot_epilogue_prototype"

    output, path, fc1_ms, fc2_ms = bench.run_convrot_epilogue_case(
        FakeSession(), activation, residual, gate, 2, torch.device("cpu")
    )
    expected = original_residual + (activation + 1) * gate
    assert path == "held_convrot_epilogue_prototype"
    assert torch.equal(output, expected)
    assert torch.equal(residual, original_residual)
    assert len(calls) == 3
    assert all(destination_ptr != residual.data_ptr() for _, destination_ptr in calls)
    assert stage_names == ["mlp_fc1", "mlp_swiglu_fc2"] * 3
    assert fc1_ms >= 0 and fc2_ms >= 0


def test_feature_tile_packing_shape_and_bytes():
    fc1 = {
        "weight": torch.empty(1024, 256, dtype=torch.int8),
        "weight_scale": torch.empty(1024),
        "group_size": 256,
    }
    fc2 = {
        "weight": torch.empty(256, 512, dtype=torch.int8),
        "weight_scale": torch.empty(256),
        "group_size": 256,
    }
    tiles = bench.prepare_convrot_tiles(fc1, fc2, 256)
    assert bench.feature_ranges(512, 256) == ((0, 256), (256, 512))
    assert [tuple(tile["fc1_weight"].shape) for tile in tiles] == [(512, 256), (512, 256)]
    assert [tuple(tile["fc2_weight"].shape) for tile in tiles] == [(256, 256), (256, 256)]
    expected = sum(
        tile[name].numel() * tile[name].element_size()
        for tile in tiles
        for name in ("fc1_weight", "fc1_scale", "fc2_weight")
    )
    assert bench.prepared_tile_bytes(tiles) == expected
    try:
        bench.feature_ranges(512, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("unaligned feature tile width must be rejected")


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


def test_tiled_dispatch_with_cpu_fake_convrot():
    fc1 = {
        "weight": torch.arange(1024 * 256, dtype=torch.int8).reshape(1024, 256),
        "weight_scale": torch.ones(1024),
        "group_size": 256,
    }
    fc2 = {
        "weight": torch.arange(256 * 512, dtype=torch.int8).reshape(256, 512),
        "weight_scale": torch.ones(256),
        "group_size": 256,
    }
    tiles = bench.prepare_convrot_tiles(fc1, fc2, 256)

    def fake_convrot(_ck, x, weight, _scale, _groups, input_act=None):
        if input_act is None:
            return torch.nn.functional.linear(x.float(), weight.float()).to(torch.bfloat16)
        gate, up = x.chunk(2, dim=-1)
        return torch.nn.functional.linear(torch.nn.functional.silu(gate.float()) * up.float(), weight.float()).to(torch.bfloat16)

    activation = torch.randn(3, 256, dtype=torch.bfloat16)
    output, path, _, _ = bench.run_tiled_convrot_case(
        object(), activation, 2, fc1, fc2, tiles, torch.device("cpu"), convrot_fn=fake_convrot
    )
    expected_expanded = fake_convrot(None, activation, fc1["weight"], fc1["weight_scale"], 256)
    expected = fake_convrot(None, expected_expanded, fc2["weight"], fc2["weight_scale"], 256, input_act="swiglu")
    assert output.shape == (3, 256)
    assert path == "tiled_convrot"
    assert torch.allclose(output, expected, atol=1e-2, rtol=1e-2)


def test_tiled_metadata_rejects_transposed_and_wrong_format():
    def metadata(**overrides):
        value = {"format": "int8_tensorwise", "convrot": True, "transposed": False, "convrot_groupsize": 256}
        value.update(overrides)
        encoded = json.dumps(value).encode()
        return torch.tensor(list(encoded), dtype=torch.uint8)

    state = {
        "fc1.weight": torch.empty(1024, 256, dtype=torch.int8),
        "fc1.weight_scale": torch.empty(1024),
        "fc1.comfy_quant": metadata(),
        "fc2.weight": torch.empty(256, 512, dtype=torch.int8),
        "fc2.weight_scale": torch.empty(256),
        "fc2.comfy_quant": metadata(),
    }
    loaded = {"state_dict": state}
    assert bench.load_convrot_mlp(loaded)["ffn_width"] == 512
    for overrides in ({"transposed": True}, {"format": "nvfp4"}):
        bad = dict(state)
        bad["fc1.comfy_quant"] = metadata(**overrides)
        try:
            bench.load_convrot_mlp({"state_dict": bad})
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported ConvRot metadata must fail closed")


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
