"""CPU-only contracts for the MiniMax H3 VRAM arm-matrix probe."""

import os
import sys
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
_HERE = os.path.dirname(os.path.abspath(__file__))
_PACK = os.path.dirname(_HERE)
_COMFY_ROOT = os.path.abspath(os.path.join(_PACK, "..", ".."))
sys.path.insert(0, _PACK)
sys.path.insert(0, _COMFY_ROOT)
sys.argv = [sys.argv[0], "--cpu"]

import torch

import _minimax_vram_probe_ab_cli as cli  # noqa: E402
import _minimax_vram_probe_ab_runtime as runtime  # noqa: E402
import _minimax_vram_probe_ab_sweep as sweep  # noqa: E402


def test_arm_selector_cartesian_product_and_validation():
    args = SimpleNamespace(
        ab_qkv="sage128,sage128_fused_qkv",
        ab_mlp="untiled,convrot2,convrot4",
    )
    assert cli.selected_arms(args) == (
        "sage128/untiled",
        "sage128/convrot2",
        "sage128/convrot4",
        "sage128_fused_qkv/untiled",
        "sage128_fused_qkv/convrot2",
        "sage128_fused_qkv/convrot4",
    )
    for value, choices, name in (
        ("sage128,sage128", cli.QKV_MODES, "ab-qkv"),
        ("convrot3", cli.MLP_MODES, "ab-mlp"),
        ("", cli.MLP_MODES, "ab-mlp"),
    ):
        try:
            cli.parse_selector(value, choices, name)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid selector was accepted: %r" % value)
    profiles, _ = cli.parse_profiles("90,73", 100)
    assert profiles == sorted(profiles)


def test_qkv_modes_control_projector_presence():
    class Config:
        def __init__(self, **kwargs):
            self.mode = kwargs["mode"]

    class Backend:
        def __init__(self, config):
            self.config = config
            self.projector = (
                object() if config.mode == "sage128_fused_qkv" else None
            )

    established = runtime.build_attention_backend(
        "sage128", backend_cls=Backend, config_cls=Config
    )
    fused = runtime.build_attention_backend(
        "sage128_fused_qkv", backend_cls=Backend, config_cls=Config
    )
    assert established.config.mode == "sage128"
    assert established.projector is None
    assert fused.config.mode == "sage128_fused_qkv"
    assert fused.projector is not None


def test_four_tile_convrot_executes_four_feature_tiles():
    hidden = 256
    ffn = 1024
    fc1_weight = torch.empty((2 * ffn, hidden), dtype=torch.int8)
    fc2_weight = torch.empty((hidden, ffn), dtype=torch.int8)
    fc1_scale = torch.empty((2 * ffn, 1), dtype=torch.float32)
    fc2_scale = torch.empty((hidden, 1), dtype=torch.float32)

    class Acquired:
        def __init__(self, weight):
            self.weight = weight
            self.bias = None
            self.released = False

        def release(self):
            self.released = True

    mlp = SimpleNamespace(fc1="fc1", fc2="fc2")
    acquired = []

    def acquire(module, sample):
        value = Acquired(
            (fc1_weight, fc1_scale) if module == "fc1" else (fc2_weight, fc2_scale)
        )
        acquired.append(value)
        return value

    def parts(weight, name):
        return weight

    calls = []

    def linear(x, weight, scale, input_act=None):
        calls.append((tuple(weight.shape), input_act))
        return torch.zeros(
            (x.shape[0], weight.shape[0]), dtype=torch.bfloat16
        )

    sample = torch.empty((1, hidden), dtype=torch.bfloat16)
    with runtime.ProbeConvRotTiledMLP(
        mlp,
        sample,
        4,
        acquire_fn=acquire,
        parts_fn=parts,
        linear_fn=linear,
    ) as tiled:
        assert len(tiled.tiles) == 4
        assert all(tile["fc1_weight"].shape == (512, hidden) for tile in tiled.tiles)
        assert all(tile["fc2_weight"].shape == (hidden, 256) for tile in tiled.tiles)
        output = tiled.forward(torch.empty((3, hidden), dtype=torch.bfloat16), 2)
    assert output.shape == (3, hidden)
    assert len([call for call in calls if call[1] is None]) == 8
    assert len([call for call in calls if call[1] == "swiglu"]) == 8
    assert all(value.released for value in acquired)
    assert tiled.tiles is None


def _measurement(ms, physical_free):
    return runtime.ForwardMeasurement(
        peak_bytes=1024,
        median_ms=float(ms),
        physical_free_start=physical_free,
        physical_free_inputs=physical_free,
        physical_free_min=physical_free,
        physical_free_end=physical_free,
        physical_free_recovered=physical_free,
        physical_samples=2,
    )


def test_sweep_retires_each_arm_independently():
    args = SimpleNamespace(
        ab_qkv="sage128",
        ab_mlp="untiled,convrot2,convrot4",
        physical_warning_mb=1,
        width=32,
        height=32,
        ab_warmup=0,
        ab_iterations=1,
        seed=1,
        physical_poll_ms=1.0,
        calibrate_to=None,
        calibrate_arm=None,
        ab_frames="grid",
        max_frames=39,
        budget=11.0,
        spill_ratio=1.35,
        past_spill=False,
    )
    labels = cli.selected_arms(args)
    forwards = {label: label for label in labels}
    calls = []

    def fake_profiles(spec, max_frames):
        return [5, 22, 39], []

    def fake_layout(args, frames):
        return {
            "frames": frames,
            "seq_len": frames,
            "video_rows_total": 1,
            "cond_video_rows": 0,
        }

    def fake_measure(forward, layout, *unused, **kwargs):
        calls.append((forward, layout["frames"]))
        if forward.endswith("untiled"):
            return _measurement(10, 0), None
        if forward.endswith("convrot2"):
            return None, "synthetic OOM"
        return _measurement(10 if layout["frames"] == 5 else 100, 2 << 20), None

    def fake_fit(points, sequence, ms, ratio, include_in_fit=True):
        if ms >= 100:
            return True, 10.0
        if include_in_fit:
            points.append((sequence, ms))
        return False, None

    original = (
        sweep.parse_profiles,
        sweep.layout_for,
        sweep.safe_measure,
        sweep.update_resident_fit,
    )
    sweep.parse_profiles = fake_profiles
    sweep.layout_for = fake_layout
    sweep.safe_measure = fake_measure
    sweep.update_resident_fit = fake_fit
    try:
        sweep.run(
            args,
            SimpleNamespace(error=lambda message: (_ for _ in ()).throw(ValueError(message))),
            {"video_patch_dim": 1},
            torch.bfloat16,
            torch.device("cpu"),
            0.0,
            False,
            forwards,
        )
    finally:
        (
            sweep.parse_profiles,
            sweep.layout_for,
            sweep.safe_measure,
            sweep.update_resident_fit,
        ) = original

    assert calls == [
        ("sage128/untiled", 5),
        ("sage128/convrot2", 5),
        ("sage128/convrot4", 5),
        ("sage128/convrot4", 22),
    ]


if __name__ == "__main__":
    test_arm_selector_cartesian_product_and_validation()
    test_qkv_modes_control_projector_presence()
    test_four_tile_convrot_executes_four_feature_tiles()
    test_sweep_retires_each_arm_independently()
    print("MiniMax VRAM arm-matrix CPU tests passed")
