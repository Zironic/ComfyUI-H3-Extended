"""CPU contracts for the experimental ConvRot H3 MLP epilogue path."""

import os
import sys
from contextlib import nullcontext
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

if "--cpu" not in sys.argv:
    sys.argv.append("--cpu")
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()

from h3_activation_memory.config import (  # noqa: E402
    ActivationMemoryConfig,
    MODE_CONVROT_EPILOGUE,
)
from h3_activation_memory.convrot_epilogue import (  # noqa: E402
    ConvRotEpilogueMLP,
    convrot_epilogue_launch_policy,
    convrot_fc1_swiglu,
)
from h3_optimizations_dependency import dependency_module  # noqa: E402
from h3_sage_optimizations.plan import (  # noqa: E402
    MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
)

_providers = dependency_module("qkv.providers")
MLP_CONVROT_INT8_TWO_SLICE = _providers.MLP_CONVROT_INT8_TWO_SLICE
resolve_mlp_provider = _providers.resolve_mlp_provider


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class Inventory:
    fc1 = (object(),)
    fc2 = (object(),)
    mlp_convrot_int8_256 = True

    def homogeneous(self, name):
        return name in ("fc1", "fc2")

    def labels(self, name):
        return ("TensorWiseINT8Layout+convrot256",)


class IncompatibleInventory(Inventory):
    mlp_convrot_int8_256 = False


def test_provider_and_config():
    resolution = resolve_mlp_provider(
        Inventory(), request=MLP_MEMORY_LEGACY_CONVROT_REQUIRED
    )
    check(
        resolution.provider_id == MLP_CONVROT_INT8_TWO_SLICE,
        "saved prototype requests migrate to the production two-slice provider",
    )
    check(
        resolution.activation_mode == "mlp_chunked_convrot_2slice",
        "dependency execution no longer selects the retired epilogue prototype",
    )
    config = ActivationMemoryConfig(
        mode=MODE_CONVROT_EPILOGUE,
        strict=False,
    )
    check(
        config.convrot_epilogue and not config.convrot_2slice,
        "epilogue mode is distinct from the established two-slice mode",
    )

    try:
        resolve_mlp_provider(
            IncompatibleInventory(),
            request=MLP_MEMORY_LEGACY_CONVROT_REQUIRED,
        )
    except RuntimeError as exc:
        check(
            "required ConvRot two-slice" in str(exc),
            "explicit prototype request fails during format preflight",
        )
    else:
        raise AssertionError("epilogue prototype accepted incompatible weights")


def test_epilogue_launch_policy():
    check(
        convrot_epilogue_launch_policy(64)
        == {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        },
        "large-row epilogue launch policy is fixed",
    )
    check(
        convrot_epilogue_launch_policy(63)
        == {
            "BLOCK_M": 32,
            "BLOCK_N": 64,
            "BLOCK_K": 64,
            "GROUP_M": 8,
            "num_warps": 4,
            "num_stages": 3,
        },
        "short-row epilogue launch policy is fixed",
    )


def test_two_slice_epilogue_execution_contract():
    h = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )
    residual = torch.tensor(
        [[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32
    )
    gate = torch.tensor([0.5, 2.0], dtype=torch.float32)
    initial = residual.clone()

    tiles = (
        {
            "fc1_weight": torch.tensor(1.0),
            "fc1_scale": torch.tensor(1.0),
            "fc2_weight": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
            "fc2_scale": torch.tensor(1.0),
        },
        {
            "fc1_weight": torch.tensor(2.0),
            "fc1_scale": torch.tensor(1.0),
            "fc2_weight": torch.tensor(
                [[0.5, 0.0], [0.0, 0.25]]
            ),
            "fc2_scale": torch.tensor(1.0),
        },
    )
    calls = []
    stages = []

    def fake_fc1(value, weight, scale):
        calls.append(("fc1", float(weight)))
        return value * float(weight)

    def fake_fc2(value, weight, scale, destination, current_gate):
        calls.append(("fc2", float(scale)))
        partial = value @ weight.transpose(0, 1)
        destination.add_(partial * current_gate)
        return destination

    session = object.__new__(ConvRotEpilogueMLP)
    session.tiles = tiles
    session.shared_fc1_carrier = False
    session.fc1_swiglu = fake_fc1
    session.fc2_gated_residual = fake_fc2

    expected = initial.clone()
    for tile in tiles:
        activated = h * float(tile["fc1_weight"])
        partial = activated @ tile["fc2_weight"].transpose(0, 1)
        expected.add_(partial * gate)

    with mock.patch("comfy.ops.run_every_op"):
        path = session.fc1_swiglu_fc2_gated_(
            h,
            residual,
            gate,
            stage_factory=lambda name: (stages.append(name), nullcontext())[1],
        )

    check(
        path == "held_convrot_epilogue_prototype",
        "execution reports the prototype path",
    )
    check(
        calls == [
            ("fc1", 1.0),
            ("fc2", 1.0),
            ("fc1", 2.0),
            ("fc2", 1.0),
        ],
        "each activated feature slice is consumed before the next slice",
    )
    check(
        torch.equal(residual, expected),
        "fc2 accumulates gated partials directly into the residual",
    )
    check(
        stages == [
            "mlp_fc1",
            "mlp_swiglu_fc2",
            "mlp_fc1",
            "mlp_swiglu_fc2",
        ],
        "epilogue stage names remain stable",
    )


def test_default_shared_fc1_carrier_execution():
    h = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
    )
    residual = torch.zeros((2, 2), dtype=torch.bfloat16)
    gate = torch.tensor([0.5, 2.0], dtype=torch.bfloat16)
    tiles = (
        {
            "fc1_weight": torch.tensor(1.0),
            "fc1_scale": torch.tensor(1.0),
            "fc2_weight": torch.eye(2, dtype=torch.bfloat16),
            "fc2_scale": torch.tensor(1.0),
        },
        {
            "fc1_weight": torch.tensor(2.0),
            "fc1_scale": torch.tensor(1.0),
            "fc2_weight": torch.eye(2, dtype=torch.bfloat16),
            "fc2_scale": torch.tensor(1.0),
        },
    )
    calls = []
    stages = []

    def fake_quantize(value):
        calls.append("quantize")
        return value.to(torch.int8), torch.ones(value.shape[0])

    def fake_fc1_tensor(value, weight, scale, weight_scale):
        calls.append(("fc1_tensor", float(weight)))
        return (value.float() * float(weight)).to(torch.bfloat16)

    def fake_fc2(value, weight, scale, destination, current_gate):
        calls.append("fc2")
        destination.add_((value @ weight.transpose(0, 1)) * current_gate)
        return destination

    session = object.__new__(ConvRotEpilogueMLP)
    session.tiles = tiles
    session.shared_fc1_carrier = True
    session.fc1_swiglu = convrot_fc1_swiglu
    session.fc1_quantize = fake_quantize
    session.fc1_swiglu_tensor = fake_fc1_tensor
    session.fc2_gated_residual = fake_fc2

    with mock.patch("comfy.ops.run_every_op"):
        path = session.fc1_swiglu_fc2_gated_(
            h,
            residual,
            gate,
            stage_factory=lambda name: (
                stages.append(name),
                nullcontext(),
            )[1],
        )

    expected = torch.tensor(
        [[1.5, 12.0], [4.5, 24.0]], dtype=torch.bfloat16
    )
    check(
        calls == ["quantize", ("fc1_tensor", 1.0), "fc2", ("fc1_tensor", 2.0), "fc2"],
        "default path quantizes FC1 once and executes tensor-only FC1 per tile",
    )
    check(torch.equal(residual, expected), "shared-carrier residual math is correct")
    check(path == "held_convrot_epilogue_prototype", "optimized path label is stable")
    check(
        stages == [
            "mlp_fc1",
            "mlp_swiglu_fc2",
            "mlp_fc1",
            "mlp_swiglu_fc2",
        ],
        "optimized path keeps the existing stage order",
    )


def main():
    print("H3 ConvRot MLP epilogue prototype")
    test_provider_and_config()
    test_epilogue_launch_policy()
    test_two_slice_epilogue_execution_contract()
    test_default_shared_fc1_carrier_execution()
    print("\nall H3 MLP epilogue CPU contracts passed")


if __name__ == "__main__":
    main()
