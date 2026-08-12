"""CPU contracts for the experimental ConvRot H3 MLP epilogue path."""

import os
import sys
from contextlib import nullcontext
from unittest import mock

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_activation_memory.config import (  # noqa: E402
    ActivationMemoryConfig,
    MODE_CONVROT_EPILOGUE,
)
from h3_activation_memory.convrot_epilogue import (  # noqa: E402
    ConvRotEpilogueMLP,
)
from h3_sage_optimizations.plan import (  # noqa: E402
    MLP_MEMORY_EPILOGUE,
)
from h3_sage_optimizations.qkv.providers import (  # noqa: E402
    MLP_CONVROT_INT8_EPILOGUE,
    resolve_mlp_provider,
)


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
        Inventory(), request=MLP_MEMORY_EPILOGUE
    )
    check(
        resolution.provider_id == MLP_CONVROT_INT8_EPILOGUE,
        "explicit prototype request selects the epilogue provider",
    )
    check(
        resolution.activation_mode == MODE_CONVROT_EPILOGUE,
        "provider selects the dedicated activation-memory mode",
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
            request=MLP_MEMORY_EPILOGUE,
        )
    except RuntimeError as exc:
        check(
            "homogeneous ConvRot-256" in str(exc),
            "explicit prototype request fails during format preflight",
        )
    else:
        raise AssertionError("epilogue prototype accepted incompatible weights")


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
            stage_factory=lambda _name: nullcontext(),
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


def main():
    print("H3 ConvRot MLP epilogue prototype")
    test_provider_and_config()
    test_two_slice_epilogue_execution_contract()
    print("\nall H3 MLP epilogue CPU contracts passed")


if __name__ == "__main__":
    main()
