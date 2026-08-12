"""Opt-in CUDA parity checks for the H3 ConvRot MLP epilogue kernels."""

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_activation_memory.convrot_epilogue import (  # noqa: E402
    TRITON_AVAILABLE,
    convrot_fc1_swiglu_tensor_core,
    convrot_fc2_gated_residual_tensor_core_,
)


def relative_rmse(actual, expected):
    error = (actual.float() - expected.float()).square().mean().sqrt()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-8)
    return float((error / scale).item())


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def fc1_reference(x, weight, x_scale, weight_scale):
    n = weight.shape[0] // 2
    gate_acc = x.float() @ weight[:n].float().transpose(0, 1)
    up_acc = x.float() @ weight[n:].float().transpose(0, 1)
    if weight_scale.numel() == 1:
        gate_scale = up_scale = weight_scale.reshape(1, 1)
    else:
        gate_scale = weight_scale[:n].reshape(1, n)
        up_scale = weight_scale[n:].reshape(1, n)
    gate = (gate_acc * x_scale[:, None] * gate_scale).to(torch.bfloat16)
    up = (up_acc * x_scale[:, None] * up_scale).to(torch.bfloat16)
    return (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)


def fc2_reference_(x, weight, x_scale, weight_scale, residual, gate):
    value = x.float() @ weight.float().transpose(0, 1)
    if weight_scale.numel() == 1:
        scale = weight_scale.reshape(1, 1)
    else:
        scale = weight_scale.reshape(1, -1)
    value = (value * x_scale[:, None] * scale).to(torch.bfloat16)
    residual.addcmul_(value, gate.to(residual.dtype))
    return residual


def main():
    if os.environ.get("H3_RUN_MLP_EPILOGUE_CUDA_TESTS") != "1":
        print(
            "CUDA MLP epilogue parity: SKIP "
            "(set H3_RUN_MLP_EPILOGUE_CUDA_TESTS=1 after authorization)"
        )
        return
    if not torch.cuda.is_available() or not TRITON_AVAILABLE:
        raise RuntimeError("CUDA and Triton are required")

    torch.manual_seed(17)
    device = torch.device("cuda")
    m, k, n = 17, 256, 256
    x = torch.randint(-127, 128, (m, k), dtype=torch.int8, device=device)
    fc1_weight = torch.randint(
        -127, 128, (n * 2, k), dtype=torch.int8, device=device
    )
    row_scale = torch.rand((m,), dtype=torch.float32, device=device) * 0.02
    fc1_scale = torch.rand(
        (n * 2,), dtype=torch.float32, device=device
    ) * 0.02

    expected = fc1_reference(x, fc1_weight, row_scale, fc1_scale)
    actual = convrot_fc1_swiglu_tensor_core(
        x,
        fc1_weight,
        row_scale,
        fc1_scale,
    )
    torch.cuda.synchronize(device)
    check(
        relative_rmse(actual, expected) < 0.02,
        "fc1+SwiGLU epilogue matches the dequantized BF16 reference",
    )

    hidden = 256
    fc2_x = torch.randint(
        -127, 128, (m, n), dtype=torch.int8, device=device
    )
    fc2_weight = torch.randint(
        -127, 128, (hidden, n), dtype=torch.int8, device=device
    )
    fc2_row_scale = torch.rand(
        (m,), dtype=torch.float32, device=device
    ) * 0.02
    fc2_scale = torch.rand(
        (hidden,), dtype=torch.float32, device=device
    ) * 0.02
    gate = torch.randn((hidden,), dtype=torch.bfloat16, device=device)
    base = torch.randn((m, hidden), dtype=torch.bfloat16, device=device)
    expected_residual = fc2_reference_(
        fc2_x,
        fc2_weight,
        fc2_row_scale,
        fc2_scale,
        base.clone(),
        gate,
    )
    actual_residual = base.clone()
    returned = convrot_fc2_gated_residual_tensor_core_(
        fc2_x,
        fc2_weight,
        fc2_row_scale,
        fc2_scale,
        actual_residual,
        gate,
    )
    torch.cuda.synchronize(device)
    check(
        returned.data_ptr() == actual_residual.data_ptr(),
        "fc2 epilogue returns the mutated residual without an output carrier",
    )
    check(
        relative_rmse(actual_residual, expected_residual) < 0.02,
        "fc2+gated-residual epilogue matches the BF16 reference",
    )
    print("\nall H3 MLP epilogue CUDA checks passed")


if __name__ == "__main__":
    main()
