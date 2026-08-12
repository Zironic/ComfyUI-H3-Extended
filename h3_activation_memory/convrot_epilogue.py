"""Experimental ConvRot-INT8 H3 MLP GEMM epilogues.

The established two-slice path materializes a gate/up expansion for each feature
slice and a hidden-width fc2 output before applying the AdaLN residual gate.
This prototype keeps the same two-slice weight packing but changes the tensor
boundaries:

* fc1 writes the already-activated SwiGLU result, halving the feature-slice
  activation carrier;
* fc2 applies the per-channel AdaLN gate and accumulates directly into the
  residual tensor, so no hidden-width fc2 output tensor is allocated.

Only the already-validated BF16 ConvRot-256 TensorWise-INT8 path is implemented.
Other weight layouts continue through the format-neutral chunked MLP provider.
"""

from __future__ import annotations

import torch

from .linear import ConvRotTwoSliceMLP

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime dependency
    triton = None
    tl = None
    TRITON_AVAILABLE = False


class ConvRotEpilogueError(RuntimeError):
    pass


def convrot_epilogue_launch_policy(rows):
    """Return the fixed launch policy selected for an epilogue row count."""

    if int(rows) >= 64:
        block_m = 64
        num_warps = 8
    else:
        block_m = 32
        num_warps = 4
    return {
        "BLOCK_M": block_m,
        "BLOCK_N": 64,
        "BLOCK_K": 64,
        "GROUP_M": 8,
        "num_warps": num_warps,
        "num_stages": 3,
    }


def _validate_scale(scale, rows, name, device=None):
    if not torch.is_tensor(scale) or scale.dtype != torch.float32:
        raise ConvRotEpilogueError("%s scale must be a float32 tensor" % name)
    if device is not None and scale.device != device:
        raise ConvRotEpilogueError("%s scale device differs" % name)
    if scale.numel() not in (1, int(rows)):
        raise ConvRotEpilogueError(
            "%s scale must be scalar or per-output-channel" % name
        )
    return scale.reshape(-1).contiguous()


def _quantize_convrot_input(x):
    try:
        from comfy_kitchen.backends.cuda import (
            quantize_int8_rowwise_convrot64,
        )
    except ImportError as exc:  # pragma: no cover - compatibility failure
        raise ConvRotEpilogueError(
            "ConvRot MLP epilogues require Comfy Kitchen's CUDA ConvRot quantizer"
        ) from exc

    qdata, scale = quantize_int8_rowwise_convrot64(x, 256)
    scale = scale.reshape(-1).contiguous()
    if (
        tuple(qdata.shape) != tuple(x.shape)
        or qdata.dtype != torch.int8
        or qdata.device != x.device
        or not qdata.is_contiguous()
        or scale.numel() != x.shape[0]
        or scale.dtype != torch.float32
        or scale.device != x.device
    ):
        raise ConvRotEpilogueError(
            "Comfy Kitchen returned an invalid ConvRot activation carrier"
        )
    return qdata, scale


if TRITON_AVAILABLE:

    @triton.jit
    def _fc1_swiglu_epilogue_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SCALE_SCALAR: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N

        gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, K, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < K
            x_offsets = (
                rows[:, None].to(tl.int64) * stride_xm
                + inner[None, :].to(tl.int64) * stride_xk
            )
            gate_offsets = (
                cols[:, None].to(tl.int64) * stride_wn
                + inner[None, :].to(tl.int64) * stride_wk
            )
            up_offsets = (
                (N + cols[:, None]).to(tl.int64) * stride_wn
                + inner[None, :].to(tl.int64) * stride_wk
            )
            x = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            gate_weight = tl.load(
                weight_ptr + gate_offsets,
                mask=col_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            up_weight = tl.load(
                weight_ptr + up_offsets,
                mask=col_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            gate_acc += tl.dot(
                x, tl.trans(gate_weight), out_dtype=tl.int32
            )
            up_acc += tl.dot(
                x, tl.trans(up_weight), out_dtype=tl.int32
            )

        row_scale = tl.load(
            x_scale_ptr + rows, mask=row_mask, other=0.0
        ).to(tl.float32)
        gate = gate_acc.to(tl.float32)
        up = up_acc.to(tl.float32)
        if SCALE_SCALAR:
            shared_scale = tl.load(weight_scale_ptr).to(tl.float32)
            gate *= row_scale[:, None] * shared_scale
            up *= row_scale[:, None] * shared_scale
        else:
            gate_scale = tl.load(
                weight_scale_ptr + cols, mask=col_mask, other=0.0
            ).to(tl.float32)
            up_scale = tl.load(
                weight_scale_ptr + N + cols,
                mask=col_mask,
                other=0.0,
            ).to(tl.float32)
            gate *= row_scale[:, None] * gate_scale[None, :]
            up *= row_scale[:, None] * up_scale[None, :]

        # Match the established path's BF16 fc1 carrier before applying
        # SwiGLU. The activated output is also stored in BF16 for fc2.
        gate = gate.to(tl.bfloat16).to(tl.float32)
        up = up.to(tl.bfloat16).to(tl.float32)
        activated = gate * tl.sigmoid(gate) * up

        output_offsets = (
            rows[:, None].to(tl.int64) * stride_om
            + cols[None, :].to(tl.int64) * stride_on
        )
        tl.store(
            output_ptr + output_offsets,
            activated,
            mask=row_mask[:, None] & col_mask[None, :],
        )


    @triton.jit
    def _fc2_gated_residual_epilogue_kernel(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        residual_ptr,
        gate_ptr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_rm: tl.constexpr,
        stride_rn: tl.constexpr,
        stride_g: tl.constexpr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        SCALE_SCALAR: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        col_mask = cols < N

        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        for start in range(0, K, BLOCK_K):
            inner = start + tl.arange(0, BLOCK_K)
            inner_mask = inner < K
            x_offsets = (
                rows[:, None].to(tl.int64) * stride_xm
                + inner[None, :].to(tl.int64) * stride_xk
            )
            weight_offsets = (
                cols[:, None].to(tl.int64) * stride_wn
                + inner[None, :].to(tl.int64) * stride_wk
            )
            x = tl.load(
                x_ptr + x_offsets,
                mask=row_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            weight = tl.load(
                weight_ptr + weight_offsets,
                mask=col_mask[:, None] & inner_mask[None, :],
                other=0,
            )
            accumulator += tl.dot(
                x, tl.trans(weight), out_dtype=tl.int32
            )

        row_scale = tl.load(
            x_scale_ptr + rows, mask=row_mask, other=0.0
        ).to(tl.float32)
        value = accumulator.to(tl.float32)
        if SCALE_SCALAR:
            shared_scale = tl.load(weight_scale_ptr).to(tl.float32)
            value *= row_scale[:, None] * shared_scale
        else:
            column_scale = tl.load(
                weight_scale_ptr + cols, mask=col_mask, other=0.0
            ).to(tl.float32)
            value *= row_scale[:, None] * column_scale[None, :]
        # The existing ck.int8_linear path returns BF16 before addcmul_.
        value = value.to(tl.bfloat16).to(tl.float32)

        residual_offsets = (
            rows[:, None].to(tl.int64) * stride_rm
            + cols[None, :].to(tl.int64) * stride_rn
        )
        residual = tl.load(
            residual_ptr + residual_offsets,
            mask=row_mask[:, None] & col_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        gate = tl.load(
            gate_ptr + cols * stride_g,
            mask=col_mask,
            other=0.0,
        ).to(tl.bfloat16).to(tl.float32)
        updated = residual + value * gate[None, :]
        tl.store(
            residual_ptr + residual_offsets,
            updated,
            mask=row_mask[:, None] & col_mask[None, :],
        )


def convrot_fc1_swiglu_tensor_core(
    x_int8,
    weight,
    x_scale,
    weight_scale,
    *,
    output_dtype=torch.bfloat16,
):
    """Tensor-only fc1 GEMM with a SwiGLU output epilogue."""

    if not TRITON_AVAILABLE:
        raise ConvRotEpilogueError("ConvRot MLP epilogues require Triton")
    if output_dtype != torch.bfloat16:
        raise ConvRotEpilogueError(
            "the prototype currently supports BF16 output only"
        )
    if x_int8.ndim != 2 or weight.ndim != 2:
        raise ConvRotEpilogueError("fc1 epilogue expects rank-2 tensors")
    if x_int8.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ConvRotEpilogueError("fc1 epilogue expects INT8 carriers")
    if x_int8.device != weight.device:
        raise ConvRotEpilogueError("fc1 epilogue carrier devices differ")
    if weight.shape[0] % 2:
        raise ConvRotEpilogueError(
            "fc1 epilogue weight must contain equal gate and up rows"
        )
    m, k = (int(value) for value in x_int8.shape)
    n = int(weight.shape[0]) // 2
    if int(weight.shape[1]) != k:
        raise ConvRotEpilogueError("fc1 epilogue K dimensions differ")
    if k % 256 or n % 256:
        raise ConvRotEpilogueError(
            "fc1 epilogue dimensions must be ConvRot-256 aligned"
        )
    x_scale = x_scale.reshape(-1).contiguous()
    if (
        x_scale.numel() != m
        or x_scale.dtype != torch.float32
        or x_scale.device != x_int8.device
    ):
        raise ConvRotEpilogueError("fc1 epilogue row scale is invalid")
    weight_scale = _validate_scale(
        weight_scale,
        int(weight.shape[0]),
        "fc1",
        device=x_int8.device,
    )
    output = torch.empty(
        (m, n), dtype=output_dtype, device=x_int8.device
    )
    launch = convrot_epilogue_launch_policy(m)
    grid = (
        triton.cdiv(m, launch["BLOCK_M"])
        * triton.cdiv(n, launch["BLOCK_N"]),
    )
    _fc1_swiglu_epilogue_kernel[grid](
        x_int8,
        weight,
        x_scale,
        weight_scale,
        output,
        x_int8.stride(0),
        x_int8.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        M=m,
        N=n,
        K=k,
        SCALE_SCALAR=weight_scale.numel() == 1,
        **launch,
    )
    return output


def convrot_fc2_gated_residual_tensor_core_(
    x_int8,
    weight,
    x_scale,
    weight_scale,
    residual,
    gate,
):
    """Tensor-only fc2 GEMM with in-place gate and residual epilogue."""

    if not TRITON_AVAILABLE:
        raise ConvRotEpilogueError("ConvRot MLP epilogues require Triton")
    if x_int8.ndim != 2 or weight.ndim != 2 or residual.ndim != 2:
        raise ConvRotEpilogueError("fc2 epilogue expects rank-2 tensors")
    if x_int8.dtype != torch.int8 or weight.dtype != torch.int8:
        raise ConvRotEpilogueError("fc2 epilogue expects INT8 carriers")
    if residual.dtype != torch.bfloat16:
        raise ConvRotEpilogueError(
            "the prototype currently supports BF16 residuals only"
        )
    if not (x_int8.device == weight.device == residual.device == gate.device):
        raise ConvRotEpilogueError("fc2 epilogue carrier devices differ")
    m, k = (int(value) for value in x_int8.shape)
    n = int(weight.shape[0])
    if int(weight.shape[1]) != k or tuple(residual.shape) != (m, n):
        raise ConvRotEpilogueError("fc2 epilogue dimensions are incompatible")
    if k % 256 or n % 256:
        raise ConvRotEpilogueError(
            "fc2 epilogue dimensions must be ConvRot-256 aligned"
        )
    x_scale = x_scale.reshape(-1).contiguous()
    if (
        x_scale.numel() != m
        or x_scale.dtype != torch.float32
        or x_scale.device != x_int8.device
    ):
        raise ConvRotEpilogueError("fc2 epilogue row scale is invalid")
    weight_scale = _validate_scale(
        weight_scale, n, "fc2", device=x_int8.device
    )
    gate = gate.reshape(-1)
    if gate.numel() != n or not gate.is_floating_point():
        raise ConvRotEpilogueError("fc2 epilogue gate is invalid")

    launch = convrot_epilogue_launch_policy(m)
    grid = (
        triton.cdiv(m, launch["BLOCK_M"])
        * triton.cdiv(n, launch["BLOCK_N"]),
    )
    _fc2_gated_residual_epilogue_kernel[grid](
        x_int8,
        weight,
        x_scale,
        weight_scale,
        residual,
        gate,
        x_int8.stride(0),
        x_int8.stride(1),
        weight.stride(0),
        weight.stride(1),
        residual.stride(0),
        residual.stride(1),
        gate.stride(0),
        M=m,
        N=n,
        K=k,
        SCALE_SCALAR=weight_scale.numel() == 1,
        **launch,
    )
    return residual


def convrot_fc1_swiglu(x, weight, weight_scale):
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ConvRotEpilogueError(
            "fc1+SwiGLU epilogue requires rank-2 CUDA BF16 input"
        )
    x_int8, x_scale = _quantize_convrot_input(x)
    return convrot_fc1_swiglu_tensor_core(
        x_int8,
        weight,
        x_scale,
        weight_scale,
        output_dtype=x.dtype,
    )


def convrot_fc2_gated_residual_(
    x,
    weight,
    weight_scale,
    residual,
    gate,
):
    if not x.is_cuda or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ConvRotEpilogueError(
            "fc2+gated-residual epilogue requires rank-2 CUDA BF16 input"
        )
    x_int8, x_scale = _quantize_convrot_input(x)
    return convrot_fc2_gated_residual_tensor_core_(
        x_int8,
        weight,
        x_scale,
        weight_scale,
        residual,
        gate,
    )


class ConvRotEpilogueMLP(ConvRotTwoSliceMLP):
    """Two feature slices with fused fc1/SwiGLU and fc2/residual epilogues."""

    def __init__(
        self,
        mlp,
        sample,
        *,
        fc1_swiglu=None,
        fc1_quantize=None,
        fc1_swiglu_tensor=None,
        fc2_gated_residual=None,
    ):
        super().__init__(mlp, sample)
        self.shared_fc1_carrier = fc1_swiglu is None
        self.fc1_swiglu = fc1_swiglu or convrot_fc1_swiglu
        self.fc1_quantize = fc1_quantize or _quantize_convrot_input
        self.fc1_swiglu_tensor = (
            fc1_swiglu_tensor or convrot_fc1_swiglu_tensor_core
        )
        self.fc2_gated_residual = (
            fc2_gated_residual or convrot_fc2_gated_residual_
        )

    def fc1_swiglu_fc2_gated_(
        self,
        x,
        residual,
        gate,
        stage_factory=None,
    ):
        if self.tiles is None:
            raise RuntimeError("ConvRot epilogue session is not active")
        if torch.compiler.is_compiling():
            raise RuntimeError(
                "ConvRot MLP epilogues are an eager prototype and are not "
                "available inside shared H3 compilation"
            )
        if tuple(residual.shape[:-1]) != tuple(x.shape[:-1]):
            raise ConvRotEpilogueError(
                "MLP epilogue input and residual rows differ"
            )

        import comfy.ops

        comfy.ops.run_every_op()
        shared_carrier = None
        for tile in self.tiles:
            if self.shared_fc1_carrier:
                def fc1_call():
                    nonlocal shared_carrier
                    if shared_carrier is None:
                        shared_carrier = self.fc1_quantize(x)
                    x_int8, x_scale = shared_carrier
                    return self.fc1_swiglu_tensor(
                        x_int8,
                        tile["fc1_weight"],
                        x_scale,
                        tile["fc1_scale"],
                    )
            else:
                def fc1_call():
                    return self.fc1_swiglu(
                        x, tile["fc1_weight"], tile["fc1_scale"]
                    )
            if stage_factory is None:
                activated = fc1_call()
            else:
                with stage_factory("mlp_fc1"):
                    activated = fc1_call()
            try:
                if stage_factory is None:
                    self.fc2_gated_residual(
                        activated,
                        tile["fc2_weight"],
                        tile["fc2_scale"],
                        residual,
                        gate,
                    )
                else:
                    with stage_factory("mlp_swiglu_fc2"):
                        self.fc2_gated_residual(
                            activated,
                            tile["fc2_weight"],
                            tile["fc2_scale"],
                            residual,
                            gate,
                        )
            finally:
                del activated
        return "held_convrot_epilogue_prototype"
