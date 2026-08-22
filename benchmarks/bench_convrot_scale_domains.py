"""Numerically evaluate block-scaled ConvRot activation carriers for H3 FC2.

This is a benchmark-only architecture probe. It uses one real Kitchen FC1
BF16 expansion, applies Kitchen's fused SwiGLU + ConvRot quantizer independently
to each candidate scale domain, and accumulates the resulting FC2 partials in
FP32 before one BF16 gate/residual boundary. It does not time the prototype and
does not install a shipping provider.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


try:
    from benchmarks import bench_convrot_mlp_c as base
except ImportError:  # Direct script execution puts benchmarks/ on sys.path.
    import bench_convrot_mlp_c as base


HIDDEN = 5376
FFN = 14336
EXPANDED = FFN * 2
CONVROT_GROUP = 256
PRODUCTION_DOMAIN = FFN // 2
DEFAULT_SCALE_DOMAINS = (256, 512, 1024, 2048, 3584, 7168, 14336)


def parse_scale_domains(value):
    try:
        domains = tuple(
            int(item.strip()) for item in str(value).split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "scale domains must be comma-separated integers"
        ) from exc
    if not domains:
        raise argparse.ArgumentTypeError("at least one scale domain is required")
    return domains


def validate_scale_domains(ffn, domains, convrot_group=CONVROT_GROUP):
    ffn = int(ffn)
    domains = tuple(int(value) for value in domains)
    invalid = [
        value
        for value in domains
        if value <= 0 or value % convrot_group or ffn % value
    ]
    if invalid:
        raise ValueError(
            "scale domains must be positive ConvRot-group multiples dividing "
            "the FFN width: %s" % invalid
        )
    if len(set(domains)) != len(domains):
        raise ValueError("scale domains must be unique")
    return domains


def domain_ranges(ffn, domain):
    ffn = int(ffn)
    domain = int(domain)
    if domain <= 0 or ffn % domain:
        raise ValueError("scale domain must divide the FFN width")
    return tuple((start, start + domain) for start in range(0, ffn, domain))


def tensor_error(reference, actual):
    delta = actual.float() - reference.float()
    reference_rms = reference.float().square().mean().sqrt()
    return {
        "exact": bool(torch.equal(reference, actual)),
        "max_abs": float(delta.abs().max().item()),
        "relative_l2": float(
            delta.square().mean().sqrt().div(reference_rms.clamp_min(1e-12)).item()
        ),
    }


def gated_residual(output, residual, gate):
    result = residual.clone()
    result.addcmul_(output, gate)
    return result


def scale_domain_contract(rows, domain, ffn=FFN, hidden=HIDDEN):
    rows = int(rows)
    domain = int(domain)
    groups = int(ffn) // domain
    carrier_bytes = rows * int(ffn)
    scale_bytes = rows * groups * torch.empty((), dtype=torch.float32).element_size()
    return {
        "domain_width": domain,
        "scale_domains_per_row": groups,
        "fc2_partial_gemms_in_prototype": groups,
        "intended_production_fc2_launches": 1,
        "int8_carrier_bytes": carrier_bytes,
        "fp32_scale_bytes": scale_bytes,
        "candidate_boundary_bytes": carrier_bytes + scale_bytes,
        "fc2_shape": {"m": rows, "n": int(hidden), "k": int(ffn)},
    }


def prepare_production_tiles(fc1, fc2):
    ffn = int(fc2["weight"].shape[1])
    tiles = []
    for start, stop in domain_ranges(ffn, ffn // 2):
        if fc1["weight_scale"].numel() == 1:
            fc1_scale = fc1["weight_scale"].contiguous().clone()
        else:
            fc1_scale = torch.cat(
                (
                    fc1["weight_scale"][start:stop],
                    fc1["weight_scale"][ffn + start : ffn + stop],
                )
            ).contiguous()
        tiles.append(
            {
                "start": start,
                "stop": stop,
                "fc1_weight": torch.cat(
                    (
                        fc1["weight"][start:stop],
                        fc1["weight"][ffn + start : ffn + stop],
                    )
                ).contiguous(),
                "fc1_scale": fc1_scale,
                "fc2_weight": fc2["weight"][:, start:stop].contiguous(),
            }
        )
    return tuple(tiles)


def kitchen_cutlass_gemm(x, weight, x_scale, weight_scale, output_dtype):
    from comfy_kitchen.backends import cuda as cuda_backend

    output = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=output_dtype, device=x.device
    )
    weight_scale = weight_scale.reshape(-1)
    if weight_scale.numel() == 1:
        weight_scale = weight_scale.expand(weight.shape[0]).contiguous()
    empty_bias = torch.empty(0, dtype=torch.float32, device=x.device)
    used = cuda_backend._C.cutlass_int8_dequant(
        cuda_backend._wrap_for_dlpack(x),
        cuda_backend._wrap_for_dlpack(weight),
        cuda_backend._wrap_for_dlpack(x_scale.reshape(-1)),
        cuda_backend._wrap_for_dlpack(weight_scale),
        cuda_backend._wrap_for_dlpack(empty_bias),
        cuda_backend._wrap_for_dlpack(output),
        cuda_backend.DTYPE_TO_CODE[output.dtype],
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    if not used:
        raise RuntimeError(
            "Kitchen CUTLASS declined the scale-domain FC2 partial "
            "M=%d N=%d K=%d"
            % (x.shape[0], weight.shape[0], weight.shape[1])
        )
    return output


def quantize_domain(expanded, start, stop, ffn):
    from comfy_kitchen.backends.cuda import quantize_int8_rowwise_convrot64

    paired = torch.cat(
        (expanded[:, start:stop], expanded[:, ffn + start : ffn + stop]),
        dim=1,
    ).contiguous()
    qdata, scale = quantize_int8_rowwise_convrot64(
        paired, CONVROT_GROUP, input_act="swiglu"
    )
    expected_shape = (expanded.shape[0], stop - start)
    if (
        tuple(qdata.shape) != expected_shape
        or qdata.dtype != torch.int8
        or tuple(scale.shape) != (expanded.shape[0], 1)
        or scale.dtype != torch.float32
    ):
        raise RuntimeError("Kitchen returned an invalid scale-domain carrier")
    return qdata, scale


def scale_domain_fc2(expanded, fc2, domain):
    rows = int(expanded.shape[0])
    ffn = int(fc2["weight"].shape[1])
    fp32_output = torch.zeros(
        (rows, fc2["weight"].shape[0]),
        dtype=torch.float32,
        device=expanded.device,
    )
    bf16_partial_output = None
    for start, stop in domain_ranges(ffn, domain):
        qdata, scale = quantize_domain(expanded, start, stop, ffn)
        partial = kitchen_cutlass_gemm(
            qdata,
            fc2["weight"][:, start:stop].contiguous(),
            scale,
            fc2["weight_scale"],
            torch.float32,
        )
        fp32_output.add_(partial)
        partial_bf16 = partial.to(torch.bfloat16)
        if bf16_partial_output is None:
            bf16_partial_output = partial_bf16
        else:
            bf16_partial_output.add_(partial_bf16)
        del qdata, scale, partial, partial_bf16
    return fp32_output.to(torch.bfloat16), bf16_partial_output


def production_two_slice_output(ck, activation, expanded, fc1, fc2, tiles):
    output = None
    fc1_errors = []
    ffn = int(fc2["weight"].shape[1])
    for tile in tiles:
        tile_expanded = base._convrot_output(
            ck,
            activation,
            tile["fc1_weight"],
            tile["fc1_scale"],
            fc1["group_size"],
            input_act=None,
        )
        expected = torch.cat(
            (
                expanded[:, tile["start"] : tile["stop"]],
                expanded[:, ffn + tile["start"] : ffn + tile["stop"]],
            ),
            dim=1,
        )
        fc1_errors.append(tensor_error(expected, tile_expanded))
        partial = base._convrot_output(
            ck,
            tile_expanded,
            tile["fc2_weight"],
            fc2["weight_scale"],
            fc2["group_size"],
            input_act="swiglu",
        )
        if output is None:
            output = partial
        else:
            output.add_(partial)
        del expected, tile_expanded, partial
    return output, fc1_errors


def run(args):
    if not args.i_understand_this_uses_gpu:
        raise RuntimeError(
            "pass --i-understand-this-uses-gpu after the required idle-GPU preflight"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.rows <= 0:
        raise ValueError("--rows must be positive")

    loaded = base.load_convrot_mlp(args.checkpoint, args.block_index)
    if (
        loaded["hidden_width"],
        loaded["ffn_width"],
        loaded["expanded_width"],
    ) != (HIDDEN, FFN, EXPANDED):
        raise RuntimeError("checkpoint is not the exact H3 MLP shape")
    domains = validate_scale_domains(loaded["ffn_width"], args.scale_domains)
    if PRODUCTION_DOMAIN not in domains or FFN not in domains:
        raise RuntimeError("scale domains must include the 7168 and 14336 controls")

    device = torch.device("cuda")
    ck = base._load_comfy_kitchen()
    fc1 = base._move_convrot_weight(loaded["fc1"], device)
    fc2 = base._move_convrot_weight(loaded["fc2"], device)
    fc1_reference_weight = base._dequantize_weight(fc1, device)
    fc2_reference_weight = base._dequantize_weight(fc2, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    activation = torch.randn(
        (args.rows, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn(
        (args.rows, HIDDEN), generator=generator, dtype=torch.bfloat16, device=device
    )
    gate = torch.randn(
        (HIDDEN,), generator=generator, dtype=torch.bfloat16, device=device
    )

    reference_output = base._bf16_mlp(
        activation, fc1_reference_weight, fc2_reference_weight
    )
    reference_residual = gated_residual(reference_output, residual, gate)
    del fc1_reference_weight, fc2_reference_weight
    torch.cuda.empty_cache()

    expanded = base._convrot_output(
        ck,
        activation,
        fc1["weight"],
        fc1["weight_scale"],
        fc1["group_size"],
        input_act=None,
    )
    tiles = prepare_production_tiles(fc1, fc2)
    production_output, fc1_control_errors = production_two_slice_output(
        ck, activation, expanded, fc1, fc2, tiles
    )
    production_residual = gated_residual(production_output, residual, gate)
    full_output = base._convrot_output(
        ck,
        expanded,
        fc2["weight"],
        fc2["weight_scale"],
        fc2["group_size"],
        input_act="swiglu",
    )
    full_residual = gated_residual(full_output, residual, gate)

    production_reference_error = tensor_error(reference_output, production_output)
    results = []
    control_outputs = {}
    for domain in domains:
        output, bf16_partial_output = scale_domain_fc2(expanded, fc2, domain)
        output_residual = gated_residual(output, residual, gate)
        reference_error = tensor_error(reference_output, output)
        result = {
            **scale_domain_contract(args.rows, domain),
            "fp32_accumulated": {
                "output_error_vs_bf16_reference": reference_error,
                "residual_error_vs_bf16_reference": tensor_error(
                    reference_residual, output_residual
                ),
                "output_error_vs_production": tensor_error(
                    production_output, output
                ),
                "relative_l2_improvement_vs_production": 1.0
                - reference_error["relative_l2"]
                / production_reference_error["relative_l2"],
            },
            "bf16_partial_accumulated": {
                "output_error_vs_production": tensor_error(
                    production_output, bf16_partial_output
                ),
                "output_error_vs_full_width": tensor_error(
                    full_output, bf16_partial_output
                ),
            },
        }
        if domain in (PRODUCTION_DOMAIN, FFN):
            control_outputs[domain] = result["bf16_partial_accumulated"]
        results.append(result)
        del output, bf16_partial_output, output_residual

    production_control = control_outputs[PRODUCTION_DOMAIN][
        "output_error_vs_production"
    ]
    full_control = control_outputs[FFN]["output_error_vs_full_width"]
    control_tolerance = 2e-5
    if (
        production_control["relative_l2"] > control_tolerance
        or full_control["relative_l2"] > control_tolerance
    ):
        raise RuntimeError(
            "scale-domain controls failed: production=%g full=%g"
            % (
                production_control["relative_l2"],
                full_control["relative_l2"],
            )
        )

    return {
        "experiment": (
            "numerical-only ConvRot scale-domain sweep; partial Kitchen CUTLASS "
            "GEMMs emulate a future one-launch scaled mainloop and are not timed"
        ),
        "checkpoint": {
            "path": str(args.checkpoint),
            "block_index": int(args.block_index),
            "weight_prefix": loaded["fc1"]["prefix"].rsplit("fc1.", 1)[0],
            "format": "ConvRot-256 TensorWise INT8",
        },
        "gpu": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
        },
        "dimensions": {
            "rows": int(args.rows),
            "hidden": HIDDEN,
            "ffn": FFN,
            "expanded": EXPANDED,
        },
        "controls": {
            "fc1_full_vs_two_slice_expansions": fc1_control_errors,
            "domain_7168_bf16_partial_vs_production": production_control,
            "domain_14336_bf16_partial_vs_full_width": full_control,
            "tolerance_relative_l2": control_tolerance,
        },
        "baselines": {
            "production_two_slice": {
                "output_error_vs_bf16_reference": production_reference_error,
                "residual_error_vs_bf16_reference": tensor_error(
                    reference_residual, production_residual
                ),
            },
            "full_width_kitchen": {
                "output_error_vs_bf16_reference": tensor_error(
                    reference_output, full_output
                ),
                "residual_error_vs_bf16_reference": tensor_error(
                    reference_residual, full_residual
                ),
                "output_error_vs_production": tensor_error(
                    production_output, full_output
                ),
            },
        },
        "scale_domains": results,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block-index", type=int, default=0)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument(
        "--scale-domains",
        type=parse_scale_domains,
        default=DEFAULT_SCALE_DOMAINS,
        metavar="N[,N...]",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--i-understand-this-uses-gpu", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    report = run(args)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
