"""torch.compile policy for H3's mixed eager/CUDA execution path."""

import logging

import torch

from comfy_api.torch_helpers.torch_compile import COMPILE_KEY, TORCH_COMPILE_KWARGS
from comfy.patcher_extension import WrappersMP

from .block_dispatch import install_shared_block_dispatch


LOG_PREFIX = "[H3 compile]"
BACKEND_MARKER = "minimax_h3_cuda_only_inductor"
REQUEST_MARKER = "minimax_h3_shared_block_compile_requested"


def _installation_signature(value):
    if value is None:
        return None
    signature = getattr(value, "installation_signature", None)
    if callable(signature):
        signature = signature()
    if signature is not None:
        return signature
    signature = getattr(value, "signature", None)
    if callable(signature):
        signature = signature()
    if signature is not None:
        return signature
    return (
        type(value).__module__,
        type(value).__qualname__,
        getattr(value, "name", value if isinstance(value, str) else None),
    )


def _compile_signature(compile_config, backend, activation_config):
    return (
        tuple(sorted(compile_config.items())),
        _installation_signature(backend),
        _installation_signature(activation_config),
        (cuda_only_inductor.__module__, cuda_only_inductor.__qualname__),
    )


def _tensor_devices(value, devices):
    if torch.is_tensor(value):
        devices.add(value.device.type)
    elif isinstance(value, dict):
        for item in value.values():
            _tensor_devices(item, devices)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _tensor_devices(item, devices)


def graph_tensor_devices(graph_module, example_inputs):
    devices = set()
    _tensor_devices(example_inputs, devices)
    for node in graph_module.graph.nodes:
        if node.op == "get_attr":
            value = graph_module
            for name in node.target.split("."):
                value = getattr(value, name)
            _tensor_devices(value, devices)
        _tensor_devices(node.meta.get("example_value"), devices)
        _tensor_devices(node.meta.get("val"), devices)
    return frozenset(devices)


def cuda_only_inductor(graph_module, example_inputs):
    devices = graph_tensor_devices(graph_module, example_inputs)
    if devices == frozenset(("cuda",)):
        with torch._inductor.config.patch({
            "triton.cudagraphs": False,
            "triton.cudagraph_trees": False,
        }):
            return torch._inductor.compile(graph_module, example_inputs)
    logging.debug(
        "%s leaving non-CUDA graph eager: devices=%s",
        LOG_PREFIX,
        sorted(devices) if devices else ["none"],
    )
    return graph_module.forward


def request_shared_block_compile(model_patcher):
    if (model_patcher.model_options.get(TORCH_COMPILE_KWARGS) is not None
            or model_patcher.get_wrappers(WrappersMP.APPLY_MODEL, COMPILE_KEY)):
        raise RuntimeError(
            "Remove TorchCompileModel before enabling H3 shared block compilation"
        )
    model_patcher.model_options[REQUEST_MARKER] = {
        "backend": "inductor",
        "fullgraph": True,
        "dynamic": False,
    }


def configure_shared_block_inductor(
    model_patcher,
    *,
    backend,
    activation_config,
    adaln_provider=None,
    block_cache=None,
):
    compile_config = model_patcher.model_options.get(REQUEST_MARKER)
    if not compile_config:
        return False
    if compile_config.get("backend") != "inductor":
        raise RuntimeError("H3 shared block compilation requires Inductor")
    if adaln_provider is not None:
        raise RuntimeError(
            "H3 shared block compilation does not yet support AdaLN precompute"
        )
    if block_cache is not None:
        raise RuntimeError(
            "H3 shared block compilation does not yet support FirstBlockCache"
        )
    desired_signature = _compile_signature(
        compile_config, backend, activation_config
    )
    installed_signature = model_patcher.model_options.get(BACKEND_MARKER)
    if installed_signature is not None:
        if installed_signature == desired_signature:
            return True
        raise RuntimeError(
            "H3 shared block compilation is already configured differently"
        )

    model_patcher.remove_wrappers_with_key(WrappersMP.APPLY_MODEL, COMPILE_KEY)
    install_shared_block_dispatch(
        model_patcher,
        backend,
        activation_config,
        cuda_only_inductor,
    )
    model_patcher.model_options.pop("disable_dynamic_vbar_prefetch", None)
    model_patcher.model_options[BACKEND_MARKER] = desired_signature
    logging.info(
        "%s Inductor will reuse one CUDA tensor program across all H3 main blocks",
        LOG_PREFIX,
    )
    return True
