from __future__ import annotations

import logging

from ..h3_runtime.context import H3RuntimeSession, install_runtime_wrapper
from .forward import make_forward
from .report import H3ChipmunkReportListener
from .state import H3ChipmunkSession

BLOCKS_ATTR = "diffusion_model.blocks"
STATUS_KEY = "minimax_h3_chipmunk"
LOG_PREFIX = "[H3 Chipmunk]"


class H3ChipmunkPatchError(RuntimeError):
    pass


def install(model_patcher, config):
    try:
        blocks = tuple(model_patcher.get_model_object(BLOCKS_ATTR))
    except Exception as exc:
        raise H3ChipmunkPatchError("MiniMax H3 blocks were not found") from exc
    if len(blocks) != 50:
        raise H3ChipmunkPatchError(f"expected 50 H3 blocks, got {len(blocks)}")

    options = model_patcher.model_options["transformer_options"] = (
        model_patcher.model_options.get("transformer_options", {}).copy()
    )
    runtime = options.get("minimax_h3_runtime_session")
    chip_session = H3ChipmunkSession()
    listener = H3ChipmunkReportListener(chip_session, config)
    if runtime is None:
        runtime = H3RuntimeSession(listeners=[listener])
        install_runtime_wrapper(model_patcher, runtime)
        options["minimax_h3_runtime_session"] = runtime
    else:
        runtime.add_listener(listener)

    existing = getattr(model_patcher, "object_patches", {})
    for index, block in enumerate(blocks):
        key = f"{BLOCKS_ATTR}.{index}.forward"
        current = existing.get(key)
        if current is not None and getattr(current, "_h3_shared_block_compile", False):
            raise H3ChipmunkPatchError(
                "Chipmunk must be installed instead of shared-block compilation; "
                "disable H3 shared compile for this experiment"
            )
        if current is not None and not (
            getattr(current, "_h3_activation_memory", False)
            or getattr(current, "_h3_chipmunk", False)
        ):
            raise H3ChipmunkPatchError(f"foreign block patch already owns {key}")
        if getattr(current, "_h3_chipmunk", False):
            if getattr(current, "_h3_chipmunk_config", None) == config.signature:
                continue
            raise H3ChipmunkPatchError("Chipmunk is already installed with another configuration")
        original = getattr(current, "_h3_activation_original", None) if current is not None else None
        original = original or block.forward
        model_patcher.add_object_patch(
            key, make_forward(block, index, config, chip_session, original_forward=original)
        )

    options[STATUS_KEY] = {
        "mode": config.mode,
        "top_fraction": float(config.top_fraction),
        "refresh_every": int(config.refresh_every),
        "feature_group": int(config.feature_group),
        "token_group_rows": int(config.token_group_rows),
        "scope": config.scope,
        "layer_range": [int(config.layer_start), int(config.layer_stop)],
        "approximate": config.mode == "reference_delta",
    }
    options["minimax_h3_chipmunk_session"] = chip_session
    logging.info(
        "%s installed: mode=%s top=%.3f refresh=%d group=%d token_group=%d scope=%s",
        LOG_PREFIX, config.mode, config.top_fraction, config.refresh_every,
        config.feature_group, config.token_group_rows, config.scope,
    )
    return chip_session
