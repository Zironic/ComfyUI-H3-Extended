"""Deliver target-aligned conditions to the H3 DiT via `add_object_patch`.

Two things stand between a prepared `TargetAlignedCondition` and the model
seeing it, both in `MiniMaxH3.extra_conds` (`comfy/model_base.py:2090-2111`):

1. `payload["cond_video_latents"]` is assigned from keyframes and then
   **overwritten** by references, so a run with both keeps only the refs.
2. `payload["layout"]` is the stock pack, with no rows for our conditions.

Both are fixed by wrapping `extra_conds` rather than forking `comfy/ldm/minimax/`,
which keeps the boundary this repo's README states and makes the change
reversible on unpatch.

`ModelPatcher.clone()` copies the patch dict but shares `self.model`
(`comfy/model_patcher.py:451`), so assigning `model.model.extra_conds = ...`
would escape the clone and leak into every other graph using that model.
`add_object_patch` `set_attr`s on patch and restores on unpatch, which is the
same mechanism `MiniMaxH3SigmaShiftZi` already uses for `model_sampling`.
"""

import logging

import comfy.conds

from .layout_ops import condition_latents, describe_layout, insert_target_conditions

LOG_PREFIX = "[H3 Extended] harness"


def patch_target_conditions(model, conditions, *, position_policy="copy_target"):
    """Clone `model` so its `extra_conds` emits target-aligned condition rows.

    With no conditions the model is returned unchanged - the baseline arm must
    run through exactly the stock path, or it is not a control.
    """
    if not conditions:
        return model

    m = model.clone()
    existing = m.get_model_object("extra_conds")
    # `get_model_object` returns the *patch* when one is registered
    # (`model_patcher.py:756-761`) and `clone()` copies object_patches, so a
    # second harness run in the same graph would otherwise wrap the wrapper.
    if getattr(existing, "_h3_harness_patch", False):
        original = existing._h3_harness_original
    else:
        original = existing

    latents = condition_latents(conditions)
    described = [c.describe() for c in conditions]

    def patched_extra_conds(**kwargs):
        out = original(**kwargs)
        payload_cond = out.get("minimax_payload")
        if payload_cond is None:
            return out
        payload = payload_cond.cond.copy()

        base_layout = payload.get("layout")
        if base_layout is None:
            # No prebuilt layout means core could not see both latent streams;
            # transforming nothing is safer than guessing the pack.
            logging.warning("%s no prebuilt packed layout - target conditions skipped",
                            LOG_PREFIX)
            return out

        refs = payload.get("refs") or []
        payload["cond_video_latents"] = [
            *latents,
            *[r["latent"] for r in refs if "latent" in r],
        ]
        payload["layout"] = insert_target_conditions(
            base_layout, conditions, position_policy=position_policy)
        payload["h3_harness_conditions"] = described

        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        logging.info("%s target conditions: %s | %s", LOG_PREFIX,
                     "; ".join(described), describe_layout(payload["layout"]))
        return out

    patched_extra_conds._h3_harness_patch = True
    patched_extra_conds._h3_harness_original = original
    m.add_object_patch("extra_conds", patched_extra_conds)
    return m
