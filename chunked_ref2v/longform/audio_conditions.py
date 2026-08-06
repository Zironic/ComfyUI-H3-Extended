"""Target-aligned generated-audio carry for MiniMax H3.

The existing visual carry patch inserts non-updating video rows at target
positions. H3 exposes the matching ``cond_audio_latents`` path, but the stock
layout has no target-aligned generated-audio rows. This module adds those rows
without changing the existing visual implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import comfy.conds

from ..layout_ops import TransformedLayout

LOG = "[H3 Extended] longform audio carry"


@dataclass
class TargetAlignedAudioCondition:
    """An audio latent clip conditioned at a target audio-latent position."""

    latent: torch.Tensor                 # [B, 32, 2, T], B == 1
    target_latent_start: int
    label: str = ""

    @property
    def latent_t(self):
        return int(self.latent.shape[-1])

    def describe(self):
        return "%s: audio latent t=%d at target position %d" % (
            self.label or "audio condition",
            self.latent_t,
            self.target_latent_start,
        )


def _segment_range(segments, kind):
    for start, stop, segment_kind in segments:
        if segment_kind == kind:
            return start, stop
    raise ValueError("packed layout has no '%s' segment" % kind)


def validate_audio_conditions(layout, conditions):
    """Validate generated-audio conditions against the target audio stream."""

    target_audio_t = int(layout.signature[4])
    seen = []
    for condition in conditions:
        latent = condition.latent
        if latent.ndim != 4:
            raise ValueError(
                "audio condition %r must be [B, C, 2, T], got %s"
                % (condition.label, tuple(latent.shape))
            )
        if latent.shape[0] != 1:
            raise ValueError(
                "audio condition %r must have batch size 1" % condition.label
            )
        if latent.shape[2] != 2:
            raise ValueError(
                "audio condition %r must have the H3 two-row audio axis, got %d"
                % (condition.label, latent.shape[2])
            )
        if condition.target_latent_start < 0:
            raise ValueError(
                "audio condition %r has a negative target position"
                % condition.label
            )
        stop = condition.target_latent_start + condition.latent_t
        if stop > target_audio_t:
            raise ValueError(
                "audio condition %r covers target positions %d-%d but the "
                "target is only %d positions long"
                % (
                    condition.label,
                    condition.target_latent_start,
                    stop - 1,
                    target_audio_t,
                )
            )
        span = (condition.target_latent_start, stop)
        for other_label, other in seen:
            if span[0] < other[1] and other[0] < span[1]:
                raise ValueError(
                    "audio conditions %r and %r overlap target positions"
                    % (other_label, condition.label)
                )
        seen.append((condition.label, span))
    return True


def insert_target_audio_conditions(layout, conditions):
    """Insert non-updating audio rows whose positions copy target audio rows.

    The new rows are placed directly after text. They use the existing
    ``ref_audio`` segment kind so the stock H3 forward gives them audio
    embeddings and the audio-conditioning timestep. ``audio_pos`` and
    ``audio_update`` are prepended in the same order as
    ``cond_audio_latents``.
    """

    conditions = list(conditions or [])
    if not conditions:
        return layout

    validate_audio_conditions(layout, conditions)
    text_start, text_stop = _segment_range(layout.segments, "text")
    if text_start != 0:
        raise ValueError("expected the text segment to lead the pack")
    target_start, _ = _segment_range(layout.segments, "audio")

    insert_at = text_stop
    pieces = []
    row_counts = []
    condition_segments = []
    row = insert_at
    for condition in conditions:
        start = target_start + condition.target_latent_start * 2
        stop = start + condition.latent_t * 2
        positions = layout.position_ids[start:stop].clone()
        expected = condition.latent_t * 2
        if positions.shape != (expected, 3):
            raise AssertionError(
                "audio condition %r produced %s position rows, expected (%d, 3)"
                % (condition.label, tuple(positions.shape), expected)
            )
        pieces.append(positions)
        row_counts.append(expected)
        condition_segments.append((row, row + expected, condition.label))
        row += expected

    inserted = sum(row_counts)
    position_ids = torch.cat(
        [layout.position_ids[:insert_at], *pieces, layout.position_ids[insert_at:]]
    )

    shifted_img_pos = layout.img_pos + (layout.img_pos >= insert_at) * inserted
    cond_audio_pos = torch.arange(insert_at, insert_at + inserted)
    shifted_audio_pos = (
        layout.audio_pos + (layout.audio_pos >= insert_at) * inserted
    )
    audio_pos = torch.cat([cond_audio_pos, shifted_audio_pos])
    audio_update = torch.cat(
        [torch.zeros(inserted, dtype=torch.bool), layout.audio_update]
    )

    segments = [(text_start, text_stop, "text")]
    offset = insert_at
    for count in row_counts:
        segments.append((offset, offset + count, "ref_audio"))
        offset += count
    for start, stop, kind in layout.segments:
        if kind == "text":
            continue
        segments.append((start + inserted, stop + inserted, kind))

    visual_segments = [
        (start + inserted, stop + inserted, label)
        for start, stop, label in getattr(layout, "condition_segments", [])
    ]
    transformed = TransformedLayout(
        seq_len=layout.seq_len + inserted,
        position_ids=position_ids,
        img_pos=shifted_img_pos,
        img_update=layout.img_update.clone(),
        audio_pos=audio_pos,
        audio_update=audio_update,
        segments=segments,
        signature=layout.signature,
        condition_rows=getattr(layout, "condition_rows", 0) + inserted,
        condition_segments=visual_segments,
    )
    transformed.audio_condition_segments = condition_segments
    _assert_consistent(
        transformed,
        layout,
        conditions,
        inserted,
        target_start + inserted,
    )
    return transformed


def _assert_consistent(transformed, base, conditions, inserted, target_start):
    if transformed.position_ids.shape[0] != transformed.seq_len:
        raise AssertionError("audio carry position_ids/seq_len mismatch")
    if transformed.img_pos.shape[0] != transformed.img_update.shape[0]:
        raise AssertionError("audio carry shifted image positions incorrectly")
    if transformed.audio_pos.shape[0] != transformed.audio_update.shape[0]:
        raise AssertionError("audio carry audio positions/mask mismatch")
    expected_non_target = int((~base.audio_update).sum()) + inserted
    if int((~transformed.audio_update).sum()) != expected_non_target:
        raise AssertionError(
            "audio carry non-target row count mismatch: got %d, expected %d"
            % (int((~transformed.audio_update).sum()), expected_non_target)
        )
    total = sum(stop - start for start, stop, _ in transformed.segments)
    if total != transformed.seq_len:
        raise AssertionError(
            "audio carry segments cover %d rows, seq_len is %d"
            % (total, transformed.seq_len)
        )

    for condition, (start, stop, _) in zip(
        conditions, transformed.audio_condition_segments
    ):
        target_row = target_start + condition.target_latent_start * 2
        target_stop = target_row + condition.latent_t * 2
        if not torch.equal(
            transformed.position_ids[start:stop],
            transformed.position_ids[target_row:target_stop],
        ):
            raise AssertionError(
                "audio condition %r does not carry target position ids"
                % condition.label
            )


def audio_condition_latents(conditions):
    return [condition.latent for condition in conditions]


def patch_target_audio_conditions(model, conditions):
    """Clone ``model`` and add target-aligned generated-audio conditions."""

    conditions = list(conditions or [])
    if not conditions:
        return model

    patched_model = model.clone()
    existing = patched_model.get_model_object("extra_conds")
    if getattr(existing, "_h3_audio_carry_patch", False):
        original = existing._h3_audio_carry_original
    else:
        original = existing

    latents = audio_condition_latents(conditions)
    described = [condition.describe() for condition in conditions]

    def patched_extra_conds(**kwargs):
        out = original(**kwargs)
        payload_cond = out.get("minimax_payload")
        if payload_cond is None:
            return out
        payload = payload_cond.cond.copy()
        layout = payload.get("layout")
        if layout is None:
            logging.warning(
                "%s no prebuilt packed layout - audio carry skipped", LOG
            )
            return out

        refs = payload.get("refs") or []
        payload["cond_audio_latents"] = [
            *latents,
            *[
                ref["audio_latent"]
                for ref in refs
                if ref.get("audio_latent") is not None
            ],
        ]
        payload["layout"] = insert_target_audio_conditions(layout, conditions)
        payload["h3_longform_audio_conditions"] = described
        out["minimax_payload"] = comfy.conds.CONDConstant(payload)
        logging.info("%s %s", LOG, "; ".join(described))
        return out

    patched_extra_conds._h3_audio_carry_patch = True
    patched_extra_conds._h3_audio_carry_original = original
    patched_model.add_object_patch("extra_conds", patched_extra_conds)
    return patched_model


__all__ = [
    "TargetAlignedAudioCondition",
    "audio_condition_latents",
    "insert_target_audio_conditions",
    "patch_target_audio_conditions",
    "validate_audio_conditions",
]
