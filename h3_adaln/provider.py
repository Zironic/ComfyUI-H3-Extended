"""Run-scoped trajectory precomputation for H3 block AdaLN projections.

Unlike the serving implementation in Sol-Engine, this provider does not delete
checkpoint weights.  It computes a table after Comfy has loaded the model, lets
Comfy continue to own/offload the original weights, and falls back to the
original projection whenever the run signature cannot be proven compatible.
"""

from __future__ import annotations

import logging
import math
import threading

import torch
import torch.nn.functional as F

from .config import MODE_AUTO, AdaLNPrecomputeConfig

LOG_PREFIX = "[H3 AdaLN]"
GIB = 1024 ** 3


class AdaLNPrecomputeError(RuntimeError):
    pass


def _tensor_bytes(tensor):
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _projection_bytes(projection):
    total = 0
    try:
        for parameter in projection.parameters():
            total += _tensor_bytes(parameter)
    except Exception:
        pass
    return total


def _step_t_values(model, sigma, transformer_options, payload, layout):
    from comfy.ldm.minimax.model import (
        AUDIO_COND_TIMESTEP,
        VISUAL_COND_TIMESTEP,
        time_shift_sigma,
    )

    shift_v = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_video",
            model.sigma_shift_video,
        )
    )
    shift_a = float(
        transformer_options.get(
            "minimax_h3_sigma_shift_audio",
            model.sigma_shift_audio,
        )
    )
    sigma_v = torch.as_tensor(sigma, dtype=torch.float32).clamp(min=1e-6)
    t_v = float(1.0 - sigma_v)
    t_a = float(1.0 - time_shift_sigma(sigma_v, shift_v, shift_a))
    vis_aug = float(payload.get("visual_cond_noise_aug", VISUAL_COND_TIMESTEP))
    aud_aug = float(payload.get("audio_cond_noise_aug", AUDIO_COND_TIMESTEP))
    kinds = {kind for _, _, kind in layout.segments}
    values = {t_v, t_a}
    if kinds.intersection({"cond", "ref_img"}):
        values.add(max(t_v, vis_aug))
    if "ref_audio" in kinds:
        values.add(max(t_a, aud_aug))
    return tuple(sorted(values))


def _embed_t_values(model, values, device, dtype):
    t_vals = torch.tensor(values, dtype=torch.float32, device=device)
    if model.use_adaln_curves:
        import comfy.model_management

        table = comfy.model_management.cast_to(model.adaln_t_table, device=device)
        pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
        i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
        return torch.lerp(
            table[i0],
            table[i0 + 1],
            (pos - i0).unsqueeze(1),
        )
    return model.time_embedder(t_vals).to(dtype)


class AdaLNProvider:
    def __init__(self, model, blocks, originals, config=None):
        self.model = model
        self.blocks = list(blocks)
        self.originals = list(originals)
        self.config = config or AdaLNPrecomputeConfig()
        self.tables = None
        self.signature = None
        self.declined_reason = None
        self.declined_signature = None
        self.stats = {
            "builds": 0,
            "hits": 0,
            "misses": 0,
            "table_bytes": 0,
            "weight_bytes": 0,
            "steps": 0,
            "blocks": len(self.blocks),
            "held_weight_sessions": 0,
            "held_weight_fallbacks": 0,
        }
        self._local = threading.local()
        self._lock = threading.RLock()

    def on_request_reset(self, request_id):
        self._local.request_id = request_id
        self._local.step_index = -1

    def before_forward(self, snapshot, transformer_options, payload):
        self._local.request_id = snapshot.request_id
        self._local.step_index = snapshot.step_index
        if not self.config.enabled or not snapshot.valid_layout:
            return
        try:
            self._maybe_build(snapshot, transformer_options, payload)
        except Exception as exc:
            self.declined_reason = "%s: %s" % (type(exc).__name__, exc)
            try:
                schedule = self._schedule(transformer_options)
                self.declined_signature = self._build_signature(
                    snapshot, transformer_options, payload, schedule
                )
            except Exception:
                self.declined_signature = None
            if self.config.strict:
                raise
            logging.warning("%s disabled: %s", LOG_PREFIX, self.declined_reason)

    def _schedule(self, transformer_options):
        schedule = transformer_options.get("sample_sigmas")
        if schedule is None or not torch.is_tensor(schedule) or schedule.numel() < 2:
            raise AdaLNPrecomputeError("sample_sigmas is unavailable")
        return schedule.detach().flatten()[:-1].to("cpu", torch.float32)

    def _build_signature(self, snapshot, transformer_options, payload, schedule):
        return (
            tuple(round(float(x), 8) for x in schedule.tolist()),
            snapshot.layout_signature,
            str(snapshot.compute_dtype),
            str(snapshot.device),
            float(transformer_options.get("minimax_h3_sigma_shift_video", self.model.sigma_shift_video)),
            float(transformer_options.get("minimax_h3_sigma_shift_audio", self.model.sigma_shift_audio)),
            float(payload.get("visual_cond_noise_aug", 0.999)),
            float(payload.get("audio_cond_noise_aug", 1.0)),
        )

    def _maybe_build(self, snapshot, transformer_options, payload):
        if self.tables is not None and self.signature is not None:
            schedule = self._schedule(transformer_options)
            signature = self._build_signature(snapshot, transformer_options, payload, schedule)
            if signature == self.signature:
                return
        with self._lock:
            schedule = self._schedule(transformer_options)
            signature = self._build_signature(snapshot, transformer_options, payload, schedule)
            if self.tables is not None and signature == self.signature:
                return
            if self.tables is None and signature == self.declined_signature:
                return
            # Release an incompatible trajectory table before constructing the
            # replacement so two multi-gigabyte tables never overlap.
            self.tables = None
            self.signature = None
            self._build(snapshot, transformer_options, payload, schedule, signature)

    def _projection_outputs(self, projection, original, embeddings):
        """Evaluate a block's whole trajectory while holding its linear once.

        Comfy may stream or cast low-VRAM weights on every module call.  Repeating
        that for every denoising step would turn a lossless optimization into a
        large I/O regression.  Reuse the same safe acquisition helper as the
        chunked MLP and release the weight after this block's table is complete.
        """
        acquired = None
        try:
            try:
                from ..h3_activation_memory.linear import acquire_linear
            except ImportError:
                from h3_activation_memory.linear import acquire_linear

            sample = (
                F.silu(embeddings[0])
                if bool(getattr(projection, "apply_silu", True))
                else embeddings[0]
            )
            acquired = acquire_linear(projection.linear, sample)
            self.stats["held_weight_sessions"] += 1
            outputs = []
            for embedding in embeddings:
                value = (
                    F.silu(embedding)
                    if bool(getattr(projection, "apply_silu", True))
                    else embedding
                )
                x = acquired.linear(value)
                x = x.view(
                    x.shape[0] * int(projection.modalities),
                    int(projection.expand) * int(projection.hidden),
                )
                outputs.append(x.chunk(int(projection.expand), dim=-1))
            return outputs
        except Exception:
            self.stats["held_weight_fallbacks"] += 1
            if self.config.strict:
                raise
            logging.warning(
                "%s held projection unavailable; using ordinary module calls",
                LOG_PREFIX,
                exc_info=True,
            )
            return [original(embedding) for embedding in embeddings]
        finally:
            if acquired is not None:
                acquired.release()

    def _build(self, snapshot, transformer_options, payload, schedule, signature):
        if snapshot.device is None or snapshot.compute_dtype is None:
            raise AdaLNPrecomputeError("runtime device or compute dtype is unavailable")
        plans = []
        for sigma in schedule.tolist():
            values = _step_t_values(
                self.model,
                sigma,
                transformer_options,
                payload,
                snapshot.layout,
            )
            plans.append(values)

        hidden = int(getattr(self.blocks[0].adaln_proj, "hidden", 0))
        if hidden <= 0:
            raise AdaLNPrecomputeError("could not determine AdaLN hidden width")
        dtype_bytes = torch.tensor([], dtype=snapshot.compute_dtype).element_size()
        row_counts = [len(values) * 3 for values in plans]
        estimate = sum(rows * 6 * hidden * dtype_bytes for rows in row_counts) * len(self.blocks)
        max_bytes = int(float(self.config.max_table_gib) * GIB)
        weight_bytes = sum(_projection_bytes(block.adaln_proj) for block in self.blocks)
        self.stats["weight_bytes"] = weight_bytes
        if estimate > max_bytes:
            raise AdaLNPrecomputeError(
                "estimated table %.3f GiB exceeds limit %.3f GiB"
                % (estimate / GIB, max_bytes / GIB)
            )
        if self.config.mode == MODE_AUTO and weight_bytes and estimate >= weight_bytes:
            raise AdaLNPrecomputeError(
                "auto declined: table %.3f GiB is not smaller than projection weights %.3f GiB"
                % (estimate / GIB, weight_bytes / GIB)
            )

        embeddings = [
            _embed_t_values(
                self.model,
                values,
                snapshot.device,
                snapshot.compute_dtype,
            )
            for values in plans
        ]
        tables = []
        actual_bytes = 0
        with torch.no_grad():
            for index, original in enumerate(self.originals):
                projection = self.blocks[index].adaln_proj
                per_step = []
                for result in self._projection_outputs(
                    projection, original, embeddings
                ):
                    stored = tuple(item.detach() for item in result)
                    actual_bytes += sum(_tensor_bytes(item) for item in stored)
                    per_step.append(stored)
                tables.append(tuple(per_step))
                logging.debug("%s precomputed block %d", LOG_PREFIX, index)

        self.tables = tuple(tables)
        self.signature = signature
        self.declined_reason = None
        self.declined_signature = None
        self.stats.update(
            {
                "builds": self.stats["builds"] + 1,
                "table_bytes": actual_bytes,
                "steps": len(schedule),
            }
        )
        logging.info(
            "%s ready: blocks=%d steps=%d table=%.3f GiB projection_weights=%.3f GiB",
            LOG_PREFIX,
            len(self.blocks),
            len(schedule),
            actual_bytes / GIB,
            weight_bytes / GIB,
        )

    def lookup(self, layer_index, t_emb):
        step = int(getattr(self._local, "step_index", -1))
        if self.tables is None or step < 0:
            self.stats["misses"] += 1
            return self.originals[layer_index](t_emb)
        try:
            result = self.tables[layer_index][step]
        except Exception:
            self.stats["misses"] += 1
            return self.originals[layer_index](t_emb)
        self.stats["hits"] += 1
        return result

    def as_status(self):
        return {
            **self.stats,
            "mode": self.config.mode,
            "ready": self.tables is not None,
            "declined_reason": self.declined_reason,
        }
