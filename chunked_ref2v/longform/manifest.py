"""Run identity, resumability, and artifact invalidation for long-form Ref2V."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

SCHEMA_VERSION = 3


def _atomic_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=str)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def file_identity(path: str, *, sample_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    """Return a cheap but strong source identity without rereading a multi-GB VOD.

    Size, mtime, and hashes of the first and last samples are sufficient to reject
    accidental reuse while avoiding a full-file hash before every resume.
    """
    st = os.stat(path)
    first = b""
    last = b""
    with open(path, "rb") as fh:
        first = fh.read(sample_bytes)
        if st.st_size > sample_bytes:
            fh.seek(max(0, st.st_size - sample_bytes))
            last = fh.read(sample_bytes)
    return {
        "path": os.path.abspath(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "head_sha256": sha256_bytes(first),
        "tail_sha256": sha256_bytes(last),
    }


def tensor_digest(tensor: Any) -> str:
    try:
        import torch
        if not isinstance(tensor, torch.Tensor):
            return sha256_text(repr(tensor))
        value = tensor.detach().to("cpu").contiguous()
        header = f"{tuple(value.shape)}|{value.dtype}".encode("utf-8")
        return sha256_bytes(header + value.numpy().tobytes())
    except Exception:
        return sha256_text(repr(tensor))


def object_fingerprint(obj: Any) -> dict[str, Any]:
    """Best-effort stable model/component provenance.

    Comfy wrappers do not expose one universal checkpoint hash. Capture class,
    common path/name fields, and explicitly attached H3 configuration instead of
    pretending class name alone is enough.
    """
    if obj is None:
        return {"class": None}
    out: dict[str, Any] = {
        "class": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
    }
    for key in ("model_name", "name", "filename", "ckpt_name", "model_path"):
        value = getattr(obj, key, None)
        if isinstance(value, (str, int, float, bool)):
            out[key] = value
    get_model_object = getattr(obj, "get_model_object", None)
    if callable(get_model_object):
        try:
            model_sampling = get_model_object("model_sampling")
        except (AttributeError, KeyError):
            model_sampling = None
        if model_sampling is not None:
            sampling_identity = {
                "class": f"{model_sampling.__class__.__module__}.{model_sampling.__class__.__qualname__}",
                "shift": getattr(model_sampling, "shift", None),
                "audio_shift": getattr(model_sampling, "audio_shift", None),
            }
            out["model_sampling"] = sampling_identity

    model_options = getattr(obj, "model_options", None)
    if isinstance(model_options, dict):
        transformer_options = model_options.get("transformer_options", {})
        if isinstance(transformer_options, dict):
            for key in (
                "minimax_h3_sigma_shift_video",
                "minimax_h3_sigma_shift_audio",
                "h3_attention_backend",
                "h3_activation_mode",
            ):
                if key in transformer_options:
                    out[key] = transformer_options[key]
    return out


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def identity_hash(identity: dict[str, Any]) -> str:
    return sha256_text(json.dumps(normalize(identity), sort_keys=True, separators=(",", ":")))


@dataclass
class RunManifest:
    root: str
    identity: dict[str, Any]

    @property
    def path(self) -> str:
        return os.path.join(self.root, "manifest.json")

    @property
    def state_path(self) -> str:
        return os.path.join(self.root, "state.json")

    def ensure(self) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "identity_hash": identity_hash(self.identity),
            "identity": normalize(self.identity),
        }
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as fh:
                existing = json.load(fh)
            if existing.get("schema_version") != SCHEMA_VERSION:
                raise RuntimeError(
                    "long-form run manifest schema differs; use a new run directory"
                )
            if existing.get("identity_hash") != payload["identity_hash"]:
                old = existing.get("identity", {})
                changed = sorted(
                    key for key in set(old) | set(payload["identity"])
                    if old.get(key) != payload["identity"].get(key)
                )
                raise RuntimeError(
                    "run directory belongs to a different configuration; changed: %s. "
                    "Use a new run directory or remove the old run explicitly."
                    % (", ".join(changed) or "unknown")
                )
            return
        os.makedirs(self.root, exist_ok=True)
        _atomic_json(self.path, payload)
        if not os.path.exists(self.state_path):
            _atomic_json(self.state_path, {"schema_version": SCHEMA_VERSION})

    def update_state(self, **fields: Any) -> None:
        state: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, encoding="utf-8") as fh:
                    state.update(json.load(fh))
            except (OSError, ValueError):
                pass
        state.update(normalize(fields))
        _atomic_json(self.state_path, state)


def contiguous_prefix(paths: Iterable[str], validator=None) -> int:
    """Return the number of valid consecutive artifacts from index zero."""
    count = 0
    for path in paths:
        if not os.path.exists(path):
            break
        if validator is not None and not validator(path):
            break
        count += 1
    return count


def remove_from(paths: Iterable[str], start: int) -> None:
    for index, path in enumerate(paths):
        if index < start:
            continue
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
