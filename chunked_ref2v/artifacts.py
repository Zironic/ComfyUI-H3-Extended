"""Run directory, manifest and asset persistence for the Ref2V harness.

Chunk A can be explicitly reused through the node's `reuse_run` input. Automatic
identity-based reuse is disabled by default because a trustworthy identity must
include checkpoint, CLIP, VAE, model patches and sampler options; Python class
names are not sufficient provenance. Set `H3_HARNESS_AUTO_REUSE=1` only when the
operator deliberately accepts that limitation.
"""

import hashlib
import json
import logging
import os
import time

import torch

import comfy.utils

LOG_PREFIX = "[H3 Extended] harness"
MANIFEST_VERSION = 2
AUTO_REUSE_ENV = "H3_HARNESS_AUTO_REUSE"


def _hash_tensor(h, tensor):
    if tensor is None:
        h.update(b"\x00none")
        return
    t = tensor.detach().to("cpu")
    h.update(str(tuple(t.shape)).encode())
    h.update(str(t.dtype).encode())
    h.update(t.contiguous().view(torch.uint8).numpy().tobytes())


def _hash_value(h, value):
    if isinstance(value, torch.Tensor):
        _hash_tensor(h, value)
    else:
        h.update(json.dumps(value, sort_keys=True, default=str).encode())


def chunk_a_identity(*, source_frames, prompt, ref_pixels, canvas, geometry,
                     seed, sampler_name, sigmas, checkpoint):
    """Best-effort digest of the inputs represented by the current node API.

    This remains useful for validating an explicitly selected `reuse_run`, but
    it is not strong enough to authorize automatic reuse across arbitrary model
    changes. See `find_reusable_run`.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(b"h3-ref2v-harness-chunk-a-v3")
    _hash_tensor(h, source_frames)
    _hash_value(h, prompt)
    for pixels in ref_pixels or []:
        _hash_tensor(h, pixels)
    _hash_value(h, list(canvas))
    _hash_value(h, [geometry.chunk_frames, geometry.overlap_frames, geometry.fps])
    _hash_value(h, int(seed))
    _hash_value(h, sampler_name)
    _hash_tensor(h, sigmas)
    _hash_value(h, checkpoint)
    return h.hexdigest()


class RunStore:
    def __init__(self, root, run_id):
        self.root = os.path.join(root, run_id)
        self.run_id = run_id
        self.common = os.path.join(self.root, "common")
        self.dynamic = os.path.join(self.root, "dynamic")
        self.experiments = os.path.join(self.root, "experiments")
        for path in (self.root, self.common, self.dynamic, self.experiments):
            os.makedirs(path, exist_ok=True)
        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.manifest = self._read_manifest()

    def _read_manifest(self):
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return {"schema_version": MANIFEST_VERSION, "run_id": self.run_id,
                    "created": time.time(), "assets": {}}
        if data.get("schema_version") != MANIFEST_VERSION:
            logging.info("%s manifest schema %s != %s, starting fresh",
                         LOG_PREFIX, data.get("schema_version"), MANIFEST_VERSION)
            return {"schema_version": MANIFEST_VERSION, "run_id": self.run_id,
                    "created": time.time(), "assets": {}}
        return data

    def write_manifest(self, **extra):
        self.manifest.update(extra)
        self.manifest["updated"] = time.time()
        self._atomic_write(self.manifest_path,
                           json.dumps(self.manifest, indent=2, default=str))

    def _atomic_write(self, path, text):
        tmp = "%s.%d.tmp" % (path, os.getpid())
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except OSError:
            _unlink(tmp)
            raise

    def write_text(self, name, text):
        self._atomic_write(os.path.join(self.root, name), text)

    def _asset_path(self, group, name):
        return os.path.join(getattr(self, group), name + ".safetensors")

    def save_tensors(self, group, name, tensors, identity=None):
        path = self._asset_path(group, name)
        payload = {k: v.detach().to("cpu").contiguous()
                   for k, v in tensors.items() if v is not None}
        tmp = "%s.%d.tmp" % (path, os.getpid())
        try:
            comfy.utils.save_torch_file(payload, tmp)
            os.replace(tmp, path)
        except Exception:
            _unlink(tmp)
            raise
        self.manifest.setdefault("assets", {})["%s/%s" % (group, name)] = {
            "identity": identity,
            "stored": time.time(),
            "keys": sorted(payload),
        }
        return path

    def load_tensors(self, group, name, identity=None):
        record = self.manifest.get("assets", {}).get("%s/%s" % (group, name))
        if record is None:
            return None
        if identity is not None and record.get("identity") != identity:
            logging.info("%s asset %s/%s belongs to a different identity - ignoring",
                         LOG_PREFIX, group, name)
            return None
        path = self._asset_path(group, name)
        if not os.path.exists(path):
            return None
        try:
            return comfy.utils.load_torch_file(path, safe_load=True)
        except Exception as exc:
            logging.warning("%s asset %s/%s failed to load (%s) - regenerating",
                            LOG_PREFIX, group, name, exc)
            return None

    def invalidate(self, group, name):
        self.manifest.get("assets", {}).pop("%s/%s" % (group, name), None)
        _unlink(self._asset_path(group, name))

    def invalidate_group(self, group):
        prefix = group + "/"
        for key in [k for k in self.manifest.get("assets", {}) if k.startswith(prefix)]:
            self.invalidate(*key.split("/", 1))

    def experiment_dir(self, experiment_id):
        path = os.path.join(self.experiments, experiment_id)
        os.makedirs(path, exist_ok=True)
        return path

    def save_experiment_tensors(self, experiment_id, name, tensors):
        path = os.path.join(self.experiment_dir(experiment_id), name + ".safetensors")
        payload = {k: v.detach().to("cpu").contiguous()
                   for k, v in tensors.items() if v is not None}
        tmp = "%s.%d.tmp" % (path, os.getpid())
        try:
            comfy.utils.save_torch_file(payload, tmp)
            os.replace(tmp, path)
        except Exception:
            _unlink(tmp)
            raise
        return path


def _unlink(path):
    try:
        os.remove(path)
    except OSError:
        pass


def save_frames(directory, frames, prefix="frame", limit=None):
    from PIL import Image
    import numpy as np

    os.makedirs(directory, exist_ok=True)
    written = []
    count = frames.shape[0] if limit is None else min(limit, frames.shape[0])
    for i in range(count):
        arr = frames[i].detach().to("cpu").float().clamp(0.0, 1.0).numpy()
        img = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8))
        path = os.path.join(directory, "%s_%05d.png" % (prefix, i))
        img.save(path, compress_level=4)
        written.append(path)
    return written


def resolve_root(output_directory=None):
    if output_directory is None:
        import folder_paths
        output_directory = folder_paths.get_output_directory()
    root = os.path.join(output_directory, "h3_ref2v_harness")
    os.makedirs(root, exist_ok=True)
    return root


def new_run_id(identity):
    return "%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), identity[:8])


def _auto_reuse_enabled():
    return os.environ.get(AUTO_REUSE_ENV, "").strip().lower() in (
        "1", "true", "yes", "on")


def find_reusable_run(root, identity):
    """Return a matching run only when automatic reuse is explicitly enabled."""
    if not _auto_reuse_enabled():
        return None
    if not os.path.isdir(root):
        return None
    candidates = []
    for name in os.listdir(root):
        manifest_path = os.path.join(root, name, "manifest.json")
        if not os.path.exists(manifest_path):
            continue
        try:
            with open(manifest_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        if data.get("chunk_a_identity") == identity:
            candidates.append((data.get("updated", 0), name))
    if not candidates:
        return None
    return max(candidates)[1]
