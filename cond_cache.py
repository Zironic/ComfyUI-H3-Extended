"""Content-addressed disk cache for the Qwen3-VL conditioning pass.

The H3 conditioning nodes hand the tokenizer a *presentation* — label text plus
reference image/video pixel blocks — and run Qwen3-VL-32B over it to get hidden
states. That forward pass is the most expensive thing either node does, and it
depends on nothing but the token stream and the TE weights: not on width,
height, length, sampler settings, or seed.

ComfyUI's execution cache is keyed on the whole node, so nudging `length` re-runs
the encoder even though not one token changed, and a restart discards it
entirely. Keying on the token stream itself instead means:

- changing `length` / `width` / `height` / sampler settings reuses the embeds
- restarting the server reuses the embeds
- a hit avoids staging the 14.6 GB text encoder onto the 12 GB card at all

Anything the cache cannot identify with certainty — a LoRA-patched TE, a hook
schedule, an unrecognised token or conditioning payload, an unknown checkpoint
provenance — falls back to a plain encode. A wrong hit is much worse than a miss.

Entries are safetensors files under the user directory, so they land on D: with
the rest of the data. Log lines are prefixed `[H3 Extended] cond cache`.

Bounded by `sweep()`, which runs once on first use and again after every store:
orphaned temp files, then entries unused past the age limit, then oldest-used
first until under the size cap. See that function for why each limit exists.

Env overrides (the node's `cond_cache` widget is the normal control):
  H3_COND_CACHE_DISABLE=1         force off regardless of the widget
  H3_COND_CACHE_DIR=<path>        override the location
  H3_COND_CACHE_GB=<float>        size cap, default 20
  H3_COND_CACHE_MAX_AGE_DAYS=<f>  expire by time since last use, default 30, 0 disables
"""

import enum
import hashlib
import json
import logging
import os
import re
import time

import numpy as np
import torch

import comfy.utils
import folder_paths

CACHE_FORMAT = "h3_cond_cache/1"
DEFAULT_MAX_GB = 20.0
DEFAULT_MAX_AGE_DAYS = 30.0
STALE_TMP_SECONDS = 3600
ENTRY_SUFFIX = ".safetensors"
TMP_SUFFIX = ".tmp"
DIGEST_CHARS = 32
MARKER = ".h3_cond_cache"
ENTRY_RE = re.compile(r"[0-9a-f]{%d}\%s" % (DIGEST_CHARS, ENTRY_SUFFIX))
TMP_RE = re.compile(r"[0-9a-f]{%d}\%s\.\d+\%s" % (DIGEST_CHARS, ENTRY_SUFFIX, TMP_SUFFIX))
LOG = "[H3 Extended] cond cache"

_ENV_DISABLE = "H3_COND_CACHE_DISABLE"
_ENV_DIR = "H3_COND_CACHE_DIR"
_ENV_GB = "H3_COND_CACHE_GB"
_ENV_AGE = "H3_COND_CACHE_MAX_AGE_DAYS"

MODES = ["auto", "off", "refresh"]

_swept = False
_UNSET = object()
_claimed = _UNSET


def _env_disabled():
    return os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on")


def _env_float(name, default):
    try:
        return max(0.0, float(os.environ.get(name, default)))
    except ValueError:
        return default


def _max_bytes():
    return _env_float(_ENV_GB, DEFAULT_MAX_GB) * (1024 ** 3)


def _max_age_seconds():
    """0 disables age-based expiry."""
    return _env_float(_ENV_AGE, DEFAULT_MAX_AGE_DAYS) * 86400


def _claim(directory):
    """Return `directory` if this cache owns it, else None.

    Ownership is a property of the *folder*, established once and recorded in a
    marker file, rather than something re-derived per file at delete time. A
    directory is ours if we created it, if it is empty, or if it already carries
    the marker. A pre-existing directory holding only cache-shaped files is
    adopted, which covers upgrades from before the marker existed.

    Anything else — most importantly `H3_COND_CACHE_DIR` aimed at a real models
    folder — is refused outright, and the cache disables itself for the process.
    Refusing to use a directory is a much better failure than sweeping one.
    """
    marker = os.path.join(directory, MARKER)
    if os.path.exists(marker):
        return directory

    if not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    else:
        strays = [n for n in os.listdir(directory)
                  if not (ENTRY_RE.fullmatch(n) or TMP_RE.fullmatch(n))]
        if strays:
            logging.error(
                "%s: refusing to use %s — it holds files this cache did not write "
                "(%s%s). Point %s at a dedicated folder, or remove it to let one be "
                "created. The cache is disabled for this session.",
                LOG, directory, ", ".join(sorted(strays)[:3]),
                ", ..." if len(strays) > 3 else "", _ENV_DIR)
            return None

    with open(marker, "w", encoding="utf-8") as f:
        f.write("Cache folder for %s.\nEverything in here is disposable and is "
                "deleted automatically.\nDeleting this file makes the cache "
                "disown the folder.\n" % CACHE_FORMAT)
    return directory


def _cache_dir():
    """The owned cache directory, or None if it could not be claimed."""
    global _claimed
    if _claimed is _UNSET:
        d = os.environ.get(_ENV_DIR) or os.path.join(folder_paths.get_user_directory(), "h3_cond_cache")
        try:
            _claimed = _claim(d)
        except OSError:
            logging.exception("%s: could not prepare %s", LOG, d)
            _claimed = None
    return _claimed


# ---------------------------------------------------------------- hashing

def _hash_tensor(h, t):
    """Hash a tensor's logical contents, shape and dtype.

    Note the deliberate avoidance of untyped_storage(): the tokenizer hands us
    slice views (e.g. frames[i:i+2]) that are already contiguous, so
    .contiguous() is a no-op and the underlying storage still covers the whole
    video. numpy's view of the tensor carries the correct shape and strides.
    """
    t = t.detach()
    if t.device.type != "cpu":
        t = t.cpu()
    h.update(("t|%s|%s|" % (tuple(t.shape), t.dtype)).encode())
    try:
        arr = t.numpy()
    except (TypeError, RuntimeError):  # bf16 and friends have no numpy dtype
        arr = t.to(torch.float32).numpy()
    h.update(np.ascontiguousarray(arr).reshape(-1).view(np.uint8))


def _stable_repr(obj, depth=0):
    """Deterministic, cheap description of a config value (no tensor contents)."""
    if depth > 6:
        return "..."
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        return repr(obj)
    if torch.is_tensor(obj):
        return "tensor%s:%s" % (tuple(obj.shape), obj.dtype)
    if isinstance(obj, dict):
        return "{%s}" % ",".join("%r:%s" % (k, _stable_repr(obj[k], depth + 1)) for k in sorted(obj, key=repr))
    if isinstance(obj, (list, tuple, set)):
        items = sorted(obj, key=repr) if isinstance(obj, set) else obj
        return "[%s]" % ",".join(_stable_repr(v, depth + 1) for v in items)
    if isinstance(obj, enum.Enum):
        return "%s.%s" % (type(obj).__name__, obj.name)
    if isinstance(obj, (torch.dtype, torch.device)):
        return str(obj)
    return "<%s>" % type(obj).__name__


def _model_fingerprint(clip):
    """Identify the TE weights, or None if provenance is unknown.

    Core's loaders record how to rebuild a patcher in `cached_patcher_init`,
    which for a CLIP is `(load_clip_model_patcher, (ckpt_paths, embedding_dir,
    clip_type, model_options))` and survives `clone()`. That gives the actual
    checkpoint files without hashing 14.6 GB of weights.
    """
    init = getattr(clip.patcher, "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    args = init[1]
    paths = args[0]
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, (list, tuple)) or not paths:
        return None

    parts = [CACHE_FORMAT, type(clip.cond_stage_model).__name__, "layer=%s" % (clip.layer_idx,)]
    for p in paths:
        try:
            st = os.stat(p)
        except OSError:
            return None
        parts.append("%s|%d|%d" % (os.path.basename(p), st.st_size, int(st.st_mtime)))
    parts.append(_stable_repr(list(args[2:])))  # clip_type, model_options (dtype, quantization, ...)
    return "\n".join(parts)


def tokens_digest(clip, tokens):
    """blake2b over the TE identity and the full token stream, or None to bypass."""
    fp = _model_fingerprint(clip)
    if fp is None:
        return None

    h = hashlib.blake2b(digest_size=32)
    h.update(fp.encode())
    for key in sorted(tokens):
        h.update(("\nkey|%s" % key).encode())
        for batch in tokens[key]:
            h.update(b"\nbatch")
            for entry in batch:
                token, weight = entry[0], entry[1]
                h.update(("\nw|%r" % float(weight)).encode())
                if isinstance(token, (int, np.integer)):
                    h.update(("i|%d" % int(token)).encode())
                elif torch.is_tensor(token):
                    h.update(b"e|")
                    _hash_tensor(h, token)
                elif isinstance(token, dict):
                    h.update(b"d|")
                    for k in sorted(token):
                        v = token[k]
                        h.update(("|%s=" % k).encode())
                        if torch.is_tensor(v):
                            _hash_tensor(h, v)
                        else:
                            h.update(_stable_repr(v).encode())
                else:
                    return None  # unrecognised token payload -> do not cache
    return h.hexdigest()


# ---------------------------------------------------------------- storage

def _pack(cond):
    """[[tensor, dict]] -> (state_dict, metadata), or None if not representable."""
    if not isinstance(cond, list) or len(cond) != 1:
        return None
    entry = cond[0]
    if len(entry) != 2 or not torch.is_tensor(entry[0]) or not isinstance(entry[1], dict):
        return None

    sd = {"cond": entry[0].contiguous()}
    none_keys = []
    for k, v in entry[1].items():
        if v is None:
            none_keys.append(k)
        elif torch.is_tensor(v):
            sd["extra." + k] = v.contiguous()
        else:
            return None  # non-tensor payload (hooks, percentages, ...) -> do not cache
    return sd, {"format": CACHE_FORMAT, "none_keys": json.dumps(none_keys)}


def _unpack(sd, metadata):
    """Rebuild [[tensor, dict]], copying out of the safetensors mmap.

    The clone is not optional: safetensors hands back mmap-backed tensors, and on
    Windows a mapped file cannot be deleted or replaced. Holding those views for
    the length of a sampling run would block both eviction and rewrites of the
    entry.
    """
    if metadata is None or metadata.get("format") != CACHE_FORMAT or "cond" not in sd:
        return None
    extra = {}
    for k in json.loads(metadata.get("none_keys", "[]")):
        extra[k] = None
    for k, v in sd.items():
        if k.startswith("extra."):
            extra[k[len("extra."):]] = v.clone()
    return [[sd["cond"].clone(), extra]]


# ---------------------------------------------------------------- janitor

def _scan(directory):
    """(entries, temps) as (path, stat) pairs. Unreadable files are skipped.

    Files are recognised by the exact shape this module writes — 32 hex digits
    plus suffix — never by extension alone. The sweep deletes what it finds
    here, and `H3_COND_CACHE_DIR` pointed at the wrong directory must not turn
    that into a model shredder.
    """
    entries, temps = [], []
    try:
        with os.scandir(directory) as it:
            for e in it:
                try:
                    if not e.is_file():
                        continue
                    st = e.stat()
                except OSError:
                    continue
                if ENTRY_RE.fullmatch(e.name):
                    entries.append((e.path, st))
                elif TMP_RE.fullmatch(e.name):
                    temps.append((e.path, st))
    except OSError:
        pass
    return entries, temps


def _remove(path, size, freed):
    try:
        os.remove(path)
        freed[0] += size
        return True
    except OSError:
        # a live mmap on Windows, or a concurrent writer; the next sweep retries
        return False


# Deliberately no per-file ownership check before deleting. _claim() already
# established that this folder is ours and holds nothing else, so a file in it
# matching ENTRY_RE is one of ours — including a corrupt one, which is exactly
# the file a content check could not recognise and would therefore strand.


def sweep(directory=None):
    """Enforce the temp-file, age and size limits. Returns a one-line summary.

    Three separate ways the cache is kept bounded:

    - **stale temps** — a store writes `<digest>.<pid>.tmp` and then `os.replace`s
      it into position. A process killed in between (an OOM cascade takes the
      prompt worker with it) leaves an orphan that nothing else would ever look
      at, so anything older than an hour goes.
    - **age** — mtime is refreshed on every hit, so this expires entries by time
      since *last use*, not since creation. An entry used weekly never expires.
    - **size** — oldest-used first until under the cap.

    Every removal is individually best-effort; one failure never aborts the
    sweep.
    """
    if directory is None:
        directory = _cache_dir()
    if directory is None:
        return "disabled, no owned cache folder"
    entries, temps = _scan(directory)
    now = time.time()
    freed = [0]
    removed_temps = 0
    removed_aged = 0
    removed_capped = 0

    for path, st in temps:
        if now - st.st_mtime > STALE_TMP_SECONDS and _remove(path, st.st_size, freed):
            removed_temps += 1

    max_age = _max_age_seconds()
    kept = []
    for path, st in entries:
        if max_age > 0 and now - st.st_mtime > max_age and _remove(path, st.st_size, freed):
            removed_aged += 1
        else:
            kept.append((path, st))

    cap = _max_bytes()
    total = sum(st.st_size for _, st in kept)
    if total > cap:
        kept.sort(key=lambda f: f[1].st_mtime)  # oldest use first
        survivors = []
        for path, st in kept:
            if total > cap and _remove(path, st.st_size, freed):
                total -= st.st_size
                removed_capped += 1
            else:
                survivors.append((path, st))
        kept = survivors

    summary = "%d entries, %.1f GB of %.1f GB cap" % (len(kept), total / 1024 ** 3, cap / 1024 ** 3)
    removals = []
    if removed_capped:
        removals.append("%d over cap" % removed_capped)
    if removed_aged:
        removals.append("%d unused for over %.0f days" % (removed_aged, max_age / 86400))
    if removed_temps:
        removals.append("%d orphaned temp files" % removed_temps)
    if removals:
        summary += "; removed %s, freed %.1f GB" % (", ".join(removals), freed[0] / 1024 ** 3)
    return summary


def _sweep_once():
    """First-use sweep, so a restart tidies up even if nothing is ever stored."""
    global _swept
    if _swept:
        return
    _swept = True
    try:
        logging.info("%s: %s", LOG, sweep())
    except Exception:
        logging.exception("%s: startup sweep failed (the cache still works)", LOG)


def purge(directory=None):
    """Delete every entry and temp file. Returns (removed, bytes_freed).

    Only ever operates on a folder this cache owns; the marker file itself is
    left in place so the folder stays claimed.
    """
    if directory is None:
        directory = _cache_dir()
    if directory is None:
        return 0, 0
    entries, temps = _scan(directory)
    freed = [0]
    removed = sum(1 for path, st in entries
                  if _remove(path, st.st_size, freed))
    removed += sum(1 for path, st in temps if _remove(path, st.st_size, freed))
    logging.info("%s: purged %d files, freed %.1f GB", LOG, removed, freed[0] / 1024 ** 3)
    return removed, freed[0]


# ---------------------------------------------------------------- public

def encode(clip, tokens, mode="auto", label=None):
    """Drop-in for clip.encode_from_tokens_scheduled(tokens), cached to disk.

    mode: "auto" read+write, "off" bypass entirely, "refresh" re-encode and
    overwrite any existing entry. Falls back to a plain encode whenever the
    result cannot be keyed or stored with certainty.
    """
    def plain():
        return clip.encode_from_tokens_scheduled(tokens)

    if mode == "off" or _env_disabled():
        return plain()
    # LoRA-patched or hook-scheduled text encoders change the output without
    # changing the tokens; a content hash cannot see that.
    if getattr(clip.patcher, "patches", None):
        logging.info("%s: bypassed, text encoder has weight patches", LOG)
        return plain()
    if getattr(clip.patcher, "forced_hooks", None) is not None and getattr(clip, "use_clip_schedule", False):
        logging.info("%s: bypassed, hook schedule in use", LOG)
        return plain()

    try:
        t0 = time.perf_counter()
        digest = tokens_digest(clip, tokens)
        hash_s = time.perf_counter() - t0
    except Exception:
        logging.exception("%s: hashing failed, encoding normally", LOG)
        return plain()

    if digest is None:
        logging.info("%s: bypassed, unidentified text encoder or token payload", LOG)
        return plain()

    directory = _cache_dir()
    if directory is None:
        return plain()  # _claim() already explained why, once

    _sweep_once()
    path = os.path.join(directory, digest[:DIGEST_CHARS] + ENTRY_SUFFIX)

    if mode != "refresh" and os.path.exists(path):
        try:
            sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            cond = _unpack(sd, metadata)
            del sd  # drop the mmap views; see _unpack
            if cond is not None:
                try:
                    os.utime(path, None)  # refresh for LRU eviction
                except OSError:
                    pass
                logging.info("%s: hit %s, reusing %d Qwen3-VL tokens (hashed in %.2fs, "
                             "text encoder not loaded)", LOG, digest[:12], cond[0][0].shape[-2], hash_s)
                return cond
            logging.warning("%s: unreadable entry %s, re-encoding", LOG, digest[:12])
        except Exception:
            logging.exception("%s: failed to read %s, re-encoding", LOG, path)

    logging.info("%s: %s %s, running Qwen3-VL (hashed in %.2fs)", LOG,
                 "refreshing" if mode == "refresh" else "miss", digest[:12], hash_s)
    cond = plain()

    tmp = "%s.%d%s" % (path, os.getpid(), TMP_SUFFIX)
    try:
        packed = _pack(cond)
        if packed is None:
            logging.info("%s: result not representable, not stored", LOG)
            return cond
        sd, metadata = packed
        if label:
            metadata["label"] = str(label)[:512]
        metadata["created"] = str(int(time.time()))
        comfy.utils.save_torch_file(sd, tmp, metadata=metadata)
        os.replace(tmp, path)
        logging.info("%s: stored %s (%.0f MB)", LOG, digest[:12], os.path.getsize(path) / 1e6)
    except Exception:
        logging.exception("%s: failed to store entry (the conditioning is unaffected)", LOG)
        # a partial write would otherwise sit here until it aged out as stale
        if os.path.exists(tmp):
            _remove(tmp, 0, [0])

    try:
        # after every store attempt, not only successful ones
        logging.info("%s: %s", LOG, sweep(directory))
    except Exception:
        logging.exception("%s: sweep failed (the cache still works)", LOG)

    return cond
