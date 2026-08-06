"""Production diagnostics for the H3 conditioning cache.

This module deliberately wraps, rather than replaces, ``cond_cache.encode``.
The cache's key, lookup, persistence and fallback behaviour therefore remain
unchanged.  The wrapper exposes enough component fingerprints to explain why
two apparently identical graph executions do or do not address the same entry.
"""

import hashlib
import logging
import os
import threading

import numpy as np
import torch

try:
    from . import cond_cache
except ImportError:  # the self-tests import this file as a top-level module
    import cond_cache

LOG = cond_cache.LOG
_SHORT = 12
_last_by_label = {}
_last_lock = threading.Lock()

# Captured at import, before anything can install this wrapper over the name it
# delegates to. Without this, installing at the `cond_cache.encode` seam — the
# only seam that also catches callers doing a function-local
# `from ..cond_cache import encode` — would recurse forever.
_REAL_ENCODE = cond_cache.encode


def install():
    """Route every caller of the cache through these diagnostics.

    Patching `cond_cache.encode` itself is what makes this reach callers that
    import the function inside a function body, which module-level rebinding of
    an already-imported name cannot do.
    """
    cond_cache.encode = encode
    return encode


def _digest_bytes(data):
    h = hashlib.blake2b(digest_size=16)
    h.update(data)
    return h.hexdigest()[:_SHORT]


def _digest_value(value):
    h = hashlib.blake2b(digest_size=16)
    if torch.is_tensor(value):
        cond_cache._hash_tensor(h, value)
        return h.hexdigest()[:_SHORT]
    h.update(cond_cache._stable_repr(value).encode())
    return h.hexdigest()[:_SHORT]


def _label_key(label):
    return _digest_bytes(str(label or "<no label>").encode("utf-8", "replace"))


def _describe_tensor(tensor):
    return "%s:%s:%s" % (tuple(tensor.shape), tensor.dtype, tensor.device)


def inspect_key(clip, tokens):
    """Return component fingerprints without changing the real cache key."""
    model_fp = cond_cache._model_fingerprint(clip)
    model_digest = None if model_fp is None else _digest_bytes(model_fp.encode())

    text_h = hashlib.blake2b(digest_size=16)
    vision = []
    unknown = []
    entry_index = 0

    try:
        for token_key in sorted(tokens):
            text_h.update(("key|%s" % token_key).encode())
            for batch_index, batch in enumerate(tokens[token_key]):
                text_h.update(("batch|%d" % batch_index).encode())
                for entry in batch:
                    token, weight = entry[0], entry[1]
                    text_h.update(("entry|%d|w|%r|" % (entry_index, float(weight))).encode())
                    if isinstance(token, (int, np.integer)):
                        text_h.update(("i|%d" % int(token)).encode())
                    elif torch.is_tensor(token):
                        digest = _digest_value(token)
                        vision.append({
                            "index": entry_index,
                            "kind": "tensor",
                            "digest": digest,
                            "detail": _describe_tensor(token),
                        })
                        text_h.update(("tensor-slot|%d" % entry_index).encode())
                    elif isinstance(token, dict):
                        block_h = hashlib.blake2b(digest_size=16)
                        tensor_details = []
                        for key in sorted(token):
                            value = token[key]
                            block_h.update(("|%s=" % key).encode())
                            if torch.is_tensor(value):
                                cond_cache._hash_tensor(block_h, value)
                                tensor_details.append("%s=%s" % (key, _describe_tensor(value)))
                            else:
                                block_h.update(cond_cache._stable_repr(value).encode())
                        vision.append({
                            "index": entry_index,
                            "kind": str(token.get("type", "dict")),
                            "digest": block_h.hexdigest()[:_SHORT],
                            "detail": ",".join(tensor_details) or "no tensors",
                        })
                        # Keep text fingerprint sensitive to where a visual block occurs,
                        # but not to its contents; content differences then appear under
                        # the individual vision component instead of looking like text.
                        text_h.update(("dict-slot|%d|%s" %
                                       (entry_index, token.get("type", "dict"))).encode())
                    else:
                        name = type(token).__name__
                        unknown.append("entry %d: %s" % (entry_index, name))
                        text_h.update(("unknown|%s" % name).encode())
                    entry_index += 1
    except Exception as exc:
        unknown.append("inspection raised %s: %s" % (type(exc).__name__, exc))

    try:
        final_digest = cond_cache.tokens_digest(clip, tokens)
    except Exception as exc:
        final_digest = None
        unknown.append("real key raised %s: %s" % (type(exc).__name__, exc))

    return {
        "model": model_digest,
        "text": text_h.hexdigest()[:_SHORT],
        "vision": vision,
        "unknown": unknown,
        "final": None if final_digest is None else final_digest[:_SHORT],
    }


def _changes(previous, current):
    if previous is None:
        return "first observation for this label"
    changed = []
    if previous["model"] != current["model"]:
        changed.append("model %s->%s" % (previous["model"], current["model"]))
    if previous["text"] != current["text"]:
        changed.append("text %s->%s" % (previous["text"], current["text"]))

    old = previous["vision"]
    new = current["vision"]
    if len(old) != len(new):
        changed.append("vision-count %d->%d" % (len(old), len(new)))
    for i in range(min(len(old), len(new))):
        if old[i]["digest"] != new[i]["digest"]:
            changed.append("vision[%d]@entry%d %s->%s" % (
                i, new[i]["index"], old[i]["digest"], new[i]["digest"]))
        elif old[i]["detail"] != new[i]["detail"]:
            changed.append("vision[%d]-layout %s->%s" % (
                i, old[i]["detail"], new[i]["detail"]))
    if previous["final"] != current["final"] and not changed:
        changed.append("final-only %s->%s" % (previous["final"], current["final"]))
    return ", ".join(changed) if changed else "no component changed"


def _log_bypass_context(clip, mode):
    if mode == "off":
        logging.info("%s diagnostic: bypass mode=off", LOG)
        return True
    if cond_cache._env_disabled():
        logging.info("%s diagnostic: bypass %s is enabled", LOG,
                     cond_cache._ENV_DISABLE)
        return True
    patches = getattr(clip.patcher, "patches", None)
    if patches:
        names = sorted(str(k) for k in patches)
        logging.info("%s diagnostic: bypass weight patches count=%d first=%s",
                     LOG, len(names), names[:5])
        return True
    forced_hooks = getattr(clip.patcher, "forced_hooks", None)
    if forced_hooks is not None and getattr(clip, "use_clip_schedule", False):
        logging.info("%s diagnostic: bypass hook schedule type=%s",
                     LOG, type(forced_hooks).__name__)
        return True
    return False


def encode(clip, tokens, mode="auto", label=None):
    """Log component-level key evidence, then run the unmodified cache."""
    if _log_bypass_context(clip, mode):
        return _REAL_ENCODE(clip, tokens, mode=mode, label=label)

    diagnostic = inspect_key(clip, tokens)
    label_key = _label_key(label)
    with _last_lock:
        previous = _last_by_label.get(label_key)
        _last_by_label[label_key] = diagnostic

    directory = None
    path = None
    exists = False
    if diagnostic["final"] is not None:
        directory = cond_cache._cache_dir()
        if directory is not None:
            # inspect_key exposes the abbreviated digest for logs, while the real
            # path must use the complete cache digest.
            full_digest = cond_cache.tokens_digest(clip, tokens)
            path = os.path.join(directory,
                                full_digest[:cond_cache.DIGEST_CHARS] + cond_cache.ENTRY_SUFFIX)
            exists = os.path.exists(path)

    logging.info(
        "%s diagnostic: label=%s mode=%s final=%s model=%s text=%s "
        "vision_blocks=%d path_exists=%s changes=[%s]",
        LOG, label_key, mode, diagnostic["final"], diagnostic["model"],
        diagnostic["text"], len(diagnostic["vision"]), exists,
        _changes(previous, diagnostic))

    for i, block in enumerate(diagnostic["vision"]):
        logging.info(
            "%s diagnostic: label=%s vision[%d] entry=%d kind=%s digest=%s %s",
            LOG, label_key, i, block["index"], block["kind"],
            block["digest"], block["detail"])

    if diagnostic["unknown"]:
        logging.warning("%s diagnostic: label=%s unknown=%s", LOG, label_key,
                        "; ".join(diagnostic["unknown"]))
    if diagnostic["final"] is None:
        init = getattr(clip.patcher, "cached_patcher_init", None)
        logging.warning(
            "%s diagnostic: label=%s real key unavailable; model_fingerprint=%s "
            "cached_patcher_init=%s token_keys=%s",
            LOG, label_key, diagnostic["model"],
            "missing" if init is None else "present(len=%d)" % len(init),
            sorted(tokens) if isinstance(tokens, dict) else type(tokens).__name__)
    elif path is not None:
        logging.info("%s diagnostic: label=%s cache_path=%s", LOG, label_key, path)

    return _REAL_ENCODE(clip, tokens, mode=mode, label=label)
