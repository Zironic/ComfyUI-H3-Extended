"""Disk cache for the VAE encodes that feed H3 reference conditioning.

The conditioning cache is consulted *after* every reference has already been
through the VAE, because its key is built from the tokenizer presentation and
that is assembled alongside the latents. So a conditioning hit still paid for
the whole VAE pass — and the reference latents themselves were never cached at
all, since they reach the DiT as `minimax_refs` rather than as part of the
encoder output.

Caching at the VAE call puts the check before the work: hash the resized pixels,
and on a hit skip both the encode and staging the ~5 GB video VAE onto a 12 GB
card. A run whose references are unchanged then touches neither VAE nor Qwen.

Safe to cache because the H3 VAEs are deterministic. `MiniMaxH3VideoVAE.encode`
takes `torch.chunk(moments, 2)[0]` — the posterior mean — and the audio VAE
documents the same ("the returned posterior mean is used directly (no
sampling)"). Neither draws from the distribution, so a cached latent is the
value a re-encode would have produced, not merely an equally valid sample.

Entries share the conditioning cache's owned folder and janitor: same marker,
same age and size limits, same LRU. They are small next to a conditioning entry
— roughly 80 KB for a reference image and 2 MB for a 73-frame clip, against
~40 MB for the Qwen hidden states.

Env: H3_LATENT_CACHE_DISABLE=1 turns this off on its own;
H3_COND_CACHE_DISABLE=1 turns off both caches.
"""

import hashlib
import logging
import os
import time

import torch

import comfy.utils

try:
    from . import cond_cache
except ImportError:  # the self-tests import this file as a top-level module
    import cond_cache

LATENT_FORMAT = "h3_latent_cache/1"
LOG = "[H3 Extended] latent cache"
_ENV_DISABLE = "H3_LATENT_CACHE_DISABLE"


def _disabled():
    if os.environ.get(_ENV_DISABLE, "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    return cond_cache._env_disabled()  # the conditioning switch is a master switch


def _vae_fingerprint(vae):
    """Identify the VAE weights, or None if provenance is unknown.

    VAELoader records `(load_vae_patcher, (vae_path, metadata, device))` on the
    patcher, and a checkpoint-loaded VAE records the checkpoint path the same
    way — so, as with the text encoder, the file identifies the weights and
    nothing has to hash them.
    """
    patcher = getattr(vae, "patcher", None)
    init = getattr(patcher, "cached_patcher_init", None)
    if not init or len(init) < 2 or not init[1]:
        return None
    args = init[1]
    path = args[0]
    if not isinstance(path, str) or not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    return "\n".join([
        LATENT_FORMAT,
        type(getattr(vae, "first_stage_model", vae)).__name__,
        "%s|%d|%d" % (os.path.basename(path), st.st_size, int(st.st_mtime)),
        # metadata can change how the VAE is built (e.g. tae_latent_channels)
        cond_cache._stable_repr(list(args[1:])),
    ])


def key(vae, pixels):
    """blake2b over the VAE identity and the exact pixels, or None to bypass."""
    fingerprint = _vae_fingerprint(vae)
    if fingerprint is None:
        return None
    h = hashlib.blake2b(digest_size=32)
    h.update(fingerprint.encode())
    cond_cache._hash_tensor(h, pixels)
    return h.hexdigest()


def encode(vae, pixels, mode="auto", label=None):
    """Drop-in for `vae.encode(pixels)`, cached to disk.

    Falls back to a plain encode whenever the result cannot be keyed or stored.
    """
    def plain():
        return vae.encode(pixels)

    if mode == "off" or _disabled() or not torch.is_tensor(pixels):
        return plain()

    try:
        t0 = time.perf_counter()
        digest = key(vae, pixels)
        hash_s = time.perf_counter() - t0
    except Exception:
        logging.exception("%s: hashing failed, encoding normally", LOG)
        return plain()

    if digest is None:
        logging.info("%s: bypassed, unidentified VAE", LOG)
        return plain()

    directory = cond_cache._cache_dir()
    if directory is None:
        return plain()  # cond_cache already explained why, once

    shape = "x".join(str(d) for d in pixels.shape)
    path = os.path.join(directory, digest[:cond_cache.DIGEST_CHARS] + cond_cache.ENTRY_SUFFIX)

    if mode != "refresh" and os.path.exists(path):
        try:
            sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
            latent = sd.get("latent")
            ok = metadata is not None and metadata.get("format") == LATENT_FORMAT
            latent = latent.clone() if (ok and latent is not None) else None
            del sd  # drop the mmap views, as the conditioning cache does
            if latent is not None:
                try:
                    os.utime(path, None)  # refresh for LRU eviction
                except OSError:
                    pass
                output_device = getattr(vae, "output_device", None)
                if output_device is not None:
                    latent = latent.to(output_device)
                logging.info("%s: hit %s for %s (hashed in %.2fs, VAE not run)",
                             LOG, digest[:12], shape, hash_s)
                return latent
            logging.warning("%s: unreadable entry %s, re-encoding", LOG, digest[:12])
        except Exception:
            logging.exception("%s: failed to read %s, re-encoding", LOG, path)

    logging.info("%s: %s %s for %s, running the VAE (hashed in %.2fs)", LOG,
                 "refreshing" if mode == "refresh" else "miss", digest[:12], shape, hash_s)
    latent = plain()

    tmp = "%s.%d%s" % (path, os.getpid(), cond_cache.TMP_SUFFIX)
    try:
        if not torch.is_tensor(latent):
            return latent
        metadata = {"format": LATENT_FORMAT, "pixels": shape}
        if label:
            metadata["label"] = str(label)[:256]
        metadata["created"] = str(int(time.time()))
        comfy.utils.save_torch_file({"latent": latent.contiguous()}, tmp, metadata=metadata)
        os.replace(tmp, path)
        logging.info("%s: stored %s (%.1f MB)", LOG, digest[:12], os.path.getsize(path) / 1e6)
    except Exception:
        logging.exception("%s: failed to store entry (the latent is unaffected)", LOG)
        if os.path.exists(tmp):
            cond_cache._remove(tmp, 0, [0])

    try:
        logging.info("%s: %s", LOG, cond_cache.sweep(directory))
    except Exception:
        logging.exception("%s: sweep failed (the cache still works)", LOG)

    return latent
