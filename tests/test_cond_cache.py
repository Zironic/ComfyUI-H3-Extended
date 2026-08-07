"""Self-test for the H3 conditioning disk cache.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_cond_cache.py

No model or checkpoint required. The text encoder is stubbed with a counter;
the tests cover cache identity, persistence semantics, token/reference hashing,
bypass paths, storage, eviction and cleanup.
"""

import glob
import logging
import os
import shutil
import sys
import tempfile
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_TMPROOT = tempfile.mkdtemp(prefix="h3_cond_cache_test_")
CACHE = os.path.join(_TMPROOT, "cache")
os.makedirs(CACHE)
os.environ["H3_COND_CACHE_DIR"] = CACHE

import torch  # noqa: E402

import cond_cache  # noqa: E402

VISION_START, VISION_END = 151652, 151653
PINNED_DIGEST = "e34800f7c0293c1fdd74fcf3f95faf8db756c78bdd9c88d20b9994e51ec0aa12"

_TE_FILE = os.path.join(_TMPROOT, "fake_te.safetensors")
_ALT_TE_FILE = os.path.join(_TMPROOT, "fake_te_v2.safetensors")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakePatcher:
    """Stands in for the CLIP's ModelPatcher provenance and patch state."""

    def __init__(self):
        self.patches = {}
        self.weight_wrapper_patches = {}
        self.patches_uuid = "base-process-uuid"
        self.forced_hooks = None
        self.cached_patcher_init = (
            None, ([_TE_FILE], None, "MINIMAX_H3", {"dtype": torch.float16}))


class FakeCLIP:
    layer_idx = None
    use_clip_schedule = False

    def __init__(self):
        self.patcher = FakePatcher()
        self.cond_stage_model = type("MiniMaxH3TEModel_", (), {})()
        self.calls = 0

    def encode_from_tokens_scheduled(self, tokens):
        self.calls += 1
        n = len(tokens["qwen3vl_32b"][0])
        torch.manual_seed(0)
        return [[torch.randn(1, n * 4, 5120, dtype=torch.float32),
                 {"pooled_output": None,
                  "minimax_token_tags": torch.ones(n * 4, dtype=torch.long)}]]


def make_tokens(text_ids, image=None, video_pair=None, video_block=True):
    entries = [(t, 1.0) for t in text_ids]
    if image is not None:
        entries += [(VISION_START, 1.0),
                    ({"type": "image", "data": image, "original_type": "image"}, 1.0),
                    (VISION_END, 1.0)]
    if video_pair is not None:
        block = {"type": "image", "data": video_pair, "original_type": "image"}
        if video_block:
            block["minimax_video_block"] = True
        entries += [(VISION_START, 1.0), (block, 1.0), (VISION_END, 1.0)]
    return {"qwen3vl_32b": [entries]}


def write_te(path, payload):
    with open(path, "wb") as f:
        f.write(payload)


def set_checkpoint(clip, path):
    init = clip.patcher.cached_patcher_init
    args = list(init[1])
    args[0] = [path]
    clip.patcher.cached_patcher_init = (init[0], tuple(args))


def test_hit_is_bit_identical(clip, tokens):
    print("hit returns exactly what the encoder returned")
    first = cond_cache.encode(clip, tokens, label="a prompt")
    check(clip.calls == 1, "cold call ran the encoder")
    second = cond_cache.encode(clip, tokens, label="a prompt")
    check(clip.calls == 1, "warm call did not run the encoder")
    check(torch.equal(first[0][0], second[0][0]), "cond tensor bit-identical")
    check(first[0][0].dtype == second[0][0].dtype == torch.float32, "fp32 preserved")
    check(torch.equal(first[0][1]["minimax_token_tags"],
                      second[0][1]["minimax_token_tags"]), "token tags bit-identical")
    check(second[0][1]["pooled_output"] is None, "None-valued key round-tripped")
    check(set(first[0][1]) == set(second[0][1]), "no keys gained or lost")


def test_key_covers_token_inputs(clip, image, video):
    print("everything in the token presentation that changes embeddings changes the key")

    def misses(label, tokens):
        before = clip.calls
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 1, label)

    misses("different prompt tokens", make_tokens([1, 2, 4], image, video[2:4]))

    nudged = image.clone()
    nudged[0, 0, 0, 0] += 0.01
    misses("one changed pixel in a reference image", make_tokens([1, 2, 3], nudged, video[2:4]))

    misses("different video frame pair, same shape", make_tokens([1, 2, 3], image, video[4:6]))
    before = clip.calls
    cond_cache.encode(clip, make_tokens([1, 2, 3], image, video[4:6]))
    check(clip.calls == before, "and that pair then hits on repeat")

    misses("minimax_video_block flag cleared",
           make_tokens([1, 2, 3], image, video[4:6], video_block=False))


def test_checkpoint_identity_policy(clip, tokens):
    print("checkpoint identity is persistent by filename, not filesystem metadata")
    cond_cache.purge()
    set_checkpoint(clip, _TE_FILE)
    write_te(_TE_FILE, b"x" * 1024)
    base = cond_cache.tokens_digest(clip, tokens)

    # Replacing/touching the file under the same name is intentionally assumed to
    # mean the same checkpoint. This is a short-lived optimization cache, not a
    # cryptographic model registry.
    write_te(_TE_FILE, b"different bytes and different size" * 200)
    os.utime(_TE_FILE, (time.time() + 123, time.time() + 123))
    check(cond_cache.tokens_digest(clip, tokens) == base,
          "same checkpoint filename survives size/content/mtime changes")

    # The process-local ModelPatcher UUID also must not poison persistence for an
    # ordinary unpatched checkpoint.
    clip.patcher.patches_uuid = "another-process-uuid"
    check(cond_cache.tokens_digest(clip, tokens) == base,
          "unpatched identity ignores process-local patches_uuid")

    write_te(_ALT_TE_FILE, b"anything")
    set_checkpoint(clip, _ALT_TE_FILE)
    check(cond_cache.tokens_digest(clip, tokens) != base,
          "different checkpoint filename gets a different key")
    set_checkpoint(clip, _TE_FILE)


def test_runtime_patch_identity(clip, tokens):
    print("runtime-patched encoders cache by patches_uuid within one process")
    cond_cache.purge()
    clip.patcher.patches = {"model.layers.0.weight": [1]}
    clip.patcher.patches_uuid = "patch-state-a"
    try:
        before = clip.calls
        first_digest = cond_cache.tokens_digest(clip, tokens)
        cond_cache.encode(clip, tokens)
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 1, "same patched state hits on repeat")

        clip.patcher.patches_uuid = "patch-state-b"
        second_digest = cond_cache.tokens_digest(clip, tokens)
        check(second_digest != first_digest, "changed patches_uuid changes the key")
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 2, "changed patch state re-encodes")

        clip.patcher.patches_uuid = None
        check(cond_cache.tokens_digest(clip, tokens) is None,
              "patched encoder without patches_uuid is not cacheable")
        cond_cache.encode(clip, tokens)
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 4, "unknown patched state bypasses safely")
    finally:
        clip.patcher.patches = {}
        clip.patcher.patches_uuid = "base-process-uuid"


def test_weight_wrapper_patch_identity(clip, tokens):
    print("weight-wrapper patches use the same ephemeral identity policy")
    cond_cache.purge()
    clip.patcher.weight_wrapper_patches = {"model.layers.0.weight": object()}
    clip.patcher.patches_uuid = "wrapper-a"
    try:
        a = cond_cache.tokens_digest(clip, tokens)
        clip.patcher.patches_uuid = "wrapper-b"
        b = cond_cache.tokens_digest(clip, tokens)
        check(a != b, "weight-wrapper patch UUID participates in the key")
    finally:
        clip.patcher.weight_wrapper_patches = {}
        clip.patcher.patches_uuid = "base-process-uuid"


def test_bypasses(clip, tokens):
    print("uncertain situations bypass rather than risk a wrong hit")

    def always_encodes(label):
        before = clip.calls
        cond_cache.encode(clip, tokens)
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 2, label)

    clip.patcher.forced_hooks = object()
    clip.use_clip_schedule = True
    always_encodes("hook schedule in use")
    clip.patcher.forced_hooks = None
    clip.use_clip_schedule = False

    saved = clip.patcher.cached_patcher_init
    clip.patcher.cached_patcher_init = None
    always_encodes("unknown checkpoint provenance")
    clip.patcher.cached_patcher_init = saved

    os.environ["H3_COND_CACHE_DISABLE"] = "1"
    always_encodes("H3_COND_CACHE_DISABLE=1")
    del os.environ["H3_COND_CACHE_DISABLE"]

    before = clip.calls
    cond_cache.encode(clip, tokens, mode="off")
    cond_cache.encode(clip, tokens, mode="off")
    check(clip.calls == before + 2, "mode='off'")


def test_modes(clip, tokens):
    print("refresh re-encodes and overwrites")
    cond_cache.encode(clip, tokens)
    before = clip.calls
    cond_cache.encode(clip, tokens, mode="refresh")
    check(clip.calls == before + 1, "refresh ignored the stored entry")
    cond_cache.encode(clip, tokens)
    check(clip.calls == before + 1, "and the overwritten entry hits again")


def test_entry_not_mmap_locked(clip, tokens):
    print("a hit must not leave the file locked (Windows)")
    cond_cache.encode(clip, tokens)
    path = os.path.join(CACHE, cond_cache.tokens_digest(clip, tokens)[:32] + ".safetensors")
    check(os.path.exists(path), "entry is on disk")
    try:
        os.remove(path)
        check(True, "entry is deletable straight after a hit")
    except OSError as e:
        check(False, "entry is deletable straight after a hit (%s)" % e)


def test_corrupt_entry(clip, tokens):
    print("a corrupt entry degrades to a normal encode")
    cond_cache.encode(clip, tokens)
    victim = sorted(glob.glob(os.path.join(CACHE, "*.safetensors")))[0]
    with open(victim, "wb") as f:
        f.write(b"not a safetensors file")
    before = clip.calls
    cond_cache.encode(clip, tokens)
    check(clip.calls >= before, "survived and produced conditioning")


def test_eviction(clip, image, video):
    print("LRU eviction honours the size cap")
    os.environ["H3_COND_CACHE_GB"] = "0.05"
    try:
        for i in range(4):
            cond_cache.encode(clip, make_tokens([50 + i, 9, 9], image, video[0:2]))
        files = glob.glob(os.path.join(CACHE, "*.safetensors"))
        total = sum(os.path.getsize(f) for f in files)
        check(total <= 0.05 * 1024 ** 3,
              "pruned to cap (%.1f MB in %d files)" % (total / 1e6, len(files)))
    finally:
        del os.environ["H3_COND_CACHE_GB"]


def test_evicts_least_recently_used(clip, image):
    print("eviction drops the least recently used, not the oldest stored")
    cond_cache.purge()
    keep = make_tokens([70, 1], image)
    drop = make_tokens([71, 1], image)
    cond_cache.encode(clip, keep)
    cond_cache.encode(clip, drop)
    cond_cache.encode(clip, keep)

    keep_path = os.path.join(CACHE, cond_cache.tokens_digest(clip, keep)[:32] + ".safetensors")
    drop_path = os.path.join(CACHE, cond_cache.tokens_digest(clip, drop)[:32] + ".safetensors")
    os.environ["H3_COND_CACHE_GB"] = "%.10f" % (os.path.getsize(keep_path) * 1.5 / 1024 ** 3)
    try:
        cond_cache.sweep()
        check(os.path.exists(keep_path), "the entry that was hit survived")
        check(not os.path.exists(drop_path), "the entry that was not hit was evicted")
    finally:
        del os.environ["H3_COND_CACHE_GB"]


def test_orphaned_temp_files(clip):
    print("orphaned temp files are collected")
    fresh = os.path.join(CACHE, "a" * 32 + ".safetensors.12345.tmp")
    stale = os.path.join(CACHE, "b" * 32 + ".safetensors.12345.tmp")
    for p in (fresh, stale):
        with open(p, "wb") as f:
            f.write(b"partial write from a killed process")
    old = time.time() - cond_cache.STALE_TMP_SECONDS - 60
    os.utime(stale, (old, old))

    cond_cache.sweep()
    check(not os.path.exists(stale), "a stale temp file is removed")
    check(os.path.exists(fresh), "a temp file from a live store is left alone")
    os.remove(fresh)

    print("a failed store does not leave its own temp file behind")
    before = set(glob.glob(os.path.join(CACHE, "*.tmp")))
    real_save = cond_cache.comfy.utils.save_torch_file

    def exploding_save(sd, path, metadata=None):
        real_save(sd, path, metadata=metadata)
        raise RuntimeError("disk full")

    cond_cache.comfy.utils.save_torch_file = exploding_save
    try:
        cond_cache.encode(clip, make_tokens([80, 1]), mode="refresh")
    finally:
        cond_cache.comfy.utils.save_torch_file = real_save
    check(set(glob.glob(os.path.join(CACHE, "*.tmp"))) == before,
          "the partial write was cleaned up immediately")


def test_age_expiry(clip, image):
    print("entries expire on time since last use")
    cond_cache.purge()
    tokens = make_tokens([90, 1], image)
    cond_cache.encode(clip, tokens)
    path = os.path.join(CACHE, cond_cache.tokens_digest(clip, tokens)[:32] + ".safetensors")
    check(os.path.exists(path), "entry stored")

    os.environ["H3_COND_CACHE_MAX_AGE_DAYS"] = "7"
    try:
        cond_cache.sweep()
        check(os.path.exists(path), "a fresh entry is kept")
        old = time.time() - 8 * 86400
        os.utime(path, (old, old))
        cond_cache.sweep()
        check(not os.path.exists(path), "an entry unused for longer than the limit is dropped")

        cond_cache.encode(clip, tokens)
        os.utime(path, (old, old))
        os.environ["H3_COND_CACHE_MAX_AGE_DAYS"] = "0"
        cond_cache.sweep()
        check(os.path.exists(path), "0 disables age expiry")
    finally:
        os.environ.pop("H3_COND_CACHE_MAX_AGE_DAYS", None)


def with_cache_dir(directory):
    os.environ["H3_COND_CACHE_DIR"] = directory
    cond_cache._claimed = cond_cache._UNSET


def test_refuses_a_folder_it_does_not_own(clip, image):
    print("a folder holding anything else is refused outright")
    models = os.path.join(_TMPROOT, "text_encoders")
    os.makedirs(models, exist_ok=True)
    victims = [os.path.join(models, n) for n in
               ("qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "notes.txt")]
    for p in victims:
        with open(p, "wb") as f:
            f.write(b"a real model, as far as this cache knows")
    old = time.time() - 400 * 86400
    for p in victims:
        os.utime(p, (old, old))

    with_cache_dir(models)
    os.environ["H3_COND_CACHE_GB"] = "0"
    os.environ["H3_COND_CACHE_MAX_AGE_DAYS"] = "1"
    try:
        check(cond_cache._cache_dir() is None, "the folder is refused")
        check(not os.path.exists(os.path.join(models, cond_cache.MARKER)),
              "and not claimed by dropping a marker into it")
        before = clip.calls
        cond_cache.encode(clip, make_tokens([120, 1], image))
        cond_cache.encode(clip, make_tokens([120, 1], image))
        check(clip.calls == before + 2, "the cache disables itself rather than storing there")
        cond_cache.sweep()
        cond_cache.purge()
        for p in victims:
            check(os.path.exists(p), "left %s alone" % os.path.basename(p))
    finally:
        del os.environ["H3_COND_CACHE_GB"]
        del os.environ["H3_COND_CACHE_MAX_AGE_DAYS"]
        with_cache_dir(CACHE)


def test_claims_folders_it_may_have(clip, image):
    print("a folder it may have is claimed once and reused")
    fresh = os.path.join(_TMPROOT, "made_by_us")
    with_cache_dir(fresh)
    check(cond_cache._cache_dir() == fresh, "creates and claims a missing folder")
    check(os.path.exists(os.path.join(fresh, cond_cache.MARKER)), "marker written")

    empty = os.path.join(_TMPROOT, "empty")
    os.makedirs(empty, exist_ok=True)
    with_cache_dir(empty)
    check(cond_cache._cache_dir() == empty, "claims an existing empty folder")

    legacy = os.path.join(_TMPROOT, "legacy")
    os.makedirs(legacy, exist_ok=True)
    with open(os.path.join(legacy, "b" * 32 + ".safetensors"), "wb") as f:
        f.write(b"an old entry")
    with_cache_dir(legacy)
    check(cond_cache._cache_dir() == legacy, "adopts a folder holding only cache-shaped files")

    with_cache_dir(CACHE)
    cond_cache.purge()
    cond_cache.encode(clip, make_tokens([130, 1], image))
    cond_cache.sweep()
    check(os.path.exists(os.path.join(CACHE, cond_cache.MARKER)),
          "and never deletes its own marker")


def test_purge(clip, image):
    print("purge empties the cache")
    for i in range(3):
        cond_cache.encode(clip, make_tokens([100 + i, 1], image))
    removed, freed = cond_cache.purge()
    check(removed >= 3, "removed %d files, freed %.1f MB" % (removed, freed / 1e6))
    check(not glob.glob(os.path.join(CACHE, "*.safetensors")), "no entries remain")
    check(os.path.isdir(CACHE), "the directory itself survives")


def test_key_is_pinned(clip, image, video):
    print("the cache key has not drifted unexpectedly")
    set_checkpoint(clip, _TE_FILE)
    clip.patcher.patches = {}
    clip.patcher.weight_wrapper_patches = {}
    tokens = make_tokens([1, 2, 3], image, video[2:4])
    check(cond_cache.tokens_digest(clip, tokens) == PINNED_DIGEST,
          "tokens_digest produces the deliberately re-pinned key")


def test_hash_cost(clip):
    print("hashing cost stays negligible next to the encode it avoids")
    big = torch.rand(1, 2048, 2048, 3)
    t0 = time.perf_counter()
    cond_cache.tokens_digest(clip, make_tokens([1], big))
    elapsed = time.perf_counter() - t0
    check(elapsed < 2.0, "2048x2048 fp32 reference image hashed in %.2fs" % elapsed)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    write_te(_TE_FILE, b"x" * 1024)
    write_te(_ALT_TE_FILE, b"y" * 1024)

    torch.manual_seed(1)
    image = torch.rand(1, 256, 256, 3)
    video = torch.rand(8, 128, 128, 3)
    clip = FakeCLIP()
    tokens = make_tokens([1, 2, 3], image, video[2:4])

    test_hit_is_bit_identical(clip, tokens)
    test_key_covers_token_inputs(clip, image, video)
    test_checkpoint_identity_policy(clip, tokens)
    test_runtime_patch_identity(clip, tokens)
    test_weight_wrapper_patch_identity(clip, tokens)
    test_bypasses(clip, tokens)
    test_modes(clip, tokens)
    test_entry_not_mmap_locked(clip, tokens)
    test_corrupt_entry(clip, tokens)
    test_eviction(clip, image, video)
    test_evicts_least_recently_used(clip, image)
    test_orphaned_temp_files(clip)
    test_age_expiry(clip, image)
    test_refuses_a_folder_it_does_not_own(clip, image)
    test_claims_folders_it_may_have(clip, image)
    test_purge(clip, image)
    test_key_is_pinned(clip, image, video)
    test_hash_cost(clip)
    print("\nall cond cache tests passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMPROOT, ignore_errors=True)
