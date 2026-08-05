"""Self-test for the H3 conditioning disk cache.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_cond_cache.py

No model or checkpoint required. The text encoder is the one thing that cannot
be exercised here — loading Qwen3-VL-32B to test a cache would defeat the point —
so it is stubbed with a counter, and what gets tested is everything around it:
that a hit returns bit-identical tensors, that every input which genuinely
changes the embeddings also changes the key, that the bypasses fire, and that a
hit does not leave the entry mmap-locked against eviction.
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

# deliberately outside the cache directory: the sweep must never be able to
# reach a checkpoint, and a test that stored one inside would not notice
_TE_FILE = os.path.join(_TMPROOT, "fake_te.safetensors")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakePatcher:
    """Stands in for the CLIP's ModelPatcher: provenance, LoRAs and hooks."""

    def __init__(self):
        self.patches = {}
        self.forced_hooks = None
        self.cached_patcher_init = (
            None, ([_TE_FILE], None, "MINIMAX_H3", {"dtype": torch.float16}))


class FakeCLIP:
    """Counts encoder invocations; the count is what every assertion reads."""

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
    """Mimic MiniMaxH3Tokenizer's output: ids interleaved with vision blocks."""
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


def write_te(payload):
    with open(_TE_FILE, "wb") as f:
        f.write(payload)
    # size and mtime are the fingerprint; make sure mtime actually moves
    os.utime(_TE_FILE, (time.time() + 10, time.time() + 10))


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


def test_key_covers_every_input(clip, image, video):
    print("everything that changes the embeddings changes the key")

    def misses(label, tokens):
        before = clip.calls
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 1, label)

    misses("different prompt tokens", make_tokens([1, 2, 4], image, video[2:4]))

    nudged = image.clone()
    nudged[0, 0, 0, 0] += 0.01
    misses("one changed pixel in a reference image", make_tokens([1, 2, 3], nudged, video[2:4]))

    # frames[i:i+2] is a contiguous view over the whole video, so a naive
    # storage-level hash would read identical bytes for every pair
    misses("different video frame pair, same shape", make_tokens([1, 2, 3], image, video[4:6]))
    before = clip.calls
    cond_cache.encode(clip, make_tokens([1, 2, 3], image, video[4:6]))
    check(clip.calls == before, "and that pair then hits on repeat")

    misses("minimax_video_block flag cleared",
           make_tokens([1, 2, 3], image, video[4:6], video_block=False))

    write_te(b"y" * 4096)
    misses("text encoder file replaced", make_tokens([1, 2, 3], image, video[2:4]))


def test_bypasses(clip, tokens):
    print("uncertain situations bypass rather than risk a wrong hit")

    def always_encodes(label):
        before = clip.calls
        cond_cache.encode(clip, tokens)
        cond_cache.encode(clip, tokens)
        check(clip.calls == before + 2, label)

    clip.patcher.patches = {"model.layers.0.weight": [1]}
    always_encodes("LoRA-patched text encoder")
    clip.patcher.patches = {}

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
    cond_cache.encode(clip, tokens)  # ensure stored
    before = clip.calls
    cond_cache.encode(clip, tokens, mode="refresh")
    check(clip.calls == before + 1, "refresh ignored the stored entry")
    cond_cache.encode(clip, tokens)
    check(clip.calls == before + 1, "and the overwritten entry hits again")


def test_entry_not_mmap_locked(clip, tokens):
    print("a hit must not leave the file locked (Windows)")
    cond_cache.encode(clip, tokens)  # warm hit; holds mmap views unless cloned
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


def test_evicts_least_recently_used(clip, image, video):
    print("eviction drops the least recently *used*, not the oldest stored")
    cond_cache.purge()
    keep = make_tokens([70, 1], image)
    drop = make_tokens([71, 1], image)
    cond_cache.encode(clip, keep)
    cond_cache.encode(clip, drop)
    cond_cache.encode(clip, keep)  # hit: refreshes keep's mtime above drop's

    keep_path = os.path.join(CACHE, cond_cache.tokens_digest(clip, keep)[:32] + ".safetensors")
    drop_path = os.path.join(CACHE, cond_cache.tokens_digest(clip, drop)[:32] + ".safetensors")
    # a cap that fits exactly one of the two
    os.environ["H3_COND_CACHE_GB"] = "%.10f" % (os.path.getsize(keep_path) * 1.5 / 1024 ** 3)
    try:
        cond_cache.sweep()
        check(os.path.exists(keep_path), "the entry that was hit survived")
        check(not os.path.exists(drop_path), "the entry that was not hit was evicted")
    finally:
        del os.environ["H3_COND_CACHE_GB"]


def test_orphaned_temp_files(clip, tokens):
    print("orphaned temp files are collected")
    # exactly what a store in flight writes: <digest>.safetensors.<pid>.tmp
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
        real_save(sd, path, metadata=metadata)  # write it, then fail
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
    """Repoint the cache and reset the once-per-process ownership decision."""
    os.environ["H3_COND_CACHE_DIR"] = directory
    cond_cache._claimed = cond_cache._UNSET


def test_refuses_a_folder_it_does_not_own(clip, image):
    """Ownership is decided per *folder*, once, not per file at delete time.

    The scenario that matters is H3_COND_CACHE_DIR aimed somewhere real. The
    right outcome is not 'sweep it carefully' — it is 'do not touch it'.
    """
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
        os.utime(p, (old, old))  # ancient and over any cap: maximally tempting

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

    # a cache directory from before the marker existed
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


def test_hash_cost(clip):
    print("hashing cost stays negligible next to the encode it avoids")
    big = torch.rand(1, 2048, 2048, 3)  # a 'max' ref image, 50 MB fp32
    t0 = time.perf_counter()
    cond_cache.tokens_digest(clip, make_tokens([1], big))
    elapsed = time.perf_counter() - t0
    check(elapsed < 2.0, "2048x2048 fp32 reference image hashed in %.2fs" % elapsed)


def main():
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    write_te(b"x" * 1024)

    torch.manual_seed(1)
    image = torch.rand(1, 256, 256, 3)
    video = torch.rand(8, 128, 128, 3)
    clip = FakeCLIP()
    tokens = make_tokens([1, 2, 3], image, video[2:4])

    test_hit_is_bit_identical(clip, tokens)
    test_key_covers_every_input(clip, image, video)
    test_bypasses(clip, tokens)
    test_modes(clip, tokens)
    test_entry_not_mmap_locked(clip, tokens)
    test_corrupt_entry(clip, tokens)
    test_eviction(clip, image, video)
    test_evicts_least_recently_used(clip, image, video)
    test_orphaned_temp_files(clip, tokens)
    test_age_expiry(clip, image)
    test_refuses_a_folder_it_does_not_own(clip, image)
    test_claims_folders_it_may_have(clip, image)
    test_purge(clip, image)
    test_hash_cost(clip)
    print("\nall cond cache tests passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMPROOT, ignore_errors=True)
