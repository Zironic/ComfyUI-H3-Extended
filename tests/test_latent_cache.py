"""Self-test for the VAE latent cache.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_latent_cache.py

The VAE is stubbed with a counter, so what is tested is the part that decides
whether it runs at all: that identical pixels skip it, that any change to the
pixels or the weights does not, and that a hit is bit-identical to the encode it
replaced. Also asserts the property the whole cache rests on — that the check
happens *before* the encode, not after.
"""

import logging
import os
import shutil
import sys
import h3_test_tempfile as tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_TMPROOT = tempfile.mkdtemp(prefix="h3_latent_cache_test_")
os.environ["H3_COND_CACHE_DIR"] = os.path.join(_TMPROOT, "cache")

import torch  # noqa: E402

import cond_cache  # noqa: E402
import latent_cache  # noqa: E402

_VAE_FILE = os.path.join(_TMPROOT, "fake_vae.safetensors")


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakePatcher:
    def __init__(self, path):
        self.cached_patcher_init = (None, (path, {"tae_latent_channels": 24}, None))


class FakeVAE:
    """Deterministic stand-in: the latent is a fixed function of the pixels."""

    def __init__(self, path=_VAE_FILE, salt=0.0):
        self.patcher = FakePatcher(path)
        self.first_stage_model = type("MiniMaxH3VideoVAE", (), {})()
        self.output_device = torch.device("cpu")
        self.calls = 0
        self.salt = salt

    def encode(self, pixels):
        self.calls += 1
        t = pixels.shape[0]
        per_frame = pixels.reshape(t, -1).mean(dim=1).float()  # [t]
        return per_frame.view(1, 1, t, 1).expand(1, 24, t, 1).clone() + self.salt


def write_vae(payload, path=_VAE_FILE):
    with open(path, "wb") as f:
        f.write(payload)


def test_hit_skips_the_vae(vae, pixels):
    print("identical pixels do not reach the VAE")
    first = latent_cache.encode(vae, pixels, label="ref video")
    check(vae.calls == 1, "cold call ran the VAE")
    second = latent_cache.encode(vae, pixels, label="ref video")
    check(vae.calls == 1, "warm call did not run the VAE")
    check(torch.equal(first, second), "latent is bit-identical")
    check(first.dtype == second.dtype, "dtype preserved (%s)" % second.dtype)
    check(second.device == torch.device("cpu"), "returned on the VAE's output device")


def test_key_covers_pixels_and_weights(vae, pixels):
    print("anything that changes the latent changes the key")

    def misses(label, v, px):
        before = v.calls
        latent_cache.encode(v, px)
        check(v.calls == before + 1, label)

    nudged = pixels.clone()
    nudged[0, 0, 0, 0] += 0.01
    misses("one changed pixel", vae, nudged)

    misses("a different frame count", vae, pixels[:2])

    # a slice view over a larger buffer, the trap the conditioning cache hit
    big = torch.rand(16, 8, 8, 3)
    misses("a slice view", vae, big[2:6])
    before = vae.calls
    latent_cache.encode(vae, big[2:6])
    check(vae.calls == before, "and the same slice then hits")
    misses("a different slice of the same buffer, same shape", vae, big[6:10])

    write_vae(b"y" * 2048)
    misses("the VAE file changed", vae, pixels)


def test_bypasses(vae, pixels):
    print("uncertain situations bypass rather than risk a wrong hit")

    def always_encodes(label, v=vae, px=pixels):
        before = v.calls
        latent_cache.encode(v, px)
        latent_cache.encode(v, px)
        check(v.calls == before + 2, label)

    unknown = FakeVAE()
    unknown.patcher.cached_patcher_init = None
    always_encodes("a VAE with unknown provenance", unknown)

    missing = FakeVAE(path=os.path.join(_TMPROOT, "not_there.safetensors"))
    always_encodes("a VAE whose file is gone", missing)

    os.environ["H3_LATENT_CACHE_DISABLE"] = "1"
    always_encodes("H3_LATENT_CACHE_DISABLE=1")
    del os.environ["H3_LATENT_CACHE_DISABLE"]

    os.environ["H3_COND_CACHE_DISABLE"] = "1"
    always_encodes("H3_COND_CACHE_DISABLE=1 is a master switch")
    del os.environ["H3_COND_CACHE_DISABLE"]

    before = vae.calls
    latent_cache.encode(vae, pixels, mode="off")
    latent_cache.encode(vae, pixels, mode="off")
    check(vae.calls == before + 2, "mode='off'")


def test_two_vaes_do_not_collide(pixels):
    print("two different VAEs never share an entry")
    video_path = os.path.join(_TMPROOT, "video_vae.safetensors")
    audio_path = os.path.join(_TMPROOT, "audio_vae.safetensors")
    write_vae(b"v" * 1024, video_path)
    write_vae(b"a" * 4096, audio_path)

    video = FakeVAE(path=video_path, salt=0.0)
    audio = FakeVAE(path=audio_path, salt=7.0)

    a = latent_cache.encode(video, pixels)
    b = latent_cache.encode(audio, pixels)
    check(video.calls == 1 and audio.calls == 1, "both VAEs ran on the same pixels")
    check(not torch.equal(a, b), "and returned their own distinct latents")
    check(torch.equal(latent_cache.encode(audio, pixels), b),
          "the second VAE still hits its own entry")
    check(audio.calls == 1, "without re-running")


def test_refresh(vae, pixels):
    print("refresh re-encodes and overwrites")
    latent_cache.encode(vae, pixels)
    before = vae.calls
    latent_cache.encode(vae, pixels, mode="refresh")
    check(vae.calls == before + 1, "refresh ignored the stored entry")
    latent_cache.encode(vae, pixels)
    check(vae.calls == before + 1, "and the overwritten entry hits again")


def test_shares_the_conditioning_janitor(vae, pixels):
    print("entries live under the conditioning cache's janitor")
    directory = cond_cache._cache_dir()
    latent_cache.encode(vae, pixels)
    entries, _ = cond_cache._scan(directory)
    check(entries, "the sweep sees latent entries as its own (%d found)" % len(entries))
    check(os.path.exists(os.path.join(directory, cond_cache.MARKER)),
          "in the same owned folder, under the same marker")

    os.environ["H3_COND_CACHE_GB"] = "0"
    try:
        cond_cache.sweep()
        entries, _ = cond_cache._scan(directory)
        check(not entries, "and a zero cap evicts them like any other entry")
    finally:
        del os.environ["H3_COND_CACHE_GB"]


def test_entry_not_mmap_locked(vae, pixels):
    print("a hit does not leave the entry locked (Windows)")
    cond_cache.purge()
    latent_cache.encode(vae, pixels)
    latent_cache.encode(vae, pixels)  # warm hit
    path = os.path.join(cond_cache._cache_dir(),
                        latent_cache.key(vae, pixels)[:cond_cache.DIGEST_CHARS]
                        + cond_cache.ENTRY_SUFFIX)
    try:
        os.remove(path)
        check(True, "entry is deletable straight after a hit")
    except OSError as e:
        check(False, "entry is deletable straight after a hit (%s)" % e)


def test_check_precedes_the_encode(vae, pixels):
    """The whole point: the decision happens before the expensive call."""
    print("the key is computable without running the VAE")
    cond_cache.purge()
    fresh = FakeVAE()
    digest = latent_cache.key(fresh, pixels)
    check(digest is not None, "a key exists for pixels that have never been encoded")
    check(fresh.calls == 0, "and computing it did not touch the VAE")

    latent_cache.encode(fresh, pixels)
    check(fresh.calls == 1, "the encode then ran once")
    probe = FakeVAE()
    check(latent_cache.key(probe, pixels) == digest, "the key is stable across VAE instances")
    latent_cache.encode(probe, pixels)
    check(probe.calls == 0, "so a fresh instance hits without ever loading the VAE")


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    write_vae(b"x" * 1024)

    torch.manual_seed(3)
    pixels = torch.rand(4, 8, 8, 3)
    vae = FakeVAE()

    test_hit_skips_the_vae(vae, pixels)
    test_key_covers_pixels_and_weights(vae, pixels)
    test_bypasses(vae, pixels)
    test_two_vaes_do_not_collide(pixels)
    test_refresh(vae, pixels)
    test_shares_the_conditioning_janitor(vae, pixels)
    test_entry_not_mmap_locked(vae, pixels)
    test_check_precedes_the_encode(vae, pixels)
    print("\nall latent cache tests passed")


if __name__ == "__main__":
    try:
        main()
    finally:
        shutil.rmtree(_TMPROOT, ignore_errors=True)
