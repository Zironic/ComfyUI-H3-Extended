"""Self-test for the conditioning-cache diagnostics wiring.

Run from the ComfyUI root:

    python custom_nodes/ComfyUI-H3-Extended/tests/test_cond_cache_diagnostics.py

The assertions are mostly about reachability: installing must route every calling
style, must not recurse, and must report which key component moved. Runtime
weight patches are now keyable via ``patches_uuid`` rather than a diagnostics
bypass, so that path is covered here as well.
"""

import logging
import os
import sys
import h3_test_tempfile as tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

_TMPROOT = tempfile.mkdtemp(prefix="h3_cond_diag_test_")
os.environ["H3_COND_CACHE_DIR"] = os.path.join(_TMPROOT, "cache")

import torch  # noqa: E402

import cond_cache  # noqa: E402
import cond_cache_diagnostics as diag  # noqa: E402

_TE_FILE = os.path.join(_TMPROOT, "fake_te.safetensors")
VISION_START, VISION_END = 151652, 151653


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("  ok: %s" % msg)


class FakePatcher:
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


def make_tokens(text_ids, image=None):
    entries = [(t, 1.0) for t in text_ids]
    if image is not None:
        entries += [(VISION_START, 1.0),
                    ({"type": "image", "data": image, "original_type": "image"}, 1.0),
                    (VISION_END, 1.0)]
    return {"qwen3vl_32b": [entries]}


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        self.lines = []
        logging.getLogger().addHandler(self)
        logging.getLogger().setLevel(logging.INFO)
        return self

    def __exit__(self, *exc):
        logging.getLogger().removeHandler(self)

    def diagnostics(self):
        return [ln for ln in self.lines if "diagnostic:" in ln]


def harness_style_call(clip, tokens):
    """Exactly how chunked_ref2v/harness.py reaches the cache."""
    from cond_cache import encode as encode_conditioning
    return encode_conditioning(clip, tokens, mode="auto", label="harness")


def test_install_reaches_a_function_local_import(clip, image):
    print("installing reaches a caller that imports inside a function")
    with Capture() as cap:
        harness_style_call(clip, make_tokens([1, 2, 3], image))
    check(cap.diagnostics(), "the harness calling style produced diagnostic lines")
    check(any("vision[0]" in ln for ln in cap.diagnostics()),
          "including a per-vision-block line")


def test_no_recursion(clip, image):
    print("installing over the delegated name does not recurse")
    before = clip.calls
    with Capture() as cap:
        harness_style_call(clip, make_tokens([4, 5, 6], image))
    check(clip.calls == before + 1, "the encoder ran exactly once")
    finals = [ln for ln in cap.diagnostics() if "final=" in ln]
    check(len(finals) == 1, "and exactly one diagnostic summary was emitted")


def test_reports_the_component_that_moved(clip, image):
    print("a repeat under the same label names what changed")
    tokens = make_tokens([7, 8, 9], image)
    harness_style_call(clip, tokens)

    nudged = image.clone()
    nudged[0, 0, 0, 0] += 0.5
    with Capture() as cap:
        harness_style_call(clip, make_tokens([7, 8, 9], nudged))
    summary = " ".join(ln for ln in cap.diagnostics() if "changes=" in ln)
    check("vision[0]" in summary, "blames vision[0] for changed reference pixels")
    check("text " not in summary.split("changes=")[-1],
          "and does not blame the text, which did not change")

    with Capture() as cap:
        harness_style_call(clip, make_tokens([7, 8, 99], nudged))
    summary = " ".join(ln for ln in cap.diagnostics() if "changes=" in ln)
    check("text" in summary.split("changes=")[-1],
          "blames the text when only prompt tokens change")


def test_cache_behaviour_is_unchanged(clip, image):
    print("the wrapper does not disturb the cache itself")
    cond_cache.purge()
    tokens = make_tokens([10, 11], image)
    plain_digest = cond_cache.tokens_digest(clip, tokens)

    before = clip.calls
    harness_style_call(clip, tokens)
    check(clip.calls == before + 1, "cold call ran the encoder")
    harness_style_call(clip, tokens)
    check(clip.calls == before + 1, "warm call hit the cache through the wrapper")
    check(cond_cache.tokens_digest(clip, tokens) == plain_digest,
          "and the key is untouched by inspection")


def test_bypass_paths_are_explained(clip, image):
    print("true bypasses say which bypass they were")
    tokens = make_tokens([12, 13], image)
    with Capture() as cap:
        from cond_cache import encode as e
        e(clip, tokens, mode="off", label="x")
    check(any("bypass mode=off" in ln for ln in cap.diagnostics()), "mode=off is named")

    clip.patcher.forced_hooks = object()
    clip.use_clip_schedule = True
    try:
        with Capture() as cap:
            harness_style_call(clip, tokens)
        check(any("bypass hook schedule" in ln for ln in cap.diagnostics()),
              "hook-scheduled encoder bypass is named")
    finally:
        clip.patcher.forced_hooks = None
        clip.use_clip_schedule = False


def test_runtime_patches_are_keyed_not_bypassed(clip, image):
    print("runtime weight patches participate in the diagnostic model key")
    cond_cache.purge()
    tokens = make_tokens([14, 15], image)
    clip.patcher.patches = {"model.layers.0.weight": [1]}
    clip.patcher.patches_uuid = "patch-a"
    try:
        before = clip.calls
        with Capture() as cap:
            harness_style_call(clip, tokens)
            harness_style_call(clip, tokens)
        check(clip.calls == before + 1, "same patched state caches through diagnostics")
        check(not any("bypass weight patches" in ln for ln in cap.diagnostics()),
              "weight patches are not reported as a bypass")

        with Capture() as cap:
            clip.patcher.patches_uuid = "patch-b"
            harness_style_call(clip, tokens)
        check(clip.calls == before + 2, "new patches_uuid re-encodes")
        summary = " ".join(ln for ln in cap.diagnostics() if "changes=" in ln)
        check("model" in summary.split("changes=")[-1],
              "diagnostics attribute patches_uuid change to the model component")
    finally:
        clip.patcher.patches = {}
        clip.patcher.patches_uuid = "base-process-uuid"


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(_TE_FILE, "wb") as f:
        f.write(b"x" * 1024)

    check(cond_cache.encode is not diag.encode,
          "before install(), the raw cache is what callers resolve")
    diag.install()
    check(cond_cache.encode is diag.encode,
          "after install(), cond_cache.encode is the diagnostic wrapper")

    torch.manual_seed(1)
    image = torch.rand(1, 128, 128, 3)
    clip = FakeCLIP()

    test_install_reaches_a_function_local_import(clip, image)
    test_no_recursion(clip, image)
    test_reports_the_component_that_moved(clip, image)
    test_cache_behaviour_is_unchanged(clip, image)
    test_bypass_paths_are_explained(clip, image)
    test_runtime_patches_are_keyed_not_bypassed(clip, image)
    print("\nall cond cache diagnostics tests passed")


if __name__ == "__main__":
    import shutil
    try:
        main()
    finally:
        shutil.rmtree(_TMPROOT, ignore_errors=True)
