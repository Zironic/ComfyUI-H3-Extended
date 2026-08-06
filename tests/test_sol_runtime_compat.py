"""CPU compatibility tests for repeated long-form H3 sampler requests."""

import os
import sys
import threading
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_adaln.provider import AdaLNProvider  # noqa: E402
from h3_attention.sol.config import SolAttentionConfig  # noqa: E402
from h3_attention.sol.policy import decline_reason  # noqa: E402
from h3_block_cache.coordinator import BranchState, FirstBlockCacheCoordinator  # noqa: E402
from h3_masked_cache.compat import approximation_contamination_reason  # noqa: E402
from h3_runtime.context import H3RuntimeSession  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


def layout(seq_len=100, video_start=60):
    return SimpleNamespace(
        seq_len=seq_len,
        segments=[
            (0, 10, "text"),
            (10, video_start - 10, "ref_img"),
            (video_start - 10, video_start, "audio"),
            (video_start, seq_len, "video"),
        ],
        text_range=(0, 10),
        audio_range=(video_start - 10, video_start),
        video_range=(video_start, seq_len),
        reference_ranges=[("ref_img", 10, video_start - 10)],
        video_shape=(1, 4, 10),
        audio_t=5,
    )


def observe_one_step(session, seq_len=100):
    session._resolve_layout = lambda x, context, payload: layout(seq_len, 60)
    schedule = torch.tensor([1.0, 0.0])
    options = {
        "sample_sigmas": schedule,
        "sigmas": schedule[:1],
        "cond_or_uncond": [0],
    }
    return session.observe(
        [torch.zeros(1), torch.zeros(1)],
        None,
        torch.zeros(1, 4, 8, dtype=torch.bfloat16),
        options,
        {},
    )


def test_explicit_request_boundary():
    print("explicit request boundary")
    cache = FirstBlockCacheCoordinator()
    session = H3RuntimeSession(listeners=[cache])

    token = session.begin_outer_request()
    first = observe_one_step(session)
    session.end_outer_request(token)
    check(first.request_id == 0, "first outer sample uses request zero")

    cache.states[(0,)] = BranchState(
        request_id=0,
        cached_tail=torch.ones(2, 2),
        prev_head_residual=torch.ones(2, 2),
    )
    token = session.begin_outer_request()
    second = observe_one_step(session)
    session.end_outer_request(token)
    check(second.request_id == 1, "identical one-step sample gets a new request id")
    check(not cache.states, "new outer sample clears FirstBlockCache state")


def test_adaln_survives_request_reset():
    print("AdaLN trajectory reuse")
    provider = object.__new__(AdaLNProvider)
    provider._local = threading.local()
    provider.tables = (("cached",),)
    provider.signature = ("same schedule and layout",)
    provider.on_request_reset(7)
    check(provider.tables == (("cached",),), "request reset preserves compatible AdaLN tables")
    check(provider._local.request_id == 7 and provider._local.step_index == -1,
          "request reset only rewinds the AdaLN cursor")


def test_longform_sink_guard():
    print("long-form Sol sink guard")
    config = SolAttentionConfig(dense_steps=0, dense_layers=0)
    q = torch.zeros(1, 2, 100, 128, dtype=torch.bfloat16)
    large = SimpleNamespace(
        request_id=0, step_index=0, valid_layout=True, error=None,
        layout=layout(100, 60),
    )
    reason = decline_reason(config, large, 10, q)
    check(reason is not None and "sink fraction" in reason,
          "default guard declines a 60% exact long-form prefix")

    small_layout = layout(100, 40)
    small = SimpleNamespace(
        request_id=0, step_index=0, valid_layout=True, error=None,
        layout=small_layout,
    )
    check(decline_reason(config, small, 10, q) is None,
          "a 40% prefix remains sparse-eligible")


def test_masked_measurement_guard():
    print("masked Stage-0 contamination guard")
    sol = {"minimax_h3_memory_optimizer": {
        "attention_selected": "sol_attn", "attention_approximate": True,
        "block_cache_mode": "off", "block_cache_approximate": False,
    }}
    cache = {"minimax_h3_memory_optimizer": {
        "attention_selected": "efficient_sage_sm89", "attention_approximate": False,
        "block_cache_mode": "first_block", "block_cache_approximate": True,
    }}
    exact = {"minimax_h3_memory_optimizer": {
        "attention_selected": "efficient_sage_sm89", "attention_approximate": False,
        "block_cache_mode": "off", "block_cache_approximate": False,
    }}
    check("Sol-Attn" in approximation_contamination_reason(sol),
          "masked measurement rejects Sol-Attn")
    check("FirstBlockCache" in approximation_contamination_reason(cache),
          "masked measurement rejects FirstBlockCache")
    check(approximation_contamination_reason(exact) is None,
          "lossless optimizer configuration remains a valid Stage-0 baseline")


def main():
    test_explicit_request_boundary()
    test_adaln_survives_request_reset()
    test_longform_sink_guard()
    test_masked_measurement_guard()
    print("\nall Sol long-form compatibility tests passed")


if __name__ == "__main__":
    main()
