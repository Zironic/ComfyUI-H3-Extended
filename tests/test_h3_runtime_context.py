"""CPU tests for the shared H3 request/step/layout context."""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))

from h3_runtime.context import H3RuntimeSession, RUNTIME_KEY  # noqa: E402


def check(value, message):
    if not value:
        raise AssertionError(message)
    print("  ok: %s" % message)


class Listener:
    def __init__(self):
        self.resets = []
        self.before = []
        self.after = []

    def on_request_reset(self, request_id):
        self.resets.append(request_id)

    def before_forward(self, snapshot, options, payload):
        self.before.append((snapshot.request_id, snapshot.step_index))

    def after_forward(self, snapshot, result, options):
        self.after.append(snapshot.step_index)


def layout(seq=100):
    return SimpleNamespace(
        seq_len=seq,
        segments=[(0, 10, "text"), (10, 20, "audio"), (20, seq, "video")],
        video_shape=(2, 4, 10),
        audio_t=5,
    )


def main():
    listener = Listener()
    session = H3RuntimeSession(listeners=[listener])
    session._resolve_layout = lambda x, context, payload: layout(payload.get("seq", 100))
    x = [torch.zeros(1), torch.zeros(1)]
    context = torch.zeros(1, 10, 4, dtype=torch.bfloat16)
    schedule = torch.tensor([1.0, 0.5, 0.0])

    options = {"sample_sigmas": schedule, "sigmas": schedule[0:1], "cond_or_uncond": [0]}
    first = session.observe(x, None, context, options, {"seq": 100})
    check(first.request_id == 0 and first.step_index == 0, "first call starts request zero")
    check(options[RUNTIME_KEY] is first, "snapshot is published in transformer options")

    options = {"sample_sigmas": schedule, "sigmas": schedule[1:2], "cond_or_uncond": [0]}
    second = session.observe(x, None, context, options, {"seq": 100})
    check(second.request_id == 0 and second.step_index == 1, "step advances within request")

    # A second CFG branch at the same step does not reset the request.
    options = {"sample_sigmas": schedule, "sigmas": schedule[1:2], "cond_or_uncond": [1]}
    branch = session.observe(x, None, context, options, {"seq": 100})
    check(branch.request_id == 0 and branch.branch == (1,), "CFG branches share request state")

    # Step reversal marks a new sampler request.
    options = {"sample_sigmas": schedule, "sigmas": schedule[0:1], "cond_or_uncond": [0]}
    again = session.observe(x, None, context, options, {"seq": 100})
    check(again.request_id == 1 and again.step_index == 0, "schedule reversal resets request")

    # Layout changes also reset state.
    options = {"sample_sigmas": schedule, "sigmas": schedule[0:1], "cond_or_uncond": [0]}
    changed = session.observe(x, None, context, options, {"seq": 120})
    check(changed.request_id == 2, "packed-layout change resets request")
    session.after_forward(changed, object(), options)
    check(listener.resets == [0, 1, 2], "listeners receive every request reset")
    check(listener.after == [0], "listeners receive post-forward notification")
    print("\nall H3 runtime-context tests passed")


if __name__ == "__main__":
    main()
