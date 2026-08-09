"""CPU tests for the shared H3 request/step/layout context."""

import os
import sys
import math
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..", "..")))

from h3_runtime.context import (  # noqa: E402
    H3RuntimeSession,
    RUNTIME_KEY,
    make_apply_model_wrapper,
)
from h3_runtime.timing import publish_timing  # noqa: E402


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


def test_apply_model_wrapper():
    print("compiled APPLY_MODEL boundary")
    listener = Listener()
    session = H3RuntimeSession(listeners=[listener])
    session._resolve_layout = lambda x, context, payload: layout(payload.get("seq", 100))
    wrapper = make_apply_model_wrapper(session)
    x = [torch.zeros(1), torch.zeros(1)]
    context = torch.zeros(1, 10, 4, dtype=torch.bfloat16)
    options = {"sample_sigmas": torch.tensor([0.25, 0.0]), "cond_or_uncond": [0]}
    class Timing:
        def __init__(self):
            self.samples = []

        def begin(self, stage):
            return stage

        def end(self, token):
            self.samples.append(token)

    timing = Timing()
    publish_timing(options, timing)
    calls = []
    def executor(*args, **kwargs):
        calls.append(args[5][RUNTIME_KEY])
        return "result"

    result = wrapper(
        executor,
        x,
        torch.tensor([0.25]),
        None,
        context,
        None,
        options,
        minimax_payload={"seq": 120},
    )
    snapshot = calls[0]
    check(result == "result", "wrapped executor result is preserved")
    check(snapshot.sigma == 0.25, "raw APPLY_MODEL timestep keeps sigma units")
    check(snapshot.layout.seq_len == 120, "runtime layout is published before model execution")
    check(listener.before == [(0, 0)] and listener.after == [0], "listeners surround the model call")
    check(timing.samples == ["model_forward"], "model call is timed outside Dynamo")


def test_apply_model_wrapper_unpacks_sampler_latent():
    print("compiled APPLY_MODEL packed latent boundary")
    session = H3RuntimeSession(strict_layout=True)
    seen = []

    def resolve(streams, context, payload):
        seen.append(tuple(tuple(stream.shape) for stream in streams))
        return layout()

    session._resolve_layout = resolve
    wrapper = make_apply_model_wrapper(session)
    latent_shapes = [torch.Size((1, 4, 3, 2, 2)), torch.Size((1, 2, 8))]
    packed_width = sum(math.prod(shape[1:]) for shape in latent_shapes)
    packed = torch.zeros((1, 1, packed_width))
    context = torch.zeros((1, 10, 4), dtype=torch.bfloat16)
    options = {"sigmas": torch.tensor([0.25])}

    result = wrapper(
        lambda *args, **kwargs: "result",
        packed,
        torch.tensor([0.25]),
        None,
        context,
        None,
        options,
        latent_shapes=latent_shapes,
        minimax_payload={},
    )
    check(result == "result", "packed sampler input reaches the wrapped model unchanged")
    check(seen == [tuple(tuple(shape) for shape in latent_shapes)], "layout sees unpacked video and audio streams")


def main():
    test_apply_model_wrapper()
    test_apply_model_wrapper_unpacks_sampler_latent()
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
