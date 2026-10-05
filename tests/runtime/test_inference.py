from __future__ import annotations

import time

import numpy as np
import pytest

from tmagent.interfaces import Action
from tmagent.runtime.inference import InferenceWorker
from tmagent.runtime.profiler import LatencyProfiler
from tmagent.runtime.scheduler import ActionScheduler

from .test_helpers import DummyPolicy, FakeRealtimeGame

CONTROL_HZ = 60
FRAME_HZ = 20
R = CONTROL_HZ // FRAME_HZ


class StubLoop:
    """ControlLoop stand-in: action_at_wall returns steer = call_counter / 1000."""

    def __init__(self) -> None:
        self.queried: list[float] = []

    def action_at_wall(self, t: float) -> Action:
        self.queried.append(t)
        return Action(steer=len(self.queried) / 1000.0, gas=1.0, brake=0.0)


def run_worker(game, policy, loop=None, seconds=1.0, frame_hz=FRAME_HZ, profiler=None):
    sched = ActionScheduler(CONTROL_HZ)
    loop = loop or StubLoop()
    prof = profiler or LatencyProfiler()
    worker = InferenceWorker(game, policy, sched, loop, frame_hz, CONTROL_HZ, prof)
    worker.start()
    time.sleep(seconds)
    worker.stop()
    return worker, sched, loop, prof


def observed_wall_times(game, policy) -> np.ndarray:
    """Wall times of the observed frames (pixel value = frame index, valid below 256 frames)."""
    return np.array([game.frame_times[v] for v, _ in policy.observed])


def test_observes_about_frame_hz_and_subsamples_fast_capture():
    game = FakeRealtimeGame(fps=60.0).start_frames()
    policy = DummyPolicy()
    try:
        worker, _, _, prof = run_worker(game, policy, seconds=1.2)
    finally:
        game.close()
    n = len(policy.observed)
    assert 14 <= n <= 28  # ~24 expected at 20 Hz in 1.2 s, tolerant of CPU contention
    assert n == policy.predicts == worker.stats()["frames_observed"]
    assert policy.resets == 1
    walls = observed_wall_times(game, policy)
    gaps = np.diff(walls)
    assert np.all(gaps >= 1.0 / FRAME_HZ - 0.002 - 1e-9)  # at least a frame period minus 2 ms
    assert np.median(gaps) == pytest.approx(1.0 / FRAME_HZ, abs=0.02)  # not faster, not slower
    assert prof.stat("observe")["n"] == n
    assert prof.stat("predict")["n"] == n
    age = prof.stat("frame_age")
    assert age["n"] == n and 0.0 <= age["p50_ms"] < 100.0


def test_past_actions_length_order_and_first_zeros():
    game = FakeRealtimeGame(fps=60.0).start_frames()
    policy = DummyPolicy()
    loop = StubLoop()
    try:
        run_worker(game, policy, loop=loop, seconds=0.6)
    finally:
        game.close()
    assert len(policy.observed) >= 4
    first = policy.observed[0][1]
    assert first.shape == (R, 3) and first.dtype == np.float32 and not first.any()
    assert len(loop.queried) == R * (len(policy.observed) - 1)  # none for the first observe
    for i, (_, past) in enumerate(policy.observed[1:]):
        assert past.shape == (R, 3) and past.dtype == np.float32
        # the stub numbers its answers 1, 2, 3, ...: oldest first, one per control step
        expected = (R * i + 1 + np.arange(R)) / 1000.0
        np.testing.assert_allclose(past[:, 0], expected, rtol=1e-6)
        assert np.all(past[:, 1] == 1.0)
        q = np.array(loop.queried[R * i : R * (i + 1)])
        np.testing.assert_allclose(np.diff(q), 1.0 / CONTROL_HZ, atol=1e-9)  # oldest first
        # the newest history step is one control period before the observed frame
        assert q[-1] == pytest.approx(q[0] + (R - 1) / CONTROL_HZ)


def test_chunk_is_published_at_the_frame_time():
    game = FakeRealtimeGame(fps=60.0).start_frames()
    policy = DummyPolicy()
    try:
        _, sched, _, _ = run_worker(game, policy, seconds=0.5)
    finally:
        game.close()
    k_last = policy.observed[-1][0]
    t_frame = game.frame_times[k_last]
    a = sched.action_at(t_frame)  # row 0 applies exactly at the observed frame's wall time
    assert a.steer == pytest.approx(k_last / 255.0, abs=1e-6) and a.gas == 1.0


def test_slow_capture_observes_every_frame_and_counts_slow_frames():
    game = FakeRealtimeGame(fps=8.0).start_frames()  # 125 ms between frames, frame_hz = 20
    policy = DummyPolicy()
    try:
        worker, _, _, prof = run_worker(game, policy, seconds=1.0)
    finally:
        game.close()
    ks = np.array([v for v, _ in policy.observed])
    assert len(ks) >= 5 and np.all(np.diff(ks) == 1)  # every new frame observed, none skipped
    assert worker.stats()["slow_frames"] == len(ks) - 1
    assert prof.counter("slow_frame") == worker.stats()["slow_frames"]


def test_policy_exception_does_not_kill_the_worker():
    game = FakeRealtimeGame(fps=60.0).start_frames()
    policy = DummyPolicy(fail_every=2)  # every second predict raises
    try:
        worker, sched, _, prof = run_worker(game, policy, seconds=1.0)
    finally:
        game.close()
    st = worker.stats()
    assert st["policy_errors"] >= 3
    assert prof.counter("policy_error") == st["policy_errors"]
    assert st["frames_observed"] >= 3  # it kept publishing between the failures
    assert len(policy.observed) > st["policy_errors"]


def test_no_frames_is_fine_and_stop_is_prompt():
    game = FakeRealtimeGame()  # frame thread never started: latest_frame() is None
    policy = DummyPolicy()
    t0 = time.perf_counter()
    worker, _, _, _ = run_worker(game, policy, seconds=0.2)
    assert time.perf_counter() - t0 < 1.0
    assert policy.observed == [] and worker.stats()["frames_observed"] == 0


def test_rejects_incompatible_rates():
    with pytest.raises(ValueError):
        InferenceWorker(
            FakeRealtimeGame(), DummyPolicy(), ActionScheduler(60), StubLoop(), 25, 60
        )  # fmt: skip
