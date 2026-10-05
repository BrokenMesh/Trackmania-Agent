from __future__ import annotations

import time

import numpy as np
import pytest

from tmagent.interfaces import NEUTRAL_ACTION, Action
from tmagent.runtime.control_loop import ControlLoop
from tmagent.runtime.profiler import LatencyProfiler
from tmagent.runtime.scheduler import ActionScheduler

from .test_helpers import FakeClock, FakeRealtimeGame

HZ = 60
PERIOD = 1.0 / HZ


class ClockedGame:
    """Records (fake time, action); can advance the fake clock inside set_action (a stall)."""

    def __init__(self, clock: FakeClock, stalls: dict[int, float] | None = None) -> None:
        self.clock = clock
        self.stalls = stalls or {}
        self.sent: list[tuple[float, Action]] = []

    def set_action(self, action: Action) -> None:
        k = len(self.sent)
        self.sent.append((self.clock.t, action))
        self.clock.t += self.stalls.get(k, 0.0)


def make_fake(stalls=None, profiler=None, **kw):
    clock = FakeClock()
    game = ClockedGame(clock, stalls)
    sched = ActionScheduler(HZ)
    loop = ControlLoop(
        game, sched, HZ, profiler=profiler, clock=clock, sleep=clock.sleep, **kw
    )  # fmt: skip
    return clock, game, sched, loop


def test_runs_at_60hz_and_sends_scheduler_actions():
    game = FakeRealtimeGame(fps=60.0)
    sched = ActionScheduler(HZ)
    sched.publish(np.tile([0.25, 1.0, 0.0], (240, 1)).astype(np.float32), time.perf_counter())
    prof = LatencyProfiler()
    loop = ControlLoop(game, sched, HZ, profiler=prof)
    t0 = time.perf_counter()
    loop.start()
    time.sleep(1.0)
    loop.stop()
    elapsed = time.perf_counter() - t0
    st = loop.stats()
    expected = elapsed * HZ
    print(
        f"\ncontrol 60Hz: slots={st['slots']} ticks={st['ticks']} misses={st['deadline_misses']} "
        f"miss_pct={st['miss_pct']:.2f} jitter p50/p99={st['jitter_p50_ms']:.3f}/"
        f"{st['jitter_p99_ms']:.3f} ms set_action p99={st['set_action_p99_ms']:.3f} ms"
    )
    assert abs(st["slots"] - expected) <= 0.05 * expected + 1
    assert abs(len(game.actions) - expected) <= 0.05 * expected + 1  # ticks within +-5 %
    assert st["ticks"] == len(game.actions)
    assert all(a.steer == pytest.approx(0.25) and a.gas == 1.0 for _, a in game.actions)
    assert st["jitter_p50_ms"] < 5.0
    assert prof.counter("control_tick") == st["ticks"]
    loop.stop()  # idempotent


def test_executed_log_and_action_at_wall():
    clock, game, sched, loop = make_fake()
    sched.publish(np.array([[0.0, 1, 0]] * 3, dtype=np.float32), 1000.0)
    assert loop.action_at_wall(1000.5) == NEUTRAL_ACTION  # nothing sent yet
    loop.run(max_slots=30)
    times = [t for t, _ in game.sent]
    assert len(times) == 30
    log = loop.executed_between(times[5], times[9])
    assert [t for t, _ in log] == times[5:10]  # inclusive bounds
    assert loop.executed_between(0.0, times[0] - 1e-3) == []
    assert loop.action_at_wall(times[0] - 1e-3) == NEUTRAL_ACTION  # before the first send
    for i in (0, 7, 29):
        assert loop.action_at_wall(times[i]) == game.sent[i][1]
        assert loop.action_at_wall(times[i] + 0.5 * PERIOD) == game.sent[i][1]
    assert loop.action_at_wall(times[-1] + 10.0) == game.sent[-1][1]


def test_log_is_a_bounded_ring():
    clock, game, sched, loop = make_fake(log_size=16)
    loop.run(max_slots=50)
    assert len(game.sent) == 50
    assert len(loop.executed_between(0.0, 1e12)) == 16
    oldest = loop.executed_between(0.0, 1e12)[0][0]
    assert oldest == game.sent[-16][0]
    assert loop.action_at_wall(game.sent[10][0]) == NEUTRAL_ACTION  # fell out of the ring


def test_absolute_schedule_without_drift():
    clock, game, sched, loop = make_fake()
    loop.run(max_slots=200)
    t = np.array([x for x, _ in game.sent])
    assert len(t) == 200
    ideal = t[0] + np.arange(200) * PERIOD
    assert np.max(np.abs(t - ideal)) < 1e-3  # no accumulated drift
    st = loop.stats()
    assert st["deadline_misses"] == 0 and st["miss_pct"] == 0.0 and st["ticks"] == 200
    assert st["jitter_p99_ms"] < 1.0


def test_stall_longer_than_a_slot_skips_ahead_without_burst():
    # The 11th set_action (slot 10) takes 3.5 periods: slot 10 goes out in slot 13 time,
    # slots 11 and 12 are skipped; the loop resumes in slot 13 and never bursts.
    prof = LatencyProfiler()
    clock, game, sched, loop = make_fake(stalls={10: 3.5 * PERIOD}, profiler=prof)
    loop.run(max_slots=40)
    t = np.array([x for x, _ in game.sent])
    st = loop.stats()
    assert st["slots"] == 40
    assert st["deadline_misses"] == 3  # slot 10 (late send) + skipped slots 11, 12
    assert st["ticks"] == len(t) == 38
    assert prof.counter("deadline_miss") == 3
    assert st["miss_pct"] == pytest.approx(100 * 3 / 40)
    gaps = np.diff(t)
    assert gaps.min() > 0.4 * PERIOD  # no burst of catch-up sends
    assert gaps.max() < 4.6 * PERIOD  # skipped, not delayed further
    slot = np.floor((t - t[0]) / PERIOD + 1e-6).astype(int)
    assert len(set(slot)) == len(slot)  # at most one send per slot


def test_late_by_less_than_a_slot_is_one_miss_and_no_skip():
    clock, game, sched, loop = make_fake(stalls={5: 1.5 * PERIOD})
    loop.run(max_slots=20)
    st = loop.stats()
    assert st["deadline_misses"] == 1  # slot 5 sent after slot 6 started
    assert st["ticks"] == 20  # but nothing skipped


def test_set_action_exception_does_not_kill_the_loop():
    game = FakeRealtimeGame()
    game.fail_set_action_at = {2, 3}
    loop = ControlLoop(game, ActionScheduler(HZ), HZ)
    loop.start()
    time.sleep(0.3)
    loop.stop()
    st = loop.stats()
    assert st["set_action_errors"] == 2
    assert len(game.actions) > 5
    assert st["ticks"] == len(game.actions) - 2  # failed sends are not logged as executed


def test_stop_is_prompt_and_restartable():
    game = FakeRealtimeGame()
    loop = ControlLoop(game, ActionScheduler(HZ), 30)
    loop.start()
    time.sleep(0.1)
    t0 = time.perf_counter()
    loop.stop()
    assert time.perf_counter() - t0 < 0.5
    n = len(game.actions)
    time.sleep(0.1)
    assert len(game.actions) == n  # really stopped
    loop.start()
    time.sleep(0.1)
    loop.stop()
    assert len(game.actions) > n


def test_invalid_rate_rejected():
    with pytest.raises(ValueError):
        ControlLoop(FakeRealtimeGame(), ActionScheduler(HZ), 0)
