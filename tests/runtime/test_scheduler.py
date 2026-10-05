from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tmagent.interfaces import NEUTRAL_ACTION
from tmagent.runtime.scheduler import ActionScheduler

HZ = 10  # 0.1 s per row keeps the numbers readable


def chunk(*rows) -> np.ndarray:
    return np.array(rows, dtype=np.float32)


def test_no_chunk_is_neutral():
    assert ActionScheduler(HZ).action_at(123.0) == NEUTRAL_ACTION


def test_steer_interpolates_gas_brake_hold():
    s = ActionScheduler(HZ)
    s.publish(chunk([0.0, 1, 0], [1.0, 0, 1], [-1.0, 1, 0]), t0_wall=100.0)
    a = s.action_at(100.0)
    assert (a.steer, a.gas, a.brake) == (0.0, 1.0, 0.0)
    a = s.action_at(100.05)  # halfway between row 0 and 1
    assert a.steer == pytest.approx(0.5)
    assert (a.gas, a.brake) == (1.0, 0.0)  # held from row 0
    a = s.action_at(100.1)  # exactly row 1 (despite float rounding)
    assert a.steer == pytest.approx(1.0)
    assert (a.gas, a.brake) == (0.0, 1.0)
    a = s.action_at(100.15)
    assert a.steer == pytest.approx(0.0)
    assert (a.gas, a.brake) == (0.0, 1.0)  # still row 1
    a = s.action_at(100.2)
    assert a.steer == pytest.approx(-1.0)
    assert (a.gas, a.brake) == (1.0, 0.0)


def test_before_first_chunk_is_neutral():
    s = ActionScheduler(HZ)
    s.publish(chunk([0.5, 1, 0]), t0_wall=10.0)
    assert s.action_at(9.9) == NEUTRAL_ACTION


def test_newest_chunk_wins_and_previous_fills_before_its_t0():
    s = ActionScheduler(HZ)
    s.publish(np.tile([0.2, 1.0, 0.0], (20, 1)), t0_wall=0.0)  # covers 0 .. 1.9
    s.publish(np.tile([0.8, 0.0, 1.0], (20, 1)), t0_wall=1.0)  # starts later, newest
    a = s.action_at(0.5)  # newest has not started: previous chunk
    assert (a.steer, a.gas, a.brake) == (pytest.approx(0.2), 1.0, 0.0)
    a = s.action_at(1.0)  # newest covers t even though the old one still does too
    assert (a.steer, a.gas, a.brake) == (pytest.approx(0.8), 0.0, 1.0)
    assert s.action_at(1.5).steer == pytest.approx(0.8)


def test_newest_by_publish_order_even_with_earlier_t0():
    s = ActionScheduler(HZ)
    s.publish(np.tile([0.2, 1.0, 0.0], (5, 1)), t0_wall=5.0)
    s.publish(np.tile([0.9, 1.0, 0.0], (5, 1)), t0_wall=4.8)  # published later, starts earlier
    assert s.action_at(5.0).steer == pytest.approx(0.9)


def test_hold_after_chunk_end_then_neutral():
    s = ActionScheduler(HZ, hold_s=0.25)
    s.publish(chunk([0.1, 1, 0], [0.3, 0, 1], [0.5, 1, 1]), t0_wall=0.0)  # last row at 0.2
    a = s.action_at(0.3)  # 0.1 s after the last row: held
    assert (a.steer, a.gas, a.brake) == (pytest.approx(0.5), 1.0, 1.0)
    a = s.action_at(0.4)
    assert (a.steer, a.gas, a.brake) == (pytest.approx(0.5), 1.0, 1.0)
    assert s.action_at(0.5) == NEUTRAL_ACTION  # 0.3 s after the last row: neutral
    assert s.action_at(100.0) == NEUTRAL_ACTION


def test_previous_chunk_that_ran_out_is_neutral():
    s = ActionScheduler(HZ, hold_s=0.25)
    s.publish(chunk([0.5, 1, 0], [0.5, 1, 0]), t0_wall=0.0)
    s.publish(chunk([-0.5, 1, 0]), t0_wall=10.0)
    assert s.action_at(5.0) == NEUTRAL_ACTION


def test_publish_copies_validates_and_sanitizes():
    s = ActionScheduler(HZ)
    buf = np.tile(np.array([0.5, 1.0, 0.0], dtype=np.float32), (3, 1))
    s.publish(buf, t0_wall=0.0)
    buf[:] = -1.0  # caller reuses its buffer
    assert s.action_at(0.0).steer == pytest.approx(0.5)
    s.publish(chunk([np.nan, 2.0, -3.0], [5.0, np.nan, 0.0]), t0_wall=1.0)
    a = s.action_at(1.0)
    assert (a.steer, a.gas, a.brake) == (0.0, 1.0, 0.0)
    assert s.action_at(1.1).steer == 1.0
    for bad in (np.zeros((0, 3)), np.zeros((4, 2)), np.zeros(3)):
        with pytest.raises(ValueError):
            s.publish(bad, t0_wall=0.0)
    s.clear()
    assert s.action_at(1.0) == NEUTRAL_ACTION


def test_action_at_costs_microseconds():
    s = ActionScheduler(60)
    s.publish(np.random.default_rng(0).uniform(-1, 1, (16, 3)).astype(np.float32), 0.0)
    n = 20000
    t0 = time.perf_counter()
    for i in range(n):
        s.action_at(0.1 + (i % 100) * 1e-4)
    per_call = (time.perf_counter() - t0) / n
    assert per_call < 100e-6  # typically ~1 us; very generous bound


def test_thread_safety_smoke():
    s = ActionScheduler(60)
    stop = threading.Event()
    errors: list[BaseException] = []

    def publisher() -> None:
        try:
            i = 0
            while not stop.is_set():
                v = (i % 10) / 10.0
                s.publish(np.tile([v, 1.0, 0.0], (8, 1)).astype(np.float32), time.perf_counter())
                i += 1
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    def reader() -> None:
        try:
            n = 0
            while not stop.is_set():
                a = s.action_at(time.perf_counter())
                assert -1.0 <= a.steer <= 1.0 and a.gas in (0.0, 1.0) and a.brake == 0.0
                n += 1
            assert n > 0
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=publisher), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    time.sleep(0.3)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    assert not errors
    assert s.action_at(time.perf_counter()).gas in (0.0, 1.0)
