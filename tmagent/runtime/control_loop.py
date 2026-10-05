"""Fixed-rate control thread. It only reads the scheduler and never waits on the model."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from tmagent.interfaces import NEUTRAL_ACTION, Action, RealtimeGame
from tmagent.runtime.profiler import LatencyProfiler
from tmagent.runtime.scheduler import ActionScheduler

log = logging.getLogger(__name__)


class _ActionLog:
    """Bounded ring of (wall time, action) with non-decreasing times, bisect lookup."""

    def __init__(self, size: int) -> None:
        self._size = size
        self._t = [0.0] * size
        self._a: list[Action] = [NEUTRAL_ACTION] * size
        self._n = 0  # total appended; logical index i lives in slot i % size
        self._lock = threading.Lock()

    def append(self, t: float, action: Action) -> None:
        with self._lock:
            i = self._n % self._size
            self._t[i] = t
            self._a[i] = action
            self._n += 1

    def _bisect(self, t: float, right: bool) -> int:
        """First logical index with time > t (right) or >= t (left); lock must be held."""
        lo, hi = max(0, self._n - self._size), self._n
        while lo < hi:
            mid = (lo + hi) // 2
            tm = self._t[mid % self._size]
            if tm < t or (right and tm == t):
                lo = mid + 1
            else:
                hi = mid
        return lo

    def between(self, t_from: float, t_to: float) -> list[tuple[float, Action]]:
        with self._lock:
            lo, hi = self._bisect(t_from, False), self._bisect(t_to, True)
            return [(self._t[i % self._size], self._a[i % self._size]) for i in range(lo, hi)]

    def at(self, t: float) -> Action:
        with self._lock:
            i = self._bisect(t, True) - 1
            if i < max(0, self._n - self._size):
                return NEUTRAL_ACTION
            return self._a[i % self._size]


class ControlLoop:
    """Calls `game.set_action(scheduler.action_at(now))` at `control_hz`.

    Slot n is due at t_n = t_start + n / control_hz (absolute schedule, no drift):
    coarse sleep, then spin for the last `spin_s`. A deadline miss is an action for
    slot n sent after slot n + 1 started. If the loop wakes more than one slot late it
    skips to the current slot (the skipped slots count as misses) and never bursts.
    `clock` and `sleep` are injectable for tests.
    """

    def __init__(
        self,
        game: RealtimeGame,
        scheduler: ActionScheduler,
        control_hz: int,
        profiler: LatencyProfiler | None = None,
        spin_s: float = 0.002,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
        log_size: int = 1 << 15,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._game = game
        self._scheduler = scheduler
        self._period = 1.0 / control_hz
        self._prof = profiler if profiler is not None else LatencyProfiler()
        self._spin_s = spin_s
        self._clock = clock
        self._sleep = sleep
        self._log = _ActionLog(max(1, log_size))
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._ticks = self._slots = self._misses = self._errors = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self.run, name="tm-control", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
            if t.is_alive():
                log.warning("control thread did not stop within 2 s")

    def run(self, max_slots: int | None = None) -> None:
        """Run the loop in the calling thread until stop() or `max_slots` slots elapsed."""
        clock, period = self._clock, self._period
        self._ticks = self._slots = self._misses = self._errors = 0
        t_start = clock()
        n = 0
        while not self._stop_evt.is_set() and (max_slots is None or n < max_slots):
            t_n = t_start + n * period
            if not self._wait_until(t_n):
                break
            now = clock()
            if now - t_n >= period:  # woke more than a slot late: skip ahead, never burst
                cur = max(n + 1, int((now - t_start) / period))
                self._miss(cur - n)
                n = cur
                self._slots = n
                if max_slots is not None and n >= max_slots:
                    break
                t_n = t_start + n * period
            if self._tick(now, max(0.0, now - t_n)) >= t_start + (n + 1) * period:
                # slot n's action went out after slot n + 1 started
                self._miss(1)
            n += 1
            self._slots = n

    def _wait_until(self, t_target: float) -> bool:
        """Coarse sleep, then spin (yielding the GIL) until t_target; False if stopped."""
        clock, spin = self._clock, self._spin_s
        while not self._stop_evt.is_set():
            remaining = t_target - clock()
            if remaining <= 0:
                return True
            self._sleep(remaining - spin if remaining > spin else 0)
        return False

    def _tick(self, now: float, jitter: float) -> float:
        """Send the action for `now`; returns the clock reading after the send."""
        prof = self._prof
        try:
            action = self._scheduler.action_at(now)
            self._game.set_action(action)
        except Exception:
            self._errors += 1
            prof.count("set_action_error")
            if self._errors <= 3 or self._errors % 1000 == 0:
                log.exception("set_action failed (%d so far)", self._errors)
        else:
            self._log.append(now, action)
            self._ticks += 1
        end = self._clock()
        prof.record("control_set_action", end - now)
        prof.record("control_jitter", jitter)
        prof.count("control_tick")
        return end

    def _miss(self, k: int) -> None:
        self._misses += k
        self._prof.count("deadline_miss", k)

    # -- introspection -----------------------------------------------------

    def executed_between(self, t_from: float, t_to: float) -> list[tuple[float, Action]]:
        """Actions sent with t_from <= wall time <= t_to, oldest first (bounded history)."""
        return self._log.between(t_from, t_to)

    def action_at_wall(self, t: float) -> Action:
        """The action in effect at wall time t: the last one sent at or before t."""
        return self._log.at(t)

    def stats(self) -> dict:
        jit = self._prof.stat("control_jitter")
        sa = self._prof.stat("control_set_action")
        slots = self._slots
        return {
            "ticks": self._ticks,
            "slots": slots,
            "deadline_misses": self._misses,
            "miss_pct": 100.0 * self._misses / slots if slots else 0.0,
            "jitter_p50_ms": jit["p50_ms"] if jit else 0.0,
            "jitter_p99_ms": jit["p99_ms"] if jit else 0.0,
            "set_action_p99_ms": sa["p99_ms"] if sa else 0.0,
            "set_action_errors": self._errors,
        }
