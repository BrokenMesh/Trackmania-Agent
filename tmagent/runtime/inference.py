"""Asynchronous inference worker: newest frame -> policy -> chunk -> scheduler."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from tmagent.interfaces import ACTION_DIM, ChunkPolicy, Frame, RealtimeGame
from tmagent.runtime.control_loop import ControlLoop
from tmagent.runtime.profiler import LatencyProfiler
from tmagent.runtime.scheduler import ActionScheduler

log = logging.getLogger(__name__)

_MIN_POLL_S = 0.001
_MAX_POLL_S = 0.01
_SLOW_FACTOR = 1.25  # an observed frame gap above this many frame periods is a "slow_frame"
_SLACK_S = 0.002  # observe a frame this much earlier than one full frame period


class InferenceWorker:
    """Observes frames at `frame_hz` on its own thread and publishes action chunks.

    A frame is observed if it is new (wall_time changed) and at least
    `1 / frame_hz - 2 ms` after the last observed one: faster capture is
    sub-sampled, slower capture observes every new frame (gaps above 1.25 frame
    periods are counted as "slow_frame"). Policy errors are logged and counted; the
    worker keeps running and the control loop holds or neutralizes on its own.
    """

    def __init__(
        self,
        game: RealtimeGame,
        policy: ChunkPolicy,
        scheduler: ActionScheduler,
        control_loop: ControlLoop,
        frame_hz: int,
        control_hz: int,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        if frame_hz <= 0 or control_hz % frame_hz != 0:
            raise ValueError(f"control_hz ({control_hz}) must be a multiple of frame_hz")
        self._game = game
        self._policy = policy
        self._scheduler = scheduler
        self._loop = control_loop
        self._control_hz = control_hz
        self._r = control_hz // frame_hz
        self._frame_dt = 1.0 / frame_hz
        self._prof = profiler if profiler is not None else LatencyProfiler()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._observed = self._errors = self._slow = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            reset()
        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="tm-inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        t, self._thread = self._thread, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=5.0)
            if t.is_alive():
                log.warning("inference thread did not stop within 5 s")

    def stats(self) -> dict:
        return {
            "frames_observed": self._observed,
            "slow_frames": self._slow,
            "policy_errors": self._errors,
        }

    def _past_actions(self, wall: float) -> np.ndarray:
        """float32 [R, 3]: executed actions at wall - (R - r) / control_hz, oldest first."""
        out = np.zeros((self._r, ACTION_DIM), dtype=np.float32)
        for r in range(self._r):
            a = self._loop.action_at_wall(wall - (self._r - r) / self._control_hz)
            out[r] = (a.steer, a.gas, a.brake)
        return out

    def _run(self) -> None:
        min_gap = self._frame_dt - _SLACK_S
        last_seen: float | None = None  # wall_time of the newest frame looked at
        last_obs: float | None = None  # wall_time of the last observed frame
        while not self._stop_evt.is_set():
            wait = _MIN_POLL_S
            try:
                frame = self._game.latest_frame()
                if frame is not None and frame.wall_time != last_seen:
                    last_seen = frame.wall_time
                    gap = None if last_obs is None else frame.wall_time - last_obs
                    if gap is None or gap >= min_gap:
                        if gap is not None and gap > _SLOW_FACTOR * self._frame_dt:
                            self._slow += 1
                            self._prof.count("slow_frame")
                        last_obs = frame.wall_time
                        self._step(frame, first=gap is None)
                if last_obs is not None:  # next useful frame is at least min_gap away
                    wait = last_obs + min_gap - time.perf_counter() - 0.003
                    wait = min(max(wait, _MIN_POLL_S), _MAX_POLL_S)
            except Exception:
                self._errors += 1
                self._prof.count("policy_error")
                if self._errors <= 3 or self._errors % 100 == 0:
                    log.exception("inference step failed (%d so far)", self._errors)
            self._stop_evt.wait(wait)

    def _step(self, frame: Frame, first: bool) -> None:
        prof = self._prof
        with prof.section("observe"):
            if first:
                past = np.zeros((self._r, ACTION_DIM), dtype=np.float32)
            else:
                past = self._past_actions(frame.wall_time)
            self._policy.observe(frame.image, past)
        with prof.section("predict"):
            chunk = self._policy.predict()
        self._scheduler.publish(chunk, t0_wall=frame.wall_time)
        prof.record("frame_age", time.perf_counter() - frame.wall_time)
        self._observed += 1
