"""LiveSession: scheduler + control loop + inference worker around one game and policy."""

from __future__ import annotations

import logging
import sys
from types import TracebackType

from tmagent.config import Config
from tmagent.interfaces import NEUTRAL_ACTION, ChunkPolicy, RealtimeGame
from tmagent.runtime.control_loop import ControlLoop
from tmagent.runtime.inference import InferenceWorker
from tmagent.runtime.profiler import LatencyProfiler
from tmagent.runtime.scheduler import ActionScheduler

log = logging.getLogger(__name__)

# The default 5 ms GIL switch interval can delay the control thread's wake-up while the
# inference thread runs Python code; keep it short for the duration of a session.
_SWITCH_INTERVAL_S = 0.0005


class LiveSession:
    """Runs live play: `start()` the control loop, then the inference worker.

    `stop()` stops the worker, then the loop, then sends NEUTRAL_ACTION. Usable as a
    context manager.
    """

    def __init__(
        self,
        game: RealtimeGame,
        policy: ChunkPolicy,
        cfg: Config,
        profiler: LatencyProfiler | None = None,
    ) -> None:
        rt = cfg.runtime
        self.game = game
        self.profiler = profiler if profiler is not None else LatencyProfiler()
        self.scheduler = ActionScheduler(rt.control_hz, rt.hold_s)
        self.control = ControlLoop(
            game, self.scheduler, rt.control_hz, self.profiler, spin_s=rt.spin_s
        )
        self.worker = InferenceWorker(
            game, policy, self.scheduler, self.control, cfg.data.frame_hz, rt.control_hz,
            self.profiler,
        )  # fmt: skip
        self._running = False
        self._old_switch: float | None = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._old_switch = sys.getswitchinterval()
        sys.setswitchinterval(min(self._old_switch, _SWITCH_INTERVAL_S))
        self.scheduler.clear()
        self.control.start()
        self.worker.start()

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        self.worker.stop()
        self.control.stop()
        try:
            self.game.set_action(NEUTRAL_ACTION)
        except Exception:
            log.exception("failed to send the final neutral action")
        if self._old_switch is not None:
            sys.setswitchinterval(self._old_switch)

    def stats(self) -> dict:
        """Control loop stats at top level plus "inference" counters and the profiler summary."""
        return {
            **self.control.stats(),
            "inference": self.worker.stats(),
            "profiler": self.profiler.summary(),
        }

    def __enter__(self) -> LiveSession:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
