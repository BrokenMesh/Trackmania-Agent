"""Shared fakes for the runtime tests (plus a smoke test of the fakes themselves)."""

from __future__ import annotations

import threading
import time

import numpy as np

from tmagent.interfaces import Action, Frame, GameState


class FakeRealtimeGame:
    """RealtimeGame stand-in: a thread produces frames at `fps`, set_action calls are logged.

    Frame k carries the value k % 256 in every pixel.
    """

    def __init__(self, fps: float = 60.0, size: int = 8) -> None:
        self.fps = fps
        self._size = size
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.n_frames = 0
        self.frame_times: list[float] = []  # wall_time of frame k
        self.actions: list[tuple[float, Action]] = []
        self.finished = False
        self.fail_set_action_at: set[int] = set()
        self.loaded: str | None = None
        self.restarts = 0
        self.closed = False

    def start_frames(self) -> FakeRealtimeGame:
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        return self

    def _produce(self) -> None:
        t0 = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            due = t0 + k / self.fps
            while time.perf_counter() < due and not self._stop.is_set():
                time.sleep(max(0.0, min(0.002, due - time.perf_counter() - 0.0005)))
            img = np.full((self._size, self._size, 3), k % 256, dtype=np.uint8)
            self._frame = Frame(image=img, race_time_ms=int(k * 1000 / self.fps), wall_time=due)
            self.frame_times.append(due)
            self.n_frames = k + 1
            k += 1

    # RealtimeGame protocol
    def load_map(self, map_ref: str) -> None:
        self.loaded = map_ref

    def restart(self) -> None:
        self.restarts += 1

    def set_action(self, action: Action) -> None:
        n = len(self.actions)
        self.actions.append((time.perf_counter(), action))
        if n in self.fail_set_action_at:
            raise RuntimeError("set_action boom")

    def get_state(self) -> GameState:
        z = np.zeros(3, dtype=np.float32)
        return GameState(
            race_time_ms=0, position=z, velocity=z, speed_kmh=0.0, checkpoint=0,
            finished=self.finished,
        )  # fmt: skip

    def latest_frame(self) -> Frame | None:
        return self._frame

    def close(self) -> None:
        self.closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


class DummyPolicy:
    """ChunkPolicy whose chunk is derived from the last observed image value.

    steer = pixel / 255 for all rows, gas = 1, brake = 0.
    """

    def __init__(self, chunk_len: int = 8, predict_s: float = 0.0, fail_every: int = 0) -> None:
        self.chunk_len = chunk_len
        self.predict_s = predict_s
        self.fail_every = fail_every
        self.resets = 0
        self.observed: list[tuple[int, np.ndarray]] = []  # (pixel value, past_actions)
        self.predicts = 0
        self._last = 0

    def reset(self) -> None:
        self.resets += 1

    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        self._last = int(image[0, 0, 0])
        self.observed.append((self._last, np.array(past_actions)))

    def predict(self) -> np.ndarray:
        self.predicts += 1
        if self.fail_every and self.predicts % self.fail_every == 0:
            raise ValueError("predict boom")
        if self.predict_s:
            time.sleep(self.predict_s)
        chunk = np.zeros((self.chunk_len, 3), dtype=np.float32)
        chunk[:, 0] = self._last / 255.0
        chunk[:, 1] = 1.0
        return chunk


class FakeClock:
    """Deterministic clock: every reading advances `cost` s, sleep(s) advances s."""

    def __init__(self, cost: float = 2e-5, start: float = 1000.0) -> None:
        self.t = start
        self.cost = cost

    def __call__(self) -> float:
        self.t += self.cost
        return self.t

    def sleep(self, seconds: float) -> None:
        self.t += max(0.0, seconds)


def test_fake_game_produces_frames_and_logs_actions():
    game = FakeRealtimeGame(fps=100.0).start_frames()
    try:
        assert game.latest_frame() is None or game.latest_frame().image.shape == (8, 8, 3)
        time.sleep(0.15)
        f1 = game.latest_frame()
        assert f1 is not None and game.n_frames >= 5
        game.set_action(Action(0.5, 1.0, 0.0))
        assert game.actions[0][1].steer == 0.5
    finally:
        game.close()


def test_fake_clock_advances():
    c = FakeClock(cost=0.001, start=0.0)
    assert c() == 0.001
    c.sleep(1.0)
    assert abs(c() - 1.002) < 1e-12
