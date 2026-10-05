"""Shared test helpers (stub game, synthetic datasets) plus tests of the helpers."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tmagent.config import DataConfig
from tmagent.data.episode_io import append_index, save_episode
from tmagent.data.render import render_episode
from tmagent.interfaces import Action, Episode, Frame, GameState, InputTimeline, SyncGame


class StubGame:
    """Deterministic 1D car. Frames encode the race tick in their pixel values.

    Position x advances by gas + 0.25 * steer per tick. The image is filled with
    (tick % 256, tick // 256, 7 * tick % 256), see decode_tick(). lag_ticks > 0 makes
    frames (pixels and race_time_ms) stale by that many ticks; report_time=False
    reports race_time_ms = -1 (unknown).
    """

    def __init__(
        self,
        size: tuple[int, int] = (32, 24),  # (W, H) of the frames it produces
        channels: int = 3,
        finish_x: float | None = None,
        lag_ticks: int = 0,
        report_time: bool = True,
    ) -> None:
        self.size, self.channels, self.finish_x = size, channels, finish_x
        self.lag_ticks, self.report_time = lag_ticks, report_time
        self.map_ref: str | None = None
        self.tick = 0
        self.x = 0.0
        self.dx = 0.0
        self.finish_ms: int | None = None
        self.loaded = 0

    def load_map(self, map_ref: str) -> None:
        self.map_ref = map_ref
        self.loaded += 1

    def _state(self) -> GameState:
        return GameState(
            race_time_ms=self.tick * 10,
            position=np.array([self.x, 0.0, 0.0], dtype=np.float32),
            velocity=np.array([self.dx * 100, 0.0, 0.0], dtype=np.float32),
            speed_kmh=self.dx * 36.0,
            checkpoint=0,
            finished=self.finish_ms is not None,
        )

    def start_race(self) -> GameState:
        self.tick, self.x, self.dx, self.finish_ms = 0, 0.0, 0.0, None
        return self._state()

    def step(self, action: Action, n_ticks: int = 1) -> GameState:
        for _ in range(n_ticks):
            self.dx = action.gas + 0.25 * action.steer
            self.x += self.dx
            self.tick += 1
            if self.finish_x is not None and self.finish_ms is None and self.x >= self.finish_x:
                self.finish_ms = self.tick * 10
        return self._state()

    def grab_frame(self) -> Frame:
        w, h = self.size
        tick = max(self.tick - self.lag_ticks, 0)
        img = np.empty((h, w, 3), dtype=np.uint8)
        img[..., 0] = tick % 256
        img[..., 1] = tick // 256
        img[..., 2] = (7 * tick) % 256
        if self.channels == 1:
            img = img[..., :1].copy()
        shown = tick * 10 if self.report_time else -1
        return Frame(image=img, race_time_ms=shown, wall_time=time.perf_counter())

    def close(self) -> None:
        pass


def decode_tick(image: np.ndarray) -> int:
    """Tick index encoded in a StubGame frame (H, W, C) or (C, H, W)-free (H, W, C)."""
    return int(image[0, 0, 0]) + 256 * int(image[0, 0, 1])


def make_timeline(n_ticks: int, meta: dict | None = None) -> InputTimeline:
    """Deterministic, varied, valid actions: steer in [-1, 1], binary gas/brake."""
    i = np.arange(n_ticks)
    steer = ((i * 7) % 21 - 10) / 10.0
    gas = (i % 3 != 0).astype(np.float64)
    brake = (i % 5 == 0).astype(np.float64)
    actions = np.stack([steer, gas, brake], axis=1).astype(np.float32)
    return InputTimeline(actions=actions, meta=dict(meta or {}))


def small_cfg(**kw) -> DataConfig:
    """Tiny frames so tests are fast; 20 Hz frames at 60 Hz control by default."""
    kw.setdefault("resolution", [32, 24])
    kw.setdefault("history_s", 0.0)
    return DataConfig(**kw)


def make_episode(
    cfg: DataConfig,
    n_ticks: int = 300,
    map_uid: str = "mapA",
    source: str = "fake:0",
    game: StubGame | None = None,
    **meta: object,
) -> Episode:
    """Render a synthetic episode through StubGame + render_episode."""
    game = game or StubGame(size=tuple(cfg.resolution), channels=cfg.channels)
    tl = make_timeline(n_ticks)
    m = {"map_uid": map_uid, "map_name": map_uid, "source": source, **meta}
    return render_episode(game, tl, map_uid, cfg, m)


def write_dataset(
    root: Path, cfg: DataConfig, maps: dict[str, int], per_map: int = 1
) -> list[tuple[Episode, Path]]:
    """Render, save and index `per_map` episodes for each {map_uid: n_ticks}."""
    out = []
    for map_uid, n_ticks in maps.items():
        for j in range(per_map):
            ep = make_episode(cfg, n_ticks, map_uid=map_uid, source=f"fake:{map_uid}:{j}")
            path = save_episode(ep, root)
            append_index(root, ep, path)
            out.append((ep, path))
    return out


def test_stub_is_a_sync_game_and_deterministic() -> None:
    cfg = small_cfg()
    assert isinstance(StubGame(), SyncGame)
    a, b = make_episode(cfg, 60), make_episode(cfg, 60)
    assert np.array_equal(a.frames, b.frames)
    assert np.array_equal(a.positions, b.positions)


def test_decode_tick_roundtrip() -> None:
    g = StubGame()
    g.start_race()
    for t in (0, 1, 255, 256, 300):
        g.tick = t
        assert decode_tick(g.grab_frame().image) == t
