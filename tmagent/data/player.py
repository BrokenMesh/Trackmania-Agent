"""Record a human driving the real game: player segments, saved as training episodes.

Used by takeover mode (tmagent/eval/takeover.py) and by recording sessions (`record_map`):
the game runs in realtime with the plugin's passthrough (the player's keyboard drives), the
recorder samples state, keys and streamed frames, and every segment / attempt is resampled
to the dataset grids and stored like any other episode.

Keys are read with GetAsyncKeyState (~500 Hz polling), not per physics tick: the episodes are
fine for training, but unlike replays they cannot be re-driven tick-exactly.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import Config, DataConfig
from tmagent.data.episode_io import append_index, save_episode
from tmagent.data.render import conform_frame
from tmagent.data.timeline import control_times_ms, frame_times_ms, grid_len
from tmagent.game.tmnf.window import DRIVE_KEYS, VK_H, keys_down
from tmagent.interfaces import Episode, GameState

POLL_S = 0.002
WAIT_POLL_S = 0.05
MIN_SEGMENT_MS = 1000  # shorter segments are not worth a training episode
RESPAWN_JUMP_M = 15.0  # position jump between two samples that counts as a respawn
RESTART_BACK_MS = 50  # race time going back by more than this = the race was restarted
KeyReader = Callable[[], tuple[bool, bool, bool, bool, bool]]  # left right gas brake H


def read_keyboard() -> tuple[bool, bool, bool, bool, bool]:
    """(left, right, gas, brake, H) from the real keyboard (TMNF default arrows)."""
    return keys_down((*DRIVE_KEYS, VK_H))  # type: ignore[return-value]


def keys_to_action(left: bool, right: bool, gas: bool, brake: bool) -> np.ndarray:
    """Keyboard state as an action row; left + right held together -> left (as in replays)."""
    steer = -1.0 if left else (1.0 if right else 0.0)
    return np.array([steer, float(gas), float(brake)], np.float32)


class Requests:
    """Thread-safe one-slot request box for a UI thread, plus the current phase."""

    def __init__(self, phase: str) -> None:
        self._lock = threading.Lock()
        self._request: str | None = None
        self.phase = phase

    def request(self, what: str) -> None:
        with self._lock:
            self._request = what

    def pop(self) -> str | None:
        with self._lock:
            req, self._request = self._request, None
            return req


@dataclass
class PlayerSegment:
    """What happened while the player drove, in race time."""

    start_ms: int
    inputs: list[tuple[int, np.ndarray, np.ndarray, float]] = field(default_factory=list)
    frames: list[tuple[int, np.ndarray]] = field(default_factory=list)
    finished: bool = False
    finish_ms: int | None = None  # race time of the finish, if finished
    respawns: int = 0

    @property
    def end_ms(self) -> int:
        return self.inputs[-1][0] if self.inputs else self.start_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def last_action(self) -> np.ndarray:
        return self.inputs[-1][1] if self.inputs else np.zeros(3, np.float32)

    def add_state(self, st: GameState, action: np.ndarray) -> None:
        pos = np.asarray(st.position, np.float32).reshape(3)
        if self.inputs and np.linalg.norm(pos - self.inputs[-1][2]) > RESPAWN_JUMP_M:
            self.respawns += 1
        self.inputs.append((int(st.race_time_ms), action, pos, float(st.speed_kmh)))

    def to_episode(self, cfg: DataConfig, meta: dict[str, Any]) -> Episode:
        """Resample to the dataset grids, with times relative to the segment start."""
        span = self.duration_ms + 1
        f_times = frame_times_ms(grid_len(span, cfg.frame_hz), cfg.frame_hz)
        c_times = control_times_ms(grid_len(span, cfg.control_hz), cfg.control_hz)
        in_t = np.array([t for t, *_ in self.inputs], np.int64) - self.start_ms
        fr_t = np.array([t for t, _ in self.frames], np.int64) - self.start_ms
        at = np.clip(np.searchsorted(in_t, c_times, side="right") - 1, 0, len(in_t) - 1)
        af = np.clip(np.searchsorted(fr_t, f_times, side="right") - 1, 0, len(fr_t) - 1)
        frames = np.stack([conform_frame(self.frames[i][1], cfg)[0] for i in af])
        out_meta = {
            **meta,
            "race_time_ms": int(self.duration_ms),
            "finished": self.finished,
            "finish_time_ms": self.finish_ms,
            "frame_hz": int(cfg.frame_hz),
            "control_hz": int(cfg.control_hz),
            "resolution": [int(cfg.resolution[0]), int(cfg.resolution[1])],
            "channels": int(cfg.channels),
            "segment_start_ms": int(self.start_ms),
            "respawns": int(self.respawns),
            "desync": False,
        }
        return Episode(
            frames=frames,
            frame_times_ms=f_times,
            actions=np.stack([self.inputs[i][1] for i in at]).astype(np.float32),
            action_times_ms=c_times,
            positions=np.stack([self.inputs[i][2] for i in at]).astype(np.float32),
            speeds_kmh=np.array([self.inputs[i][3] for i in at], np.float32),
            meta=out_meta,
        )


def store_segment(seg: PlayerSegment, cfg: Config, meta: dict[str, Any]) -> Path | None:
    """Save a segment as an episode (meta needs episode_id, map_uid, map_name, source) and
    index it; None if it is shorter than MIN_SEGMENT_MS or has no frames."""
    if seg.duration_ms < MIN_SEGMENT_MS or not seg.frames:
        return None
    base = {"player": "human", "camera": cfg.game.camera, "renderer": "player"}
    ep = seg.to_episode(cfg.data, {**base, **meta})
    path = save_episode(ep, cfg.data.root)
    append_index(cfg.data.root, ep, path)
    return path


def drive_segment(
    game: Any,
    seg: PlayerSegment,
    read_keys: KeyReader,
    stop: Callable[[GameState, tuple], str | None],
    on_frame: Callable[[Any], None] | None = None,
) -> tuple[str, GameState]:
    """Sample the player's driving into `seg` (realtime passthrough must be on) until
    stop(state, keys) returns a reason; returns (reason, last state).

    Frames are taken from the realtime stream; on_frame sees each new one (e.g. a policy
    that keeps observing during a takeover).
    """
    last_frame_t = -1
    while True:
        st = game.get_state()
        keys = read_keys()
        reason = stop(st, keys)
        if reason is not None:
            return reason, st
        seg.add_state(st, keys_to_action(*keys[:4]))
        frame = game.latest_frame()
        if frame is not None and frame.race_time_ms > last_frame_t:
            last_frame_t = frame.race_time_ms
            if frame.race_time_ms >= seg.start_ms:
                seg.frames.append((frame.race_time_ms, frame.image))
                if on_frame is not None:
                    on_frame(frame)
        time.sleep(POLL_S)


class _AttemptEnd:
    """Stop rule of one recorded attempt (finish, restart key / request, next map)."""

    def __init__(
        self,
        control: Requests,
        seg: PlayerSegment,
        attempt: int,
        on_tick: Callable[[dict[str, Any]], None] | None,
    ) -> None:
        self.control, self.seg, self.attempt, self.on_tick = control, seg, attempt, on_tick
        self.last_t = 0

    def __call__(self, st: GameState, keys: tuple) -> str | None:
        if self.on_tick is not None:
            info = {"phase": "driving", "attempt": self.attempt, "race_time_ms": st.race_time_ms}
            self.on_tick(info)
        req = self.control.pop()
        if req in ("next", "restart"):
            return req
        if st.finished:
            self.seg.finished, self.seg.finish_ms = True, int(st.race_time_ms)
            return "finished"
        if st.race_time_ms < self.last_t - RESTART_BACK_MS:
            return "restart"  # the game's own restart key
        self.last_t = st.race_time_ms
        return None


def record_map(
    game: Any,
    map_ref: str,
    control: Requests,
    on_attempt: Callable[[PlayerSegment, str], None],
    on_tick: Callable[[dict[str, Any]], None] | None = None,
    read_keys: KeyReader = read_keyboard,
) -> int:
    """Record the player's attempts on one map until a "next" request; returns the count.

    Every attempt starts paused at race time 0 (control.phase "ready") and begins when the
    player presses a drive key; it runs in realtime with passthrough ("driving") and ends at
    the finish, at a restart (the game's own restart key, or a "restart" request) or at
    "next". on_attempt(segment, end) gets every attempt, also when on_tick raises (end
    "stopped"); on_tick sees {"phase", "attempt", "race_time_ms"} while waiting / driving.
    """
    game.load_map(map_ref)
    attempt = 0
    while True:
        game.start_race()
        control.phase = "ready"
        while not any(read_keys()[:4]):
            if control.pop() == "next":
                return attempt
            if on_tick is not None:
                on_tick({"phase": "ready", "attempt": attempt + 1, "race_time_ms": 0})
            time.sleep(WAIT_POLL_S)
        attempt += 1
        control.phase = "driving"
        game.hand_to_player()
        seg = PlayerSegment(start_ms=0)
        stop = _AttemptEnd(control, seg, attempt, on_tick)
        try:
            end, _ = drive_segment(game, seg, read_keys, stop)
        except BaseException:
            on_attempt(seg, "stopped")
            raise
        on_attempt(seg, end)
        if end == "next":
            return attempt
