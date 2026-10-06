"""Interactive drives: the policy drives in sync mode, a human can take over and hand back.

Phases (TakeoverControl.phase):
  "agent"    sync mode, the policy is stepped tick by tick (as in harness.run_episode_sync)
  "player"   realtime mode with plugin passthrough: the player's keyboard drives at game speed
  "waiting"  the agent got stuck / went off track; the game is paused until the player takes
             over or lets the agent retry
The player takes over with an arrow key (or a request) and hands back with H (or a request).
Every player segment is recorded (frames, keys, positions) and can be saved as a training
episode: these are the recoveries plain behaviour cloning never sees.
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
from tmagent.eval.harness import _episode_result, drive_frame_period, policy_rows
from tmagent.eval.progress import ProgressTracker, ReferencePath
from tmagent.game.tmnf.window import DRIVE_KEYS, VK_H, keys_down
from tmagent.interfaces import ChunkPolicy, Episode, GameState

PLAYER_POLL_S = 0.002
WAIT_POLL_S = 0.05
MIN_SEGMENT_MS = 1000  # shorter player segments are not worth a training episode
KeyReader = Callable[[], tuple[bool, bool, bool, bool, bool]]  # left right gas brake hand_back


def read_keyboard() -> tuple[bool, bool, bool, bool, bool]:
    """(left, right, gas, brake, hand_back) from the real keyboard (arrows, H)."""
    return keys_down((*DRIVE_KEYS, VK_H))  # type: ignore[return-value]


def keys_to_action(left: bool, right: bool, gas: bool, brake: bool) -> np.ndarray:
    """Keyboard state as an action row; left + right held together -> left (as in replays)."""
    steer = -1.0 if left else (1.0 if right else 0.0)
    return np.array([steer, float(gas), float(brake)], np.float32)


class TakeoverControl:
    """Thread-safe requests from a UI: "take_over", "hand_back", "retry"."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request: str | None = None
        self.phase = "agent"

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

    @property
    def end_ms(self) -> int:
        return self.inputs[-1][0] if self.inputs else self.start_ms

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def last_action(self) -> np.ndarray:
        return self.inputs[-1][1] if self.inputs else np.zeros(3, np.float32)

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
            "frame_hz": int(cfg.frame_hz),
            "control_hz": int(cfg.control_hz),
            "resolution": [int(cfg.resolution[0]), int(cfg.resolution[1])],
            "channels": int(cfg.channels),
            "segment_start_ms": int(self.start_ms),
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


def save_segment(
    seg: PlayerSegment, cfg: Config, map_uid: str, map_name: str, ckpt: str
) -> Path | None:
    """Store a player segment as an episode (source "takeover") and index it; None if too short."""
    if seg.duration_ms < MIN_SEGMENT_MS or not seg.frames:
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    meta = {
        "episode_id": f"{map_uid}-takeover-{stamp}-{seg.start_ms}",
        "map_uid": map_uid,
        "map_name": map_name,
        "source": f"takeover:{stamp}",
        "player": "human",
        "camera": cfg.game.camera,
        "renderer": "takeover",
        "agent_ckpt": ckpt,
    }
    ep = seg.to_episode(cfg.data, meta)
    path = save_episode(ep, cfg.data.root)
    append_index(cfg.data.root, ep, path)
    return path


def _drive_player(
    game: Any,
    policy: ChunkPolicy,
    cfg: Config,
    tracker: ProgressTracker,
    control: TakeoverControl,
    state: GameState,
    read_keys: KeyReader,
    on_frame: Callable[[ProgressTracker], None] | None,
) -> tuple[GameState, PlayerSegment]:
    """Realtime + passthrough until hand back (H / request) or the finish; the policy keeps
    observing the frames so its history is current when it drives again."""
    d, r = cfg.data, cfg.data.actions_per_frame
    control.phase = "player"
    game.hand_to_player()
    seg = PlayerSegment(start_ms=int(state.race_time_ms))
    frame_ms = 1000.0 / d.frame_hz
    next_obs = float(state.race_time_ms)
    last_frame_t = -1
    while True:
        if on_frame is not None:
            on_frame(tracker)
        st = game.get_state()
        left, right, gas, brake, hand_back = read_keys()
        action = keys_to_action(left, right, gas, brake)
        pos = np.asarray(st.position, np.float32).reshape(3)
        seg.inputs.append((int(st.race_time_ms), action, pos, float(st.speed_kmh)))
        tracker.update(st, judge=False)
        frame = game.latest_frame()
        if frame is not None and frame.race_time_ms > last_frame_t:
            last_frame_t = frame.race_time_ms
            seg.frames.append((frame.race_time_ms, frame.image))
            if frame.race_time_ms >= next_obs:
                policy.observe(frame.image, np.tile(action, (r, 1)))
                next_obs += frame_ms
        if tracker.done:
            seg.finished = tracker.reason == "finished"
            break
        if hand_back or control.pop() == "hand_back":
            break
        time.sleep(PLAYER_POLL_S)
    state = game.take_back()
    tracker.revive()
    control.phase = "agent"
    return state, seg


def run_takeover_episode(
    game: Any,
    policy: ChunkPolicy,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
    control: TakeoverControl,
    on_frame: Callable[[ProgressTracker], None] | None = None,
    on_segment: Callable[[PlayerSegment], None] | None = None,
    read_keys: KeyReader = read_keyboard,
) -> dict:
    """One interactive episode; the game needs hand_to_player() / take_back() (TMNFGame).

    Ends at the finish, at a timeout, or when on_frame raises. Stuck / offtrack do not end
    it: the game pauses in phase "waiting" until a takeover (arrow key / request) or a
    "retry" request.
    """
    d = cfg.data
    rng = np.random.default_rng(seed)
    policy.reset()
    control.phase = "agent"
    game.load_map(map_ref)
    state = game.start_race()
    tracker = ProgressTracker(ref, cfg.eval)
    tracker.update(state)
    past = np.zeros((d.actions_per_frame, 3), np.float32)
    k = 0
    takeovers, player_ms = 0, 0
    while not tracker.done:
        if on_frame is not None:
            on_frame(tracker)
        req = control.pop()
        if req == "take_over" or any(read_keys()[:4]):
            state, seg = _drive_player(
                game, policy, cfg, tracker, control, state, read_keys, on_frame
            )
            takeovers += 1
            player_ms += seg.duration_ms
            if on_segment is not None:
                on_segment(seg)
            past = np.tile(seg.last_action(), (d.actions_per_frame, 1))
            k = int(state.race_time_ms) * d.frame_hz // 1000
            continue
        if control.phase == "waiting":
            if req == "retry":
                control.phase = "agent"
            else:
                time.sleep(WAIT_POLL_S)
            continue
        frame = game.grab_frame()
        policy.observe(frame.image, past)
        rows = policy_rows(policy, cfg, rng)
        state = drive_frame_period(game, rows, state, k, cfg, tracker)
        past = rows
        k += 1
        if tracker.done and tracker.reason in ("stuck", "offtrack"):
            tracker.revive()
            control.phase = "waiting"
    control.phase = "agent"
    result = _episode_result(map_ref, seed, "takeover", tracker, ref)
    return {**result, "frames": k, "takeovers": takeovers, "player_ms": player_ms}
