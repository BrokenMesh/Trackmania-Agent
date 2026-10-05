"""Shared contracts between tmagent modules.

Every module (data, model, train, runtime, eval, game) talks to the others only
through the types and conventions defined here. Changing anything in this file
requires an entry in docs/decisions.md.

Time conventions (see docs/ARCHITECTURE.md, "Timing"):
- All game times are integer milliseconds of race time (0 = race start).
- TMNF physics advances in fixed ticks of PHYSICS_TICK_MS.
- A frame stamped t shows the game state at race time t, BEFORE the action
  stamped t is applied. The action stamped t is the input held from t until
  the next control step. So the policy observes frame t and predicts actions
  t, t+dt, ..., t+(chunk_len-1)*dt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

PHYSICS_TICK_MS = 10  # TMNF simulation step. Verified source: docs/research.md

# Action vector layout used everywhere (datasets, model outputs, runtime).
ACTION_DIM = 3
STEER, GAS, BRAKE = 0, 1, 2


@dataclass(frozen=True)
class Action:
    """One control input.

    steer: [-1, 1], negative = left, positive = right.
    gas:   [0, 1]  (keyboard players produce only 0 or 1).
    brake: [0, 1]  (binary in practice).
    """

    steer: float = 0.0
    gas: float = 0.0
    brake: float = 0.0

    def to_array(self) -> np.ndarray:
        return np.array([self.steer, self.gas, self.brake], dtype=np.float32)

    @staticmethod
    def from_array(a: np.ndarray) -> Action:
        a = np.asarray(a, dtype=np.float32).reshape(ACTION_DIM)
        return Action(
            steer=float(np.clip(a[STEER], -1.0, 1.0)),
            gas=float(np.clip(a[GAS], 0.0, 1.0)),
            brake=float(np.clip(a[BRAKE], 0.0, 1.0)),
        )


NEUTRAL_ACTION = Action()


@dataclass
class GameState:
    """Simulation state as reported by the game bridge."""

    race_time_ms: int
    position: np.ndarray  # (3,) float32 world coordinates
    velocity: np.ndarray  # (3,) float32 world units per second
    speed_kmh: float
    checkpoint: int  # number of checkpoints passed in the current run
    finished: bool
    num_checkpoints: int | None = None  # total incl. finish, if known
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Frame:
    """One captured image."""

    image: np.ndarray  # uint8, shape (H, W, C), C in {1, 3}
    race_time_ms: int  # race time shown in the image, -1 if unknown
    wall_time: float  # time.perf_counter() when the capture completed


@runtime_checkable
class SyncGame(Protocol):
    """Tick-synchronous game access (used for rendering replays and eval).

    The game does not advance unless step() is called. Implementations:
    tmagent.game.fake.FakeGame (tests), tmagent.game.tmnf (TMNF via TMInterface).
    """

    def load_map(self, map_ref: str) -> None: ...

    def start_race(self) -> GameState:
        """Restart the current map and return the state at race_time_ms == 0."""
        ...

    def step(self, action: Action, n_ticks: int = 1) -> GameState:
        """Hold `action` for n_ticks physics ticks; return the resulting state."""
        ...

    def grab_frame(self) -> Frame:
        """Render the current state at the configured capture resolution."""
        ...

    def close(self) -> None: ...


@runtime_checkable
class RealtimeGame(Protocol):
    """Free-running game access (used for live play).

    The game advances on its own clock. set_action() must return quickly
    (< 1 ms) and never block on the model.
    """

    def load_map(self, map_ref: str) -> None: ...

    def restart(self) -> None: ...

    def set_action(self, action: Action) -> None: ...

    def get_state(self) -> GameState: ...

    def latest_frame(self) -> Frame | None:
        """Most recent captured frame, or None if none yet. Non-blocking."""
        ...

    def close(self) -> None: ...


@dataclass
class InputTimeline:
    """Driver inputs of one run on the physics-tick grid.

    actions[i] is the input held during tick i, i.e. from race time
    i * PHYSICS_TICK_MS (inclusive) to (i + 1) * PHYSICS_TICK_MS. Produced from
    replay files (tmagent.game.tmnf.replay) or scripted drivers (FakeGame).
    """

    actions: np.ndarray  # float32 (N_ticks, ACTION_DIM)
    meta: dict[str, Any] = field(default_factory=dict)  # map_uid, player, source, ...


@runtime_checkable
class ChunkPolicy(Protocol):
    """What the runtime needs from a model (tmagent.model.streaming.StreamingPolicy).

    Call order per frame step: observe(...) then predict().
    """

    chunk_len: int

    def reset(self) -> None: ...

    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        """image: uint8 (H, W, C) at the data resolution.
        past_actions: float32 (R, ACTION_DIM), the R = control_hz // frame_hz
        actions executed since the previous observe (oldest first); zeros on
        the first call."""
        ...

    def predict(self) -> np.ndarray:
        """float32 (chunk_len, ACTION_DIM); row 0 applies at the time of the
        last observed frame, row j at that time + j / control_hz seconds."""
        ...


# --------------------------------------------------------------------------
# Episode storage (written by tools/render_replays.py, read by tmagent.data)
# --------------------------------------------------------------------------


@dataclass
class Episode:
    """One rendered run on one map.

    frames are sampled at frame_hz, actions and states at control_hz, both on
    exact multiples of their period starting at race time 0.
    """

    frames: np.ndarray  # uint8 (T_f, H, W, C)
    frame_times_ms: np.ndarray  # int64 (T_f,)
    actions: np.ndarray  # float32 (T_a, ACTION_DIM)
    action_times_ms: np.ndarray  # int64 (T_a,)
    positions: np.ndarray  # float32 (T_a, 3) car position at action_times_ms
    speeds_kmh: np.ndarray  # float32 (T_a,)
    meta: dict[str, Any]  # see EPISODE_META_KEYS


EPISODE_META_KEYS = (
    "episode_id",  # str, unique
    "map_uid",  # str, split key
    "map_name",  # str
    "source",  # str, e.g. "tmx:<replay_id>", "fake:<seed>", "selfplay:<run>"
    "player",  # str or "" if unknown
    "race_time_ms",  # int, final time if finished else last time
    "finished",  # bool
    "frame_hz",  # int
    "control_hz",  # int
    "resolution",  # [W, H]
    "channels",  # 1 or 3
    "camera",  # str, e.g. "cam1"
    "renderer",  # str, tool + version / git hash
)

# Optional meta keys written by tmagent.data.render / episode_io.
EPISODE_META_OPTIONAL_KEYS = (
    "format",  # int, storage format version (2 = per-frame zlib, see ARCHITECTURE.md)
    "finish_time_ms",  # int | None, race time when the game reported finished
    "expected_time_ms",  # int | None, finish time stored in the replay
    "desync",  # bool, re-driven run did not reproduce the replay's finish time
    "frame_time_mismatch",  # int, frames whose reported race time != tick time at grab
    "resized_frames",  # int, frames resized because the capture size was wrong
    "respawns",  # int, respawn events in the source replay
)
