"""ActionScheduler: time-stamped action chunks -> the action to send at any wall time."""

from __future__ import annotations

import math
import threading
from typing import NamedTuple

import numpy as np

from tmagent.interfaces import ACTION_DIM, NEUTRAL_ACTION, Action

_EPS = 1e-6  # row-index tolerance so that t == t0 + j / hz selects row j despite rounding


class _Chunk(NamedTuple):
    t0: float
    t_last: float  # wall time of the last row
    rows: list[list[float]]  # [C][3] python floats (fast scalar access)


class ActionScheduler:
    """Holds the two newest chunks; `action_at` is pure arithmetic (microseconds).

    Row j of a chunk published with `t0_wall` applies at `t0_wall + j / control_hz`.
    Steer is linearly interpolated between rows; gas and brake are sample-and-hold
    (row floor). Chunks and the lock are only touched to swap references.
    """

    def __init__(self, control_hz: int, hold_s: float = 0.25) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self._hz = float(control_hz)
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self._cur: _Chunk | None = None
        self._prev: _Chunk | None = None

    def publish(self, chunk: np.ndarray, t0_wall: float) -> None:
        """Install a float32 [C, 3] chunk; the newest publish wins where it covers t."""
        arr = np.array(chunk, dtype=np.float32)  # copy: the caller may reuse its buffer
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] != ACTION_DIM:
            raise ValueError(f"chunk must have shape [C>=1, {ACTION_DIM}], got {arr.shape}")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
        arr[:, 0] = np.clip(arr[:, 0], -1.0, 1.0)
        arr[:, 1:] = np.clip(arr[:, 1:], 0.0, 1.0)
        new = _Chunk(t0_wall, t0_wall + (arr.shape[0] - 1) / self._hz, arr.tolist())
        with self._lock:
            self._prev, self._cur = self._cur, new

    def clear(self) -> None:
        """Forget all chunks (action_at returns NEUTRAL_ACTION until the next publish)."""
        with self._lock:
            self._prev = self._cur = None

    def action_at(self, t_wall: float) -> Action:
        with self._lock:
            cur, prev = self._cur, self._prev
        if cur is None:
            return NEUTRAL_ACTION
        if t_wall >= cur.t0:
            chunk = cur
        elif prev is not None and t_wall >= prev.t0:
            chunk = prev  # the newest chunk has not started yet
        else:
            return NEUTRAL_ACTION
        return self._eval(chunk, t_wall)

    def _eval(self, chunk: _Chunk, t: float) -> Action:
        rows = chunk.rows
        if t > chunk.t_last + self._hold_s:
            return NEUTRAL_ACTION
        x = (t - chunk.t0) * self._hz
        last = len(rows) - 1
        i = int(math.floor(x + _EPS))
        if i >= last:  # at or past the last row: hold it (until hold_s ran out above)
            r = rows[last]
            return Action(r[0], r[1], r[2])
        frac = min(max(x - i, 0.0), 1.0)
        a, b = rows[i], rows[i + 1]
        return Action(a[0] + frac * (b[0] - a[0]), a[1], a[2])
