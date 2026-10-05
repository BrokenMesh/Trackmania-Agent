"""Reference path, track progress and episode-end detection for closed-loop evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmagent.config import EvalConfig
from tmagent.interfaces import GameState

GRACE_MS = 2000  # no stuck detection during the first 2 s of a race
OFFTRACK_MS = 1000  # off-track must last this long (> offtrack_dist) to end the episode
BEHIND_M = 20.0  # projection window reaches this far behind the hint
TRACK_WINDOW_M = 100.0  # how far ahead of its last projection the tracker looks


class ReferencePath:
    """Polyline (N, 3) with arc-length parametrization and windowed projection."""

    def __init__(self, points: np.ndarray, time_ms: int | None = None) -> None:
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
            raise ValueError(f"reference points must have shape (N>=2, 3), got {pts.shape}")
        self.points = pts
        self.time_ms = time_ms  # reference run time if known
        d = np.diff(pts, axis=0)
        self._d = d
        self._len = np.linalg.norm(d, axis=1)
        self._inv = 1.0 / np.maximum(self._len**2, 1e-12)
        self.s = np.concatenate([[0.0], np.cumsum(self._len)])
        self.length = float(self.s[-1])

    def project(
        self, pos: np.ndarray, hint_s: float | None = None, window: float = 200.0
    ) -> tuple[float, float]:
        """Nearest point on the path -> (arc length s, distance).

        With `hint_s` only segments within [hint_s - 20, hint_s + window] are searched,
        so the projection cannot jump to a different part of a self-approaching track.
        """
        n = len(self._d)
        lo, hi = 0, n
        if hint_s is not None:
            lo = max(int(np.searchsorted(self.s, hint_s - BEHIND_M, side="right")) - 1, 0)
            hi = min(int(np.searchsorted(self.s, hint_s + window, side="left")), n)
            lo = min(lo, n - 1)
            hi = max(hi, lo + 1)
        p = np.asarray(pos, dtype=np.float64)[:3] - self.points[lo:hi]
        d = self._d[lo:hi]
        t = np.clip(np.einsum("ij,ij->i", p, d) * self._inv[lo:hi], 0.0, 1.0)
        q = p - t[:, None] * d
        d2 = np.einsum("ij,ij->i", q, q)
        k = int(np.argmin(d2))
        return float(self.s[lo + k] + t[k] * self._len[lo + k]), float(np.sqrt(d2[k]))

    def progress(self, s: float) -> float:
        """Arc length -> fraction of the path in [0, 1]."""
        return float(min(max(s / self.length, 0.0), 1.0)) if self.length > 0 else 0.0


class ProgressTracker:
    """Feed it GameStates in race-time order; it tracks progress and decides when to stop.

    Done reasons: "finished", "timeout" (race time > timeout_s), "stuck" (speed below
    stuck_speed_kmh for stuck_s, only counted after the first 2 s) and "offtrack"
    (distance to the path above offtrack_dist for more than 1 s).
    """

    def __init__(self, ref: ReferencePath, cfg: EvalConfig) -> None:
        self.ref, self.cfg = ref, cfg
        self.done = False
        self.reason: str | None = None
        self.s = 0.0  # last projection (arc length); the car starts at the path start
        self.max_s = 0.0
        self.dist = 0.0
        self.max_dist = 0.0
        self.offtrack_ms = 0
        self.finish_time_ms: int | None = None
        self.time_ms = 0
        self._dist_sum, self._n = 0.0, 0
        self._last_t: int | None = None
        self._slow_since: int | None = None
        self._off_since: int | None = None

    @property
    def progress(self) -> float:
        return self.ref.progress(self.max_s)

    @property
    def mean_dist(self) -> float:
        return self._dist_sum / self._n if self._n else 0.0

    def finish(self, reason: str) -> None:
        """End the episode from outside (e.g. a wall-clock guard)."""
        if not self.done:
            self.done, self.reason = True, reason

    def update(self, state: GameState) -> None:
        """Account for one state; sets `done` and `reason` when the episode should end."""
        if self.done:
            return
        t = int(state.race_time_ms)
        dt = 0 if self._last_t is None else max(t - self._last_t, 0)
        self._last_t, self.time_ms = t, t
        self.s, self.dist = self.ref.project(state.position, self.s, TRACK_WINDOW_M)
        self.max_s = max(self.max_s, self.s)
        self.max_dist = max(self.max_dist, self.dist)
        self._dist_sum += self.dist
        self._n += 1
        cfg = self.cfg
        if state.finished:
            self.max_s = self.ref.length
            self.finish_time_ms = t
            self.finish("finished")
            return
        if self.dist > cfg.offtrack_dist:
            self.offtrack_ms += dt
            if self._off_since is None:
                self._off_since = t
            if t - self._off_since > OFFTRACK_MS:
                self.finish("offtrack")
                return
        else:
            self._off_since = None
        if t >= GRACE_MS and state.speed_kmh < cfg.stuck_speed_kmh:
            if self._slow_since is None:
                self._slow_since = t
            if t - self._slow_since >= cfg.stuck_s * 1000:
                self.finish("stuck")
                return
        else:
            self._slow_since = None
        if t > cfg.timeout_s * 1000:
            self.finish("timeout")

    def result(self) -> dict:
        """Episode metrics collected so far."""
        return {
            "finished": self.reason == "finished",
            "finish_time_ms": self.finish_time_ms,
            "race_time_ms": self.time_ms,
            "progress": self.progress,
            "reason": self.reason,
            "mean_dist": self.mean_dist,
            "max_dist": self.max_dist,
            "offtrack_s": self.offtrack_ms / 1000.0,
        }


def load_reference(map_ref: str, data_root: str | Path) -> ReferencePath:
    """Reference path of a map: fake maps come from the fake track, others from
    `<data_root>/refs/<map_uid>.npy` (positions (N, 3); map_uid = file name up to the first
    dot). An optional `refs/<map_uid>.json` {"race_time_ms": int, "source": str} sets the
    reference run time (`ReferencePath.time_ms`)."""
    if map_ref.startswith("fake:"):
        from tmagent.game.fake import track_reference

        return ReferencePath(track_reference(map_ref))
    refs = Path(data_root) / "refs"
    uid = Path(map_ref).name.split(".")[0]
    path = refs / f"{uid}.npy"
    if not path.is_file():
        raise FileNotFoundError(f"no reference path for {map_ref!r}: expected {path}")
    time_ms = None
    meta = path.with_suffix(".json")
    if meta.is_file():
        time_ms = int(json.loads(meta.read_text())["race_time_ms"])
    return ReferencePath(np.load(path), time_ms=time_ms)
