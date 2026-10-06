"""Line-following expert for DAgger: labels any car state with a corrective keyboard action.

The expert follows one human reference run of the map (positions, actions, speeds at the
control rate). Near the line it copies the human input at that point of the track; off the
line (or misaligned) it steers towards a lookahead point on the line (pure pursuit with a
dead band). Gas / brake follow the human, gas is released when the car is much faster than
the human was there.

VERIFIED convention (TMNF episodes): with heading = atan2(dz, dx) a left key press turns the
heading towards negative angles, so steer = sign(angle from heading to target). The fake game
uses the x/y plane and the opposite sign (ExpertConfig.plane, right_turn_sign).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmagent.data.episode_io import load_episode, read_index


@dataclass
class ExpertConfig:
    on_line_m: float = 1.5  # lateral error below this (and aligned): copy the human
    aligned_rad: float = 0.12  # heading error below this counts as aligned
    lookahead_s: float = 0.45  # pure pursuit lookahead in seconds of travel
    min_lookahead_m: float = 6.0
    dead_band_rad: float = 0.03  # pure pursuit: |angle| below this -> no steer
    overspeed_kmh: float = 25.0  # release gas when faster than the human by this much
    search_back: int = 30  # projection window around the last match, in reference samples
    search_ahead: int = 240
    plane: tuple[int, int] = (0, 2)  # ground-plane axes of positions (TMNF: x, z; y is up)
    right_turn_sign: float = 1.0  # sign of the heading change of a right turn (TMNF: +1)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    """Signed angle from 2D vector a to b."""
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


class LineExpert:
    """Corrective expert along one reference run; call reset() at the start of each episode."""

    def __init__(
        self, positions: np.ndarray, actions: np.ndarray, speeds_kmh: np.ndarray,
        cfg: ExpertConfig | None = None,
    ) -> None:  # fmt: skip
        self.p = np.asarray(positions, np.float64)
        self.a = np.asarray(actions, np.float32)
        self.v = np.asarray(speeds_kmh, np.float64)
        self.cfg = cfg or ExpertConfig()
        seg = np.linalg.norm(np.diff(self.p, axis=0), axis=1)
        self.arc = np.concatenate([[0.0], np.cumsum(seg)])
        self.i = 0
        self._axes = list(self.cfg.plane)

    def _2d(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v, np.float64)[..., self._axes]

    @classmethod
    def from_reference(
        cls, root: str | Path, map_uid: str, cfg: ExpertConfig | None = None
    ) -> LineExpert:
        """Expert on the map's reference run (refs/<uid>.json, written by render_replays)."""
        info = json.loads((Path(root) / "refs" / f"{map_uid}.json").read_text())
        entries = {e["episode_id"]: e for e in read_index(root)}
        ep = load_episode(Path(root) / entries[info["episode_id"]]["path"])
        return cls(ep.positions, ep.actions, ep.speeds_kmh, cfg)

    def reset(self) -> None:
        self.i = 0

    def nearest(self, position: np.ndarray) -> tuple[int, float]:
        """Closest reference sample in a window around the last match, and its distance."""
        lo = max(self.i - self.cfg.search_back, 0)
        hi = min(self.i + self.cfg.search_ahead, len(self.p))
        d = np.linalg.norm(self.p[lo:hi] - np.asarray(position, np.float64), axis=1)
        j = int(np.argmin(d))
        self.i = lo + j
        return self.i, float(d[j])

    def act(self, position: np.ndarray, velocity: np.ndarray, speed_kmh: float) -> np.ndarray:
        """[steer, gas, brake] for this state (steer in {-1, 0, 1})."""
        c = self.cfg
        i, dist = self.nearest(position)
        human = self.a[i].copy()
        nxt = min(i + 1, len(self.p) - 1)
        line_dir = self._2d(self.p[nxt] - self.p[max(nxt - 1, 0)])
        heading = self._2d(velocity) if speed_kmh > 5.0 else line_dir
        if not heading.any():
            heading = np.array([1.0, 0.0])
        look = max(c.min_lookahead_m, speed_kmh / 3.6 * c.lookahead_s)
        j = min(int(np.searchsorted(self.arc, self.arc[i] + look)), len(self.p) - 1)
        to_target = self._2d(self.p[j] - np.asarray(position, np.float64))
        alpha = _angle(heading, to_target) if to_target.any() else 0.0
        aligned = line_dir.any() and abs(_angle(heading, line_dir)) < c.aligned_rad
        out = human
        if not (dist < c.on_line_m and aligned):
            out[0] = 0.0 if abs(alpha) < c.dead_band_rad else float(np.sign(alpha) * c.right_turn_sign)
        if speed_kmh > self.v[i] + c.overspeed_kmh:
            out[1] = 0.0
        return out

    @property
    def progress(self) -> float:
        return float(self.arc[self.i] / self.arc[-1]) if self.arc[-1] > 0 else 0.0
