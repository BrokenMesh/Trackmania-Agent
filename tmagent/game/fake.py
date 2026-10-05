"""FakeGame: a deterministic 2D racer that stands in for TMNF in tests and CI.

Implements both `SyncGame` (tick-synchronous) and `RealtimeGame` (background threads).
World units are meters; the horizontal plane is (x, y) and z is always 0. Maps are
`fake:oval` (one closed lap), `fake:s_curve` and `fake:random:<seed>` (open tracks).

Physics (every PHYSICS_TICK_MS): kinematic bicycle model without randomness. Gas and
brake set the longitudinal acceleration, steering sets the yaw rate (scaled with speed,
capped by a lateral grip limit), off-track the car gets strong drag and 40 km/h max.
After the finish line the race is frozen: step() returns the final state unchanged.

Frames are a car-centric view rendered with numpy from a per-map raster of the track:
the heading points up, the car sits at the bottom center, about 80 m are visible ahead,
and a thin speed bar on the top rows plays the role of the HUD.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from tmagent.config import DataConfig, GameConfig
from tmagent.interfaces import PHYSICS_TICK_MS, Action, Frame, GameState, InputTimeline

DT = PHYSICS_TICK_MS / 1000.0
TRACK_WIDTH = 12.0  # m
NUM_CHECKPOINTS = 5  # incl. the finish line
SPACING = 2.0  # centerline sample spacing, m
RUN_EXT = 16.0  # road before the start / after the finish of open tracks, m

# Car model.
WHEELBASE = 2.8
MAX_STEER_RAD = 0.45  # wheel angle at full lock, low speed
A_LAT = 22.0  # lateral acceleration limit, m/s^2
ENGINE_ACC = 14.0  # m/s^2 at standstill, tapers to 0 at V_ENGINE
V_ENGINE = 60.0
BRAKE_DEC = 25.0
DRAG_LIN, DRAG_QUAD = 0.05, 0.0015
STEER_TAU = 0.05  # first-order steering lag, s
OFF_DRAG, OFF_ROLL = 1.5, 2.0  # extra deceleration off track: OFF_DRAG * v + OFF_ROLL
OFF_MAX_V = 40.0 / 3.6
OFF_DECEL = 30.0  # how fast speed is pulled down to OFF_MAX_V, m/s^2
_TAN_MAX_STEER = math.tan(MAX_STEER_RAD)
_STEER_ALPHA = 1.0 - math.exp(-DT / STEER_TAU)

# Rendering.
RASTER_RES = 0.5  # m per raster pixel
RASTER_MARGIN = 110.0  # grass around the track in the raster, m
VIEW_FORWARD_M = 80.0
EDGE_M = 0.7  # width of the white edge line
# Raster labels.
GRASS, ASPHALT, EDGE, CHECKPOINT, FINISH = range(5)
_PALETTE_RGB = np.array(
    [(40, 120, 45), (110, 110, 118), (235, 235, 235), (255, 215, 0), (200, 40, 220)], np.uint8
)
_PALETTE_GRAY = np.array([75, 135, 240, 195, 25], np.uint8)[:, None]
_CAR_RGB = ((235, 60, 20), (255, 200, 150), (30, 30, 40))  # body, windshield, wheels
_CAR_GRAY = (15, 250, 0)
_BAR_RGB, _BAR_GRAY = (0, 220, 255), 255
_BAR_MAX_KMH = 180.0


def max_yaw_rate(v: float) -> float:
    """Yaw rate in rad/s at full steer lock: bicycle model capped by lateral grip."""
    if v < 1e-6:
        return 0.0
    return min(v / WHEELBASE * _TAN_MAX_STEER, A_LAT / v)


def steer_ratio(v: float) -> float:
    """v / max_yaw_rate(v): the tightest turning radius at speed v (m)."""
    return max(WHEELBASE / _TAN_MAX_STEER, v * v / A_LAT)


# --------------------------------------------------------------------------
# Track geometry
# --------------------------------------------------------------------------


def _resample(pts: np.ndarray, spacing: float, closed: bool = False) -> np.ndarray:
    """Uniform arc-length resampling of a polyline; the last point is always kept."""
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 2)
    t = np.linspace(0.0, s[-1], n + 1)
    out = np.stack([np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])], axis=1)
    if closed:
        out[-1] = out[0]
    return out


def _from_curvature(segments: list[tuple[float, float]], smooth_m: float) -> np.ndarray:
    """Integrate a piecewise-constant curvature profile [(length m, 1/radius)], smoothed."""
    ds = 0.5
    kappa = np.concatenate([np.full(int(round(length / ds)), k) for length, k in segments])
    w = max(int(smooth_m / ds), 1)
    padded = np.concatenate([np.full(w, kappa[0]), kappa, np.full(w, kappa[-1])])
    kappa = np.convolve(padded, np.ones(w) / w, mode="same")[w:-w]
    theta = np.cumsum(kappa) * ds
    pts = np.stack([np.cumsum(np.cos(theta)), np.cumsum(np.sin(theta))], axis=1) * ds
    return _resample(np.vstack([[0.0, 0.0], pts]), SPACING)


def _oval() -> np.ndarray:
    a, b = 130.0, 75.0
    th = np.linspace(-np.pi / 2, 3 * np.pi / 2, 4000)
    dense = np.stack([a * np.cos(th), b * np.sin(th)], axis=1)
    return _resample(dense, SPACING, closed=True)


def _s_curve() -> np.ndarray:
    left, right = 1 / 60.0, -1 / 60.0
    arc = lambda deg, k: (math.radians(deg) / abs(k), k)  # noqa: E731
    segs = [(60.0, 0.0), arc(70, left), arc(140, right), arc(70, left), (60.0, 0.0)]
    return _from_curvature(segs, smooth_m=24.0)


def _catmull_rom(p: np.ndarray, samples: int = 24) -> np.ndarray:
    """Uniform Catmull-Rom spline through the control points p (n, 2)."""
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1 : i + 3]
        out.append(
            0.5
            * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (3 * p1 - p0 - 3 * p2 + p3) * t**3
            )
        )
    out.append(p[-2][None])
    return np.vstack(out)


def _curvature(pts: np.ndarray) -> np.ndarray:
    """Signed curvature (1/m, left positive) at every point of a polyline, lightly smoothed."""
    d = np.diff(pts, axis=0)
    ang = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    seg = np.hypot(d[:, 0], d[:, 1])
    k = np.diff(ang) / (0.5 * (seg[1:] + seg[:-1]))
    k = np.concatenate([[k[0]], k, [k[-1]]])
    return np.convolve(np.pad(k, 2, mode="edge"), np.ones(5) / 5, mode="valid")


def _random_track(seed: int) -> np.ndarray:
    """Smooth open track: random-walk control points, Catmull-Rom spline, then
    rejection of draws with a tight corner (< 30 m radius) or a near self-approach."""
    for attempt in range(1000):
        rng = np.random.default_rng([seed % 2**32, attempt])
        n = int(rng.integers(9, 13))
        heading, p, ctrl = 0.0, np.zeros(2), [np.zeros(2)]
        for i in range(n):
            if i > 0:
                heading += float(np.clip(rng.normal(0.0, 0.5), -0.9, 0.9))
            p = p + rng.uniform(70.0, 110.0) * np.array([math.cos(heading), math.sin(heading)])
            ctrl.append(p)
        pts = _resample(_catmull_rom(np.array(ctrl)), SPACING)
        if np.abs(_curvature(pts)).max() > 1 / 30.0:
            continue
        d = np.hypot(*(pts[:, None] - pts[None]).transpose(2, 0, 1))
        near_in_index = np.abs(np.arange(len(pts))[:, None] - np.arange(len(pts))[None]) < 45
        if d[~near_in_index].min() > TRACK_WIDTH + 6.0:
            return pts
    raise RuntimeError(f"no valid random track for seed {seed}")


class _Polyline:
    """Polyline with arc length and a windowed nearest-segment search."""

    BEHIND, AHEAD = 4, 16  # search window around the hint, in segments

    def __init__(self, pts: np.ndarray, closed: bool, fallback: float) -> None:
        self.pts, self.closed, self.fallback = pts, closed, fallback
        d = np.diff(pts, axis=0)
        self.seg_len = np.hypot(d[:, 0], d[:, 1])
        self.length = float(self.seg_len.sum())
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.n = len(d)
        self._ax, self._ay = pts[:-1, 0].copy(), pts[:-1, 1].copy()
        self._dx, self._dy = d[:, 0].copy(), d[:, 1].copy()
        self._inv = 1.0 / np.maximum(self.seg_len**2, 1e-12)
        self._off = np.arange(-self.BEHIND, self.AHEAD)

    def _search(self, x: float, y: float, idx: np.ndarray) -> tuple[int, float, float]:
        px, py = x - self._ax[idx], y - self._ay[idx]
        dx, dy = self._dx[idx], self._dy[idx]
        t = np.clip((px * dx + py * dy) * self._inv[idx], 0.0, 1.0)
        qx, qy = px - t * dx, py - t * dy
        d2 = qx * qx + qy * qy
        k = int(np.argmin(d2))
        return int(idx[k]), float(t[k]), math.sqrt(float(d2[k]))

    def nearest(self, x: float, y: float, hint: int) -> tuple[int, float, float]:
        """(segment index, position t in [0, 1] on it, distance); full search as a fallback."""
        idx = hint + self._off
        if self.closed:
            idx %= self.n
        else:
            np.clip(idx, 0, self.n - 1, out=idx)
        i, t, d = self._search(x, y, idx)
        if d > self.fallback:
            i, t, d = self._search(x, y, np.arange(self.n))
        return i, t, d

    def s_at(self, i: int, t: float) -> float:
        return float(self.s[i] + t * self.seg_len[i])

    def point_at(self, s: float) -> tuple[float, float]:
        return float(np.interp(s, self.s, self.pts[:, 0])), float(
            np.interp(s, self.s, self.pts[:, 1])
        )


def _disc(r: int) -> np.ndarray:
    yy, xx = np.ogrid[-r : r + 1, -r : r + 1]
    return xx * xx + yy * yy <= r * r


def _stamp(label: np.ndarray, px: np.ndarray, py: np.ndarray, r: int, value: int) -> None:
    mask = _disc(r)
    for cx, cy in zip(px.tolist(), py.tolist(), strict=True):
        label[cy - r : cy + r + 1, cx - r : cx + r + 1][mask] = value


@dataclass(frozen=True)
class Gate:
    """Checkpoint or finish line: crossed in direction `t` within +-`half` along `n`."""

    cx: float
    cy: float
    tx: float
    ty: float
    nx: float
    ny: float
    half: float


class Track:
    """Immutable track data shared by all FakeGame instances on the same map."""

    def __init__(self, map_ref: str, center: np.ndarray, closed: bool) -> None:
        self.map_ref, self.closed, self.width = map_ref, closed, TRACK_WIDTH
        self.center = center  # (N, 2) official centerline, closed: last == first
        poly = center
        if not closed:
            t0 = center[1] - center[0]
            t1 = center[-1] - center[-2]
            t0, t1 = t0 / np.hypot(*t0), t1 / np.hypot(*t1)
            k = np.arange(int(RUN_EXT / SPACING), 0, -1)[:, None] * SPACING
            poly = np.vstack([center[0] - k * t0, center, center[-1] + k[::-1] * t1])
        self.road = _Polyline(poly, closed, fallback=self.width / 2)
        self.kappa = _curvature(poly)  # per road point
        self.start = (float(center[0, 0]), float(center[0, 1]))
        d0 = center[1] - center[0]
        self.start_yaw = float(math.atan2(d0[1], d0[0]))
        self.length = float(np.hypot(*np.diff(center, axis=0).T).sum())
        self.gates = self._make_gates()
        self._build_raster()

    def _make_gates(self) -> list[Gate]:
        cl = _Polyline(self.center, self.closed, self.width)
        gates = []
        for i in range(1, NUM_CHECKPOINTS + 1):
            s = cl.length * i / NUM_CHECKPOINTS
            j = min(int(np.searchsorted(cl.s, s, side="right")) - 1, cl.n - 1)
            c = cl.point_at(s)
            d = cl.pts[j + 1] - cl.pts[j]
            if i == NUM_CHECKPOINTS and self.closed:  # finish == start: average both sides
                d = d + (cl.pts[1] - cl.pts[0])
            d = d / np.hypot(*d)
            gates.append(Gate(c[0], c[1], d[0], d[1], -d[1], d[0], self.width / 2 + 1.0))
        return gates

    def reference(self) -> np.ndarray:
        """Centerline (N, 3) float32 with z = 0, from the start to the finish line."""
        return np.column_stack([self.center, np.zeros(len(self.center))]).astype(np.float32)

    def _build_raster(self) -> None:
        pts = self.road.pts
        lo = pts.min(axis=0) - RASTER_MARGIN - self.width
        hi = pts.max(axis=0) + RASTER_MARGIN + self.width
        self.x0, self.y0 = float(lo[0]), float(lo[1])
        self.rw = int(math.ceil((hi[0] - lo[0]) / RASTER_RES))
        self.rh = int(math.ceil((hi[1] - lo[1]) / RASTER_RES))
        label = np.zeros((self.rh, self.rw), np.uint8)
        dense = _resample(pts, RASTER_RES * 0.8, self.closed)
        px = ((dense[:, 0] - self.x0) / RASTER_RES).astype(np.intp)
        py = ((dense[:, 1] - self.y0) / RASTER_RES).astype(np.intp)
        _stamp(label, px, py, int(round(self.width / 2 / RASTER_RES)), EDGE)
        _stamp(label, px, py, int(round((self.width / 2 - EDGE_M) / RASTER_RES)), ASPHALT)
        for i, g in enumerate(self.gates):
            thick = 2.4 if i == len(self.gates) - 1 else 1.6
            self._paint_gate(label, g, thick, FINISH if i == len(self.gates) - 1 else CHECKPOINT)
        self.flat = label.ravel()
        self.label = label

    def _paint_gate(self, label: np.ndarray, g: Gate, thick: float, value: int) -> None:
        r = int((g.half + 2.0) / RASTER_RES)
        cx, cy = int((g.cx - self.x0) / RASTER_RES), int((g.cy - self.y0) / RASTER_RES)
        ys, xs = np.mgrid[cy - r : cy + r + 1, cx - r : cx + r + 1]
        wx = self.x0 + (xs + 0.5) * RASTER_RES - g.cx
        wy = self.y0 + (ys + 0.5) * RASTER_RES - g.cy
        along, across = wx * g.tx + wy * g.ty, wx * g.nx + wy * g.ny
        mask = (np.abs(along) <= thick / 2) & (np.abs(across) <= self.width / 2 - EDGE_M)
        label[cy - r : cy + r + 1, cx - r : cx + r + 1][mask] = value


@lru_cache(maxsize=32)
def get_track(map_ref: str) -> Track:
    """Build (and cache) the track of a map ref: fake:oval | fake:s_curve | fake:random:<seed>."""
    parts = map_ref.split(":")
    if parts[0] == "fake" and len(parts) == 2 and parts[1] == "oval":
        return Track(map_ref, _oval(), closed=True)
    if parts[0] == "fake" and len(parts) == 2 and parts[1] == "s_curve":
        return Track(map_ref, _s_curve(), closed=False)
    if parts[0] == "fake" and len(parts) == 3 and parts[1] == "random":
        try:
            seed = int(parts[2])
        except ValueError:
            seed = None
        if seed is not None:
            return Track(map_ref, _random_track(seed), closed=False)
    raise ValueError(
        f"FakeGame cannot load {map_ref!r} (use fake:oval, fake:s_curve or fake:random:<seed>)"
    )


def track_reference(map_ref: str) -> np.ndarray:
    """Centerline polyline [N, 3] (z = 0) of a fake map, start to finish, for evaluation."""
    return get_track(map_ref).reference()


# --------------------------------------------------------------------------
# FakeGame
# --------------------------------------------------------------------------


class FakeGame:
    """Sync + realtime fake game. Use `load_map`, then either `start_race` + `step`
    (sync mode) or `restart` (realtime mode: background physics and capture threads).
    `load_map` and `close` reset the mode; sync and realtime calls do not mix."""

    def __init__(
        self, game_cfg: GameConfig, data_cfg: DataConfig, capture_hz: float = 60.0
    ) -> None:
        self.game_cfg, self.data_cfg, self.capture_hz = game_cfg, data_cfg, capture_hz
        self.speed = float(game_cfg.game_speed)
        self._w, self._h = int(data_cfg.resolution[0]), int(data_cfg.resolution[1])
        self._c = int(data_cfg.channels)
        if self._c not in (1, 3):
            raise ValueError("channels must be 1 or 3")
        self._palette = _PALETTE_RGB if self._c == 3 else _PALETTE_GRAY
        self._init_view()
        self._lock = threading.Lock()
        self._mode: str | None = None  # None | "sync" | "realtime"
        self._track: Track | None = None
        self._action = (0.0, 0.0, 0.0)
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ---- view ------------------------------------------------------------

    def _init_view(self) -> None:
        w, h = self._w, self._h
        y_car = h - 1 - max(2, int(round(0.08 * h)))
        scale = VIEW_FORWARD_M / y_car  # m per image pixel
        self._scale = scale
        u = (np.arange(w) - w // 2) * scale / RASTER_RES
        v = (y_car - np.arange(h)) * scale / RASTER_RES
        self._xg = np.broadcast_to(u[None, :], (h, w)).astype(np.float32)  # lateral, right +
        self._fg = np.broadcast_to(v[:, None], (h, w)).astype(np.float32)  # forward
        self._y_car = y_car
        # Car sprite (front up), at least 3 x 5 px.
        cw = max(3, int(round(2.0 / scale)) | 1)
        cl = max(5, int(round(4.5 / scale)))
        spr = np.empty((cl, cw, self._c), np.uint8)
        body, wind, wheel = _CAR_RGB if self._c == 3 else [(g,) for g in _CAR_GRAY]
        spr[:] = body
        spr[: max(1, cl // 4)] = wind
        spr[-max(1, cl // 5) :, :1] = wheel
        spr[-max(1, cl // 5) :, -1:] = wheel
        self._sprite = spr
        self._sy0, self._sx0 = y_car - cl // 2, w // 2 - cw // 2
        self._bar_h = max(1, h // 48)

    def _render(self, x: float, y: float, yaw: float, speed_kmh: float) -> np.ndarray:
        tr = self._track
        assert tr is not None
        c, s = math.cos(yaw), math.sin(yaw)
        # World offset of an image pixel = forward * (c, s) + lateral * (s, -c).
        col = self._fg * c + self._xg * s + (x - tr.x0) / RASTER_RES
        row = self._fg * s - self._xg * c + (y - tr.y0) / RASTER_RES
        np.clip(col, 0, tr.rw - 1, out=col)
        np.clip(row, 0, tr.rh - 1, out=row)
        idx = row.astype(np.intp) * tr.rw + col.astype(np.intp)
        img = self._palette[tr.flat.take(idx)]
        sy, sx = self._sy0, self._sx0
        h, w = self._sprite.shape[:2]
        img[sy : sy + h, sx : sx + w] = self._sprite
        n = int(round(min(speed_kmh / _BAR_MAX_KMH, 1.0) * self._w))
        img[: self._bar_h, :n] = _BAR_RGB if self._c == 3 else _BAR_GRAY
        return img

    # ---- shared simulation core -----------------------------------------

    def _ensure_track(self) -> Track:
        if self._track is None:
            name = self.game_cfg.fake_track
            self.load_map(name if name.startswith("fake:") else f"fake:{name}")
        assert self._track is not None
        return self._track

    def _reset_car(self) -> None:
        tr = self._ensure_track()
        self._x, self._y = tr.start
        self._yaw, self._v, self._steer = tr.start_yaw, 0.0, 0.0
        self._time_ms, self._gate, self._finished = 0, 0, False
        self._seg, self._on_track, self._dist = 0, True, 0.0

    def _advance(self, steer: float, gas: float, brake: float, n: int) -> None:
        """Run n physics ticks holding (steer, gas, brake); stops at the finish line."""
        tr = self._ensure_track()
        road, gates = tr.road, tr.gates
        half = tr.width / 2
        x, y, yaw, v, st = self._x, self._y, self._yaw, self._v, self._steer
        for _ in range(n):
            if self._finished:
                break
            st += (steer - st) * _STEER_ALPHA
            a = gas * ENGINE_ACC * max(0.0, 1.0 - v / V_ENGINE) - DRAG_LIN * v - DRAG_QUAD * v * v
            if brake > 0.0 and v > 0.0:
                a -= brake * BRAKE_DEC
            on_track = self._on_track
            if not on_track:
                a -= OFF_DRAG * v + OFF_ROLL
            v = max(0.0, v + a * DT)
            if not on_track and v > OFF_MAX_V:
                v = max(OFF_MAX_V, v - OFF_DECEL * DT)
            rate = -st * max_yaw_rate(v)
            yaw += rate * DT
            px, py = x, y
            x += v * math.cos(yaw - 0.5 * rate * DT) * DT
            y += v * math.sin(yaw - 0.5 * rate * DT) * DT
            self._seg, _, self._dist = road.nearest(x, y, self._seg)
            self._on_track = self._dist <= half
            self._time_ms += PHYSICS_TICK_MS
            if self._gate < len(gates):
                g = gates[self._gate]
                d0 = (px - g.cx) * g.tx + (py - g.cy) * g.ty
                d1 = (x - g.cx) * g.tx + (y - g.cy) * g.ty
                if d0 < 0.0 <= d1:
                    f = -d0 / (d1 - d0)
                    lat = (px + f * (x - px) - g.cx) * g.nx + (py + f * (y - py) - g.cy) * g.ny
                    if abs(lat) <= g.half:
                        self._gate += 1
                        self._finished = self._gate == len(gates)
        self._x, self._y, self._yaw, self._v, self._steer = x, y, yaw, v, st

    def _state(self) -> GameState:
        v, yaw = self._v, self._yaw
        return GameState(
            race_time_ms=self._time_ms,
            position=np.array([self._x, self._y, 0.0], np.float32),
            velocity=np.array([v * math.cos(yaw), v * math.sin(yaw), 0.0], np.float32),
            speed_kmh=v * 3.6,
            checkpoint=self._gate,
            finished=self._finished,
            num_checkpoints=NUM_CHECKPOINTS,
            extra={
                "yaw": yaw,
                "on_track": self._on_track,
                "dist": self._dist,
                "steer": self._steer,
            },
        )

    def _frame(self) -> Frame:
        with self._lock:
            x, y, yaw, v, t = self._x, self._y, self._yaw, self._v, self._time_ms
        img = self._render(x, y, yaw, v * 3.6)
        return Frame(image=img, race_time_ms=t, wall_time=time.perf_counter())

    @staticmethod
    def _clean(action: Action) -> tuple[float, float, float]:
        return (
            min(1.0, max(-1.0, float(action.steer))),
            min(1.0, max(0.0, float(action.gas))),
            min(1.0, max(0.0, float(action.brake))),
        )

    @property
    def track(self) -> Track:
        return self._ensure_track()

    # ---- SyncGame --------------------------------------------------------

    def load_map(self, map_ref: str) -> None:
        """Select a map (stops realtime threads); call start_race() or restart() next."""
        self._stop_threads()
        self._track = get_track(map_ref)
        self._mode = None
        self._latest = None
        self._reset_car()

    def start_race(self) -> GameState:
        if self._mode == "realtime":
            raise RuntimeError("start_race() in realtime mode; call load_map() or close() first")
        self._mode = "sync"
        with self._lock:
            self._reset_car()
            return self._state()

    def step(self, action: Action, n_ticks: int = 1) -> GameState:
        self._require_sync("step")
        steer, gas, brake = self._clean(action)
        with self._lock:
            self._advance(steer, gas, brake, int(n_ticks))
            return self._state()

    def grab_frame(self) -> Frame:
        self._require_sync("grab_frame")
        return self._frame()

    def _require_sync(self, what: str) -> None:
        if self._mode != "sync":
            raise RuntimeError(
                f"{what}() needs sync mode: "
                + ("realtime threads are running" if self._mode else "call start_race() first")
            )

    # ---- RealtimeGame ----------------------------------------------------

    def restart(self) -> None:
        """Reset the race and start the physics and capture threads (non-blocking)."""
        if self._mode == "sync":
            raise RuntimeError("restart() in sync mode; call load_map() or close() first")
        self._stop_threads()
        with self._lock:
            self._reset_car()
        self._action = (0.0, 0.0, 0.0)
        self._latest = self._frame()
        self._mode = "realtime"
        self._stop = threading.Event()
        self._threads = [
            threading.Thread(target=self._physics_loop, args=(self._stop,), daemon=True),
            threading.Thread(target=self._capture_loop, args=(self._stop,), daemon=True),
        ]
        for th in self._threads:
            th.start()

    def set_action(self, action: Action) -> None:
        if self._mode == "sync":
            raise RuntimeError("set_action() in sync mode; use step()")
        self._action = self._clean(action)

    def get_state(self) -> GameState:
        """Copy of the current state (any mode)."""
        with self._lock:
            return self._state()

    def latest_frame(self) -> Frame | None:
        if self._mode == "sync":
            raise RuntimeError("latest_frame() in sync mode; use grab_frame()")
        return self._latest

    def close(self) -> None:
        """Stop the realtime threads (idempotent). The game can be reused afterwards."""
        self._stop_threads()
        self._mode = None

    def _stop_threads(self) -> None:
        self._stop.set()
        for th in self._threads:
            if th is not threading.current_thread():
                th.join(timeout=2.0)
        self._threads = []

    def _physics_loop(self, stop: threading.Event) -> None:
        t0, done = time.perf_counter(), 0
        tick = DT / self.speed
        while not stop.is_set():
            target = int((time.perf_counter() - t0) / tick)
            if target > done:
                steer, gas, brake = self._action
                with self._lock:
                    self._advance(steer, gas, brake, target - done)
                done = target
            stop.wait(max(0.0, t0 + (done + 1) * tick - time.perf_counter()))

    def _capture_loop(self, stop: threading.Event) -> None:
        period = 1.0 / self.capture_hz
        nxt = time.perf_counter()
        while not stop.is_set():
            self._latest = self._frame()
            nxt += period
            delay = nxt - time.perf_counter()
            if delay <= 0.0:
                nxt = time.perf_counter()
            else:
                stop.wait(delay)


# --------------------------------------------------------------------------
# Scripted expert
# --------------------------------------------------------------------------

_V_CRUISE = 36.0  # m/s
_A_TARGET = 13.0  # lateral acceleration the expert plans for, m/s^2
_B_COMFORT = 9.0  # braking deceleration the expert plans for, m/s^2
_LOOKAHEAD_PTS = 75  # points (150 m) scanned for the speed profile
_KEYBOARD_PERIOD = 4  # ticks between keyboard steer decisions


def scripted_driver(
    map_ref: str,
    data_cfg: DataConfig,
    seed: int = 0,
    noise: float = 0.0,
    skill: float = 1.0,
    max_s: float = 120.0,
    keyboard: bool = False,
    hold_ticks: int = 10,
) -> tuple[InputTimeline, dict]:
    """Pure-pursuit expert driven on a sync FakeGame; returns (timeline, info).

    Gas and brake are binary with hysteresis like a keyboard player. Steering is analog,
    or -1/0/1 pulses (sigma-delta) with keyboard=True. `noise` is the std of a smooth
    steering perturbation (seeded by `seed`); `skill` < 1 scales the target speed.
    The driver decides every `hold_ticks` ticks and holds the action in between; with a
    multiple of 10 (100 ms) the timeline survives resampling at 50 Hz and at 60 Hz with
    20 Hz frames unchanged, so open-loop replays through the control grid are exact.
    info = {finished, race_time_ms}.
    """
    if hold_ticks < 1:
        raise ValueError("hold_ticks must be >= 1")
    game = FakeGame(GameConfig(), data_cfg)
    game.load_map(map_ref)
    road, kappa = game.track.road, game.track.kappa
    state = game.start_race()
    rng = np.random.default_rng(seed)
    rho = math.exp(-hold_ticks * DT / 0.25)  # steering noise correlation per decision
    kb_every = hold_ticks * math.ceil(_KEYBOARD_PERIOD / hold_ticks)
    steer = gas = brake = 0.0
    n_max = int(max_s * 1000 // PHYSICS_TICK_MS)
    acts = np.zeros((n_max, 3), np.float32)
    seg, ou, acc, kb_steer, gas_on, brake_on, n = 0, 0.0, 0.0, 0.0, False, False, 0
    for i in range(n_max):
        if i % hold_ticks == 0:
            x, y = float(state.position[0]), float(state.position[1])
            yaw, v = state.extra["yaw"], state.speed_kmh / 3.6
            seg, t, _ = road.nearest(x, y, seg)
            s = road.s_at(seg, t)
            # Pure pursuit toward a point `ld` ahead on the road.
            ld = min(max(6.0 + 0.45 * v, 6.0), 30.0)
            tx, ty = road.point_at(min(s + ld, road.length))
            wx, wy = tx - x, ty - y
            alpha = math.atan2(wy, wx) - yaw
            alpha = math.atan2(math.sin(alpha), math.cos(alpha))
            steer = -2.0 * math.sin(alpha) / max(math.hypot(wx, wy), 1e-3) * steer_ratio(v)
            if noise > 0.0:
                ou = rho * ou + noise * math.sqrt(1.0 - rho * rho) * float(rng.standard_normal())
                steer += ou
            steer = min(1.0, max(-1.0, steer))
            if keyboard:
                if i % kb_every == 0:
                    u = steer + acc
                    kb_steer = float(min(1.0, max(-1.0, round(u))))
                    acc = u - kb_steer
                steer = kb_steer
            # Target speed: slow enough for every corner ahead given comfortable braking.
            j = min(seg + _LOOKAHEAD_PTS, len(kappa) - 1)
            if j > seg:
                v_corner = np.sqrt(_A_TARGET / np.maximum(np.abs(kappa[seg + 1 : j + 1]), 1e-4))
                reach = np.sqrt(v_corner**2 + 2 * _B_COMFORT * (road.s[seg + 1 : j + 1] - s))
                target = min(_V_CRUISE, float(reach.min()))
            else:
                target = _V_CRUISE
            target *= skill
            if v < target - 0.5:
                gas_on = True
            elif v > target + 0.5:
                gas_on = False
            if v > target + 2.5:
                brake_on = True
            elif v < target + 1.0:
                brake_on = False
            gas, brake = float(gas_on and not brake_on), float(brake_on)
        acts[i] = (steer, gas, brake)
        state = game.step(Action(steer, gas, brake))
        n = i + 1
        if state.finished:
            break
    meta = {
        "map_uid": map_ref,
        "player": f"bot-{seed}",
        "source": f"fake:{seed}",
        "keyboard": keyboard,
        "noise": noise,
        "skill": skill,
        "hold_ticks": hold_ticks,
    }
    info = {"finished": bool(state.finished), "race_time_ms": int(state.race_time_ms)}
    return InputTimeline(actions=acts[:n].copy(), meta=meta), info
