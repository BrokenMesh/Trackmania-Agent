"""Drive a SyncGame with a recorded input timeline and capture an Episode.

Every physics tick i (race time t0 = 10 i) the loop grabs a frame for each frame
time f in [t0, t0 + 10) (the latest tick <= f), records the action / position /
speed for each control time c in [t0, t0 + 10), then steps the game with the
tick's action. Both grids are the exact formula grids of tmagent.data.timeline.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import DataConfig
from tmagent.data.timeline import control_times_ms, frame_times_ms, grid_len
from tmagent.interfaces import PHYSICS_TICK_MS, Action, Episode, InputTimeline, SyncGame

_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _area_weights(n_in: int, n_out: int) -> np.ndarray:
    """(n_out, n_in) box-filter weights, rows normalized to sum 1."""
    edges = np.linspace(0.0, n_in, n_out + 1)
    lo = np.arange(n_in)
    w = np.minimum(edges[1:, None], lo + 1) - np.maximum(edges[:-1, None], lo)
    w = np.clip(w, 0.0, None)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def resize_area(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Area-average resize of a uint8 (H, W, C) image to (h, w, C)."""
    x = np.tensordot(_area_weights(img.shape[0], h), img.astype(np.float32), axes=(1, 0))
    x = np.tensordot(_area_weights(img.shape[1], w), x, axes=(1, 1)).transpose(1, 0, 2)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def conform_frame(img: np.ndarray, cfg: DataConfig) -> tuple[np.ndarray, bool]:
    """Convert a captured image to uint8 (H, W, cfg.channels) at cfg.resolution.

    Returns (image, resized). RGB -> gray uses luma, gray -> RGB repeats the
    channel, alpha is dropped, size mismatches are area-resized.
    """
    if img.dtype != np.uint8:
        raise ValueError(f"frame must be uint8, got {img.dtype}")
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3 or img.shape[2] not in (1, 3, 4):
        raise ValueError(f"frame must be (H, W, C) with C in 1/3/4, got {img.shape}")
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.shape[2] == 3 and cfg.channels == 1:
        img = np.clip(np.rint(img.astype(np.float32) @ _LUMA), 0, 255).astype(np.uint8)[..., None]
    elif img.shape[2] == 1 and cfg.channels == 3:
        img = np.repeat(img, 3, axis=2)
    w, h = cfg.resolution
    resized = img.shape[:2] != (h, w)
    if resized:
        img = resize_area(img, w, h)
    return np.ascontiguousarray(img), resized


def render_episode(
    game: SyncGame,
    timeline: InputTimeline,
    map_ref: str,
    cfg: DataConfig,
    meta: dict[str, Any],
    stop_after_finish_ms: int = 500,
    expected_time_ms: int | None = None,
) -> Episode:
    """Re-drive `timeline` on `game` and capture frames, actions and states.

    Stops stop_after_finish_ms after the game first reports finished, or at the
    end of the timeline. meta (over timeline.meta) fills EPISODE_META_KEYS; extra
    keys: finish_time_ms (None if not finished), resized_frames, desync (True if
    expected_time_ms is given and the finish time is not within +-10 ms of it)
    and frame_time_mismatch (number of frames whose Frame.race_time_ms, when
    >= 0, differs from the tick time 10 * i at grab).
    """
    n = len(timeline.actions)
    total_ms = n * PHYSICS_TICK_MS
    f_times = frame_times_ms(grid_len(total_ms, cfg.frame_hz), cfg.frame_hz)
    c_times = control_times_ms(grid_len(total_ms, cfg.control_hz), cfg.control_hz)

    game.load_map(map_ref)
    state = game.start_race()
    frames: list[np.ndarray] = []
    acts: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    speeds: list[float] = []
    resized = mismatched = 0
    finish_ms: int | None = None
    for i in range(n):
        t_end = (i + 1) * PHYSICS_TICK_MS
        while len(frames) < len(f_times) and f_times[len(frames)] < t_end:
            frame = game.grab_frame()
            img, was_resized = conform_frame(frame.image, cfg)
            frames.append(img)
            resized += was_resized
            if frame.race_time_ms >= 0 and frame.race_time_ms != i * PHYSICS_TICK_MS:  # -1: unknown
                mismatched += 1
        while len(acts) < len(c_times) and c_times[len(acts)] < t_end:
            acts.append(timeline.actions[i])
            positions.append(np.asarray(state.position, dtype=np.float32).reshape(3))
            speeds.append(float(state.speed_kmh))
        state = game.step(Action.from_array(timeline.actions[i]))
        if state.finished and finish_ms is None:
            finish_ms = int(state.race_time_ms)
        if finish_ms is not None and state.race_time_ms >= finish_ms + stop_after_finish_ms:
            break

    w, h = cfg.resolution
    ep_meta = _fill_meta(
        {**timeline.meta, **meta},
        cfg,
        map_ref=map_ref,
        race_time_ms=finish_ms if finish_ms is not None else int(state.race_time_ms),
        finish_ms=finish_ms,
        resized=resized,
        mismatched=mismatched,
        desync=expected_time_ms is not None
        and (finish_ms is None or abs(finish_ms - expected_time_ms) > PHYSICS_TICK_MS),
    )
    return Episode(
        frames=np.stack(frames) if frames else np.zeros((0, h, w, cfg.channels), np.uint8),
        frame_times_ms=f_times[: len(frames)],
        actions=np.asarray(acts, dtype=np.float32).reshape(-1, 3),
        action_times_ms=c_times[: len(acts)],
        positions=np.asarray(positions, dtype=np.float32).reshape(-1, 3),
        speeds_kmh=np.asarray(speeds, dtype=np.float32),
        meta=ep_meta,
    )


def _fill_meta(
    m: dict[str, Any],
    cfg: DataConfig,
    map_ref: str,
    race_time_ms: int,
    finish_ms: int | None,
    resized: int,
    mismatched: int,
    desync: bool,
) -> dict[str, Any]:
    map_uid = str(m.get("map_uid") or Path(str(map_ref)).name)
    source = str(m.get("source") or "unknown")
    digest = hashlib.sha1(source.encode()).hexdigest()[:10]
    out = dict(m)
    out.update(
        episode_id=str(m.get("episode_id") or f"{map_uid}-{digest}"),
        map_uid=map_uid,
        map_name=str(m.get("map_name") or map_uid),
        source=source,
        player=str(m.get("player") or ""),
        race_time_ms=race_time_ms,
        finished=finish_ms is not None,
        frame_hz=int(cfg.frame_hz),
        control_hz=int(cfg.control_hz),
        resolution=[int(cfg.resolution[0]), int(cfg.resolution[1])],
        channels=int(cfg.channels),
        camera=str(m.get("camera") or "unknown"),
        renderer=str(m.get("renderer") or "unknown"),
        finish_time_ms=finish_ms,
        resized_frames=int(resized),
        frame_time_mismatch=int(mismatched),
        desync=bool(desync),
    )
    return out
