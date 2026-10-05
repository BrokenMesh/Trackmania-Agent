"""Driver input events -> 10 ms tick timeline -> control-rate actions.

Tick rule: tick i covers [i*10, (i+1)*10) ms and holds the input state in
effect at time i*10. An event at time t therefore applies to tick t/10 when t is
a multiple of 10, otherwise to the first tick starting at or after t
(ceil(t / 10)). Events are applied in time order; same-time events keep their
input order, so the later one wins.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from tmagent.interfaces import ACTION_DIM, PHYSICS_TICK_MS, InputTimeline

_ANALOG_STEER, _LEFT, _RIGHT, _ACCEL, _BRAKE, _GAS = (
    "steer",
    "steer_left",
    "steer_right",
    "accelerate",
    "brake",
    "gas",
)
_KNOWN = {_ANALOG_STEER, _LEFT, _RIGHT, _ACCEL, _BRAKE, _GAS}


def timeline_from_events(
    events: Iterable[tuple[int, str, float]],
    duration_ms: int | None = None,
    meta: dict[str, Any] | None = None,
) -> InputTimeline:
    """Build the per-tick timeline from (time_ms, name, value) events.

    Names: "steer" (analog [-1, 1]), "steer_left" / "steer_right" / "accelerate" /
    "brake" (binary, value > 0.5 is pressed), "gas" (analog [0, 1]). Names starting
    with "_" are ignored, other unknown names raise ValueError. Steer is
    right - left for the binary source; the analog and binary steer sources are
    last-writer-wins. Gas = max(accelerate, analog gas). Times < 0 are clamped to 0.

    Ticks: duration_ms // 10 if given, else last event time // 10 + 1 (0 ticks
    without events). Events whose tick lies beyond the last tick are dropped.
    """
    evs: list[tuple[int, str, float]] = []
    for t, name, value in events:
        if name.startswith("_"):
            continue
        if name not in _KNOWN:
            raise ValueError(f"unknown input event name {name!r}")
        if not np.isfinite(value):
            raise ValueError(f"non-finite value for event {name!r} at {t} ms")
        evs.append((max(int(t), 0), name, float(value)))
    evs.sort(key=lambda e: e[0])  # stable: same-time events keep input order

    if duration_ms is not None:
        n = max(int(duration_ms), 0) // PHYSICS_TICK_MS
    else:
        n = evs[-1][0] // PHYSICS_TICK_MS + 1 if evs else 0

    left = right = accel = brake = analog_steer = analog_gas = 0.0
    use_analog = False
    snaps: dict[int, tuple[float, float, float]] = {}
    for t, name, v in evs:
        tick = -(-t // PHYSICS_TICK_MS)  # ceil
        if tick >= n:
            break
        if name == _ANALOG_STEER:
            analog_steer, use_analog = float(np.clip(v, -1.0, 1.0)), True
        elif name == _LEFT:
            left, use_analog = float(v > 0.5), False
        elif name == _RIGHT:
            right, use_analog = float(v > 0.5), False
        elif name == _ACCEL:
            accel = float(v > 0.5)
        elif name == _BRAKE:
            brake = float(v > 0.5)
        else:
            analog_gas = float(np.clip(v, 0.0, 1.0))
        steer = analog_steer if use_analog else right - left
        snaps[tick] = (steer, max(accel, analog_gas), brake)  # latest event wins per tick

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    ticks = sorted(snaps)
    for j, tick in enumerate(ticks):
        end = ticks[j + 1] if j + 1 < len(ticks) else n
        actions[tick:end] = snaps[tick]
    return InputTimeline(actions=actions, meta=dict(meta or {}))


def _grid_ms(n: int, hz: int) -> np.ndarray:
    if hz <= 0:
        raise ValueError(f"rate must be positive, got {hz}")
    return np.rint(np.arange(n, dtype=np.int64) * 1000 / hz).astype(np.int64)


def control_times_ms(n: int, hz: int) -> np.ndarray:
    """Control step times round(i * 1000 / hz), i = 0..n-1, as int64 ms."""
    return _grid_ms(n, hz)


def frame_times_ms(n: int, hz: int) -> np.ndarray:
    """Frame times round(k * 1000 / hz), k = 0..n-1, as int64 ms."""
    return _grid_ms(n, hz)


def grid_len(total_ms: int, hz: int) -> int:
    """Number of grid points i with round(i * 1000 / hz) < total_ms."""
    if total_ms <= 0:
        return 0
    n = int(np.ceil(total_ms * hz / 1000)) + 2
    return int(np.searchsorted(_grid_ms(n, hz), total_ms, side="left"))


def resample_to_control(timeline: InputTimeline, control_hz: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample-and-hold the tick timeline at the control rate.

    Returns (actions float32 [T_a, 3], times int64 [T_a]); the action at control
    time t is timeline.actions[t // 10] and T_a covers all control times
    < N_ticks * 10.
    """
    total_ms = len(timeline.actions) * PHYSICS_TICK_MS
    times = control_times_ms(grid_len(total_ms, control_hz), control_hz)
    actions = timeline.actions[times // PHYSICS_TICK_MS].astype(np.float32)
    return actions, times


def resample_loss(timeline: InputTimeline, control_hz: int) -> dict[str, float | int]:
    """How much of the tick timeline survives control-rate labels.

    The labels are `resample_to_control(timeline, control_hz)`; a tick is replayed with
    the latest label whose control time lies in or before it (the execution mapping of
    tmagent.eval.harness.control_row_for_tick). Returns
      ticks            number of physics ticks,
      mismatch_frac    fraction of ticks whose replayed input differs from the original,
      changes          input changes (per channel) on the tick grid,
      lost_changes     changes that are not visible in the control-rate labels
                       (changes - changes between consecutive labels; a tap shorter than
                       the control period that falls between two samples counts as 2),
      lost_changes_frac  lost_changes / changes (0 without changes).
    """
    acts = timeline.actions
    n = len(acts)
    labels, times = resample_to_control(timeline, control_hz)
    if n == 0:
        return {
            "ticks": 0,
            "mismatch_frac": 0.0,
            "changes": 0,
            "lost_changes": 0,
            "lost_changes_frac": 0.0,
        }
    label_ticks = times // PHYSICS_TICK_MS  # nondecreasing, starts at tick 0
    src = label_ticks[np.searchsorted(label_ticks, np.arange(n), side="right") - 1]
    mismatch = (acts[src] != acts).any(axis=1)
    changes = int((acts[1:] != acts[:-1]).sum())
    seen = int((labels[1:] != labels[:-1]).sum())
    return {
        "ticks": n,
        "mismatch_frac": float(mismatch.mean()),
        "changes": changes,
        "lost_changes": changes - seen,
        "lost_changes_frac": (changes - seen) / changes if changes else 0.0,
    }
