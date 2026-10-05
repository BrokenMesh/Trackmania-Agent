"""TMNF replay inputs -> generic input events -> InputTimeline.

Two sources, both normalized to (time_ms, name, value) events with the names of
tmagent.data.timeline ("accelerate", "brake", "steer_left", "steer_right", "steer",
"gas"; names starting with "_" are ignored there):

- `.Replay.Gbx` ghosts via pygbx (optional, GPL-3, imported lazily; decisions D-008),
- TMInterface input-script text (`<ms> press up`, ...), no third-party code needed.

Facts and sources: docs/research.md ("Input event semantics").
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from tmagent.game.tmnf import protocol as P

Event = tuple[int, str, float]

PYGBX_HINT = (
    "pygbx is required to read .Replay.Gbx files: "
    "pip install git+https://github.com/donadigo/pygbx python-lzo "
    "(GPL-3, optional dependency, see docs/decisions.md D-008)"
)

# Replay analog steer decoding yields TMI convention (negative = left). The sign
# applied to reach tmagent's convention (negative = left) lives in protocol.py.
# UNVERIFIED on real replays: check a left-hand corner of a known replay.
ANALOG_STEER_SIGN = 1 if P.STEER_NEGATIVE_IS_LEFT else -1

# TMI docs: analog gas > 19661 (of 65536) counts as accelerate, < -19661 as brake.
# TMNF has no analog acceleration strength, so "gas" events become 0/1.
GAS_THRESHOLD = 19661 / 65536
FULL = float(P.STEER_FULL_SCALE)

_DIGITAL = {
    "Accelerate": "accelerate",
    "Brake": "brake",
    "SteerLeft": "steer_left",
    "SteerRight": "steer_right",
}
_SCRIPT_KEYS = {
    "up": "accelerate",
    "down": "brake",
    "left": "steer_left",
    "right": "steer_right",
}


def decode_analog(enabled: int, flags: int) -> int:
    """GBX.NET analog decoding of a ghost control entry -> int in [-65536, 65536].

    data = enabled | flags << 16; dir = (data >> 16) & 0xFF; val = data & 0xFFFF;
    dir == 0xFF -> 65536 - val; dir == 1 -> -65536; otherwise -val * (dir + 1)
    (can exceed the range for dir >= 2; callers clip after normalizing).
    """
    data = (int(enabled) | (int(flags) << 16)) & 0xFFFFFFFF
    direction, val = (data >> 16) & 0xFF, data & 0xFFFF
    if direction == 0xFF:
        return 65536 - val
    if direction == 1:
        return -65536
    return -val * (direction + 1)


def _norm_steer(raw: int) -> float:
    return max(-1.0, min(1.0, ANALOG_STEER_SIGN * raw / FULL))


class _GasDecoder:
    """Analog gas -> "gas" 0/1 events, plus "brake" while gas < -threshold."""

    def __init__(self) -> None:
        self._brake_from_gas = False

    def __call__(self, t: int, norm: float) -> list[Event]:
        out: list[Event] = [(t, "gas", 1.0 if norm > GAS_THRESHOLD else 0.0)]
        brake = norm < -GAS_THRESHOLD
        if brake != self._brake_from_gas:  # only touch "brake" when gas changes its state
            out.append((t, "brake", 1.0 if brake else 0.0))
            self._brake_from_gas = brake
        return out


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name.lstrip("_")).lower()


def events_from_control_entries(
    entries: Iterable[Any],
    control_names: Sequence[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> list[Event]:
    """Normalize pygbx-style control entries (.time, .event_name, .enabled, .flags).

    time is already relative to the race start (raw - 100000). `control_names` is only
    used for entries that carry an index (`.event_index`) instead of `.event_name`.
    Respawn presses become "_respawn" events and increment meta["respawns"].
    Unknown names are skipped with a warning (once per name).
    """
    out: list[Event] = []
    gas = _GasDecoder()
    warned: set[str] = set()
    for e in entries:
        name = getattr(e, "event_name", None)
        if name is None and control_names is not None:
            name = control_names[getattr(e, "event_index")]  # noqa: B009
        t, enabled, flags = int(e.time), int(e.enabled), int(getattr(e, "flags", 0))
        data = (enabled | (flags << 16)) & 0xFFFFFFFF
        if name in _DIGITAL:
            out.append((t, _DIGITAL[name], 1.0 if data != 0 else 0.0))
        elif name == "Steer":
            out.append((t, "steer", _norm_steer(decode_analog(enabled, flags))))
        elif name == "Gas":
            out.extend(gas(t, decode_analog(enabled, flags) / FULL))
        elif name == "Respawn":
            if data != 0 and meta is not None:
                meta["respawns"] = meta.get("respawns", 0) + 1
            out.append((t, "_respawn", 1.0 if data != 0 else 0.0))
        elif isinstance(name, str) and name.startswith("_Fake"):
            out.append((t, "_" + _snake(name), 1.0 if data != 0 else 0.0))
        elif name in ("Horn", "AccelerateReal", "BrakeReal"):
            out.append((t, "_" + _snake(name), 1.0 if data != 0 else 0.0))
        elif name not in warned:
            warned.add(str(name))
            warnings.warn(
                f"unknown replay input event {name!r} skipped", RuntimeWarning, stacklevel=2
            )
    return out


def load_replay(path: str | os.PathLike) -> tuple[list[Event], dict[str, Any]]:
    """Read a `.Replay.Gbx` (needs pygbx) -> (events, meta).

    meta: map_uid, map_name, map_author (embedded challenge, ghost.uid as uid fallback),
    player, race_time_ms, num_respawns, respawns (counted from events), cp_times,
    game_version, source ("file:<name>").
    """
    try:
        from pygbx import Gbx, GbxType
    except ImportError as e:
        raise ImportError(PYGBX_HINT) from e
    path = Path(path)
    gbx = Gbx(str(path))
    ghost = gbx.get_class_by_id(GbxType.CTN_GHOST)
    if ghost is None:
        raise ValueError(f"{path.name}: no ghost found (is this a .Replay.Gbx with a driven run?)")
    challenge = None
    try:
        rec = gbx.get_class_by_id(GbxType.REPLAY_RECORD)
        challenge = rec.track.get_class_by_id(GbxType.CHALLENGE)
    except Exception as e:  # missing embedded map is not fatal, uid falls back to the ghost
        warnings.warn(f"{path.name}: embedded challenge not readable ({e!r})", stacklevel=2)

    def attr(obj: Any, name: str, default: Any = None) -> Any:
        v = getattr(obj, name, None) if obj is not None else None
        return default if v is None else v

    meta: dict[str, Any] = {
        "map_uid": str(attr(challenge, "map_uid", "") or attr(ghost, "uid", "")),
        "map_name": str(attr(challenge, "map_name", "")),
        "map_author": str(attr(challenge, "map_author", "")),
        "player": str(attr(ghost, "login", "")),
        "race_time_ms": int(attr(ghost, "race_time", 0)),
        "num_respawns": int(attr(ghost, "num_respawns", 0)),
        "cp_times": [int(t) for t in attr(ghost, "cp_times", [])],
        "game_version": attr(ghost, "game_version", ""),
        "source": f"file:{path.name}",
    }
    events = events_from_control_entries(
        attr(ghost, "control_entries", []), attr(ghost, "control_names", None), meta
    )
    meta.setdefault("respawns", 0)
    return events, meta


# ------------------------------------------------------------ TMI input script

_TIME = r"\d+(?:\.\d+)?"
_LINE = re.compile(rf"^({_TIME})(?:-({_TIME}))?\s+(\w+)(?:\s+(\S+))?$")


def _ms(token: str) -> int:
    # Plain integers are milliseconds. A decimal point means seconds (TMI "decimal
    # time" format). UNVERIFIED: only plain-ms scripts were seen in the docs.
    return round(float(token) * 1000) if "." in token else int(token)


def parse_tmi_input_script(text: str) -> list[Event]:
    """Parse TMInterface input-script text into events.

    Lines: `<ms> press up|down|left|right`, `<ms> rel <key>`, `<t0>-<t1> press <key>`
    (press at t0, release at t1), `<ms> steer <int>`, `<ms> gas <int>`; `#` starts a
    comment, blank lines are ignored. up = accelerate, down = brake. Any other line
    raises ValueError naming the line number.
    """
    events: list[Event] = []
    gas = _GasDecoder()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _LINE.match(line.lower())
        if not m:
            raise ValueError(f"line {lineno}: cannot parse input script line {raw!r}")
        t0s, t1s, cmd, arg = m.groups()
        t0 = _ms(t0s)
        t1 = _ms(t1s) if t1s is not None else None
        if t1 is not None and t1 < t0:
            raise ValueError(f"line {lineno}: range end before start in {raw!r}")
        if cmd in ("press", "rel"):
            if arg not in _SCRIPT_KEYS:
                raise ValueError(f"line {lineno}: unknown key {arg!r} (up|down|left|right)")
            if t1 is not None and cmd == "rel":
                raise ValueError(f"line {lineno}: a time range is only valid with press")
            name = _SCRIPT_KEYS[arg]
            events.append((t0, name, 1.0 if cmd == "press" else 0.0))
            if t1 is not None:
                events.append((t1, name, 0.0))
        elif cmd in ("steer", "gas"):
            if t1 is not None:
                raise ValueError(f"line {lineno}: a time range is not valid with {cmd}")
            try:
                value = int(arg)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                raise ValueError(f"line {lineno}: {cmd} needs an integer, got {arg!r}") from None
            if abs(value) > P.STEER_FULL_SCALE:
                raise ValueError(f"line {lineno}: {cmd} value {value} outside +-65536")
            if cmd == "steer":
                events.append((t0, "steer", _norm_steer(value)))
            else:
                events.extend(gas(t0, value / FULL))
        else:
            raise ValueError(f"line {lineno}: unknown command {cmd!r} in {raw!r}")
    events.sort(key=lambda e: e[0])  # stable: same-time events keep script order
    return events


# ---------------------------------------------------------------- timeline


def replay_to_timeline(
    source: str | os.PathLike | Iterable[Event], meta: dict[str, Any] | None = None
):
    """`.Replay.Gbx` / TMI script path, or an event list -> InputTimeline.

    The timeline covers ceil(race_time_ms / 10) ticks when the replay's race time is
    known (meta["race_time_ms"]); explicit `meta` entries override loaded ones.
    """
    from tmagent.data.timeline import timeline_from_events

    merged: dict[str, Any] = {}
    if isinstance(source, str | os.PathLike):
        p = Path(source)
        if p.suffix.lower() == ".gbx":
            events, loaded = load_replay(p)
        else:
            events, loaded = (
                parse_tmi_input_script(p.read_text(encoding="utf-8")),
                {"source": f"script:{p.name}"},
            )
        merged.update(loaded)
    else:
        events = list(source)
    merged.update(meta or {})
    rt = merged.get("race_time_ms")
    duration = -(-int(rt) // 10) * 10 if rt else None
    return timeline_from_events(events, duration_ms=duration, meta=merged)
