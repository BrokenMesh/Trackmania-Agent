"""TMNFGame: SyncGame + RealtimeGame on top of the TMAgentLink client.

Sync mode (rendering replays, evaluation): the game is paused between commands;
`step()` runs exactly n physics ticks, `grab_frame()` captures the paused state.
Realtime mode (live play): the game runs at `game_cfg.game_speed`, `set_action()`
only stores/sends the newest input (never waits), frames and states are pushed by
the plugin and read via `latest_frame()` / `get_state()` without blocking.

The mode is switched lazily by the first call of the other API. Both APIs share one
connection, so use one of them at a time.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePath

import numpy as np

from tmagent.config import DataConfig, GameConfig
from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.client import (
    ConnectionLost,
    TMAgentClient,
    TMAgentError,
    convert_pixels,
)
from tmagent.interfaces import Action, Frame, GameState

MAP_SUFFIX = ".Challenge.Gbx"
STREAM_FPS_FACTOR = 2  # realtime frame push rate = factor * data.frame_hz (less staleness)
STATE_PUSH_EVERY_TICKS = 1  # realtime STATE push period in physics ticks
FRAME_SETTLE_RENDERS = 1  # Render() calls the plugin waits before a capture (UNVERIFIED need)

# TMI console variables applied once after connecting. Names are from the Linesight
# bridge for TMI 2.1.x (VERIFIED there); an unknown name only prints a console error.
SETUP_COMMANDS = (
    "set skip_map_load_screens true",
    "set unfocused_fps_limit false",
    "set autorewind false",
    "set auto_reload_plugins false",
    "set disable_forced_camera true",
)


# --------------------------------------------------------------- map lookup


def resolve_map(map_ref: str, map_dir: str | Path) -> Path:
    """Map ref -> existing .Challenge.Gbx file.

    Order: an existing file path; a path relative to map_dir; `map_dir/index.json`
    ({uid: filename}); a unique filename whose stem equals the ref (case-insensitive);
    a unique filename containing the ref (map uids are often part of the name).
    """
    p = Path(map_ref).expanduser()
    if p.is_file():
        return p.resolve()
    d = Path(map_dir).expanduser()
    if (d / map_ref).is_file():
        return (d / map_ref).resolve()
    index = d / "index.json"
    if index.is_file():
        fname = json.loads(index.read_text(encoding="utf-8")).get(map_ref)
        if fname is not None:
            if (d / fname).is_file():
                return (d / fname).resolve()
            raise FileNotFoundError(f"{index} maps {map_ref!r} to missing file {fname!r}")
    files = sorted(d.rglob(f"*{MAP_SUFFIX}")) if d.is_dir() else []
    ref = map_ref.lower().removesuffix(MAP_SUFFIX.lower())
    exact = [f for f in files if f.name[: -len(MAP_SUFFIX)].lower() == ref]
    partial = [f for f in files if ref in f.name.lower()]
    for group in (exact, partial):
        if len(group) == 1:
            return group[0].resolve()
        if len(group) > 1:
            names = ", ".join(f.name for f in group[:5])
            raise ValueError(f"map ref {map_ref!r} is ambiguous in {d}: {names}")
    raise FileNotFoundError(
        f"map {map_ref!r} is neither a file nor found in game.map_dir={str(d)!r} "
        f"({len(files)} {MAP_SUFFIX} files scanned)"
    )


def map_command_path(path: Path, style: str = "auto") -> str:
    """Path string sent with LOAD_MAP (the plugin runs `map <string>`).

    UNVERIFIED: whether TMI's `map` takes absolute paths. Linesight uses paths
    relative to `Documents/TrackMania/Tracks/Challenges`. "auto" sends the part after
    `Tracks/Challenges` when the file is inside it, else the absolute path;
    "absolute" / "relative" force one form (relative falls back to absolute).
    """
    if style not in ("auto", "absolute", "relative"):
        raise ValueError(f"map path style must be auto|absolute|relative, got {style!r}")
    parts = PurePath(path).parts
    rel = None
    if style != "absolute":
        low = [s.lower() for s in parts]
        for i in range(len(low) - 2):
            if low[i] == "tracks" and low[i + 1] == "challenges":
                rel = os.sep.join(parts[i + 2 :])
                break
    return rel if rel else str(path)


# -------------------------------------------------------- action conversion


def action_to_input(
    action: Action, steer_mode: str = "binary", steer_threshold: float = 0.5
) -> P.InputCmd:
    """Action -> plugin input. binary: steer keys by threshold; analog: TMI steer int."""
    acc, brk = action.gas >= 0.5, action.brake >= 0.5
    if steer_mode == "analog":
        return P.InputCmd(
            accelerate=acc, brake=brk, analog=True, steer=P.steer_to_wire(action.steer)
        )
    if steer_mode != "binary":
        raise ValueError(f"steer_mode must be 'binary' or 'analog', got {steer_mode!r}")
    return P.InputCmd(
        left=action.steer < -steer_threshold,
        right=action.steer > steer_threshold,
        accelerate=acc,
        brake=brk,
    )


def to_game_state(st: P.StateMsg) -> GameState:
    return GameState(
        race_time_ms=st.race_time_ms,
        position=np.array(st.position, dtype=np.float32),
        velocity=np.array(st.velocity, dtype=np.float32),
        speed_kmh=float(st.speed_kmh),
        checkpoint=st.cp_count,
        finished=st.finished,
        num_checkpoints=st.cp_target if st.cp_target > 0 else None,
        extra={"in_race": st.in_race, "paused": st.paused, "plugin_seq": st.seq},
    )


# ------------------------------------------------------------------- game


class TMNFGame:
    """TrackMania Nations Forever through the TMAgentLink plugin (see PROTOCOL.md).

    Connects lazily on first use (`connect()` does it explicitly). The first-run
    knobs restart_method, map_path_style, frame_settle_renders and
    capture_flip_vertical come from GameConfig; keyword arguments override them.
    """

    def __init__(
        self,
        game_cfg: GameConfig,
        data_cfg: DataConfig,
        *,
        client: TMAgentClient | None = None,
        restart_method: str | None = None,
        map_path_style: str | None = None,
        frame_settle_renders: int | None = None,
        flip_vertical: bool | None = None,
    ) -> None:
        self.cfg, self.data_cfg = game_cfg, data_cfg
        restart_method = restart_method or game_cfg.restart_method
        map_path_style = map_path_style or game_cfg.map_path_style
        if frame_settle_renders is None:
            frame_settle_renders = game_cfg.frame_settle_renders
        if flip_vertical is None:
            flip_vertical = game_cfg.capture_flip_vertical
        self.client = client or TMAgentClient(
            game_cfg.tmi_host, game_cfg.tmi_port, game_cfg.connect_timeout_s
        )
        methods = {"rewind": P.RestartMethod.REWIND, "give_up": P.RestartMethod.GIVE_UP}
        if restart_method not in methods:
            raise ValueError(
                f"restart_method must be one of {sorted(methods)}, got {restart_method!r}"
            )
        self.restart_method = methods[restart_method]
        self.map_path_style = map_path_style
        self.frame_settle_renders = frame_settle_renders
        self.flip_vertical = flip_vertical
        self.map_info: tuple[str, str] | None = None  # (uid, name) reported by the plugin
        self._mode: P.Mode | None = None
        self._last_input: P.InputCmd | None = None
        self._frame_cache: tuple[P.FrameMsg, Frame] | None = None

    # --------------------------------------------------------- connection

    def connect(self) -> None:
        """Connect + handshake + apply TMI settings; enters sync mode. Idempotent."""
        c = self.client
        if c.connected and self._mode is not None:
            return
        c.connect()
        self._mode = None
        for cmd in SETUP_COMMANDS:
            c.execute(cmd)
        m = re.fullmatch(r"cam(\d+)", self.cfg.camera.strip())
        c.execute(f"cam {m.group(1)}" if m else self.cfg.camera)
        self._enter(P.Mode.SYNC)

    @property
    def plugin_build(self) -> str:
        return self.client.hello.build if self.client.hello else ""

    def _enter(self, mode: P.Mode) -> None:
        """Switch the plugin between sync (paused, stepped) and realtime (free-running)."""
        c, cfg, w_h = self.client, self.cfg, tuple(self.data_cfg.resolution)
        if mode is P.Mode.SYNC:
            c.stream_frames(False)
            c.stream_state(0)
            c.set_mode(P.Mode.SYNC)  # pauses at the next tick
            c.set_speed(cfg.render_speed)
            c.execute(f"set countdown_speed {cfg.render_speed:g}")
        else:
            c.set_speed(cfg.game_speed)
            c.execute(f"set countdown_speed {cfg.game_speed:g}")
            c.set_mode(P.Mode.REALTIME)
            c.set_input(P.InputCmd())
            c.stream_frames(True, w_h[0], w_h[1], STREAM_FPS_FACTOR * self.data_cfg.frame_hz)
            c.stream_state(STATE_PUSH_EVERY_TICKS)
        self._last_input = P.InputCmd()
        self._mode = mode

    def _ensure(self, mode: P.Mode) -> TMAgentClient:
        if self._mode is not None and not self.client.connected:
            self._mode = None  # a session existed: its race state is gone, do not hide that
            raise ConnectionLost(
                "lost the connection to the TMAgentLink plugin; call connect(), load_map() "
                "(if the game was restarted) and start_race() again"
            )
        if self._mode is None:
            self.connect()
        if self._mode is not mode:
            self._enter(mode)
        return self.client

    def close(self) -> None:
        self.client.close()
        self._mode = None

    def __enter__(self) -> TMNFGame:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --------------------------------------------------------------- shared

    def load_map(self, map_ref: str) -> None:
        """Resolve `map_ref`, load it, and wait until the race is ready at t=0.

        Sync mode leaves the game paused at race time 0; realtime keeps it running.
        """
        path = resolve_map(map_ref, self.cfg.map_dir)
        c = self._ensure(self._mode or P.Mode.SYNC)
        text = c.load_map(map_command_path(path, self.map_path_style))
        uid, _, name = text.partition("\t")
        self.map_info = (uid, name)
        c.reset_pushed()  # drop pushed frames/states of the previous map

    def start_race(self) -> GameState:
        """Sync: restart to race time 0 (paused) and return that state."""
        c = self._ensure(P.Mode.SYNC)
        st = c.restart(self.restart_method)
        if st.race_time_ms != 0:
            raise TMAgentError(
                f"restart returned race time {st.race_time_ms} ms, expected 0 (the plugin's saved "
                "start state is not at race time 0, see PROTOCOL.md 'LOAD_MAP')"
            )
        return to_game_state(st)

    # ----------------------------------------------------------------- sync

    def step(self, action: Action, n_ticks: int = 1) -> GameState:
        """Hold `action` for n_ticks ticks (stops early at the finish); paused afterwards."""
        if n_ticks < 1:
            raise ValueError(f"n_ticks must be >= 1, got {n_ticks}")
        c = self._ensure(P.Mode.SYNC)
        inp = action_to_input(action, self.cfg.steer_mode, self.cfg.steer_threshold)
        return to_game_state(c.step(n_ticks, inp))

    def grab_frame(self) -> Frame:
        """Capture the paused state at data.resolution / data.channels."""
        c = self._ensure(P.Mode.SYNC)
        w, h = self.data_cfg.resolution
        fm = c.request_frame(w, h, self.frame_settle_renders)
        if (fm.width, fm.height) != (w, h):
            raise TMAgentError(f"plugin captured {fm.width}x{fm.height}, requested {w}x{h}")
        img = convert_pixels(fm, self.data_cfg.channels, self.flip_vertical)
        return Frame(image=img, race_time_ms=fm.race_time_ms, wall_time=fm.wall_time)

    # ------------------------------------------------------------- realtime

    def restart(self) -> None:
        """Realtime: restart the race (rewind to t=0, game keeps running)."""
        c = self._ensure(P.Mode.REALTIME)
        st = c.restart(self.restart_method)
        c.reset_pushed(st)  # drop frames/states of the previous run
        self._last_input = P.InputCmd()  # the plugin clears the held input on restart

    def set_action(self, action: Action) -> None:
        """Realtime: hold `action` from the next tick. Sends only on change, never waits."""
        c = self._ensure(P.Mode.REALTIME) if self._mode is not P.Mode.REALTIME else self.client
        inp = action_to_input(action, self.cfg.steer_mode, self.cfg.steer_threshold)
        if inp != self._last_input:
            c.set_input(inp)
            self._last_input = inp

    def get_state(self) -> GameState:
        """Realtime: newest pushed state (one blocking GET_STATE if none arrived yet)."""
        c = self._ensure(P.Mode.REALTIME)
        pushed = c.latest_state()
        return to_game_state(pushed[0] if pushed else c.get_state())

    def latest_frame(self) -> Frame | None:
        """Realtime: newest pushed frame converted to data.channels, or None. Non-blocking."""
        c = self.client
        if self._mode is not P.Mode.REALTIME or not c.connected:
            return None
        fm = c.latest_frame()
        if fm is None:
            return None
        cached = self._frame_cache
        if cached is not None and cached[0] is fm:
            return cached[1]
        frame = Frame(
            image=convert_pixels(fm, self.data_cfg.channels, self.flip_vertical),
            race_time_ms=fm.race_time_ms,
            wall_time=fm.wall_time,
        )
        self._frame_cache = (fm, frame)
        return frame
