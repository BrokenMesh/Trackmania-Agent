"""Python stand-in for the TMAgentLink plugin, used by tests and `tools/tmnf_smoke.py --fake`.

Speaks the protocol of PROTOCOL.md on a thread, backed by a deterministic toy car
on a straight road (no dependency on other tmagent modules). Semantics mirror the
plugin: sync mode is paused between commands and STEP runs exactly n ticks (or
until the finish); realtime mode ticks on its own wall clock at 100 * speed ticks/s.

Rendered frames encode the race time in the centre pixel (BGRA bytes B, G, R =
t & 255, (t >> 8) & 255, (t >> 16) & 255, so an RGB image reads t = R<<16 | G<<8 | B)
and have a white top row, so tests can check both time sync and row order.

This is NOT the real game: it validates the Python side and the protocol only.
"""

from __future__ import annotations

import math
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np

from tmagent.game.tmnf import protocol as P

TICK_MS = 10
DT = TICK_MS / 1000.0
ACCEL = 20.0  # m/s^2 with gas
BRAKE = 40.0  # m/s^2 with brake
DRAG = 0.1  # 1/s
TURN_RATE = 1.5  # rad/s at full steer


@dataclass
class _Car:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    v: float = 0.0

    def tick(self, inp: P.InputCmd) -> None:
        if inp.accelerate:
            self.v += ACCEL * DT
        if inp.brake:
            self.v = max(0.0, self.v - BRAKE * DT)
        self.v *= 1.0 - DRAG * DT
        if inp.analog:
            steer = inp.steer / P.STEER_FULL_SCALE
            steer = steer if P.STEER_NEGATIVE_IS_LEFT else -steer
        else:
            steer = float(inp.right) - float(inp.left)
        self.heading += steer * TURN_RATE * DT
        self.x += self.v * math.cos(self.heading) * DT
        self.y += self.v * math.sin(self.heading) * DT


def reference_finish_time(
    actions: np.ndarray, finish_x: float = 80.0, steer_threshold: float = 0.5
) -> int | None:
    """Finish time (ms) of the fake car driven by per-tick actions [N, 3] (steer, gas, brake).

    Independent of the server and client code paths: lets tests/smoke check the whole
    replay -> input -> physics chain against an expected number. None if no finish.
    """
    car = _Car()
    for i, (steer, gas, brake) in enumerate(np.asarray(actions, dtype=np.float64)):
        car.tick(
            P.InputCmd(
                left=bool(steer < -steer_threshold), right=bool(steer > steer_threshold),
                accelerate=bool(gas >= 0.5), brake=bool(brake >= 0.5),
            )
        )  # fmt: skip
        if car.x >= finish_x:
            return (i + 1) * TICK_MS
    return None


def fake_frame_race_time(image_rgb: np.ndarray) -> int:
    """Decode the race time from the centre pixel of an RGB frame made by the fake."""
    h, w = image_rgb.shape[:2]
    r, g, b = (int(v) for v in image_rgb[h // 2, w // 2, :3])
    return (r << 16) | (g << 8) | b


class FakePluginServer:
    """Threaded fake plugin. Use as a context manager; `port` is valid after start()."""

    def __init__(
        self,
        port: int = 0,
        host: str = "127.0.0.1",
        *,
        protocol_version: int = P.PROTOCOL_VERSION,
        build: str = "FakePlugin 0.1",
        finish_x: float = 80.0,
        checkpoint_xs: tuple[float, ...] = (25.0, 55.0),
    ) -> None:
        self.host, self._want_port = host, port
        self.protocol_version, self.build = protocol_version, build
        self.finish_x, self.checkpoint_xs = finish_x, tuple(checkpoint_xs)
        self.loaded_map: str | None = None
        self.executed: list[str] = []
        self.commands: list[P.Cmd] = []  # every command received, in order
        self._lock = threading.RLock()  # car/session state
        self._wlock = threading.Lock()  # socket writes
        self._listen: socket.socket | None = None
        self._client: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._reset_session()

    # ------------------------------------------------------------ lifecycle

    @property
    def port(self) -> int:
        assert self._listen is not None, "server not started"
        return self._listen.getsockname()[1]

    def start(self) -> FakePluginServer:
        self._listen = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listen.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listen.bind((self.host, self._want_port))
        self._listen.listen(4)
        self._listen.settimeout(0.05)
        self._stop.clear()
        for target in (self._accept_loop, self._tick_loop):
            t = threading.Thread(target=target, daemon=True, name=f"fake-{target.__name__}")
            t.start()
            self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        self.drop_client()
        if self._listen is not None:
            self._listen.close()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

    def __enter__(self) -> FakePluginServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def drop_client(self) -> None:
        """Close the current client connection abruptly (simulates a crash/disconnect)."""
        with self._lock:
            c, self._client = self._client, None
        if c is not None:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()

    # ---------------------------------------------------------------- state

    def _reset_session(self) -> None:
        with self._lock:
            self.mode = P.Mode.SYNC
            self.speed = 1.0
            self.paused = False
            self.input = P.InputCmd()
            self.car = _Car()
            self.race_time = 0
            self.cp_count = 0
            self.finished = False
            self.in_race = False
            self.seq = 0
            self.state_every = 0
            self.frame_stream: tuple[int, int, int] | None = None  # w, h, max_fps
            self._last_push_frame = 0.0
            self._rt_base: tuple[float, int] | None = None  # wall time, ticks done at that time

    def _reset_race(self) -> None:
        self.car = _Car()
        self.race_time = 0
        self.cp_count = 0
        self.finished = False
        self.in_race = True
        self.input = P.InputCmd()

    @property
    def cp_target(self) -> int:
        return len(self.checkpoint_xs) + 1

    def _state(self) -> P.StateMsg:
        c = self.car
        vx, vy = c.v * math.cos(c.heading), c.v * math.sin(c.heading)
        return P.StateMsg(
            race_time_ms=self.race_time, finished=self.finished, in_race=self.in_race,
            paused=self.paused, cp_count=self.cp_count, cp_target=self.cp_target,
            position=(c.x, 0.0, c.y), velocity=(vx, 0.0, vy), speed_kmh=c.v * 3.6, seq=self.seq,
        )  # fmt: skip

    def _tick(self) -> None:
        """Advance one physics tick with the current input (lock held)."""
        self.seq += 1
        if self.finished:
            return  # race time freezes at the finish, like TMI
        self.car.tick(self.input)
        self.race_time += TICK_MS
        while (
            self.cp_count < len(self.checkpoint_xs)
            and self.car.x >= self.checkpoint_xs[self.cp_count]
        ):  # noqa: E501
            self.cp_count += 1
        if self.car.x >= self.finish_x:
            self.cp_count, self.finished = self.cp_target, True

    def _frame(self, w: int, h: int) -> P.FrameMsg:
        t = self.race_time
        px = np.empty((h, w, 4), np.uint8)
        px[:] = (t & 255, (t >> 8) & 255, (t >> 16) & 255, 255)
        px[0] = 255  # white top row: marks the row order
        return P.FrameMsg(t, w, h, P.PIXEL_BGRA8, self.seq, px.tobytes())

    # -------------------------------------------------------------- sending

    def _send(self, data: bytes) -> None:
        c = self._client
        if c is None:
            return
        try:
            with self._wlock:
                c.sendall(data)
        except OSError:
            pass

    # -------------------------------------------------------------- threads

    def _accept_loop(self) -> None:
        assert self._listen is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._listen.accept()
            except (TimeoutError, OSError):
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.drop_client()  # newest connection wins, like the plugin
            self._reset_session()
            with self._lock:
                self._client = conn
            t = threading.Thread(target=self._serve, args=(conn,), daemon=True, name="fake-serve")
            t.start()
            self._threads.append(t)

    def _serve(self, conn: socket.socket) -> None:
        reader = P.MessageReader()
        conn.settimeout(0.1)
        while not self._stop.is_set() and self._client is conn:
            try:
                data = conn.recv(65536)
            except TimeoutError:
                continue
            except OSError:
                break
            if not data:
                break
            try:
                msgs = reader.feed(data)
            except P.ProtocolError:
                break
            for msg in msgs:
                self._handle(msg)
        if self._client is conn:
            self.drop_client()
            self._reset_session()

    def _tick_loop(self) -> None:
        """Realtime clock: 100 * speed ticks per second of wall time."""
        while not self._stop.is_set():
            time.sleep(0.002)
            with self._lock:
                if self._client is None or self.mode != P.Mode.REALTIME or not self.in_race:
                    self._rt_base = None
                    continue
                now = time.perf_counter()
                if self._rt_base is None:
                    self._rt_base = (now, 0)
                t0, done = self._rt_base
                due = int((now - t0) * 100.0 * self.speed) - done
                for _ in range(max(0, due)):
                    self._tick()
                    if self.state_every > 0 and self.seq % self.state_every == 0:
                        self._send(P.enc_push_state(self._state()))
                self._rt_base = (t0, done + max(0, due))
                fs = self.frame_stream
                if fs and now - self._last_push_frame >= 1.0 / max(1, fs[2]):
                    self._last_push_frame = now
                    self._send(P.enc_push_frame(self._frame(fs[0], fs[1])))

    # ------------------------------------------------------------- commands

    def _handle(self, msg: P.Message) -> None:
        try:
            cmd, rid, args = P.dec_command(msg)
        except (P.ProtocolError, ValueError, IndexError) as e:
            self._send(P.enc_error(0, f"bad command: {e}"))
            return
        with self._lock:
            self.commands.append(cmd)
            try:
                self._dispatch(cmd, rid, args)
            except Exception as e:  # keep the fake alive; report like the plugin would
                self._send(P.enc_error(rid, f"fake plugin error: {e!r}"))

    def _dispatch(self, cmd: P.Cmd, rid: int, args: tuple) -> None:
        ok, err, send = P.enc_ack, P.enc_error, self._send
        if cmd is P.Cmd.HELLO:
            send(P.enc_hello_reply(rid, self.protocol_version, self.build))
        elif cmd is P.Cmd.SET_MODE:
            self.mode = args[0]
            self.paused = self.mode is P.Mode.SYNC and self.in_race
            if self.mode is P.Mode.SYNC:
                self.frame_stream, self.state_every = None, 0
            self._rt_base = None
            send(ok(rid))
        elif cmd is P.Cmd.LOAD_MAP:
            path = args[0]
            name = path.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".Challenge.Gbx")
            if not name or name.startswith("missing"):  # test hook: unknown map
                send(err(rid, f"map not found: {path!r}"))
                return
            self.loaded_map = path
            self._reset_race()
            self.paused = self.mode is P.Mode.SYNC
            self._rt_base = None
            send(ok(rid, f"FAKEUID_{name}\t{name}"))
        elif cmd is P.Cmd.RESTART:
            if not self.in_race:
                send(err(rid, "not in a race"))
                return
            self._reset_race()
            self.paused = self.mode is P.Mode.SYNC
            self._rt_base = None
            send(P.enc_state(rid, self._state()))
        elif cmd is P.Cmd.STEP:
            n, inp = args
            if self.mode is not P.Mode.SYNC:
                send(err(rid, "STEP requires sync mode"))
            elif not self.in_race:
                send(err(rid, "no race ready; LOAD_MAP first"))
            else:
                self.input = inp
                for _ in range(max(0, n)):
                    if self.finished:
                        break
                    self._tick()
                send(P.enc_state(rid, self._state()))
        elif cmd is P.Cmd.SET_INPUT:
            self.input = args[0]
        elif cmd is P.Cmd.REQUEST_FRAME:
            w, h, _settle = args
            send(P.enc_frame(rid, self._frame(w, h)))
        elif cmd is P.Cmd.STREAM_FRAMES:
            on, w, h, fps = args
            self.frame_stream = (w, h, fps) if on and self.mode is P.Mode.REALTIME else None
            send(ok(rid))
        elif cmd is P.Cmd.STREAM_STATE:
            self.state_every = args[0] if self.mode is P.Mode.REALTIME else 0
            send(ok(rid))
        elif cmd is P.Cmd.SET_SPEED:
            self.speed = float(args[0])
            self._rt_base = None
            send(ok(rid))
        elif cmd is P.Cmd.EXECUTE:
            self.executed.append(args[0])
            send(ok(rid))
        elif cmd is P.Cmd.GET_STATE:
            send(P.enc_state(rid, self._state()))
        elif cmd is P.Cmd.PING:
            send(ok(rid, f"guard_rewinds=0 tick={self.seq} op=0"))
        elif cmd is P.Cmd.CLOSE:
            send(ok(rid))
            threading.Timer(0.05, self.drop_client).start()
