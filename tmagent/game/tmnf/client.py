"""TCP client for the TMAgentLink plugin (framing, reader thread, request/response).

One reader thread parses the byte stream and demultiplexes it: replies are matched
to the waiting request by req_id (late replies of timed-out requests are dropped),
pushed STATE/FRAME messages only update the "latest" slots. Requests are serialized
(one outstanding at a time); set_input() bypasses that lock so a control thread is
never blocked behind a long request such as LOAD_MAP.
"""

from __future__ import annotations

import itertools
import socket
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from tmagent.game.tmnf import protocol as P

# UNVERIFIED: row order of Graphics::CaptureScreenshot. Assumed top-down (row 0 =
# top of the window); the Linesight bridge converts it to gray without flipping.
# If saved PNGs of the first run are upside down, set this True (single switch).
CAPTURE_FLIP_VERTICAL = False


class TMAgentError(RuntimeError):
    """Base class of all bridge errors."""


class ConnectionLost(TMAgentError):
    """The plugin closed the connection or the socket failed."""


class TMAgentTimeout(TMAgentError):
    """A request got no reply in time."""


class ProtocolMismatch(TMAgentError):
    """Plugin and client speak different protocol versions."""


class PluginError(TMAgentError):
    """The plugin replied with ERROR."""


@dataclass(frozen=True)
class HelloInfo:
    protocol_version: int
    build: str


class _Slot:
    __slots__ = ("error", "event", "rep", "value")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.rep: P.Rep | None = None
        self.value: object = None
        self.error: Exception | None = None


def convert_pixels(
    fm: P.FrameMsg, channels: int = 3, flip_vertical: bool = CAPTURE_FLIP_VERTICAL
) -> np.ndarray:
    """Raw BGRA frame -> uint8 (H, W, 3) RGB or (H, W, 1) gray (BT.601 luma)."""
    if fm.pixel_format != P.PIXEL_BGRA8:
        raise ValueError(f"unsupported pixel format {fm.pixel_format}")
    if channels not in (1, 3):
        raise ValueError("channels must be 1 or 3")
    a = np.frombuffer(fm.pixels, np.uint8).reshape(fm.height, fm.width, 4)
    if flip_vertical:
        a = a[::-1]
    if channels == 3:
        return np.ascontiguousarray(a[:, :, 2::-1])  # B,G,R -> R,G,B (drops A)
    b, g, r = (a[:, :, i].astype(np.uint16) for i in range(3))
    return ((29 * b + 150 * g + 77 * r) >> 8).astype(np.uint8)[:, :, None]


class TMAgentClient:
    """Client side of the TMAgentLink protocol.

    timeout: seconds to keep retrying the initial TCP connect + handshake.
    request_timeout: default reply timeout for ordinary requests.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = P.DEFAULT_PORT,
        timeout: float = 30.0,
        *,
        request_timeout: float = 10.0,
        client_name: str = "tmagent",
    ) -> None:
        self.host, self.port, self.timeout = host, port, timeout
        self.request_timeout, self.client_name = request_timeout, client_name
        self.hello: HelloInfo | None = None
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._send_lock = threading.Lock()
        self._req_lock = threading.Lock()
        self._ids = itertools.count(1)
        self._pending: dict[int, _Slot] = {}
        self._cv = threading.Condition()  # guards the push slots below
        self._latest_state: tuple[P.StateMsg, float] | None = None
        self._latest_frame: P.FrameMsg | None = None
        self.states_received = 0
        self.frames_received = 0
        self.stale_replies = 0
        self._error: Exception | None = None

    # ------------------------------------------------------------ connection

    @property
    def connected(self) -> bool:
        return self._sock is not None and self._error is None

    def connect(self) -> HelloInfo:
        """Connect (retrying until `timeout`) and run the HELLO handshake.

        Idempotent while connected; call again after ConnectionLost to reconnect.
        """
        if self.connected and self.hello is not None:
            return self.hello
        self._teardown()
        deadline = time.monotonic() + self.timeout
        last: OSError | None = None
        while True:
            try:
                sock = socket.create_connection((self.host, self.port), timeout=2.0)
                break
            except OSError as e:
                last = e
                if time.monotonic() >= deadline:
                    raise TMAgentError(
                        f"cannot connect to the TMAgentLink plugin at {self.host}:{self.port} "
                        f"within {self.timeout:g} s ({last}). Is TrackMania running with the "
                        "plugin enabled and `set tmagent_port` matching game.tmi_port? "
                        "See docs/setup_windows.md."
                    ) from e
                time.sleep(0.25)
        sock.settimeout(None)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock, self._error = sock, None
        self._reader = threading.Thread(
            target=self._read_loop, args=(sock,), daemon=True, name="tmagent-reader"
        )  # noqa: E501
        self._reader.start()
        try:
            rep, (version, build) = self._request(
                lambda rid: P.enc_hello(rid, self.client_name),
                timeout=max(5.0, deadline - time.monotonic()),
            )
        except TMAgentError as e:
            self._teardown()
            raise TMAgentError(f"handshake with {self.host}:{self.port} failed: {e}") from e
        if version != P.PROTOCOL_VERSION:
            self._teardown()
            raise ProtocolMismatch(
                f"plugin '{build}' speaks protocol v{version}, this client speaks "
                f"v{P.PROTOCOL_VERSION}. Copy tmagent/game/tmnf/plugin/TMAgentLink.as from this "
                "checkout into Documents/TMInterface/Plugins/ and restart the game."
            )
        self.hello = HelloInfo(version, build)
        return self.hello

    def close(self) -> None:
        """Say CLOSE (best effort) and tear the connection down."""
        if self.connected:
            try:
                self._request(P.enc_close, timeout=1.0)
            except TMAgentError:
                pass
        self._teardown()

    def _teardown(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            sock.close()
        t, self._reader = self._reader, None
        if t is not None and t is not threading.current_thread():
            t.join(timeout=2.0)
        self._fail(ConnectionLost("connection closed"))
        self.hello = None
        self._error = None  # a fresh connect() starts clean
        with self._cv:
            self._latest_state, self._latest_frame = None, None

    def __enter__(self) -> TMAgentClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------------------------------------------------------- reader thread

    def _read_loop(self, sock: socket.socket) -> None:
        reader = P.MessageReader()
        try:
            while True:
                data = sock.recv(1 << 20)
                if not data:
                    raise ConnectionLost("the plugin closed the connection")
                now = time.perf_counter()  # receipt time of everything in this chunk
                for msg in reader.feed(data):
                    self._dispatch(msg, now)
        except (OSError, P.ProtocolError, ConnectionLost) as e:
            if sock is self._sock:  # not a deliberate close
                self._fail(e if isinstance(e, ConnectionLost) else ConnectionLost(str(e)))

    def _dispatch(self, msg: P.Message, now: float) -> None:
        rep, rid, val = P.dec_reply(msg)
        if rep is P.Rep.PUSH_STATE:
            with self._cv:
                self._latest_state = (val, now)  # type: ignore[assignment]
                self.states_received += 1
                self._cv.notify_all()
        elif rep is P.Rep.PUSH_FRAME:
            val.wall_time = now  # type: ignore[attr-defined]
            with self._cv:
                self._latest_frame = val  # type: ignore[assignment]
                self.frames_received += 1
                self._cv.notify_all()
        else:
            if rep is P.Rep.FRAME:
                val.wall_time = now  # type: ignore[attr-defined]
            elif rep is P.Rep.STATE:  # a STATE reply is also the newest known state
                with self._cv:
                    self._latest_state = (val, now)  # type: ignore[assignment]
                    self._cv.notify_all()
            slot = self._pending.get(rid)
            if slot is None:
                self.stale_replies += 1
                return
            slot.rep, slot.value = rep, val
            slot.event.set()

    def _fail(self, err: Exception) -> None:
        """Mark the connection dead and wake every waiting request."""
        self._error = self._error or err
        for slot in list(self._pending.values()):
            slot.error = self._error
            slot.event.set()
        with self._cv:
            self._cv.notify_all()

    # ---------------------------------------------------------------- requests

    def _send(self, data: bytes) -> None:
        sock = self._sock
        if sock is None or self._error is not None:
            raise self._error or ConnectionLost("not connected")
        try:
            with self._send_lock:
                sock.sendall(data)
        except OSError as e:
            err = ConnectionLost(f"send failed: {e}")
            self._fail(err)
            raise err from e

    def _request(
        self, build: Callable[[int], bytes], timeout: float | None = None
    ) -> tuple[P.Rep, object]:
        """Send one command and wait for its reply; ERROR replies raise PluginError."""
        timeout = self.request_timeout if timeout is None else timeout
        with self._req_lock:
            rid = next(self._ids)
            slot = self._pending[rid] = _Slot()
            try:
                self._send(build(rid))
                if not slot.event.wait(timeout):
                    raise TMAgentTimeout(f"no reply from the plugin within {timeout:g} s")
                if slot.error is not None:
                    raise slot.error
            finally:
                self._pending.pop(rid, None)
        if slot.rep is P.Rep.ERROR:
            raise PluginError(str(slot.value))
        assert slot.rep is not None
        return slot.rep, slot.value

    def set_mode(self, mode: P.Mode) -> None:
        self._request(lambda rid: P.enc_set_mode(rid, mode))

    def load_map(self, path: str, timeout: float = 120.0) -> str:
        """Load a map; returns the plugin's "uid<TAB>name" once the race is ready at t=0."""
        _, text = self._request(lambda rid: P.enc_load_map(rid, path), timeout)
        return str(text)

    def restart(
        self, method: P.RestartMethod = P.RestartMethod.REWIND, timeout: float = 60.0
    ) -> P.StateMsg:
        _, st = self._request(lambda rid: P.enc_restart(rid, method), timeout)
        return st  # type: ignore[return-value]

    def step(self, n_ticks: int, inp: P.InputCmd, timeout: float | None = None) -> P.StateMsg:
        """Sync mode: hold `inp` for n_ticks ticks (or until the finish); paused afterwards."""
        t = self.request_timeout + 0.02 * n_ticks if timeout is None else timeout
        _, st = self._request(lambda rid: P.enc_step(rid, n_ticks, inp), t)
        return st  # type: ignore[return-value]

    def set_input(self, inp: P.InputCmd) -> None:
        """Realtime: replace the held input from the next tick on. No reply, no waiting."""
        self._send(P.enc_set_input(inp))

    def request_frame(
        self, w: int, h: int, settle: int = 1, timeout: float | None = None
    ) -> P.FrameMsg:
        _, fm = self._request(lambda rid: P.enc_request_frame(rid, w, h, settle), timeout)
        return fm  # type: ignore[return-value]

    def stream_frames(self, on: bool, w: int = 0, h: int = 0, max_fps: int = 0) -> None:
        self._request(lambda rid: P.enc_stream_frames(rid, on, w, h, max_fps))

    def stream_state(self, every_n_ticks: int) -> None:
        self._request(lambda rid: P.enc_stream_state(rid, every_n_ticks))

    def set_speed(self, speed: float) -> None:
        self._request(lambda rid: P.enc_set_speed(rid, speed))

    def execute(self, command: str) -> None:
        self._request(lambda rid: P.enc_execute(rid, command))

    def get_state(self) -> P.StateMsg:
        _, st = self._request(P.enc_get_state)
        return st  # type: ignore[return-value]

    def ping(self) -> float:
        """Round-trip time in seconds of a no-op request."""
        t0 = time.perf_counter()
        self._request(P.enc_ping)
        return time.perf_counter() - t0

    def diagnostics(self) -> dict[str, int]:
        """Plugin counters from the PING reply text ("key=value ..."), e.g. guard_rewinds."""
        _, text = self._request(P.enc_ping)
        pairs = (kv.split("=", 1) for kv in str(text).split() if "=" in kv)
        return {k: int(v) for k, v in pairs if v.lstrip("-").isdigit()}

    # ------------------------------------------------------------ push slots

    def latest_state(self) -> tuple[P.StateMsg, float] | None:
        """Newest pushed (state, receipt perf_counter), or None. Non-blocking."""
        with self._cv:
            return self._latest_state

    def latest_frame(self) -> P.FrameMsg | None:
        """Newest pushed frame, or None. Non-blocking."""
        with self._cv:
            return self._latest_frame

    def reset_pushed(self, state: P.StateMsg | None = None) -> None:
        """Forget pushed frames/states (e.g. after a restart); optionally seed the state."""
        with self._cv:
            self._latest_frame = None
            self._latest_state = (state, time.perf_counter()) if state is not None else None

    def wait_for_frames(self, count: int, timeout: float) -> bool:
        """Block until frames_received >= count (test/smoke helper)."""
        end = time.monotonic() + timeout
        with self._cv:
            while self.frames_received < count:
                left = end - time.monotonic()
                if left <= 0 or self._error is not None:
                    return self.frames_received >= count
                self._cv.wait(left)
        return True
