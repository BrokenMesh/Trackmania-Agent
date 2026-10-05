"""TMAgentLink wire protocol: message ids, payload layouts, encode/decode.

Spec and rationale: tmagent/game/tmnf/PROTOCOL.md. The AngelScript plugin
(plugin/TMAgentLink.as) implements the same layouts by hand, so every change here
must bump PROTOCOL_VERSION and be mirrored there.

All integers/floats are little-endian. A message is `int32 type, int32 payload
length, payload`. Every command payload starts with `int32 req_id`; replies to a
command echo it. Pushed messages (PUSH_STATE, PUSH_FRAME) carry no req_id.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

PROTOCOL_VERSION = 1
DEFAULT_PORT = 8477  # same as GameConfig.tmi_port
MAX_PAYLOAD = 64 << 20  # sanity limit; a 1920x1080 BGRA frame is ~8 MB

# TMI analog steer value for full lock. Sign convention: negative = left.
# UNVERIFIED on a real install (TMI docs say negative = left); flip here only.
STEER_FULL_SCALE = 65536
STEER_NEGATIVE_IS_LEFT = True


class Cmd(IntEnum):
    """Client -> plugin message types."""

    HELLO = 1
    SET_MODE = 2
    LOAD_MAP = 3
    RESTART = 4
    STEP = 5
    SET_INPUT = 6  # no reply
    REQUEST_FRAME = 7
    STREAM_FRAMES = 8
    SET_SPEED = 9
    EXECUTE = 10
    GET_STATE = 11
    CLOSE = 12
    STREAM_STATE = 13
    PING = 14


class Rep(IntEnum):
    """Plugin -> client message types."""

    ACK = 101
    ERROR = 102
    STATE = 103
    FRAME = 104
    HELLO = 105
    PUSH_STATE = 106
    PUSH_FRAME = 107


class Mode(IntEnum):
    SYNC = 0
    REALTIME = 1


class RestartMethod(IntEnum):
    REWIND = 0  # rewind to the state saved at race time 0 (falls back to GIVE_UP)
    GIVE_UP = 1  # the game's own restart (countdown, then pause at 0)


PIXEL_BGRA8 = 0  # FRAME pixel format: width*height*4 bytes, B,G,R,A

_HEADER = struct.Struct("<ii")
_I32 = struct.Struct("<i")
_INPUT = struct.Struct("<BBBBB3xi")  # left right accelerate brake analog (pad) steer
_STATE = struct.Struct("<iBBBBiifffffffi")
_FRAME_HEAD = struct.Struct("<iiiii")  # race_time_ms width height pixel_format seq
INPUT_SIZE = _INPUT.size  # 12
STATE_SIZE = _STATE.size  # 48
FRAME_HEAD_SIZE = _FRAME_HEAD.size  # 20
HEADER_SIZE = _HEADER.size  # 8


class ProtocolError(ValueError):
    """Malformed message."""


class Message(NamedTuple):
    type: int
    payload: bytes


@dataclass(frozen=True)
class InputCmd:
    """Input state held until replaced. steer is used only when analog is True."""

    left: bool = False
    right: bool = False
    accelerate: bool = False
    brake: bool = False
    analog: bool = False
    steer: int = 0  # [-STEER_FULL_SCALE, STEER_FULL_SCALE] in TMI convention

    def pack(self) -> bytes:
        steer = max(-STEER_FULL_SCALE, min(STEER_FULL_SCALE, int(self.steer)))
        return _INPUT.pack(
            int(self.left), int(self.right), int(self.accelerate), int(self.brake),
            int(self.analog), steer,
        )  # fmt: skip

    @staticmethod
    def unpack(data: bytes) -> InputCmd:
        if len(data) != INPUT_SIZE:
            raise ProtocolError(f"input payload must be {INPUT_SIZE} bytes, got {len(data)}")
        left, right, acc, brk, analog, steer = _INPUT.unpack(data)
        return InputCmd(bool(left), bool(right), bool(acc), bool(brk), bool(analog), steer)


@dataclass(frozen=True)
class StateMsg:
    """Plugin-side simulation state (STATE reply / PUSH_STATE)."""

    race_time_ms: int = 0
    finished: bool = False
    in_race: bool = False
    paused: bool = False
    cp_count: int = 0
    cp_target: int = -1  # -1 = unknown until the first checkpoint event
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_kmh: float = 0.0
    seq: int = 0  # plugin OnRunStep counter since the client connected

    def pack(self) -> bytes:
        return _STATE.pack(
            self.race_time_ms, int(self.finished), int(self.in_race), int(self.paused), 0,
            self.cp_count, self.cp_target, *self.position, *self.velocity,
            self.speed_kmh, self.seq,
        )  # fmt: skip

    @staticmethod
    def unpack(data: bytes) -> StateMsg:
        if len(data) != STATE_SIZE:
            raise ProtocolError(f"state payload must be {STATE_SIZE} bytes, got {len(data)}")
        (rt, fin, inr, pau, _pad, cpc, cpt, px, py, pz, vx, vy, vz, spd, seq) = _STATE.unpack(data)
        return StateMsg(
            rt, bool(fin), bool(inr), bool(pau), cpc, cpt, (px, py, pz), (vx, vy, vz), spd, seq
        )  # noqa: E501


@dataclass
class FrameMsg:
    """A captured image: raw pixels as sent by the plugin (see PIXEL_BGRA8)."""

    race_time_ms: int
    width: int
    height: int
    pixel_format: int
    seq: int
    pixels: bytes
    wall_time: float = 0.0  # perf_counter at receipt, filled in by the client

    def pack(self) -> bytes:
        head = _FRAME_HEAD.pack(
            self.race_time_ms, self.width, self.height, self.pixel_format, self.seq
        )  # noqa: E501
        return head + self.pixels

    @staticmethod
    def unpack(data: bytes) -> FrameMsg:
        if len(data) < FRAME_HEAD_SIZE:
            raise ProtocolError("frame payload shorter than its header")
        rt, w, h, fmt, seq = _FRAME_HEAD.unpack_from(data)
        pixels = bytes(data[FRAME_HEAD_SIZE:])
        if fmt == PIXEL_BGRA8 and len(pixels) != w * h * 4:
            raise ProtocolError(f"frame has {len(pixels)} bytes, expected {w}x{h}x4 = {w * h * 4}")
        return FrameMsg(rt, w, h, fmt, seq, pixels)


# ---------------------------------------------------------------- framing


def pack_message(mtype: int, payload: bytes = b"") -> bytes:
    return _HEADER.pack(int(mtype), len(payload)) + payload


class MessageReader:
    """Incremental stream parser: feed() bytes, get complete Messages back."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[Message]:
        self._buf += data
        out: list[Message] = []
        while len(self._buf) >= HEADER_SIZE:
            mtype, length = _HEADER.unpack_from(self._buf)
            if length < 0 or length > MAX_PAYLOAD:
                raise ProtocolError(f"bad payload length {length} (type {mtype})")
            if len(self._buf) < HEADER_SIZE + length:
                break
            out.append(Message(mtype, bytes(self._buf[HEADER_SIZE : HEADER_SIZE + length])))
            del self._buf[: HEADER_SIZE + length]
        return out


# ------------------------------------------------------- command encoding


def _cmd(cmd: Cmd, req_id: int, body: bytes = b"") -> bytes:
    return pack_message(cmd, _I32.pack(req_id) + body)


def enc_hello(req_id: int, client_name: str = "tmagent") -> bytes:
    return _cmd(Cmd.HELLO, req_id, _I32.pack(PROTOCOL_VERSION) + client_name.encode())


def enc_set_mode(req_id: int, mode: Mode) -> bytes:
    return _cmd(Cmd.SET_MODE, req_id, _I32.pack(int(mode)))


def enc_load_map(req_id: int, path: str) -> bytes:
    return _cmd(Cmd.LOAD_MAP, req_id, path.encode())


def enc_restart(req_id: int, method: RestartMethod = RestartMethod.REWIND) -> bytes:
    return _cmd(Cmd.RESTART, req_id, _I32.pack(int(method)))


def enc_step(req_id: int, n_ticks: int, inp: InputCmd) -> bytes:
    return _cmd(Cmd.STEP, req_id, _I32.pack(n_ticks) + inp.pack())


def enc_set_input(inp: InputCmd) -> bytes:
    return _cmd(Cmd.SET_INPUT, 0, inp.pack())


def enc_request_frame(req_id: int, w: int, h: int, settle: int = 0) -> bytes:
    return _cmd(Cmd.REQUEST_FRAME, req_id, struct.pack("<iii", w, h, settle))


def enc_stream_frames(req_id: int, on: bool, w: int, h: int, max_fps: int) -> bytes:
    return _cmd(Cmd.STREAM_FRAMES, req_id, struct.pack("<iiii", int(on), w, h, max_fps))


def enc_set_speed(req_id: int, speed: float) -> bytes:
    return _cmd(Cmd.SET_SPEED, req_id, struct.pack("<f", speed))


def enc_execute(req_id: int, command: str) -> bytes:
    return _cmd(Cmd.EXECUTE, req_id, command.encode())


def enc_get_state(req_id: int) -> bytes:
    return _cmd(Cmd.GET_STATE, req_id)


def enc_close(req_id: int) -> bytes:
    return _cmd(Cmd.CLOSE, req_id)


def enc_stream_state(req_id: int, every_n_ticks: int) -> bytes:
    return _cmd(Cmd.STREAM_STATE, req_id, _I32.pack(every_n_ticks))


def enc_ping(req_id: int) -> bytes:
    return _cmd(Cmd.PING, req_id)


def dec_command(msg: Message) -> tuple[Cmd, int, tuple]:
    """Parse a command message into (cmd, req_id, args). Used by the fake plugin."""
    try:
        return _dec_command(msg)
    except (struct.error, ValueError) as e:
        if isinstance(e, ProtocolError):
            raise
        raise ProtocolError(f"malformed command {msg.type}: {e}") from None


def _dec_command(msg: Message) -> tuple[Cmd, int, tuple]:
    try:
        cmd = Cmd(msg.type)
    except ValueError:
        raise ProtocolError(f"unknown command type {msg.type}") from None
    p = msg.payload
    if len(p) < 4:
        raise ProtocolError(f"{cmd.name}: payload shorter than req_id")
    (req_id,) = _I32.unpack_from(p)
    body = p[4:]
    if cmd is Cmd.HELLO:
        return cmd, req_id, (_I32.unpack_from(body)[0], body[4:].decode(errors="replace"))
    if cmd is Cmd.SET_MODE:
        return cmd, req_id, (Mode(_I32.unpack(body)[0]),)
    if cmd in (Cmd.LOAD_MAP, Cmd.EXECUTE):
        return cmd, req_id, (body.decode(errors="replace"),)
    if cmd is Cmd.RESTART:
        return cmd, req_id, (RestartMethod(_I32.unpack(body)[0]),)
    if cmd is Cmd.STEP:
        return cmd, req_id, (_I32.unpack_from(body)[0], InputCmd.unpack(body[4:]))
    if cmd is Cmd.SET_INPUT:
        return cmd, req_id, (InputCmd.unpack(body),)
    if cmd is Cmd.REQUEST_FRAME:
        return cmd, req_id, struct.unpack("<iii", body)
    if cmd is Cmd.STREAM_FRAMES:
        on, w, h, fps = struct.unpack("<iiii", body)
        return cmd, req_id, (bool(on), w, h, fps)
    if cmd is Cmd.SET_SPEED:
        return cmd, req_id, struct.unpack("<f", body)
    if cmd is Cmd.STREAM_STATE:
        return cmd, req_id, (_I32.unpack(body)[0],)
    return cmd, req_id, ()  # GET_STATE, CLOSE, PING


# --------------------------------------------------------- reply encoding


def _rep(rep: Rep, req_id: int, body: bytes = b"") -> bytes:
    return pack_message(rep, _I32.pack(req_id) + body)


def enc_ack(req_id: int, text: str = "") -> bytes:
    return _rep(Rep.ACK, req_id, text.encode())


def enc_error(req_id: int, text: str) -> bytes:
    return _rep(Rep.ERROR, req_id, text.encode())


def enc_state(req_id: int, st: StateMsg) -> bytes:
    return _rep(Rep.STATE, req_id, st.pack())


def enc_frame(req_id: int, fr: FrameMsg) -> bytes:
    return _rep(Rep.FRAME, req_id, fr.pack())


def enc_hello_reply(req_id: int, version: int, build: str) -> bytes:
    return _rep(Rep.HELLO, req_id, _I32.pack(version) + build.encode())


def enc_push_state(st: StateMsg) -> bytes:
    return pack_message(Rep.PUSH_STATE, st.pack())


def enc_push_frame(fr: FrameMsg) -> bytes:
    return pack_message(Rep.PUSH_FRAME, fr.pack())


def dec_reply(msg: Message) -> tuple[Rep, int, object]:
    """Parse a plugin message into (type, req_id, value); req_id is 0 for pushes.

    value: str for ACK/ERROR, StateMsg, FrameMsg, (version, build) for HELLO.
    """
    try:
        return _dec_reply(msg)
    except struct.error as e:
        raise ProtocolError(f"malformed reply {msg.type}: {e}") from None


def _dec_reply(msg: Message) -> tuple[Rep, int, object]:
    try:
        rep = Rep(msg.type)
    except ValueError:
        raise ProtocolError(f"unknown reply type {msg.type}") from None
    p = msg.payload
    if rep is Rep.PUSH_STATE:
        return rep, 0, StateMsg.unpack(p)
    if rep is Rep.PUSH_FRAME:
        return rep, 0, FrameMsg.unpack(p)
    if len(p) < 4:
        raise ProtocolError(f"{rep.name}: payload shorter than req_id")
    (req_id,) = _I32.unpack_from(p)
    body = p[4:]
    if rep in (Rep.ACK, Rep.ERROR):
        return rep, req_id, body.decode(errors="replace")
    if rep is Rep.STATE:
        return rep, req_id, StateMsg.unpack(body)
    if rep is Rep.FRAME:
        return rep, req_id, FrameMsg.unpack(body)
    return rep, req_id, (_I32.unpack_from(body)[0], body[4:].decode(errors="replace"))  # HELLO


def steer_to_wire(steer: float) -> int:
    """tmagent steer in [-1, 1] (negative = left) -> TMI analog steer int."""
    v = round(max(-1.0, min(1.0, float(steer))) * STEER_FULL_SCALE)
    return v if STEER_NEGATIVE_IS_LEFT else -v
