"""Wire format: roundtrips, golden bytes, stream parsing, errors."""

from __future__ import annotations

import struct

import pytest

from tmagent.game.tmnf import protocol as P


def one(data: bytes) -> P.Message:
    (msg,) = P.MessageReader().feed(data)
    return msg


def test_layout_sizes():
    assert P.INPUT_SIZE == 12 and P.STATE_SIZE == 48 and P.FRAME_HEAD_SIZE == 20


def test_input_golden_bytes_and_roundtrip():
    inp = P.InputCmd(left=True, accelerate=True, analog=True, steer=-65536)
    raw = inp.pack()
    assert raw == bytes([1, 0, 1, 0, 1, 0, 0, 0]) + struct.pack("<i", -65536)
    assert P.InputCmd.unpack(raw) == inp
    assert P.InputCmd(steer=10**9).pack()[-4:] == struct.pack("<i", 65536)  # clamped
    with pytest.raises(P.ProtocolError):
        P.InputCmd.unpack(raw[:-1])


def test_state_golden_bytes_and_roundtrip():
    st = P.StateMsg(
        race_time_ms=1230, finished=True, in_race=True, paused=False, cp_count=2, cp_target=5,
        position=(1.0, 2.0, 3.0), velocity=(4.0, 5.0, 6.0), speed_kmh=7.5, seq=99,
    )  # fmt: skip
    raw = st.pack()
    assert raw[:8] == struct.pack("<i", 1230) + bytes([1, 1, 0, 0])
    assert raw[8:16] == struct.pack("<ii", 2, 5)
    assert struct.unpack_from("<7f", raw, 16) == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.5)
    assert raw[44:] == struct.pack("<i", 99)
    assert P.StateMsg.unpack(raw) == st


def test_frame_roundtrip_and_size_check():
    fm = P.FrameMsg(230, 4, 3, P.PIXEL_BGRA8, 7, bytes(range(48)))
    back = P.FrameMsg.unpack(fm.pack())
    assert (back.race_time_ms, back.width, back.height, back.seq) == (230, 4, 3, 7)
    assert back.pixels == fm.pixels
    with pytest.raises(P.ProtocolError):
        P.FrameMsg.unpack(P.FrameMsg(0, 4, 3, 0, 0, b"short").pack())


def test_header_golden_bytes():
    assert P.pack_message(P.Cmd.PING, b"\x01\x00\x00\x00") == bytes.fromhex(
        "0e000000 04000000 01000000"
    )


COMMANDS = [
    (P.enc_hello(5, "me"), P.Cmd.HELLO, (P.PROTOCOL_VERSION, "me")),
    (P.enc_set_mode(5, P.Mode.REALTIME), P.Cmd.SET_MODE, (P.Mode.REALTIME,)),
    (
        P.enc_load_map(5, "C:\\maps\\A.Challenge.Gbx"),
        P.Cmd.LOAD_MAP,
        ("C:\\maps\\A.Challenge.Gbx",),
    ),
    (P.enc_restart(5, P.RestartMethod.GIVE_UP), P.Cmd.RESTART, (P.RestartMethod.GIVE_UP,)),
    (P.enc_step(5, 42, P.InputCmd(right=True)), P.Cmd.STEP, (42, P.InputCmd(right=True))),
    (P.enc_set_input(P.InputCmd(brake=True)), P.Cmd.SET_INPUT, (P.InputCmd(brake=True),)),
    (P.enc_request_frame(5, 128, 96, 1), P.Cmd.REQUEST_FRAME, (128, 96, 1)),
    (P.enc_stream_frames(5, True, 64, 48, 30), P.Cmd.STREAM_FRAMES, (True, 64, 48, 30)),
    (P.enc_set_speed(5, 2.5), P.Cmd.SET_SPEED, (2.5,)),
    (P.enc_execute(5, "set x 1"), P.Cmd.EXECUTE, ("set x 1",)),
    (P.enc_get_state(5), P.Cmd.GET_STATE, ()),
    (P.enc_close(5), P.Cmd.CLOSE, ()),
    (P.enc_stream_state(5, 2), P.Cmd.STREAM_STATE, (2,)),
    (P.enc_ping(5), P.Cmd.PING, ()),
]


@pytest.mark.parametrize(("data", "cmd", "args"), COMMANDS, ids=[c[1].name for c in COMMANDS])
def test_command_roundtrip(data, cmd, args):
    got_cmd, rid, got_args = P.dec_command(one(data))
    assert got_cmd is cmd
    assert rid == (0 if cmd is P.Cmd.SET_INPUT else 5)
    assert tuple(got_args) == args


def test_reply_roundtrip():
    st = P.StateMsg(race_time_ms=10, speed_kmh=1.5)
    fm = P.FrameMsg(10, 2, 2, 0, 1, bytes(16))
    assert P.dec_reply(one(P.enc_ack(3, "uid\tname"))) == (P.Rep.ACK, 3, "uid\tname")
    assert P.dec_reply(one(P.enc_error(3, "boom"))) == (P.Rep.ERROR, 3, "boom")
    assert P.dec_reply(one(P.enc_state(3, st))) == (P.Rep.STATE, 3, st)
    assert P.dec_reply(one(P.enc_hello_reply(3, 1, "b"))) == (P.Rep.HELLO, 3, (1, "b"))
    assert P.dec_reply(one(P.enc_push_state(st))) == (P.Rep.PUSH_STATE, 0, st)
    rep, rid, val = P.dec_reply(one(P.enc_frame(3, fm)))
    assert (rep, rid, val.pixels) == (P.Rep.FRAME, 3, fm.pixels)
    rep, rid, val = P.dec_reply(one(P.enc_push_frame(fm)))
    assert (rep, rid, val.width) == (P.Rep.PUSH_FRAME, 0, 2)


def test_reader_handles_fragmented_and_coalesced_streams():
    data = P.enc_ack(1, "a") + P.enc_ack(2, "bb") + P.enc_state(3, P.StateMsg())
    reader, out = P.MessageReader(), []
    for i in range(0, len(data), 3):  # 3-byte chunks split headers and payloads
        out += reader.feed(data[i : i + 3])
    assert [P.dec_reply(m)[1] for m in out] == [1, 2, 3]
    assert [m.type for m in P.MessageReader().feed(data)] == [101, 101, 103]


def test_reader_rejects_bad_length():
    with pytest.raises(P.ProtocolError):
        P.MessageReader().feed(struct.pack("<ii", 1, -5))
    with pytest.raises(P.ProtocolError):
        P.MessageReader().feed(struct.pack("<ii", 1, P.MAX_PAYLOAD + 1))


def test_short_payloads_raise_protocol_error_not_struct_error():
    with pytest.raises(P.ProtocolError):
        P.dec_command(P.Message(int(P.Cmd.STEP), b"\x01\x00\x00\x00\x05"))
    with pytest.raises(P.ProtocolError):
        P.dec_command(
            P.Message(int(P.Cmd.SET_MODE), b"\x01\x00\x00\x00\x07\x00\x00\x00")
        )  # bad mode
    with pytest.raises(P.ProtocolError):
        P.dec_reply(P.Message(int(P.Rep.HELLO), b"\x01\x00\x00\x00\x01"))


def test_unknown_types_raise():
    with pytest.raises(P.ProtocolError):
        P.dec_command(P.Message(999, b"\0\0\0\0"))
    with pytest.raises(P.ProtocolError):
        P.dec_reply(P.Message(999, b"\0\0\0\0"))
    with pytest.raises(P.ProtocolError):
        P.dec_command(P.Message(int(P.Cmd.PING), b"\0"))


@pytest.mark.parametrize(
    ("steer", "wire"),
    [(-1.0, -65536), (0.0, 0), (0.5, 32768), (1.0, 65536), (3.0, 65536), (-9.0, -65536)],
)
def test_steer_to_wire(steer, wire):
    assert P.steer_to_wire(steer) == (wire if P.STEER_NEGATIVE_IS_LEFT else -wire)
