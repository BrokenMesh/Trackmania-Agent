"""TMAgentClient against FakePluginServer: handshake, sync stepping, frames, realtime, reconnect."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.client import (
    ConnectionLost,
    PluginError,
    ProtocolMismatch,
    TMAgentClient,
    TMAgentError,
    TMAgentTimeout,
    convert_pixels,
)
from tmagent.game.tmnf.fake_server import FakePluginServer, fake_frame_race_time

GAS = P.InputCmd(accelerate=True)


@pytest.fixture
def server():
    with FakePluginServer() as srv:
        yield srv


@pytest.fixture
def client(server):
    c = TMAgentClient(port=server.port, timeout=3.0, request_timeout=3.0)
    c.connect()
    yield c
    c.close()


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return cond()


def test_handshake_reports_build_and_version(client):
    assert client.hello.protocol_version == P.PROTOCOL_VERSION
    assert client.hello.build == "FakePlugin 0.1"
    assert client.connect() is client.hello  # idempotent


def test_version_mismatch_is_a_clear_error():
    with FakePluginServer(protocol_version=P.PROTOCOL_VERSION + 1, build="OldPlugin 0.0") as srv:
        c = TMAgentClient(port=srv.port, timeout=2.0)
        with pytest.raises(ProtocolMismatch, match=r"OldPlugin 0.0.*TMAgentLink\.as"):
            c.connect()
        assert not c.connected


def test_connect_failure_names_the_remedy():
    c = TMAgentClient(port=1, timeout=0.3)  # nothing listens on port 1
    with pytest.raises(TMAgentError, match="plugin enabled"):
        c.connect()


def test_connect_retries_until_the_server_appears():
    srv = FakePluginServer()
    srv.start()
    port = srv.port
    srv.stop()  # port now closed; start a server on it again shortly
    result: list = []

    def late_start():
        time.sleep(0.5)
        s2 = FakePluginServer(port=port)
        s2.start()
        result.append(s2)

    threading.Thread(target=late_start, daemon=True).start()
    c = TMAgentClient(port=port, timeout=5.0)
    try:
        assert c.connect().build == "FakePlugin 0.1"
    finally:
        c.close()
        for s in result:
            s.stop()


def test_load_map_start_race_step_race_time_math(client):
    client.set_mode(P.Mode.SYNC)
    assert client.load_map("/maps/Alpha.Challenge.Gbx") == "FAKEUID_Alpha\tAlpha"
    st = client.restart()
    assert (st.race_time_ms, st.paused, st.speed_kmh, st.finished) == (0, True, 0.0, False)
    total = 0
    for n in (1, 7, 100, 3):
        st = client.step(n, GAS)
        total += n
        assert st.race_time_ms == 10 * total
        assert st.paused and st.in_race and st.seq == total
    assert st.speed_kmh > 0 and st.position[0] > 0
    # stepping is additive: 100 single ticks == one 100-tick step (deterministic physics)
    a = client.restart()
    assert a.race_time_ms == 0
    one_shot = client.step(100, GAS)
    client.restart()
    for _ in range(100):
        last = client.step(1, GAS)
    assert last.race_time_ms == one_shot.race_time_ms == 1000
    assert last.position == one_shot.position and last.speed_kmh == one_shot.speed_kmh


def test_step_stops_at_the_finish_and_restart_recovers(client):
    client.set_mode(P.Mode.SYNC)
    client.load_map("/maps/A.Challenge.Gbx")
    client.restart()
    st = client.step(5000, GAS)
    assert st.finished and st.cp_count == st.cp_target == 3
    assert 0 < st.race_time_ms < 50_000
    assert client.step(10, GAS).race_time_ms == st.race_time_ms  # no-op after the finish
    st = client.restart()
    assert (st.race_time_ms, st.finished, st.cp_count) == (0, False, 0)


def test_plugin_errors_raise_plugin_error(client):
    with pytest.raises(PluginError, match="no race ready"):
        client.step(1, GAS)
    with pytest.raises(PluginError, match="not in a race"):
        client.restart()
    with pytest.raises(PluginError, match="map not found"):
        client.load_map("/maps/missing.Challenge.Gbx")
    assert client.ping() >= 0  # the connection survives errors


def test_ping_diagnostics_are_parsed(client):
    d = client.diagnostics()
    assert d == {"guard_rewinds": 0, "tick": 0, "op": 0}


def test_execute_and_get_state(client, server):
    client.execute("set skip_map_load_screens true")
    assert server.executed == ["set skip_map_load_screens true"]
    assert client.get_state().race_time_ms == 0


def test_frames_encode_race_time_and_convert_channels(client):
    client.set_mode(P.Mode.SYNC)
    client.load_map("/maps/A.Challenge.Gbx")
    client.restart()
    total = 0
    for k in (1, 5, 37, 100):  # frame captured after step k shows race time 10 * total
        client.step(k, GAS)
        total += k
        fm = client.request_frame(16, 12, 1)
        assert fm.race_time_ms == 10 * total
        rgb = convert_pixels(fm, 3, flip_vertical=False)
        assert fake_frame_race_time(rgb) == 10 * total
    assert rgb.shape == (12, 16, 3) and rgb.dtype == np.uint8
    assert rgb[0].min() == 255  # white top row: rows are top-down
    flipped = convert_pixels(fm, 3, flip_vertical=True)
    assert flipped[-1].min() == 255 and flipped[0].min() < 255
    gray = convert_pixels(fm, 1, flip_vertical=False)
    assert gray.shape == (12, 16, 1) and gray.dtype == np.uint8
    b, g, r = (int(v) for v in rgb[6, 8, ::-1])
    assert int(gray[6, 8, 0]) == (29 * b + 150 * g + 77 * r) >> 8
    assert (gray[0] == 255).all()


def test_bgra_to_rgb_channel_order():
    px = np.zeros((2, 2, 4), np.uint8)
    px[..., 0], px[..., 1], px[..., 2], px[..., 3] = 10, 20, 30, 255  # B, G, R, A
    fm = P.FrameMsg(0, 2, 2, P.PIXEL_BGRA8, 0, px.tobytes())
    assert convert_pixels(fm, 3).reshape(-1, 3)[0].tolist() == [30, 20, 10]
    with pytest.raises(ValueError):
        convert_pixels(fm, 4)


def test_realtime_set_input_has_effect_and_pushes_arrive(server):
    c = TMAgentClient(port=server.port, timeout=3.0)
    c.connect()
    try:
        c.load_map("/maps/A.Challenge.Gbx")
        c.set_speed(5.0)
        c.set_mode(P.Mode.REALTIME)
        c.stream_frames(True, 16, 12, 60)
        c.stream_state(1)
        assert wait_for(lambda: c.frames_received >= 3)
        s0, _ = c.latest_state()
        assert s0.speed_kmh == 0.0 and not s0.paused
        c.set_input(GAS)  # no reply, takes effect from the next tick
        assert wait_for(lambda: c.latest_state()[0].speed_kmh > 5.0)
        st, wall = c.latest_state()
        assert st.seq > s0.seq and abs(time.perf_counter() - wall) < 1.0
        fm = c.latest_frame()
        assert (fm.width, fm.height) == (16, 12) and fm.wall_time > 0
        c.stream_frames(False)
        n = c.frames_received
        time.sleep(0.2)
        assert c.frames_received <= n + 1  # the stream stopped
        c.set_mode(P.Mode.SYNC)
        s1 = c.get_state()
        assert s1.paused
        time.sleep(0.1)
        assert c.get_state().race_time_ms == s1.race_time_ms  # sync mode is paused
    finally:
        c.close()


def test_step_in_realtime_mode_is_an_error(client):
    client.load_map("/maps/A.Challenge.Gbx")
    client.set_mode(P.Mode.REALTIME)
    with pytest.raises(PluginError, match="sync mode"):
        client.step(1, GAS)


def test_disconnect_then_reconnect(server):
    c = TMAgentClient(port=server.port, timeout=3.0, request_timeout=2.0)
    c.connect()
    c.load_map("/maps/A.Challenge.Gbx")
    server.drop_client()
    assert wait_for(lambda: not c.connected)
    with pytest.raises(ConnectionLost):
        c.get_state()
    with pytest.raises(ConnectionLost):
        c.set_input(GAS)
    c.connect()  # reconnect on the same client object
    assert c.connected and c.get_state().race_time_ms == 0
    c.close()
    assert not c.connected
    c.connect()  # and again after a clean close
    assert c.ping() >= 0
    c.close()


def test_new_connection_replaces_the_old_one(server):
    a = TMAgentClient(port=server.port, timeout=2.0)
    a.connect()
    b = TMAgentClient(port=server.port, timeout=2.0)
    b.connect()
    assert b.get_state() is not None
    assert wait_for(lambda: not a.connected)
    b.close()
    a.close()


def test_request_timeout_and_late_reply_is_discarded():
    # A server that swallows the first PING and answers the second.
    import socket

    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    port = lsock.getsockname()[1]

    def serve():
        conn, _ = lsock.accept()
        reader, pings = P.MessageReader(), 0
        held = None
        while True:
            data = conn.recv(4096)
            if not data:
                return
            for msg in reader.feed(data):
                cmd, rid, _ = P.dec_command(msg)
                if cmd is P.Cmd.HELLO:
                    conn.sendall(P.enc_hello_reply(rid, P.PROTOCOL_VERSION, "slow"))
                elif cmd is P.Cmd.PING:
                    pings += 1
                    if pings == 1:
                        held = rid  # answer later, after the client gave up
                    else:
                        conn.sendall(P.enc_ack(held, "late") + P.enc_ack(rid, "on time"))

    threading.Thread(target=serve, daemon=True).start()
    c = TMAgentClient(port=port, timeout=2.0)
    c.connect()
    try:
        with pytest.raises(TMAgentTimeout):
            c._request(P.enc_ping, timeout=0.2)
        assert c.ping() >= 0  # got "on time"; the late ACK was dropped, not mistaken for it
        assert c.stale_replies == 1
    finally:
        c.close()
        lsock.close()


def test_set_input_is_never_blocked_by_a_pending_request(server):
    c = TMAgentClient(port=server.port, timeout=2.0)
    c.connect()
    try:
        started = threading.Event()

        def hold_request_lock():
            with c._req_lock:
                started.set()
                time.sleep(0.5)

        threading.Thread(target=hold_request_lock, daemon=True).start()
        started.wait(1.0)
        t0 = time.perf_counter()
        c.set_input(GAS)
        assert time.perf_counter() - t0 < 0.2
    finally:
        c.close()
