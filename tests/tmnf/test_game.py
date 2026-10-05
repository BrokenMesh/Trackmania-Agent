"""TMNFGame: protocol conformance, action conversion, map lookup, sync and realtime flows."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest

from tmagent.config import DataConfig, GameConfig
from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.client import ConnectionLost, TMAgentError
from tmagent.game.tmnf.fake_server import FakePluginServer, fake_frame_race_time
from tmagent.game.tmnf.game import (
    TMNFGame,
    action_to_input,
    map_command_path,
    resolve_map,
    to_game_state,
)
from tmagent.interfaces import Action, Frame, GameState, RealtimeGame, SyncGame

W, H = 16, 12


def make_game(server, tmp_path, **cfg):
    gcfg = GameConfig(
        backend="tmnf", tmi_port=server.port, connect_timeout_s=3.0, game_speed=5.0,
        render_speed=10.0, map_dir=str(tmp_path), **cfg,
    )  # fmt: skip
    dcfg = DataConfig(resolution=[W, H], channels=3, frame_hz=20)
    (tmp_path / "Alpha.Challenge.Gbx").write_bytes(b"GBX")
    return TMNFGame(gcfg, dcfg)


@pytest.fixture
def server():
    with FakePluginServer() as srv:
        yield srv


@pytest.fixture
def game(server, tmp_path):
    g = make_game(server, tmp_path)
    yield g
    g.close()


def test_satisfies_both_protocols_without_connecting(server, tmp_path):
    g = make_game(server, tmp_path)
    assert isinstance(g, SyncGame)
    assert isinstance(g, RealtimeGame)
    assert not g.client.connected  # lazy connection


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (Action(steer=-0.6, gas=1.0), dict(left=True, accelerate=True)),
        (Action(steer=0.6, gas=1.0), dict(right=True, accelerate=True)),
        (Action(steer=0.5, gas=0.49), dict()),  # threshold is strict, gas < 0.5 is off
        (Action(steer=-0.5), dict()),
        (Action(steer=0.51, brake=0.5), dict(right=True, brake=True)),
        (Action(gas=0.5, brake=0.49), dict(accelerate=True)),
    ],
)
def test_action_to_input_binary(action, expected):
    assert action_to_input(action, "binary", 0.5) == P.InputCmd(**expected)


def test_action_to_input_binary_threshold_is_configurable():
    assert action_to_input(Action(steer=0.3), "binary", 0.2).right
    assert not action_to_input(Action(steer=0.3), "binary", 0.4).right


@pytest.mark.parametrize(
    ("steer", "raw"), [(0.0, 0), (0.5, 32768), (-1.0, -65536), (1.0, 65536), (2.0, 65536)]
)
def test_action_to_input_analog(steer, raw):
    sign = 1 if P.STEER_NEGATIVE_IS_LEFT else -1
    got = action_to_input(Action(steer=steer, gas=1.0), "analog")
    assert got == P.InputCmd(accelerate=True, analog=True, steer=sign * raw)
    assert not got.left and not got.right


def test_constructor_validates_restart_method(server, tmp_path):
    with pytest.raises(ValueError, match="restart_method"):
        TMNFGame(GameConfig(), DataConfig(), restart_method="teleport")


def test_action_to_input_rejects_unknown_mode():
    with pytest.raises(ValueError, match="steer_mode"):
        action_to_input(Action(), "gamepad")


def test_to_game_state_conversion():
    st = P.StateMsg(
        race_time_ms=1230,
        finished=True,
        cp_count=2,
        cp_target=4,
        position=(1, 2, 3),
        speed_kmh=9.0,
        seq=5,
    )
    gs = to_game_state(st)
    assert isinstance(gs, GameState) and gs.race_time_ms == 1230 and gs.finished
    assert gs.checkpoint == 2 and gs.num_checkpoints == 4 and gs.position.dtype == np.float32
    assert to_game_state(P.StateMsg(cp_target=-1)).num_checkpoints is None


# ---------------------------------------------------------------- map lookup


def test_resolve_map_by_path_name_uid_and_index(tmp_path):
    (tmp_path / "Alpha.Challenge.Gbx").write_bytes(b"x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "Beta_UID123abc.Challenge.Gbx").write_bytes(b"x")
    (tmp_path / "Gamma.Challenge.Gbx").write_bytes(b"x")
    (tmp_path / "index.json").write_text(
        json.dumps({"MAPUID1": "Gamma.Challenge.Gbx", "BAD": "nope.Challenge.Gbx"})
    )
    alpha = (tmp_path / "Alpha.Challenge.Gbx").resolve()
    assert resolve_map(str(alpha), "/nonexistent") == alpha  # an existing path wins
    assert resolve_map("Alpha.Challenge.Gbx", tmp_path) == alpha  # relative to map_dir
    assert resolve_map("alpha", tmp_path) == alpha  # stem, case-insensitive
    assert (
        resolve_map("UID123ABC", tmp_path).name == "Beta_UID123abc.Challenge.Gbx"
    )  # uid in the name
    assert resolve_map("MAPUID1", tmp_path).name == "Gamma.Challenge.Gbx"  # index.json
    with pytest.raises(FileNotFoundError, match="missing file"):
        resolve_map("BAD", tmp_path)
    with pytest.raises(FileNotFoundError, match="scanned"):
        resolve_map("nothing-like-this", tmp_path)
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_map("a", tmp_path)  # Alpha, Gamma, Beta... all contain "a"


def test_resolve_map_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_map("x", tmp_path / "nope")


def test_map_command_path_styles():
    p = Path("/home/u/Documents/TrackMania/Tracks/Challenges/My Challenges/A.Challenge.Gbx")
    assert map_command_path(p) == "My Challenges/A.Challenge.Gbx"
    assert map_command_path(p, "relative") == "My Challenges/A.Challenge.Gbx"
    assert map_command_path(p, "absolute") == str(p)
    other = Path("/data/maps/A.Challenge.Gbx")
    assert map_command_path(other) == str(other)
    assert map_command_path(other, "relative") == str(other)  # no Challenges dir: falls back
    with pytest.raises(ValueError):
        map_command_path(p, "weird")


# -------------------------------------------------------------------- sync


def test_sync_flow_state_and_frames(game, server):
    game.load_map("Alpha")
    assert game.map_info == ("FAKEUID_Alpha", "Alpha")
    assert server.loaded_map.endswith("Alpha.Challenge.Gbx")
    assert "set skip_map_load_screens true" in server.executed and "cam 1" in server.executed
    st = game.start_race()
    assert st.race_time_ms == 0 and st.speed_kmh == 0.0 and st.position.shape == (3,)
    total = 0
    for k in (1, 5, 37, 100):
        st = game.step(Action(steer=0.0, gas=1.0), k)
        total += k
        assert st.race_time_ms == 10 * total
        fr = game.grab_frame()  # frame captured after step k shows race time k*10
        assert isinstance(fr, Frame) and fr.race_time_ms == 10 * total
        assert fr.image.shape == (H, W, 3) and fr.image.dtype == np.uint8
        assert fake_frame_race_time(fr.image) == 10 * total
        assert abs(time.perf_counter() - fr.wall_time) < 2.0
    assert st.speed_kmh > 0
    with pytest.raises(ValueError):
        game.step(Action(), 0)
    assert game.start_race().race_time_ms == 0


def test_sync_steering_modes_reach_the_plugin(server, tmp_path):
    for mode, action, expect in [
        ("binary", Action(steer=-1.0, gas=1.0), P.InputCmd(left=True, accelerate=True)),
        (
            "analog",
            Action(steer=-0.5, gas=1.0),
            P.InputCmd(accelerate=True, analog=True, steer=P.steer_to_wire(-0.5)),
        ),
    ]:
        g = make_game(server, tmp_path, steer_mode=mode)
        try:
            g.load_map("Alpha")
            g.start_race()
            g.step(action, 3)
            assert server.input == expect
        finally:
            g.close()


def test_grab_frame_gray_and_flip(server, tmp_path):
    g = make_game(server, tmp_path)
    g.data_cfg.channels = 1
    g.flip_vertical = True
    try:
        g.load_map("Alpha")
        g.start_race()
        g.step(Action(gas=1.0), 100)
        fr = g.grab_frame()
        assert fr.image.shape == (H, W, 1)
        assert (fr.image[-1] == 255).all()  # white marker row moved to the bottom
    finally:
        g.close()


def test_size_mismatch_is_reported(server, tmp_path, monkeypatch):
    g = make_game(server, tmp_path)
    try:
        g.load_map("Alpha")
        g.start_race()
        monkeypatch.setattr(
            g.client, "request_frame", lambda w, h, s: P.FrameMsg(0, 2, 2, 0, 0, bytes(16))
        )
        with pytest.raises(TMAgentError, match="captured 2x2"):
            g.grab_frame()
    finally:
        g.close()


def test_connection_loss_is_explicit_and_recoverable(game, server):
    game.load_map("Alpha")
    game.start_race()
    server.drop_client()
    end = time.monotonic() + 3
    while game.client.connected and time.monotonic() < end:
        time.sleep(0.01)
    with pytest.raises(ConnectionLost, match="start_race"):
        game.step(Action(gas=1.0), 1)
    game.connect()
    game.load_map("Alpha")
    assert game.start_race().race_time_ms == 0


# ---------------------------------------------------------------- realtime


def test_realtime_flow(game, server):
    game.load_map("Alpha")
    game.restart()  # switches to realtime: speed 5x, frames + states pushed
    assert [c.name for c in server.commands].count("STREAM_FRAMES") >= 1
    assert server.speed == 5.0 and server.frame_stream == (W, H, 40)  # 2 * frame_hz
    deadline = time.monotonic() + 3
    frame = None
    while frame is None and time.monotonic() < deadline:
        frame = game.latest_frame()
        time.sleep(0.01)
    assert frame is not None and frame.image.shape == (H, W, 3)
    assert game.get_state().speed_kmh == 0.0
    t0 = time.perf_counter()
    for _ in range(100):  # set_action is cheap and deduplicated
        game.set_action(Action(steer=0.0, gas=1.0))
    assert (time.perf_counter() - t0) / 100 < 0.001
    assert [c.name for c in server.commands].count("SET_INPUT") <= 3
    deadline = time.monotonic() + 3
    while game.get_state().speed_kmh <= 5.0 and time.monotonic() < deadline:
        time.sleep(0.01)
    st = game.get_state()
    assert st.speed_kmh > 5.0 and st.race_time_ms > 0 and not st.extra["paused"]
    frame2 = game.latest_frame()
    assert frame2.wall_time >= frame.wall_time
    game.restart()
    assert game.get_state().race_time_ms < st.race_time_ms
    game.set_action(Action(steer=1.0, gas=1.0))  # a different action after a restart still sends
    end = time.monotonic() + 2
    while not server.input.right and time.monotonic() < end:
        time.sleep(0.01)
    assert server.input.right and server.input.accelerate


def test_realtime_to_sync_switch_pauses(game, server):
    game.load_map("Alpha")
    game.restart()
    game.set_action(Action(gas=1.0))
    time.sleep(0.2)
    st = game.start_race()  # back to sync: paused, race time 0
    assert st.race_time_ms == 0 and server.paused
    t1 = game.step(Action(), 10).race_time_ms
    assert t1 == 100
    assert game.latest_frame() is None  # realtime-only API outside realtime mode
