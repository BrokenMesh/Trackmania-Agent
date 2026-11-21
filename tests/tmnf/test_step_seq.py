"""STEP_SEQ: one round trip per batch of ticks, same simulation as tick-by-tick STEPs."""

from __future__ import annotations

import numpy as np
import pytest

from tmagent.config import DataConfig, GameConfig, load_config
from tmagent.eval.harness import run_episode_sync, step_ticks
from tmagent.eval.progress import ReferencePath
from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.fake_server import FakePluginServer
from tmagent.game.tmnf.game import TMNFGame
from tmagent.interfaces import Action

W, H = 16, 12
NEW_BUILD = "TMAgentLink 0.3.0 protocol 1"
OLD_BUILD = "TMAgentLink 0.2.0 protocol 1"


def make_game(server, tmp_path) -> TMNFGame:
    gcfg = GameConfig(
        backend="tmnf", tmi_port=server.port, connect_timeout_s=3.0, game_speed=5.0,
        render_speed=10.0, map_dir=str(tmp_path), steer_mode="pwm",
    )  # fmt: skip
    (tmp_path / "Alpha.Challenge.Gbx").write_bytes(b"GBX")
    return TMNFGame(gcfg, DataConfig(resolution=[W, H], channels=3, frame_hz=20))


def actions(n: int) -> list[Action]:
    rng = np.random.default_rng(0)
    return [
        Action(steer=float(rng.uniform(-1, 1)), gas=1.0, brake=float(rng.random() < 0.1))
        for _ in range(n)
    ]


def drive(build: str, tmp_path, acts: list[Action]):
    with FakePluginServer(build=build) as server:
        game = make_game(server, tmp_path)
        try:
            game.load_map("Alpha")
            game.start_race()
            states = step_ticks(game, acts)
            return states, list(server.commands), game.has_step_seq
        finally:
            game.close()


def test_protocol_roundtrip():
    inputs = [P.InputCmd(left=True, accelerate=True), P.InputCmd(right=True, brake=True)]
    (msg,) = P.MessageReader().feed(P.enc_step_seq(7, inputs))
    cmd, rid, (decoded,) = P.dec_command(msg)
    assert cmd is P.Cmd.STEP_SEQ and rid == 7 and decoded == inputs

    states = [
        P.StateMsg(
            race_time_ms=10 * i, in_race=True, position=(float(i), 0.0, 0.0), speed_kmh=3.5, seq=i
        )
        for i in range(3)
    ]
    (msg,) = P.MessageReader().feed(P.enc_states(7, states))
    rep, rid, value = P.dec_reply(msg)
    assert rep is P.Rep.STATES and rid == 7 and value == states
    with pytest.raises(ValueError):
        P.enc_step_seq(1, [P.InputCmd()] * (P.MAX_STEP_SEQ + 1))


def test_step_seq_equals_tick_by_tick_and_uses_one_round_trip_per_batch(tmp_path):
    acts = actions(150)  # more than MAX_STEP_SEQ: three batches
    seq, seq_cmds, has_seq = drive(NEW_BUILD, tmp_path, acts)
    old, old_cmds, has_old = drive(OLD_BUILD, tmp_path, acts)
    assert has_seq and not has_old
    assert [(s.race_time_ms, tuple(s.position), s.speed_kmh) for s in seq] == [
        (s.race_time_ms, tuple(s.position), s.speed_kmh) for s in old
    ]
    assert seq_cmds.count(P.Cmd.STEP_SEQ) == 3 and P.Cmd.STEP not in seq_cmds
    assert old_cmds.count(P.Cmd.STEP) == len(old) and P.Cmd.STEP_SEQ not in old_cmds


def test_step_seq_stops_at_the_finish(tmp_path):
    gas = [Action(steer=0.0, gas=1.0, brake=0.0)] * 2000
    seq, _, _ = drive(NEW_BUILD, tmp_path, gas)
    assert seq[-1].finished and not any(s.finished for s in seq[:-1]) and len(seq) < 2000


class GasPolicy:
    """Full gas, steering that wiggles through PWM (fractional values) but averages to 0."""

    def reset(self) -> None:
        self.k = 0

    def observe(self, image, past_actions) -> None:
        self.k += 1

    def predict(self) -> np.ndarray:
        steer = 0.3 if self.k % 2 else -0.3
        return np.tile(np.array([steer, 1.0, 0.0], np.float32), (8, 1))


def test_eval_episode_is_identical_with_and_without_step_seq(tmp_path):
    cfg = load_config(None, [
        "game.backend=tmnf", "game.steer_mode=pwm", f"game.map_dir={tmp_path.as_posix()}",
        f"data.resolution=[{W}, {H}]", "data.channels=3", "data.frame_hz=20", "data.control_hz=60",
        "eval.timeout_s=10",
    ])  # fmt: skip
    ref = ReferencePath(np.array([[0.0, 0.0, 0.0], [80.0, 0.0, 0.0]]))
    results = {}
    for b in (NEW_BUILD, OLD_BUILD):
        with FakePluginServer(build=b) as server:
            cfg.game.tmi_port = server.port
            game = make_game(server, tmp_path)
            try:
                res = run_episode_sync(game, GasPolicy(), "Alpha", cfg, 0, ref)
            finally:
                game.close()
            results[b] = (res["finished"], res["race_time_ms"], round(res["progress"], 6))
    assert results[NEW_BUILD] == results[OLD_BUILD]
    assert results[NEW_BUILD][0]  # finished: the comparison covers a whole run
