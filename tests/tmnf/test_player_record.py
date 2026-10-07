"""Recording sessions (tmagent/data/player.record_map) on the fake plugin."""

from __future__ import annotations

import time

import pytest

from tmagent.config import load_config
from tmagent.data.episode_io import load_episode
from tmagent.data.player import Requests, record_map, store_segment
from tmagent.data.quality import check_episode, split_issues
from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.fake_server import FakePluginServer
from tmagent.game.tmnf.game import TMNFGame

W, H = 16, 12


@pytest.fixture
def server():
    with FakePluginServer() as srv:
        yield srv


@pytest.fixture
def cfg(server, tmp_path):
    c = load_config(None, [
        "game.backend=tmnf", f"game.tmi_port={server.port}", "game.connect_timeout_s=3",
        "game.game_speed=2", "game.render_speed=10", f"game.map_dir={tmp_path.as_posix()}",
        f"data.root={(tmp_path / 'data').as_posix()}", f"data.resolution=[{W}, {H}]",
        "data.channels=3", "data.frame_hz=20", "data.control_hz=60", "data.history_s=0.25",
    ])  # fmt: skip
    (tmp_path / "Alpha.Challenge.Gbx").write_bytes(b"GBX")
    return c


@pytest.fixture
def game(cfg):
    g = TMNFGame(cfg.game, cfg.data)
    yield g
    g.close()


class Script:
    """Plays the user: per attempt, hold gas, then end it in a scripted way."""

    def __init__(self, server, control: Requests, plan: list[str]) -> None:
        self.server, self.control, self.plan = server, control, plan
        self.t0: float | None = None
        self.done = 0

    def read_keys(self):
        if self.control.phase == "ready":
            self.t0 = None
            if self.done >= len(self.plan):
                self.control.request("next")
                return (False,) * 5
            self.server.player_input = P.InputCmd(accelerate=True)
            return (False, False, True, False, False)
        if self.t0 is None:
            self.t0 = time.monotonic()
        elapsed = time.monotonic() - self.t0
        step = self.plan[self.done]
        if step == "restart_request" and elapsed > 0.7:
            self.control.request("restart")
        elif step == "game_restart" and elapsed > 0.7:
            with self.server._lock:
                self.server._reset_race()  # what the game's restart key does to race time
        return (False, False, True, False, False)

    def on_attempt(self, seg, end):
        self.done += 1


def test_record_map_ends_attempts_on_restart_finish_and_next(game, server, cfg):
    control = Requests("ready")
    plan = ["restart_request", "game_restart", "finish"]
    script = Script(server, control, plan)
    attempts = []

    def on_attempt(seg, end):
        attempts.append((seg, end))
        script.on_attempt(seg, end)

    n = record_map(game, "Alpha", control, on_attempt, read_keys=script.read_keys)
    assert n == 3
    assert [end for _, end in attempts] == ["restart", "restart", "finished"]
    first, _ = attempts[0]
    assert first.duration_ms > 1000 and first.frames and not first.finished
    last, _ = attempts[-1]
    assert last.finished and last.finish_ms is not None and last.finish_ms > 0

    path = store_segment(last, cfg, {
        "episode_id": "FAKEUID_Alpha-player-s1-003", "map_uid": "FAKEUID_Alpha",
        "map_name": "Alpha", "source": "player:s1", "attempt": 3,
    })  # fmt: skip
    ep = load_episode(path)
    assert split_issues(check_episode(ep, cfg.data))[0] == []
    assert ep.meta["finished"] and ep.meta["attempt"] == 3 and ep.meta["player"] == "human"
    assert ep.meta["finish_time_ms"] == last.finish_ms


def test_record_map_next_while_waiting_records_nothing(game, server):
    control = Requests("ready")
    control.request("next")
    n = record_map(game, "Alpha", control, lambda seg, end: None, read_keys=lambda: (False,) * 5)
    assert n == 0


def test_stop_during_an_attempt_still_hands_it_over(game, server):
    control = Requests("ready")
    seen = []

    class Stop(Exception):
        pass

    def on_tick(info):
        if info["phase"] == "driving" and info["race_time_ms"] > 500:
            raise Stop

    def keys():
        server.player_input = P.InputCmd(accelerate=True)
        return (False, False, True, False, False)

    with pytest.raises(Stop):
        record_map(game, "Alpha", control, lambda s, e: seen.append(e), on_tick, read_keys=keys)
    assert seen == ["stopped"]
