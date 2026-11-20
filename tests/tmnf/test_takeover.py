"""Takeover mode on the fake plugin: passthrough, interactive episodes, saved player segments."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from tmagent.config import load_config
from tmagent.data.episode_io import load_episode, read_index
from tmagent.data.player import keys_to_action
from tmagent.data.quality import check_episode, split_issues
from tmagent.eval.progress import ReferencePath
from tmagent.eval.takeover import (
    PlayerSegment,
    TakeoverControl,
    run_takeover_episode,
    save_segment,
)
from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.fake_server import FakePluginServer
from tmagent.game.tmnf.game import TMNFGame
from tmagent.interfaces import Action, GameState

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
        "eval.stuck_s=1.0", "eval.timeout_s=60",
    ])  # fmt: skip
    (tmp_path / "Alpha.Challenge.Gbx").write_bytes(b"GBX")
    return c


@pytest.fixture
def game(cfg):
    g = TMNFGame(cfg.game, cfg.data)
    yield g
    g.close()


class StillPolicy:
    """Never presses anything: the agent gets stuck."""

    def reset(self) -> None:
        self.observed = 0

    def observe(self, image, past_actions) -> None:
        self.observed += 1

    def predict(self) -> np.ndarray:
        return np.zeros((8, 3), np.float32)


def test_keys_to_action_left_wins():
    assert keys_to_action(True, True, True, False).tolist() == [-1.0, 1.0, 0.0]
    assert keys_to_action(False, True, False, True).tolist() == [1.0, 0.0, 1.0]


def test_passthrough_lets_the_player_drive_and_take_back_pauses(game, server):
    game.load_map("Alpha")
    game.start_race()
    server.player_input = P.InputCmd(accelerate=True)
    game.hand_to_player()
    assert server.passthrough and server.mode is P.Mode.REALTIME
    end = time.monotonic() + 3
    while game.get_state().speed_kmh < 5 and time.monotonic() < end:
        time.sleep(0.01)
    assert game.get_state().speed_kmh >= 5  # the plugin's own held input is neutral
    st = game.take_back()
    assert server.paused and not server.passthrough and st.race_time_ms > 0
    after = game.step(Action(), 3)
    assert after.race_time_ms == st.race_time_ms + 30


def test_passthrough_needs_realtime(game, server):
    game.load_map("Alpha")
    game.start_race()
    with pytest.raises(Exception, match="realtime"):
        game.client.set_passthrough(True)


def test_interactive_episode_waits_on_stuck_records_and_saves_the_player(game, server, cfg):
    control = TakeoverControl()
    phases: list[str] = []
    segments: list[PlayerSegment] = []
    t_player: list[float] = []

    def read_keys():
        if control.phase == "waiting" and not segments:
            server.player_input = P.InputCmd(accelerate=True)
            return (False, False, True, False, False)  # gas pressed -> takeover
        if control.phase == "player":
            if not t_player:
                t_player.append(time.monotonic())
            held = time.monotonic() - t_player[0] < 0.6
            server.player_input = P.InputCmd(accelerate=held)
            return (False, False, held, False, not held)  # then H hands back
        return (False, False, False, False, False)

    def on_frame(tracker):
        phases.append(control.phase)

    ref = ReferencePath(np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]))
    policy = StillPolicy()
    result = run_takeover_episode(
        game, policy, "Alpha", cfg, 0, ref, control, on_frame=on_frame,
        on_segment=segments.append, read_keys=read_keys,
    )  # fmt: skip
    # the agent got stuck, the player pushed it, then the car coasted over the fake finish
    assert result["finished"] and result["takeovers"] == 1 and result["player_ms"] > 1000
    assert phases[0] == "agent" and "waiting" in phases and "player" in phases
    assert phases[-1] == "agent"
    seg = segments[0]
    assert seg.duration_ms > 1000 and seg.frames and policy.observed > 0
    assert max(a[1][1] for a in seg.inputs) == 1.0  # gas recorded

    path = save_segment(seg, cfg, "FAKEUID_Alpha", "Alpha", "ckpt.pt")
    ep = load_episode(path)
    problems, _ = split_issues(check_episode(ep, cfg.data))
    assert problems == []
    assert ep.meta["source"].startswith("takeover:") and ep.meta["segment_start_ms"] > 0
    assert ep.frame_times_ms[0] == 0 and ep.actions[:, 1].max() == 1.0
    assert [e["episode_id"] for e in read_index(cfg.data.root)] == [ep.meta["episode_id"]]


def test_retry_resumes_the_agent_and_short_segments_are_not_saved(
    game, server, cfg, tmp_path: Path
):
    control = TakeoverControl()
    seen: list[str] = []

    class Done(Exception):
        pass

    def on_frame(tracker):
        seen.append(control.phase)
        if control.phase == "waiting" and seen.count("waiting") == 1:
            control.request("retry")
        elif control.phase == "waiting" and "agent" in seen[seen.index("waiting") :]:
            raise Done

    ref = ReferencePath(np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]))
    with pytest.raises(Done):
        run_takeover_episode(
            game, StillPolicy(), "Alpha", cfg, 0, ref, control, on_frame=on_frame,
            read_keys=lambda: (False,) * 5,
        )  # fmt: skip
    short = PlayerSegment(
        start_ms=0, inputs=[(0, np.zeros(3, np.float32), np.zeros(3, np.float32), 0.0)]
    )
    assert save_segment(short, cfg, "u", "n", "c") is None


def _segment(positions_by_ms: dict[int, float]) -> PlayerSegment:
    """Samples every 2 ms (like the 500 Hz poll), x position given per race-time range."""
    seg = PlayerSegment(start_ms=0)
    bounds = sorted(positions_by_ms)
    for t in range(0, 3000, 2):
        x = positions_by_ms[max(b for b in bounds if b <= t)] + t * 0.01
        st = GameState(t, np.array([x, 0.0, 0.0]), np.zeros(3), 36.0, 0, False)
        seg.add_state(st, np.zeros(3, np.float32))
    return seg


def test_a_respawn_cuts_the_segment_to_the_part_after_it():
    seg = _segment({0: 0.0, 1500: 300.0})
    seg.frames = [(t, np.zeros((H, W, 3), np.uint8)) for t in range(0, 3000, 50)]
    assert seg.respawns == 1
    after = seg.after_last_respawn()
    assert after.start_ms == 1500 and after.inputs[0][0] == 1500
    assert after.frames[0][0] == 1500 and after.respawns == 0


def test_glitches_and_unconfirmed_jumps_are_not_respawns():
    one_sample_glitch = _segment({0: 0.0, 1000: 500.0, 1002: 0.0})
    jump_at_the_end = _segment({0: 0.0, 2900: 300.0})
    for seg in (one_sample_glitch, jump_at_the_end):
        assert seg.respawns == 0 and seg.after_last_respawn() is seg


def test_a_game_restart_during_a_takeover_starts_a_new_segment(game, server, cfg):
    control = TakeoverControl()
    segments: list[PlayerSegment] = []
    t_player: list[float] = []
    resets = []

    def read_keys():
        if control.phase == "waiting" and not t_player:
            return (False, False, True, False, False)
        if control.phase == "player":
            if not t_player:
                t_player.append(time.monotonic())
            elapsed = time.monotonic() - t_player[0]
            if elapsed > 0.6 and not resets:
                with server._lock:
                    server._reset_race()  # what the game's restart key does to race time
                resets.append(elapsed)
            held = elapsed < 1.6
            server.player_input = P.InputCmd(accelerate=held)
            return (False, False, held, False, not held)
        return (False, False, False, False, False)

    class CountingPolicy(StillPolicy):
        resets = 0

        def reset(self) -> None:
            super().reset()
            CountingPolicy.resets += 1

    ref = ReferencePath(np.array([[0.0, 0.0, 0.0], [500.0, 0.0, 0.0]]))
    result = run_takeover_episode(
        game, CountingPolicy(), "Alpha", cfg, 0, ref, control,
        on_segment=segments.append, read_keys=read_keys,
    )  # fmt: skip
    assert result["finished"] and result["takeovers"] == 1 and CountingPolicy.resets == 2
    first, second = segments
    assert first.start_ms > 0 and first.duration_ms > 0
    assert second.start_ms < first.end_ms and second.duration_ms > 1000
    times = [t for t, *_ in second.inputs]
    assert times == sorted(times)

    ep = load_episode(save_segment(second, cfg, "FAKEUID_Alpha", "Alpha", "ckpt.pt"))
    assert split_issues(check_episode(ep, cfg.data))[0] == []
    assert ep.meta["segment_start_ms"] == second.start_ms and ep.meta["respawns"] == 0
