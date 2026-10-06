"""LineExpert and DAgger collection on the fake game."""

from __future__ import annotations

import numpy as np
import pytest

from tmagent.config import load_config
from tmagent.data.dataset import check_alignment
from tmagent.data.render import render_episode
from tmagent.eval.dagger import DaggerConfig, collect_episode
from tmagent.eval.expert import ExpertConfig, LineExpert
from tmagent.eval.progress import ProgressTracker, ReferencePath
from tmagent.game import make_game
from tmagent.game.fake import scripted_driver
from tmagent.interfaces import Action

MAP = "fake:s_curve"
FAKE = ExpertConfig(plane=(0, 1), right_turn_sign=-1.0)  # FakeGame: x/y plane, mirrored


@pytest.fixture
def setup():
    cfg = load_config("configs/fake.yaml", ["data.resolution=[32, 24]", "data.history_s=0.25"])
    game = make_game(cfg.game, cfg.data)
    tl, info = scripted_driver(MAP, cfg.data)
    ep = render_episode(game, tl, MAP, cfg.data, {})
    return cfg, game, ep, info


def drive_expert(cfg, game, ep, bad_ticks: int = 0) -> dict:
    expert = LineExpert(ep.positions, ep.actions, ep.speeds_kmh, FAKE)
    game.load_map(MAP)
    st = game.start_race()
    tracker = ProgressTracker(ReferencePath(ep.positions), cfg.eval)
    n = 0
    while not tracker.done:
        if n < bad_ticks:
            action = np.array([1.0, 1.0, 0.0], np.float32)  # full right: leave the line
        else:
            action = expert.act(st.position, st.velocity, st.speed_kmh)
        st = game.step(Action.from_array(action), 1)
        tracker.update(st)
        n += 1
    return tracker.result()


def test_expert_follows_the_reference(setup):
    cfg, game, ep, info = setup
    res = drive_expert(cfg, game, ep)
    assert res["finished"] and abs(res["race_time_ms"] - info["race_time_ms"]) < 500


def test_expert_recovers_after_leaving_the_line(setup):
    cfg, game, ep, _ = setup
    res = drive_expert(cfg, game, ep, bad_ticks=80)
    assert res["finished"] and res["max_dist"] > 2.0  # it was off the line and came back


def test_expert_copies_the_human_on_the_line(setup):
    _, _, ep, _ = setup
    expert = LineExpert(ep.positions, ep.actions, ep.speeds_kmh, FAKE)
    i = len(ep.positions) // 2
    expert.i = i - 5  # the projection searches a window around the last match
    vel = (ep.positions[i + 1] - ep.positions[i]) * 60.0
    out = expert.act(ep.positions[i], vel, float(ep.speeds_kmh[i]))
    assert np.array_equal(out, ep.actions[i])


class StillPolicy:
    def reset(self) -> None:
        pass

    def observe(self, image, past_actions) -> None:
        pass

    def predict(self) -> np.ndarray:
        return np.zeros((8, 3), np.float32)


def test_collect_episode_labels_with_the_expert_and_keeps_going(setup):
    cfg, game, ep, _ = setup
    expert = LineExpert(ep.positions, ep.actions, ep.speeds_kmh, FAKE)
    out, stats = collect_episode(
        game, StillPolicy(), expert, MAP, cfg, 0, ReferencePath(ep.positions), DaggerConfig()
    )
    check_alignment(out, cfg.data)
    assert stats["progress"] > 0.5  # expert blocks carry a do-nothing agent along the track
    assert 0.3 < stats["expert_share"] < 1.0
    assert out.actions[:, 1].max() == 1.0  # labels are expert actions, not the agent's zeros
    assert out.meta["frame_hz"] == cfg.data.frame_hz
