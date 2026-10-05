"""scripted_driver: the pure-pursuit expert used to generate fake data."""

from __future__ import annotations

import numpy as np
import pytest

from tmagent.config import DataConfig, GameConfig
from tmagent.game.fake import FakeGame, scripted_driver
from tmagent.interfaces import Action, InputTimeline

DC = DataConfig()
MAPS = ["fake:oval", "fake:s_curve", "fake:random:3"]


@pytest.mark.parametrize("map_ref", MAPS)
def test_finishes_each_track(map_ref):
    tl, info = scripted_driver(map_ref, DC)
    assert isinstance(tl, InputTimeline) and info["finished"]
    assert 10_000 < info["race_time_ms"] < 60_000
    assert tl.actions.dtype == np.float32 and tl.actions.shape == (info["race_time_ms"] // 10, 3)
    assert tl.meta["map_uid"] == map_ref
    assert tl.meta["player"] == "bot-0" and tl.meta["source"] == "fake:0"


def test_replaying_the_timeline_reproduces_the_run():
    tl, info = scripted_driver("fake:s_curve", DC)
    g = FakeGame(GameConfig(), DC)
    g.load_map("fake:s_curve")
    g.start_race()
    st = None
    for a in tl.actions:
        st = g.step(Action.from_array(a))
    assert st.finished and st.race_time_ms == info["race_time_ms"]


def test_reproducible_and_seeded():
    a, _ = scripted_driver("fake:oval", DC, seed=1, noise=0.3)
    b, _ = scripted_driver("fake:oval", DC, seed=1, noise=0.3)
    c, _ = scripted_driver("fake:oval", DC, seed=2, noise=0.3)
    np.testing.assert_array_equal(a.actions, b.actions)
    assert a.meta == b.meta and a.meta["player"] == "bot-1"
    assert a.actions.shape != c.actions.shape or not np.array_equal(a.actions, c.actions)
    d, _ = scripted_driver("fake:oval", DC, seed=7)  # no noise: the seed does not matter
    e, _ = scripted_driver("fake:oval", DC, seed=8)
    np.testing.assert_array_equal(d.actions, e.actions)


def test_keyboard_actions_are_discrete_and_still_finish():
    tl, info = scripted_driver("fake:random:3", DC, keyboard=True)
    assert info["finished"]
    steer, gas, brake = tl.actions.T
    assert set(np.unique(steer)) <= {-1.0, 0.0, 1.0} and len(np.unique(steer)) == 3
    assert set(np.unique(gas)) <= {0.0, 1.0} and set(np.unique(brake)) <= {0.0, 1.0}
    assert not np.any((gas == 1.0) & (brake == 1.0))
    assert 0 < brake.mean() < 0.5 and 0.2 < gas.mean() < 1.0


def test_analog_steer_and_binary_pedals():
    tl, _ = scripted_driver("fake:oval", DC)
    assert len(np.unique(tl.actions[:, 0])) > 20
    assert np.abs(tl.actions[:, 0]).max() <= 1.0
    assert set(np.unique(tl.actions[:, 1:])) <= {0.0, 1.0}


def test_noise_and_skill():
    base, bi = scripted_driver("fake:s_curve", DC)
    noisy, ni = scripted_driver("fake:s_curve", DC, seed=3, noise=0.3)
    assert ni["finished"] and not np.array_equal(
        base.actions[:, 0], noisy.actions[:, 0][: len(base.actions)]
    )
    slow, si = scripted_driver("fake:s_curve", DC, skill=0.7)
    assert si["finished"] and si["race_time_ms"] > 1.2 * bi["race_time_ms"]


def test_gives_up_at_max_s():
    tl, info = scripted_driver("fake:oval", DC, max_s=5.0)
    assert not info["finished"] and info["race_time_ms"] == 5000 and len(tl.actions) == 500


def test_hold_ticks():
    tl, info = scripted_driver("fake:s_curve", DC)  # default: decide every 100 ms
    blocks = tl.actions[: len(tl.actions) // 10 * 10].reshape(-1, 10, 3)
    assert np.all(blocks == blocks[:, :1])
    assert tl.meta["hold_ticks"] == 10
    fast, fi = scripted_driver("fake:s_curve", DC, hold_ticks=1)
    assert fi["finished"] and len(np.unique(fast.actions[:, 0])) > len(np.unique(tl.actions[:, 0]))
    kb, ki = scripted_driver("fake:s_curve", DC, hold_ticks=3, keyboard=True)
    assert ki["finished"] and set(np.unique(kb.actions[:, 0])) <= {-1.0, 0.0, 1.0}
    with pytest.raises(ValueError):
        scripted_driver("fake:s_curve", DC, hold_ticks=0)
