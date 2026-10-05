"""FakeGame: protocols, determinism, physics, rendering, checkpoints, realtime mode."""

from __future__ import annotations

import importlib.util
import threading
import time

import numpy as np
import pytest

from tmagent.config import DataConfig, GameConfig
from tmagent.game import make_game
from tmagent.game.fake import FakeGame, scripted_driver, track_reference
from tmagent.interfaces import Action, RealtimeGame, SyncGame

MAPS = ["fake:oval", "fake:s_curve", "fake:random:3"]
CYAN = np.array([0, 220, 255])


def new_game(channels: int = 3, resolution=(128, 96), **game_kw) -> FakeGame:
    return FakeGame(
        GameConfig(**game_kw), DataConfig(resolution=list(resolution), channels=channels)
    )


def test_protocols_and_factory():
    g = new_game()
    assert isinstance(g, SyncGame) and isinstance(g, RealtimeGame)
    assert isinstance(make_game(GameConfig(), DataConfig()), FakeGame)
    with pytest.raises(ValueError):
        make_game(GameConfig(backend="nope"), DataConfig())
    if importlib.util.find_spec("tmagent.game.tmnf") is None:
        with pytest.raises(ImportError):
            make_game(GameConfig(backend="tmnf"), DataConfig())


def test_unknown_maps():
    g = new_game()
    for bad in ("oval", "fake:nope", "fake:random:x", "fake:random", "tmx:123"):
        with pytest.raises(ValueError):
            g.load_map(bad)


def test_track_reference():
    oval = track_reference("fake:oval")
    assert oval.ndim == 2 and oval.shape[1] == 3 and np.all(oval[:, 2] == 0)
    np.testing.assert_allclose(oval[0], oval[-1], atol=1e-4)  # closed: start == finish
    s_curve = track_reference("fake:s_curve")
    assert np.linalg.norm(s_curve[-1] - s_curve[0]) > 50.0  # open track
    a, b = track_reference("fake:random:1"), track_reference("fake:random:1")
    np.testing.assert_array_equal(a, b)
    assert a.shape != track_reference("fake:random:2").shape or not np.allclose(
        a, track_reference("fake:random:2")
    )
    for ref in (oval, s_curve, a):
        seg = np.linalg.norm(np.diff(ref, axis=0), axis=1)
        assert 100.0 < seg.sum() < 3000.0 and seg.max() < 3.0


def test_start_state():
    g = new_game()
    g.load_map("fake:oval")
    st = g.start_race()
    ref = track_reference("fake:oval")
    assert st.race_time_ms == 0 and st.speed_kmh == 0.0 and not st.finished
    assert st.checkpoint == 0 and st.num_checkpoints == 5
    np.testing.assert_allclose(st.position, ref[0], atol=1e-3)
    assert st.position.dtype == np.float32 and st.velocity.shape == (3,)
    st = g.step(Action(), 100)  # idle at the start/finish line must not finish the race
    assert st.race_time_ms == 1000 and not st.finished and st.checkpoint == 0


def test_physics_basics():
    g = new_game()
    g.load_map("fake:s_curve")
    g.start_race()
    st = g.step(Action(0.0, 1.0, 0.0), 200)
    assert st.race_time_ms == 2000 and 20.0 < st.speed_kmh < 100.0 and st.extra["on_track"]
    fast = st.speed_kmh
    st = g.step(Action(0.0, 0.0, 0.0), 50)  # coasting slows down a bit
    assert 0.5 * fast < st.speed_kmh < fast
    st = g.step(Action(0.0, 0.0, 1.0), 300)  # braking stops the car (no reverse)
    assert st.speed_kmh == 0.0
    # Steering: positive = right = clockwise = yaw decreases; speed scales the turn rate.
    g.start_race()
    g.step(Action(0.0, 1.0, 0.0), 100)
    yaw0 = g.get_state().extra["yaw"]
    g.step(Action(1.0, 1.0, 0.0), 50)
    assert g.get_state().extra["yaw"] < yaw0 - 0.1
    g.start_race()
    g.step(Action(0.0, 1.0, 0.0), 100)
    yaw0 = g.get_state().extra["yaw"]
    g.step(Action(-1.0, 1.0, 0.0), 50)
    assert g.get_state().extra["yaw"] > yaw0 + 0.1
    # A stationary car cannot turn.
    g.start_race()
    g.step(Action(1.0, 0.0, 0.0), 100)
    assert g.get_state().extra["yaw"] == pytest.approx(g.track.start_yaw, abs=1e-9)


def test_off_track_limits_speed():
    g = new_game()
    g.load_map("fake:s_curve")  # straight into the first bend: leaves the road
    g.start_race()
    st = g.step(Action(0.0, 1.0, 0.0), 1200)
    assert not st.extra["on_track"] and st.extra["dist"] > 6.0
    assert st.speed_kmh <= 40.0 + 1e-6 and st.speed_kmh > 10.0


def drive(g: FakeGame, actions: np.ndarray, with_frames: bool = False):
    states, frames = [], []
    for i, a in enumerate(actions):
        states.append(g.step(Action.from_array(a)))
        if with_frames and i % 25 == 0:
            frames.append(g.grab_frame())
    return states, frames


def test_determinism():
    rng = np.random.default_rng(0)
    acts = np.stack(
        [rng.uniform(-1, 1, 400), rng.integers(0, 2, 400), rng.integers(0, 2, 400)], axis=1
    ).astype(np.float32)
    runs = []
    for _ in range(2):
        g = new_game()
        g.load_map("fake:random:5")
        g.start_race()
        runs.append(drive(g, acts, with_frames=True))
    (s1, f1), (s2, f2) = runs
    for a, b in zip(s1, s2, strict=True):
        assert a.race_time_ms == b.race_time_ms and a.speed_kmh == b.speed_kmh
        np.testing.assert_array_equal(a.position, b.position)
        np.testing.assert_array_equal(a.velocity, b.velocity)
    assert len(f1) == len(f2) > 5
    for a, b in zip(f1, f2, strict=True):
        np.testing.assert_array_equal(a.image, b.image)
        assert a.race_time_ms == b.race_time_ms


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("resolution", [(128, 96), (64, 48), (224, 224)])
def test_frame_shape(channels, resolution):
    g = new_game(channels, resolution)
    g.load_map("fake:oval")
    g.start_race()
    g.step(Action(0.0, 1.0, 0.0), 120)
    fr = g.grab_frame()
    assert fr.image.shape == (resolution[1], resolution[0], channels)
    assert fr.image.dtype == np.uint8
    assert fr.race_time_ms == 1200
    assert abs(fr.wall_time - time.perf_counter()) < 5.0
    assert len(np.unique(fr.image.reshape(-1, channels), axis=0)) >= 4


def test_frame_content_rgb():
    g = new_game()
    g.load_map("fake:oval")
    g.start_race()
    img = g.grab_frame().image
    h, w, _ = img.shape
    grass, asphalt = np.array([40, 120, 45]), np.array([110, 110, 118])
    assert np.all(img[h // 3, 0] == grass)  # far left is off the road
    # Heading up: the road runs up the middle of the view, the car sits at the bottom center.
    assert np.all(img[h // 3, w // 2] == asphalt)
    assert not np.all(img[h - 12 : h - 6, w // 2] == asphalt)
    assert not (img[0] == CYAN).all(axis=1).any()  # speed bar is empty at standstill
    # The view follows the heading: a turned car sees different pixels than a straight one.
    g.step(Action(0.0, 1.0, 0.0), 100)
    straight = g.grab_frame().image
    g.step(Action(1.0, 1.0, 0.0), 60)
    turned = g.grab_frame().image
    assert not np.array_equal(straight, turned)
    # Speed bar grows with speed.
    assert (straight[0] == CYAN).all(axis=1).sum() > 0


def test_render_speed():
    g = new_game()
    g.load_map("fake:random:2")
    g.start_race()
    g.step(Action(0.0, 1.0, 0.0), 100)
    g.grab_frame()
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        g.grab_frame()
        ts.append(time.perf_counter() - t0)
    assert float(np.median(ts)) < 0.004  # target < 2 ms; tolerant for shared CI


def test_checkpoints_and_finish():
    for m in MAPS:
        tl, info = scripted_driver(m, DataConfig())
        assert info["finished"]
        g = new_game()
        g.load_map(m)
        g.start_race()
        seen, st = [0], None
        for a in tl.actions:
            st = g.step(Action.from_array(a))
            if st.checkpoint != seen[-1]:
                seen.append(st.checkpoint)
        assert seen == [0, 1, 2, 3, 4, 5], m
        assert st.finished and st.race_time_ms == info["race_time_ms"]
        before = st
        after = g.step(Action(0.0, 1.0, 0.0), 50)  # the race is frozen after the finish
        assert after.race_time_ms == before.race_time_ms
        np.testing.assert_array_equal(after.position, before.position)
        assert g.grab_frame().race_time_ms == before.race_time_ms


def test_mode_exclusion():
    g = new_game()
    g.load_map("fake:oval")
    with pytest.raises(RuntimeError):
        g.step(Action())  # needs start_race() first
    with pytest.raises(RuntimeError):
        g.grab_frame()
    g.start_race()
    for call in (g.restart, lambda: g.set_action(Action()), g.latest_frame):
        with pytest.raises(RuntimeError):
            call()
    g.get_state()  # read-only, allowed
    g.load_map("fake:oval")  # load_map resets the mode
    g.restart()
    try:
        for call in (g.start_race, lambda: g.step(Action()), g.grab_frame):
            with pytest.raises(RuntimeError):
                call()
    finally:
        g.close()
    g.start_race()  # usable in sync mode again after close()


def test_realtime_clock_and_close():
    n_threads = threading.active_count()
    g = new_game(game_speed=1.0)
    g.load_map("fake:oval")
    g.restart()
    assert threading.active_count() == n_threads + 2
    f0 = g.latest_frame()
    assert f0 is not None and f0.image.shape == (96, 128, 3)
    t0 = time.perf_counter()
    time.sleep(0.5)
    st = g.get_state()
    wall_ms = (time.perf_counter() - t0) * 1000
    assert 0.5 * wall_ms < st.race_time_ms <= wall_ms + 150  # tolerant: loaded CI machines
    assert st.speed_kmh == 0.0  # no action set
    st.position[:] = 123.0  # get_state returns a copy
    assert g.get_state().position[0] != 123.0
    g.set_action(Action(0.0, 1.0, 0.0))
    time.sleep(0.4)
    assert g.get_state().speed_kmh > 5.0
    fr = g.latest_frame()
    assert fr is not None and fr.race_time_ms > f0.race_time_ms and fr.wall_time > f0.wall_time
    g.close()
    assert threading.active_count() == n_threads
    t_closed = g.get_state().race_time_ms
    time.sleep(0.1)
    assert g.get_state().race_time_ms == t_closed  # clock stopped
    g.close()  # idempotent


def test_realtime_game_speed_and_restart():
    g = new_game(game_speed=5.0)
    g.load_map("fake:oval")
    g.restart()
    try:
        time.sleep(0.4)
        st = g.get_state()
        assert 800 < st.race_time_ms < 4500  # ~2000 ms at 5x
        g.restart()  # restarting resets the race
        assert g.get_state().race_time_ms < st.race_time_ms
        fps_frames = set()
        t_end = time.perf_counter() + 0.3
        while time.perf_counter() < t_end:
            fr = g.latest_frame()
            fps_frames.add(fr.wall_time)
            time.sleep(0.002)
        assert len(fps_frames) >= 5  # the capture thread keeps producing frames
    finally:
        g.close()
