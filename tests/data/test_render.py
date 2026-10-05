from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tmagent.data.render import conform_frame, render_episode, resize_area
from tmagent.data.timeline import control_times_ms, frame_times_ms
from tmagent.interfaces import EPISODE_META_KEYS

from .test_helpers import StubGame, decode_tick, make_timeline, small_cfg


def expected_positions(tl, ticks: np.ndarray) -> np.ndarray:
    """x at the START of each tick (before its action), per StubGame dynamics."""
    a = tl.actions.astype(np.float64)
    step = a[:, 1] + 0.25 * a[:, 0]
    x = np.concatenate([[0.0], np.cumsum(step, dtype=np.float64)])
    return x[ticks].astype(np.float32)


@pytest.mark.parametrize("frame_hz,control_hz", [(20, 60), (30, 60), (25, 50), (20, 100), (10, 60)])
def test_frames_and_actions_are_in_sync(frame_hz, control_hz):
    cfg = small_cfg(frame_hz=frame_hz, control_hz=control_hz)
    r = cfg.actions_per_frame
    tl = make_timeline(250)  # 2.5 s
    game = StubGame(size=(32, 24))
    ep = render_episode(game, tl, "maps/A01.Challenge.Gbx", cfg, {"map_uid": "A01"})

    t_f, t_a = len(ep.frames), len(ep.actions)
    assert game.map_ref == "maps/A01.Challenge.Gbx" and game.loaded == 1
    # exact formula grids, covering [0, 2500)
    assert np.array_equal(ep.frame_times_ms, frame_times_ms(t_f, frame_hz))
    assert np.array_equal(ep.action_times_ms, control_times_ms(t_a, control_hz))
    assert ep.frame_times_ms[-1] < 2500 <= frame_times_ms(t_f + 1, frame_hz)[-1]
    assert ep.action_times_ms[-1] < 2500 <= control_times_ms(t_a + 1, control_hz)[-1]
    assert t_a - r <= t_f * r <= t_a + r
    # frame_times[fi] == action_times[fi * R]
    assert np.array_equal(ep.frame_times_ms, ep.action_times_ms[::r][:t_f])
    # dtypes / shapes
    assert ep.frames.dtype == np.uint8 and ep.frames.shape == (t_f, 24, 32, 3)
    assert ep.actions.dtype == np.float32 and ep.positions.shape == (t_a, 3)
    assert ep.speeds_kmh.shape == (t_a,)
    # each frame shows the state of the latest tick <= its time
    for fi in range(t_f):
        assert decode_tick(ep.frames[fi]) == ep.frame_times_ms[fi] // 10
    # actions are sample-and-hold of the timeline; positions are recorded before the step
    ticks = ep.action_times_ms // 10
    assert np.array_equal(ep.actions, tl.actions[ticks])
    assert np.array_equal(ep.positions[:, 0], expected_positions(tl, ticks))
    assert np.all(ep.positions[:, 1:] == 0)
    # the frame at t and the action stamped t come from the same tick
    for fi in range(t_f):
        assert np.array_equal(ep.actions[fi * r], tl.actions[decode_tick(ep.frames[fi])])


def test_speed_is_state_speed_before_the_action():
    cfg = small_cfg()
    tl = make_timeline(40)
    ep = render_episode(StubGame(), tl, "m", cfg, {})
    ticks = ep.action_times_ms // 10
    a = tl.actions.astype(np.float64)
    dx = a[:, 1] + 0.25 * a[:, 0]
    want = np.concatenate([[0.0], dx])[ticks] * 36.0  # speed after tick i-1
    assert np.allclose(ep.speeds_kmh, want, atol=1e-4)


def test_stops_stop_after_finish_ms_after_first_finish():
    cfg = small_cfg()
    tl = make_timeline(600)
    probe = StubGame(finish_x=60.0)
    full = render_episode(probe, tl, "m", cfg, {"map_uid": "m"}, stop_after_finish_ms=10**9)
    finish_ms = full.meta["finish_time_ms"]
    assert finish_ms is not None and finish_ms < 6000 - 600

    ep = render_episode(StubGame(finish_x=60.0), tl, "m", cfg, {"map_uid": "m"})
    assert ep.meta["finished"] is True and ep.meta["finish_time_ms"] == finish_ms
    assert ep.meta["race_time_ms"] == finish_ms
    end = finish_ms + 500
    assert ep.action_times_ms[-1] < end <= control_times_ms(len(ep.actions) + 1, 60)[-1]
    assert ep.frame_times_ms[-1] < end <= frame_times_ms(len(ep.frames) + 1, 20)[-1]
    # a prefix of the unbounded run
    n = len(ep.actions)
    assert np.array_equal(ep.actions, full.actions[:n])
    assert np.array_equal(ep.frames, full.frames[: len(ep.frames)])

    ep0 = render_episode(StubGame(finish_x=60.0), tl, "m", cfg, {}, stop_after_finish_ms=0)
    assert ep0.action_times_ms[-1] < finish_ms <= control_times_ms(len(ep0.actions) + 1, 60)[-1]


def test_frame_time_mismatch_is_zero_for_a_synchronous_game():
    for hz in (20, 30):
        ep = render_episode(StubGame(), make_timeline(100), "m", small_cfg(frame_hz=hz), {})
        assert ep.meta["frame_time_mismatch"] == 0


def test_lagging_frames_are_counted_as_mismatches():
    cfg = small_cfg()
    ep = render_episode(StubGame(lag_ticks=1), make_timeline(100), "m", cfg, {})
    # frame 0 (tick 0) is still correct; every later frame shows the previous tick
    assert ep.meta["frame_time_mismatch"] == len(ep.frames) - 1
    assert decode_tick(ep.frames[4]) == ep.frame_times_ms[4] // 10 - 1
    ep = render_episode(StubGame(lag_ticks=3), make_timeline(100), "m", cfg, {})
    assert ep.meta["frame_time_mismatch"] == len(ep.frames) - 1  # frame 0 still matches at tick 0


def test_lag_with_30hz_frames_counts_only_frames_that_are_off():
    cfg = small_cfg(frame_hz=30)
    ep = render_episode(StubGame(lag_ticks=1), make_timeline(100), "m", cfg, {})
    assert ep.meta["frame_time_mismatch"] == len(ep.frames) - 1


def test_unknown_frame_time_is_not_a_mismatch():
    game = StubGame(lag_ticks=1, report_time=False)  # race_time_ms == -1
    ep = render_episode(game, make_timeline(100), "m", small_cfg(), {})
    assert ep.meta["frame_time_mismatch"] == 0


def test_unfinished_run_uses_last_state_time():
    cfg = small_cfg()
    ep = render_episode(StubGame(), make_timeline(100), "m", cfg, {})
    assert ep.meta["finished"] is False and ep.meta["finish_time_ms"] is None
    assert ep.meta["race_time_ms"] == 1000
    assert ep.meta["desync"] is False


@pytest.mark.parametrize(
    "expected,desync",
    [(None, False), ("exact", False), (-10, False), (10, False), (-20, True), (20, True)],
)
def test_desync_detection(expected, desync):
    cfg = small_cfg()
    tl = make_timeline(600)
    finish = render_episode(StubGame(finish_x=60.0), tl, "m", cfg, {}).meta["finish_time_ms"]
    if expected == "exact":
        expected = 0
    exp = None if expected is None else finish + expected
    ep = render_episode(StubGame(finish_x=60.0), tl, "m", cfg, {}, expected_time_ms=exp)
    assert ep.meta["desync"] is desync


def test_desync_when_expected_finish_but_run_does_not_finish():
    ep = render_episode(
        StubGame(finish_x=10**6), make_timeline(100), "m", small_cfg(), {}, expected_time_ms=900
    )
    assert ep.meta["desync"] is True and ep.meta["finished"] is False


def test_meta_filled_from_meta_timeline_and_cfg():
    cfg = small_cfg(channels=3)
    tl = make_timeline(50, meta={"map_uid": "UID123", "player": "p1", "source": "tmx:99"})
    ep = render_episode(StubGame(), tl, "path/to/Map.Challenge.Gbx", cfg, {"map_name": "My Map"})
    m = ep.meta
    assert all(k in m for k in EPISODE_META_KEYS)
    assert m["map_uid"] == "UID123" and m["player"] == "p1" and m["source"] == "tmx:99"
    assert m["map_name"] == "My Map"
    assert m["episode_id"] == "UID123-" + hashlib.sha1(b"tmx:99").hexdigest()[:10]
    assert (m["frame_hz"], m["control_hz"], m["channels"]) == (20, 60, 3)
    assert m["resolution"] == [32, 24]
    assert m["resized_frames"] == 0


def test_meta_overrides_and_defaults():
    cfg = small_cfg()
    ep = render_episode(
        StubGame(),
        make_timeline(20),
        "dir/Some.Challenge.Gbx",
        cfg,
        {"episode_id": "custom", "source": "s", "frame_hz": 999},
    )
    assert ep.meta["episode_id"] == "custom"
    assert ep.meta["map_uid"] == "Some.Challenge.Gbx"  # falls back to the map file name
    assert ep.meta["frame_hz"] == 20  # frames really are at cfg.frame_hz
    assert ep.meta["player"] == ""


def test_empty_timeline_gives_empty_episode():
    cfg = small_cfg()
    ep = render_episode(StubGame(), make_timeline(0), "m", cfg, {})
    assert ep.frames.shape == (0, 24, 32, 3) and ep.actions.shape == (0, 3)
    assert ep.frame_times_ms.shape == (0,) and ep.positions.shape == (0, 3)


def test_gray_game_frames_to_rgb_config():
    cfg = small_cfg(channels=3)
    ep = render_episode(StubGame(channels=1), make_timeline(30), "m", cfg, {})
    assert ep.frames.shape[-1] == 3
    assert np.array_equal(ep.frames[..., 0], ep.frames[..., 1])
    assert np.array_equal(ep.frames[..., 1], ep.frames[..., 2])
    assert ep.meta["resized_frames"] == 0


def test_rgb_game_frames_to_gray_config_use_luma():
    cfg = small_cfg(channels=1)
    ep = render_episode(StubGame(channels=3), make_timeline(30), "m", cfg, {})
    assert ep.frames.shape[-1] == 1 and ep.meta["channels"] == 1
    for fi, t in enumerate(ep.frame_times_ms // 10):
        r, g, b = t % 256, t // 256, (7 * t) % 256
        want = int(np.clip(np.rint(0.299 * r + 0.587 * g + 0.114 * b), 0, 255))
        assert np.all(np.abs(ep.frames[fi].astype(int) - want) <= 1)


def test_wrong_size_frames_are_resized_and_counted():
    cfg = small_cfg()  # wants 32x24
    ep = render_episode(StubGame(size=(64, 48)), make_timeline(100), "m", cfg, {})
    assert ep.frames.shape[1:] == (24, 32, 3)
    assert ep.meta["resized_frames"] == len(ep.frames)
    # constant-colour frames keep their colour, the red channel still encodes the tick
    assert decode_tick(ep.frames[3]) == ep.frame_times_ms[3] // 10
    ep = render_episode(StubGame(size=(31, 17)), make_timeline(20), "m", cfg, {})
    assert ep.frames.shape[1:] == (24, 32, 3)


def test_resize_area_is_block_mean_for_integer_factors():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(8, 12, 3), dtype=np.uint8)
    out = resize_area(img, 6, 4)
    blocks = img.reshape(4, 2, 6, 2, 3).astype(np.float64).mean(axis=(1, 3))
    assert out.shape == (4, 6, 3) and out.dtype == np.uint8
    assert np.all(np.abs(out.astype(float) - blocks) <= 0.5 + 1e-6)
    # identity size and constant upscaling
    assert np.array_equal(resize_area(img, 12, 8), img)
    const = np.full((3, 5, 1), 77, dtype=np.uint8)
    assert np.all(resize_area(const, 11, 7) == 77)


def test_conform_frame_rejects_bad_input():
    cfg = small_cfg()
    with pytest.raises(ValueError, match="uint8"):
        conform_frame(np.zeros((24, 32, 3), dtype=np.float32), cfg)
    with pytest.raises(ValueError):
        conform_frame(np.zeros((24, 32, 2), dtype=np.uint8), cfg)
    img, resized = conform_frame(np.zeros((24, 32, 4), dtype=np.uint8), cfg)  # RGBA -> RGB
    assert img.shape == (24, 32, 3) and not resized
    img, _ = conform_frame(np.zeros((24, 32), dtype=np.uint8), cfg)  # 2D gray -> RGB
    assert img.shape == (24, 32, 3)
