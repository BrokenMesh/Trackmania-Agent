from __future__ import annotations

import numpy as np
import pytest

from tmagent.data.timeline import (
    control_times_ms,
    frame_times_ms,
    grid_len,
    resample_loss,
    resample_to_control,
    timeline_from_events,
)
from tmagent.interfaces import InputTimeline


def tl(events, duration_ms=None):
    return timeline_from_events(events, duration_ms=duration_ms)


def test_tick_rule_multiples_of_ten():
    t = tl([(0, "accelerate", 1), (30, "brake", 1), (50, "brake", 0)], duration_ms=80)
    assert t.actions.shape == (8, 3) and t.actions.dtype == np.float32
    assert t.actions[:, 1].tolist() == [1] * 8
    assert t.actions[:, 2].tolist() == [0, 0, 0, 1, 1, 0, 0, 0]


def test_tick_rule_off_grid_events_apply_to_next_tick():
    # t=15 is not a tick start: it takes effect at the first tick starting at/after it.
    t = tl([(15, "accelerate", 1)], duration_ms=50)
    assert t.actions[:, 1].tolist() == [0, 0, 1, 1, 1]
    t = tl([(1, "accelerate", 1)], duration_ms=30)
    assert t.actions[:, 1].tolist() == [0, 1, 1]
    t = tl([(10, "accelerate", 1)], duration_ms=30)
    assert t.actions[:, 1].tolist() == [0, 1, 1]


def test_off_grid_events_in_same_tick_last_one_wins():
    t = tl([(11, "accelerate", 1), (19, "accelerate", 0)], duration_ms=40)
    assert t.actions[:, 1].tolist() == [0, 0, 0, 0]  # both land on tick 2, the later one wins
    t = tl([(11, "accelerate", 0), (19, "accelerate", 1)], duration_ms=40)
    assert t.actions[:, 1].tolist() == [0, 0, 1, 1]


def test_negative_times_clamped_and_unsorted_input():
    t = tl([(50, "brake", 0), (-100, "brake", 1), (0, "accelerate", 1)], duration_ms=70)
    assert t.actions[:, 2].tolist() == [1, 1, 1, 1, 1, 0, 0]
    assert t.actions[0, 1] == 1


def test_number_of_ticks():
    ev = [(0, "accelerate", 1), (95, "brake", 1)]
    assert len(tl(ev, duration_ms=1000).actions) == 100
    assert len(tl(ev, duration_ms=1005).actions) == 100
    assert len(tl(ev).actions) == 95 // 10 + 1
    assert len(tl([(100, "brake", 1)]).actions) == 11
    # no events and no duration -> empty timeline
    assert tl([]).actions.shape == (0, 3)
    assert tl([], duration_ms=40).actions.shape == (4, 3)
    assert not tl([], duration_ms=40).actions.any()


def test_events_past_the_end_are_dropped():
    t = tl([(0, "accelerate", 1), (500, "brake", 1)], duration_ms=100)
    assert t.actions.shape == (10, 3) and t.actions[:, 2].sum() == 0


def test_steer_binary_is_right_minus_left():
    t = tl(
        [
            (0, "steer_left", 1),
            (20, "steer_left", 0),
            (20, "steer_right", 1),
            (40, "steer_left", 1),
            (60, "steer_right", 0),
        ],
        duration_ms=80,
    )
    assert t.actions[:, 0].tolist() == [-1, -1, 1, 1, 0, 0, -1, -1]


def test_steer_last_writer_wins_between_analog_and_binary():
    t = tl(
        [
            (0, "steer", 0.5),
            (100, "steer_left", 1),  # binary written last -> -1
            (200, "steer", -0.25),  # analog written last -> -0.25
            (300, "steer_right", 1),  # binary again: right - left = 1 - 1 = 0
            (400, "steer_left", 0),  # binary: +1
            (500, "steer", 2.0),  # analog, clipped to 1
            (600, "steer", -3.0),  # clipped to -1
        ],
        duration_ms=700,
    )
    s = t.actions[::10, 0]
    assert s.tolist() == pytest.approx([0.5, -1, -0.25, 0, 1, 1, -1])


def test_binary_state_persists_after_analog_overrides():
    t = tl([(0, "steer_left", 1), (50, "steer", 0.3), (100, "steer_right", 0)], duration_ms=150)
    # after the analog write, a binary event (even a no-op release) makes binary the writer again
    assert t.actions[0, 0] == -1 and t.actions[5, 0] == pytest.approx(0.3)
    assert t.actions[10, 0] == -1


def test_gas_is_max_of_accelerate_and_analog_gas():
    t = tl(
        [(0, "accelerate", 1), (0, "gas", 0.3), (100, "accelerate", 0), (200, "gas", 0.8)],
        duration_ms=300,
    )
    assert t.actions[::10, 1].tolist() == pytest.approx([1, 0.3, 0.8])
    assert tl([(0, "gas", 5.0)], duration_ms=10).actions[0, 1] == 1.0  # clipped


def test_brake_binary_threshold():
    t = tl([(0, "brake", 0.4), (10, "brake", 0.6)], duration_ms=30)
    assert t.actions[:, 2].tolist() == [0, 1, 1]


def test_unknown_names_raise_and_underscore_names_are_ignored():
    with pytest.raises(ValueError, match="unknown"):
        tl([(0, "horn", 1)])
    with pytest.raises(ValueError):
        tl([(0, "steer", float("nan"))])
    t = tl([(0, "accelerate", 1), (5000, "_marker", 1)])
    assert len(t.actions) == 1  # ignored events do not extend the timeline
    assert t.actions[0, 1] == 1


def test_meta_is_copied():
    meta = {"map_uid": "m"}
    t = timeline_from_events([(0, "brake", 1)], meta=meta)
    assert t.meta == meta and t.meta is not meta
    assert isinstance(t, InputTimeline)


def test_grid_formulas():
    assert control_times_ms(7, 60).tolist() == [0, 17, 33, 50, 67, 83, 100]
    assert frame_times_ms(4, 20).tolist() == [0, 50, 100, 150]
    for hz in (20, 25, 30, 50, 60, 100):
        got = control_times_ms(500, hz)
        assert got.dtype == np.int64
        assert got.tolist() == [round(i * 1000 / hz) for i in range(500)]
        assert np.all(np.diff(got) > 0)
    assert control_times_ms(0, 60).shape == (0,)
    with pytest.raises(ValueError):
        control_times_ms(3, 0)


def test_grid_len_counts_points_before_total():
    for hz in (20, 30, 50, 60):
        for total in (0, 10, 20, 990, 1000, 1234):
            expected = sum(1 for i in range(2000) if round(i * 1000 / hz) < total)
            assert grid_len(total, hz) == expected


def _timeline(n):
    rng = np.random.default_rng(0)
    return InputTimeline(actions=rng.uniform(-1, 1, size=(n, 3)).astype(np.float32))


def test_resample_60hz_sample_and_hold():
    t = _timeline(12)  # 120 ms
    actions, times = resample_to_control(t, 60)
    assert times.tolist() == [0, 17, 33, 50, 67, 83, 100, 117]
    assert actions.dtype == np.float32 and actions.shape == (8, 3) and times.dtype == np.int64
    for a, ms in zip(actions, times, strict=True):
        assert np.array_equal(a, t.actions[ms // 10])
    # tick pattern 10/20/20 ms: ticks 0,1,3,5,6,8,10,11
    assert (times // 10).tolist() == [0, 1, 3, 5, 6, 8, 10, 11]


def test_resample_50hz_aligns_with_every_second_tick():
    t = _timeline(10)
    actions, times = resample_to_control(t, 50)
    assert times.tolist() == [0, 20, 40, 60, 80]
    assert np.array_equal(actions, t.actions[::2])


def test_resample_covers_all_control_times_below_duration():
    for n in (1, 2, 17, 100, 333):
        for hz in (50, 60):
            actions, times = resample_to_control(_timeline(n), hz)
            assert times[-1] < n * 10
            assert control_times_ms(len(times) + 1, hz)[-1] >= n * 10
            assert len(actions) == len(times)
    assert len(resample_to_control(_timeline(100), 60)[1]) == 60  # 1 s -> 60 steps


def test_resample_empty_timeline():
    actions, times = resample_to_control(InputTimeline(actions=np.zeros((0, 3), np.float32)), 60)
    assert actions.shape == (0, 3) and times.shape == (0,)


def test_resample_loss_constant_and_held_inputs_are_lossless():
    # 100 ms holds starting on multiples of 100 ms survive 60 Hz sampling exactly
    acts = np.zeros((100, 3), np.float32)
    acts[:, 1] = 1
    acts[:10, 0], acts[10:20, 0], acts[50:60, 0] = 1, -1, 1
    acts[30:70, 2] = 1
    out = resample_loss(InputTimeline(acts), 60)
    assert out["ticks"] == 100 and out["mismatch_frac"] == 0.0
    assert out["lost_changes"] == 0 and out["lost_changes_frac"] == 0.0 and out["changes"] > 0
    out = resample_loss(tl([], duration_ms=500), 60)  # no inputs at all
    assert out["mismatch_frac"] == 0.0 and out["changes"] == 0 and out["lost_changes_frac"] == 0.0


def test_resample_loss_at_the_physics_rate_is_lossless():
    acts = np.zeros((40, 3), np.float32)
    acts[7, 0] = acts[8, 1] = acts[13, 2] = acts[14, 0] = 1  # single-tick taps
    out = resample_loss(InputTimeline(acts), 100)
    assert out["mismatch_frac"] == 0.0 and out["lost_changes"] == 0 and out["changes"] == 8


def test_resample_loss_short_tap_between_samples_is_lost():
    # 50 Hz labels sit on even ticks: a one-tick brake tap on tick 21 is never sampled
    acts = np.zeros((100, 3), np.float32)
    acts[21, 2] = 1
    out = resample_loss(InputTimeline(acts), 50)
    assert out["changes"] == 2 and out["lost_changes"] == 2 and out["lost_changes_frac"] == 1.0
    assert out["mismatch_frac"] == pytest.approx(0.01)  # only tick 21 itself differs
    # a tap on an even tick is sampled; the hold then extends it over the next tick
    acts = np.zeros((100, 3), np.float32)
    acts[20, 2] = 1
    out = resample_loss(InputTimeline(acts), 50)
    assert out["lost_changes"] == 0 and out["mismatch_frac"] == pytest.approx(0.01)


def test_resample_loss_counts_changes_per_channel():
    # steer and brake both change on tick 21 and revert on tick 22: 4 changes, none visible
    acts = np.zeros((100, 3), np.float32)
    acts[21, 0] = acts[21, 2] = 1
    out = resample_loss(InputTimeline(acts), 50)
    assert out["changes"] == 4 and out["lost_changes"] == 4
    # simultaneous changes of two channels that persist are both visible at the next label
    acts = np.zeros((100, 3), np.float32)
    acts[21:, 0] = acts[21:, 1] = 1
    out = resample_loss(InputTimeline(acts), 50)
    assert out["changes"] == 2 and out["lost_changes"] == 0
    assert out["mismatch_frac"] == pytest.approx(0.01)  # only tick 21 (label ticks are even)


def test_resample_loss_merges_changes_between_two_labels():
    # 25 Hz labels sit on ticks 0, 4, 8, ...; steer goes 0 -> 1 -> -1 on ticks 21 and 22,
    # the labels at 20 and 24 show 0 and -1: 2 changes, 1 visible
    acts = np.zeros((100, 3), np.float32)
    acts[21, 0], acts[22:, 0] = 1, -1
    out = resample_loss(InputTimeline(acts), 25)
    assert out["changes"] == 2 and out["lost_changes"] == 1
    assert out["lost_changes_frac"] == pytest.approx(0.5)
    assert out["mismatch_frac"] == pytest.approx(0.03)  # ticks 21, 22, 23 hold the label at 20
