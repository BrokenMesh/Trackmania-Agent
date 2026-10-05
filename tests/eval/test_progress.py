"""ReferencePath projection, ProgressTracker end conditions, load_reference."""

from __future__ import annotations

import numpy as np
import pytest

from tmagent.config import EvalConfig
from tmagent.eval.progress import ProgressTracker, ReferencePath, load_reference
from tmagent.game.fake import track_reference
from tmagent.interfaces import GameState


def hairpin() -> ReferencePath:
    """Out along +x for 200 m, a 5 m radius U-turn, back along -x 10 m to the side."""
    out = [(x, 0.0) for x in np.arange(0.0, 200.0, 2.0)]
    a = np.linspace(-np.pi / 2, np.pi / 2, 12)
    turn = [(200 + 5 * np.cos(t), 5 + 5 * np.sin(t)) for t in a]
    back = [(x, 10.0) for x in np.arange(198.0, -1.0, -2.0)]
    pts = np.array(out + turn + back)
    return ReferencePath(np.column_stack([pts, np.zeros(len(pts))]))


def state(x, y=0.0, t=0, speed=100.0, finished=False) -> GameState:
    return GameState(
        race_time_ms=t,
        position=np.array([x, y, 0.0], np.float32),
        velocity=np.zeros(3, np.float32),
        speed_kmh=speed,
        checkpoint=0,
        finished=finished,
    )


def test_straight_projection():
    ref = ReferencePath(np.array([[0, 0, 0], [100, 0, 0], [200, 0, 0]], float))
    assert ref.length == pytest.approx(200.0)
    s, d = ref.project([50.0, 3.0, 0.0])
    assert s == pytest.approx(50.0) and d == pytest.approx(3.0)
    s, d = ref.project([-10.0, 0.0, 0.0])  # before the start: clamped
    assert s == 0.0 and d == pytest.approx(10.0)
    s, d = ref.project([250.0, 0.0, 4.0])  # beyond the end, and above the path
    assert s == pytest.approx(200.0) and d == pytest.approx(np.hypot(50.0, 4.0))
    assert ref.progress(-5) == 0.0 and ref.progress(50.0) == 0.25 and ref.progress(1e9) == 1.0


def test_window_prevents_jumping_across_a_hairpin():
    ref = hairpin()
    pos = [100.0, 6.2, 0.0]  # on the outbound leg but closer to the return leg
    s_free, d_free = ref.project(pos)
    assert s_free > 300.0 and d_free == pytest.approx(3.8)  # no hint: jumps to the return leg
    s, d = ref.project(pos, hint_s=98.0)
    assert s == pytest.approx(100.0) and d == pytest.approx(6.2)
    # Following the car along the outbound leg with the running hint never jumps
    # (while the return leg is more than `window` of arc length ahead).
    s_prev = 0.0
    for x in np.arange(0.0, 90.0, 0.5):
        s_prev, _ = ref.project([x, 6.2, 0.0], hint_s=s_prev, window=100.0)
        assert s_prev == pytest.approx(x)
    # Through the U-turn and onto the return leg.
    for x in np.arange(200.0, 0.0, -0.5):
        s_prev, d = ref.project([x, 9.0, 0.0], hint_s=s_prev, window=100.0)
    assert s_prev > ref.length - 5.0
    # The tracker does the same through its own window.
    tr = ProgressTracker(ref, EvalConfig())
    for i, x in enumerate(np.arange(0.0, 90.0, 0.5)):
        tr.update(state(x, y=6.2, t=i * 10))
        assert tr.s == pytest.approx(x)


def test_window_edges_and_degenerate_points():
    pts = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0], [10, 0, 0], [20, 0, 0]], float)
    ref = ReferencePath(pts)  # duplicate points -> zero-length segments
    for hint in (None, 0.0, 20.0, 1e6, -50.0):
        s, d = ref.project([7.0, 1.0, 0.0], hint_s=hint)
        assert np.isfinite(s) and np.isfinite(d)
    assert ref.project([7.0, 1.0, 0.0], hint_s=5.0)[0] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        ReferencePath(np.zeros((1, 3)))
    with pytest.raises(ValueError):
        ReferencePath(np.zeros((5, 2)))


def straight_ref() -> ReferencePath:
    return ReferencePath(np.array([[0, 0, 0], [1000, 0, 0]], float))


def test_tracker_finished():
    tr = ProgressTracker(straight_ref(), EvalConfig())
    tr.update(state(10.0, t=0))
    tr.update(state(500.0, t=5000))
    assert not tr.done and tr.progress == pytest.approx(0.5)
    tr.update(state(990.0, t=12340, finished=True))
    assert tr.done and tr.reason == "finished" and tr.progress == 1.0
    assert tr.result()["finish_time_ms"] == 12340 and tr.result()["finished"]
    tr.update(state(0.0, t=99999))  # ignored after done
    assert tr.result()["finish_time_ms"] == 12340


def test_tracker_progress_is_the_maximum():
    tr = ProgressTracker(straight_ref(), EvalConfig())
    for i, x in enumerate([100.0, 300.0, 200.0, 50.0]):
        tr.update(state(x, t=i * 100))
    assert tr.progress == pytest.approx(0.3) and tr.s == pytest.approx(50.0)


def test_tracker_timeout():
    tr = ProgressTracker(straight_ref(), EvalConfig(timeout_s=10.0))
    tr.update(state(10.0, t=10000))
    assert not tr.done
    tr.update(state(20.0, t=10010))
    assert tr.done and tr.reason == "timeout" and not tr.result()["finished"]


def test_tracker_stuck():
    cfg = EvalConfig(stuck_speed_kmh=5.0, stuck_s=3.0)
    tr = ProgressTracker(straight_ref(), cfg)
    for t in range(0, 5000, 100):  # standing still: only counts after the first 2 s
        tr.update(state(0.0, t=t, speed=0.0))
        assert not tr.done, t
    tr.update(state(0.0, t=5000, speed=0.0))
    assert tr.done and tr.reason == "stuck"
    # Moving again before stuck_s elapsed resets the clock.
    tr = ProgressTracker(straight_ref(), cfg)
    for t in range(0, 4000, 100):
        tr.update(state(0.0, t=t, speed=0.0 if t != 3000 else 50.0))
    assert not tr.done
    # A slow start below 5 km/h inside the first 2 s is fine.
    tr = ProgressTracker(straight_ref(), cfg)
    for t in range(0, 2000, 100):
        tr.update(state(0.0, t=t, speed=1.0))
    tr.update(state(1.0, t=2100, speed=60.0))
    assert not tr.done


def test_tracker_offtrack():
    cfg = EvalConfig(offtrack_dist=40.0)
    tr = ProgressTracker(straight_ref(), cfg)
    tr.update(state(100.0, y=0.0, t=0))
    for t in range(100, 1100, 100):  # exactly 1 s beyond the threshold: not yet
        tr.update(state(100.0, y=45.0, t=t))
    assert not tr.done and tr.offtrack_ms == 1000
    tr.update(state(100.0, y=0.0, t=1200))  # back on track resets the clock
    for t in range(1300, 2300, 100):
        tr.update(state(100.0, y=45.0, t=t))
    assert not tr.done
    tr.update(state(100.0, y=45.0, t=2400))
    assert tr.done and tr.reason == "offtrack" and tr.max_dist == pytest.approx(45.0)
    assert tr.mean_dist > 20.0
    assert tr.result()["offtrack_s"] == pytest.approx(2.2)


def test_finish_overrides_from_outside():
    tr = ProgressTracker(straight_ref(), EvalConfig())
    tr.finish("timeout")
    assert tr.done and tr.reason == "timeout"
    tr.finish("stuck")  # first reason wins
    assert tr.reason == "timeout"


def test_load_reference(tmp_path):
    ref = load_reference("fake:s_curve", tmp_path)
    np.testing.assert_allclose(ref.points, track_reference("fake:s_curve"), atol=1e-4)
    assert ref.length > 300.0
    with pytest.raises(FileNotFoundError):
        load_reference("Maps/A01-Race.Challenge.Gbx", tmp_path)
    (tmp_path / "refs").mkdir()
    pts = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)]).astype(np.float32)
    np.save(tmp_path / "refs" / "A01-Race.npy", pts)
    for ref_name in ("A01-Race", "Maps/A01-Race.Challenge.Gbx"):
        assert load_reference(ref_name, tmp_path).length == pytest.approx(100.0)
