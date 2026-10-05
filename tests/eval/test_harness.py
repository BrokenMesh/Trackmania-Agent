"""Closed-loop harness on FakeGame: episodes, timing grid, noise, evaluate() outputs."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from tmagent.config import Config
from tmagent.data.timeline import resample_to_control
from tmagent.eval.harness import (
    control_row_for_tick,
    evaluate,
    main,
    reference_time_ms,
    run_episode_realtime,
    run_episode_sync,
)
from tmagent.eval.progress import ReferencePath, load_reference
from tmagent.game.fake import FakeGame, scripted_driver
from tmagent.interfaces import Action, ChunkPolicy, InputTimeline


def make_cfg(**eval_kw) -> Config:
    cfg = Config()
    cfg.data.resolution = [64, 48]
    cfg.eval.seeds = [0]
    for k, v in eval_kw.items():
        setattr(cfg.eval, k, v)
    cfg.validate()
    return cfg


class ReplayPolicy:
    """Outputs the scripted driver's actions for the current time (tick-grid sample-and-hold)."""

    def __init__(self, timeline: InputTimeline, cfg: Config, chunk_len: int = 8) -> None:
        self.tl, self.d, self.chunk_len = timeline, cfg.data, chunk_len
        self.resets = 0
        self.k = 0

    def reset(self) -> None:
        self.resets += 1
        self.k = 0

    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        self.t = round(self.k * 1000 / self.d.frame_hz)
        self.k += 1

    def predict(self) -> np.ndarray:
        times = [self.t + round(j * 1000 / self.d.control_hz) for j in range(self.chunk_len)]
        idx = np.minimum(np.array(times) // 10, len(self.tl.actions) - 1)
        return self.tl.actions[idx]


class ConstPolicy:
    """Always the same action row; records what it observes."""

    def __init__(self, row=(0.0, 0.0, 0.0), chunk_len: int = 8) -> None:
        self.row, self.chunk_len = np.array(row, np.float32), chunk_len
        self.past: list[np.ndarray] = []
        self.shapes: list[tuple] = []

    def reset(self) -> None:
        self.past, self.shapes = [], []

    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        self.past.append(past_actions.copy())
        self.shapes.append((image.shape, image.dtype))

    def predict(self) -> np.ndarray:
        return np.tile(self.row, (self.chunk_len, 1))


class IndexPolicy(ConstPolicy):
    """Row j has steer = j / 10 so the applied row index can be read back from the action."""

    def predict(self) -> np.ndarray:
        out = np.zeros((self.chunk_len, 3), np.float32)
        out[:, 0] = np.arange(self.chunk_len) / 10.0
        out[:, 1] = 1.0
        return out


class LoggingGame(FakeGame):
    """FakeGame that logs (race time before the tick, applied steer) of every step."""

    log: list

    def start_race(self):
        self.log = []
        return super().start_race()

    def step(self, action: Action, n_ticks: int = 1):
        self.log.append((self.get_state().race_time_ms, action.steer, n_ticks))
        return super().step(action, n_ticks)


def test_policies_satisfy_protocol():
    cfg = make_cfg()
    tl, _ = scripted_driver("fake:oval", cfg.data)
    assert isinstance(ReplayPolicy(tl, cfg), ChunkPolicy) and isinstance(ConstPolicy(), ChunkPolicy)


RATES = [(20, 60), (25, 50)]  # (frame_hz, control_hz)


def rate_cfg(frame_hz: int, control_hz: int) -> Config:
    cfg = make_cfg()
    cfg.data.frame_hz, cfg.data.control_hz, cfg.runtime.control_hz = (
        frame_hz,
        control_hz,
        control_hz,
    )
    cfg.validate()
    return cfg


def test_replay_policy_finishes_the_oval():
    cfg = make_cfg()
    tl, info = scripted_driver("fake:oval", cfg.data)
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:oval", cfg.data.root)
    res = run_episode_sync(game, ReplayPolicy(tl, cfg), "fake:oval", cfg, 0, ref)
    assert res["finished"] and res["reason"] == "finished" and res["progress"] == 1.0
    assert res["map"] == "fake:oval" and res["seed"] == 0 and res["mode"] == "sync"
    assert res["finish_time_ms"] == info["race_time_ms"] and res["time_ratio"] == 1.0
    assert 0.0 <= res["mean_dist"] <= res["max_dist"] < 1.5 and res["offtrack_s"] == 0.0
    assert res["frames"] == pytest.approx(res["finish_time_ms"] / 50, abs=2)


@pytest.mark.parametrize("frame_hz, control_hz", RATES)
@pytest.mark.parametrize("map_ref", ["fake:oval", "fake:s_curve", "fake:random:3"])
def test_replay_through_the_control_grid_is_exact(map_ref, frame_hz, control_hz):
    """The scripted driver holds actions for 100 ms, so the closed-loop replay through
    the control grid (60 Hz / 20 Hz and 50 Hz / 25 Hz) is the same run as the original."""
    cfg = rate_cfg(frame_hz, control_hz)
    tl, info = scripted_driver(map_ref, cfg.data)
    executed, *_ = execute_like_harness(tl, control_hz, frame_hz)
    np.testing.assert_array_equal(executed, tl.actions)
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference(map_ref, cfg.data.root)
    res = run_episode_sync(game, ReplayPolicy(tl, cfg), map_ref, cfg, 0, ref)
    assert res["finished"] and res["finish_time_ms"] == info["race_time_ms"]
    assert res["progress"] == 1.0 and res["max_dist"] < 1.5


def test_per_tick_drivers_drift_when_resampled_to_60hz():
    """hold_ticks=1 puts toggles between the sampled ticks; replay at 60 Hz is not exact."""
    cfg = rate_cfg(20, 60)
    tl, info = scripted_driver("fake:random:3", cfg.data, hold_ticks=1)
    executed, *_ = execute_like_harness(tl, 60, 20)
    assert not np.array_equal(executed, tl.actions)
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:random:3", cfg.data.root)
    res = run_episode_sync(game, ReplayPolicy(tl, cfg), "fake:random:3", cfg, 0, ref)
    assert res["finish_time_ms"] != info["race_time_ms"]


def test_observe_inputs_and_past_actions():
    cfg = make_cfg(timeout_s=1.0)
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:oval", cfg.data.root)
    pol = ConstPolicy(row=(0.25, 1.0, 0.0), chunk_len=8)
    res = run_episode_sync(game, pol, "fake:oval", cfg, 0, ref)
    assert res["reason"] == "timeout" and len(pol.past) == res["frames"] == 21
    assert all(s == ((48, 64, 3), np.uint8) for s in pol.shapes)
    assert pol.past[0].shape == (3, 3) and pol.past[0].dtype == np.float32
    assert not pol.past[0].any()  # zeros on the first call
    for p in pol.past[1:]:
        np.testing.assert_array_equal(p, np.tile([0.25, 1.0, 0.0], (3, 1)))
    # A chunk shorter than R: the last row is held.
    pol = IndexPolicy(chunk_len=2)
    run_episode_sync(game, pol, "fake:oval", cfg, 0, ref)
    np.testing.assert_allclose(pol.past[1][:, 0], [0.0, 0.1, 0.1], atol=1e-6)
    # Single-row chunks work too.
    pol = ConstPolicy(row=(0.0, 1.0, 0.0), chunk_len=1)
    assert run_episode_sync(game, pol, "fake:oval", cfg, 0, ref)["reason"] == "timeout"


def test_chunk_rows_follow_the_control_grid():
    cfg = make_cfg(timeout_s=0.5)
    game = LoggingGame(cfg.game, cfg.data)
    ref = load_reference("fake:oval", cfg.data.root)
    pol = IndexPolicy(chunk_len=8)
    run_episode_sync(game, pol, "fake:oval", cfg, 0, ref)
    assert all(n == 1 for _, _, n in game.log)
    # 20 Hz frames, 60 Hz control (steps at 0, 17, 33 ms): ticks 0..4 run rows 0, 1, 1, 2, 2.
    first = [(t, round(steer * 10)) for t, steer, _ in game.log[:10]]
    assert first == [(0, 0), (10, 1), (20, 1), (30, 2), (40, 2)] + [
        (50, 0),
        (60, 1),
        (70, 1),
        (80, 2),
        (90, 2),
    ]
    # The executed rows 0, 1, 2 are what the next observe sees.
    np.testing.assert_allclose(pol.past[1][:, 0], [0.0, 0.1, 0.2], atol=1e-6)


def execute_like_harness(tl: InputTimeline, control_hz: int, frame_hz: int):
    """Tick-by-tick action the harness runs when a policy replays `tl` through the control grid.

    Returns (executed [N, 3], control step used at every tick, sampled actions, control times).
    """
    n_ticks, r = len(tl.actions), control_hz // frame_hz
    sampled, times = resample_to_control(tl, control_hz)  # row i = tl[times[i] // 10]
    executed = np.zeros_like(tl.actions)
    step_of_tick, k = [], 0
    for m in range(n_ticks):
        while round((k + 1) * 1000 / frame_hz) <= m * 10:  # next frame has started
            k += 1
        i = k * r + control_row_for_tick(m * 10, k, control_hz, r)
        executed[m] = sampled[min(i, len(sampled) - 1)]
        step_of_tick.append(i)
    return executed, step_of_tick, sampled, times


@pytest.mark.parametrize("control_hz, frame_hz", [(60, 20), (50, 25), (100, 20), (40, 20)])
def test_tick_mapping_inverts_resample_to_control(control_hz, frame_hz):
    """Executing frame by frame reproduces the ticks that resample_to_control sampled."""
    rng = np.random.default_rng(0)
    tl = InputTimeline(actions=rng.uniform(-1, 1, (600, 3)).astype(np.float32))
    executed, step_of_tick, sampled, times = execute_like_harness(tl, control_hz, frame_hz)
    for i, t in enumerate(times):
        tick = int(t) // 10
        assert step_of_tick[tick] == i  # control step i runs during the tick it was sampled from
        np.testing.assert_array_equal(executed[tick], tl.actions[tick])
    assert step_of_tick == sorted(step_of_tick)  # and the steps run in order


def test_tick_mapping_is_exact_at_50hz():
    rng = np.random.default_rng(1)
    base = rng.uniform(-1, 1, (300, 3)).astype(np.float32)
    pairs = InputTimeline(actions=np.repeat(base, 2, axis=0))  # constant over tick pairs
    executed, _, _, _ = execute_like_harness(pairs, 50, 25)
    np.testing.assert_array_equal(executed, pairs.actions)
    executed60, _, _, _ = execute_like_harness(pairs, 60, 20)  # 60 Hz cannot be exact
    assert not np.array_equal(executed60, pairs.actions)


def test_stuck_policy():
    cfg = make_cfg()
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:oval", cfg.data.root)
    res = run_episode_sync(game, ConstPolicy(), "fake:oval", cfg, 0, ref)
    assert res["reason"] == "stuck" and not res["finished"] and res["finish_time_ms"] is None
    assert res["race_time_ms"] == pytest.approx(5000, abs=100) and res["progress"] < 0.05
    assert res["time_ratio"] is None


def test_offtrack_policy():
    cfg = make_cfg(offtrack_dist=25.0)
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:s_curve", cfg.data.root)
    res = run_episode_sync(game, ConstPolicy((0.0, 1.0, 0.0)), "fake:s_curve", cfg, 0, ref)
    assert res["reason"] == "offtrack" and not res["finished"]
    assert 0.1 < res["progress"] < 0.9 and res["max_dist"] > 25.0 and res["offtrack_s"] > 1.0


def test_action_noise_is_seeded():
    cfg = make_cfg(action_noise=0.3, timeout_s=8.0)
    tl, _ = scripted_driver("fake:s_curve", cfg.data)
    ref = load_reference("fake:s_curve", cfg.data.root)

    def run(seed: int) -> dict:
        game = FakeGame(cfg.game, cfg.data)
        return run_episode_sync(game, ReplayPolicy(tl, cfg), "fake:s_curve", cfg, seed, ref)

    a, b, c = run(1), run(1), run(2)
    assert a == b
    assert a["mean_dist"] != c["mean_dist"]
    clean = make_cfg(timeout_s=8.0)
    game = FakeGame(clean.game, clean.data)
    res = run_episode_sync(game, ReplayPolicy(tl, clean), "fake:s_curve", clean, 1, ref)
    assert res["mean_dist"] != a["mean_dist"]


def test_reference_time():
    ref = ReferencePath(np.array([[0, 0, 0], [1, 0, 0]], float))
    t = reference_time_ms("fake:s_curve", ref)
    assert t is not None and 10_000 < t < 30_000
    assert reference_time_ms("fake:s_curve", ref) == t  # cached
    assert reference_time_ms("Maps/x.Gbx", ref) is None
    ref.time_ms = 4242
    assert reference_time_ms("Maps/x.Gbx", ref) == 4242


def test_evaluate_writes_summary_files(tmp_path):
    cfg = make_cfg(maps=["fake:oval"], seeds=[0, 1])
    tl, _ = scripted_driver("fake:oval", cfg.data)
    made = []

    def factory() -> ReplayPolicy:
        made.append(ReplayPolicy(tl, cfg))
        return made[0]

    out = tmp_path / "run"
    summary = evaluate(factory, cfg, out_dir=out)
    assert len(made) == 1 and made[0].resets == 2  # one policy, reset per episode
    assert summary["finish_rate"] == 1.0 and summary["median_progress"] == 1.0
    assert summary["mean_progress"] == 1.0 and summary["median_time_ratio"] == pytest.approx(
        1, abs=0.03
    )
    assert summary["deadline_miss_pct"] is None and summary["mode"] == "sync"
    assert summary["per_map"]["fake:oval"]["n_episodes"] == 2 and len(summary["episodes"]) == 2
    lines = (out / "episodes.jsonl").read_text().splitlines()
    eps = [json.loads(line) for line in lines]
    assert [e["seed"] for e in eps] == [0, 1] and all(e["finished"] for e in eps)
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["finish_rate"] == 1.0 and "episodes" not in on_disk
    assert on_disk["per_map"]["fake:oval"]["finish_rate"] == 1.0 and on_disk["reasons"] == {
        "finished": 2
    }


def test_evaluate_multi_map_failure_and_defaults():
    cfg = make_cfg(maps=["fake:oval", "fake:s_curve"], seeds=[0, 1, 2])
    summary = evaluate(lambda: ConstPolicy(), cfg)
    assert summary["n_episodes"] == 6 and summary["finish_rate"] == 0.0
    assert summary["median_time_ratio"] is None and summary["reasons"] == {"stuck": 6}
    assert set(summary["per_map"]) == {"fake:oval", "fake:s_curve"}
    # Empty eval.maps on the fake backend falls back to game.fake_track.
    cfg = make_cfg()
    cfg.game.fake_track = "s_curve"
    assert evaluate(lambda: ConstPolicy(), cfg)["per_map"].keys() == {"fake:s_curve"}
    cfg.eval.mode = "bogus"
    with pytest.raises(ValueError):
        evaluate(lambda: ConstPolicy(), cfg)


def test_evaluate_uses_a_given_game_and_leaves_it_open():
    cfg = make_cfg(maps=["fake:oval"], timeout_s=1.0)
    game = FakeGame(cfg.game, cfg.data)
    evaluate(lambda: ConstPolicy(), cfg, game=game)
    game.start_race()  # still usable


def test_run_episode_realtime_with_live_session():
    pytest.importorskip("tmagent.runtime.session")
    cfg = make_cfg(timeout_s=3.0)
    cfg.game.game_speed = 5.0
    game = FakeGame(cfg.game, cfg.data)
    ref = load_reference("fake:oval", cfg.data.root)
    try:
        res = run_episode_realtime(game, ConstPolicy((0.0, 1.0, 0.0)), "fake:oval", cfg, 0, ref)
    finally:
        game.close()
    assert res["mode"] == "realtime" and res["reason"] in ("timeout", "offtrack", "stuck")
    assert 0.0 <= res["progress"] <= 1.0 and res["race_time_ms"] > 500
    assert isinstance(res["stats"], dict) and "deadline_misses" in res["stats"]
    json.dumps(res, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def test_evaluate_realtime_reports_deadline_misses():
    pytest.importorskip("tmagent.runtime.session")
    cfg = make_cfg(maps=["fake:oval"], mode="realtime", timeout_s=2.0)
    cfg.game.game_speed = 5.0
    summary = evaluate(lambda: ConstPolicy((0.0, 1.0, 0.0)), cfg)
    assert summary["mode"] == "realtime" and summary["n_episodes"] == 1
    assert isinstance(summary["deadline_miss_pct"], float)
    assert 0.0 <= summary["deadline_miss_pct"] <= 100.0


def test_cli(tmp_path, monkeypatch, capsys):
    cfg_path = tmp_path / "cli.yaml"
    cfg_path.write_text(
        "name: cli\ndata:\n  resolution: [64, 48]\neval:\n  maps: [fake:oval]\n  timeout_s: 1.0\n"
    )
    stub = types.ModuleType("tmagent.model.streaming")
    loaded = []

    def load_streaming_policy(ckpt, device):
        loaded.append((ckpt, device))
        return ConstPolicy()

    stub.load_streaming_policy = load_streaming_policy
    monkeypatch.setitem(sys.modules, "tmagent.model.streaming", stub)
    out = tmp_path / "out"
    main(
        [
            "--config",
            str(cfg_path),
            "--ckpt",
            "m.pt",
            "--set",
            "eval.seeds=[0,1]",
            "runtime.device=cpu",
        ]
        + ["--out", str(out)]
    )
    assert loaded == [("m.pt", "cpu")]
    assert json.loads((out / "summary.json").read_text())["n_episodes"] == 2
    assert "n_episodes" in capsys.readouterr().out
    # Without --out a new experiments/<date>-eval-<name> run is created.
    monkeypatch.chdir(tmp_path)
    main(["--config", str(cfg_path), "--ckpt", "m.pt"])
    runs = list((tmp_path / "experiments").glob("*-eval-cli"))
    assert len(runs) == 1 and (runs[0] / "summary.json").is_file()
    assert (runs[0] / "config.yaml").is_file()


def test_load_reference_run_time(tmp_path):
    (tmp_path / "refs").mkdir()
    pts = np.column_stack([np.linspace(0, 100, 11), np.zeros(11), np.zeros(11)])
    np.save(tmp_path / "refs" / "A01-Race.npy", pts)
    assert load_reference("A01-Race", tmp_path).time_ms is None
    meta = {"race_time_ms": 41234, "source": "tmx:123"}
    (tmp_path / "refs" / "A01-Race.json").write_text(json.dumps(meta))
    ref = load_reference("A01-Race", tmp_path)
    assert ref.time_ms == 41234 and reference_time_ms("A01-Race", ref) == 41234
    assert load_reference("fake:oval", tmp_path).time_ms is None  # fake maps use the scripted time
