"""tools/eval_ui.py: map names, the job runner (queue, skip, restart, stop) on the fake game."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from tmagent.config import Config
from tmagent.game import make_game
from tools import eval_ui as ui


class GasPolicy:
    """Full gas, no steering; calls `hook(frame_count)` before each prediction."""

    def __init__(self, hook=None) -> None:
        self.hook, self.frames = hook, 0

    def reset(self) -> None:
        pass

    def observe(self, image: np.ndarray, past_actions: np.ndarray) -> None:
        self.frames += 1

    def predict(self) -> np.ndarray:
        if self.hook is not None:
            self.hook(self.frames)
        return np.tile(np.array([0.0, 1.0, 0.0], np.float32), (8, 1))


def make_runner(cfg: Config, tmp_path: Path, policy: GasPolicy) -> ui.Runner:
    cfg.eval.timeout_s = 3.0
    runner = ui.Runner(cfg, tmp_path / ui.RESULTS_NAME)
    runner._policy = lambda ckpt: policy
    runner._game = lambda: make_game(cfg.game, cfg.data)
    return runner


def wait_idle(runner: ui.Runner, timeout_s: float = 60.0) -> dict:
    end = time.time() + timeout_s
    while time.time() < end:
        st = runner.status()
        if st["current"] is None and not st["queue"]:
            return st
        time.sleep(0.05)
    raise AssertionError("runner did not finish")


def test_plain_name_strips_format_codes():
    assert ui.plain_name("$s$w$i$ff0Lost in Speed") == "Lost in Speed"
    assert ui.plain_name("$l[http://x.y]Link$l and $$5") == "Link and $5"
    assert ui.plain_name("A01-Race") == "A01-Race"


def test_runner_drives_queued_maps_and_persists_results(fake_cfg: Config, tmp_path: Path):
    runner = make_runner(fake_cfg, tmp_path, GasPolicy())
    runner.submit("ckpt.pt", [{"uid": "fake:oval", "name": "Oval"}], repeats=2)
    st = wait_idle(runner)
    assert [(r["name"], r["seed"]) for r in st["results"]] == [("Oval", 0), ("Oval", 1)]
    assert all("progress" in r and r["progress"] > 0 for r in st["results"])
    reloaded = ui.Runner(fake_cfg, tmp_path / ui.RESULTS_NAME)
    assert len(reloaded.results) == 2
    reloaded.clear_results()
    assert not (tmp_path / ui.RESULTS_NAME).exists()


def test_skip_aborts_and_restart_reruns_the_map(fake_cfg: Config, tmp_path: Path):
    calls = {"restarted": False}
    runner = None

    def hook(frames: int) -> None:
        if frames == 5 and not calls["restarted"]:
            calls["restarted"] = True
            runner.restart()

    runner = make_runner(fake_cfg, tmp_path, GasPolicy(hook))
    runner.submit("ckpt.pt", [{"uid": "fake:oval", "name": "Oval"}], repeats=1)
    st = wait_idle(runner)
    assert (
        len(st["results"]) == 1 and "aborted" not in st["results"][0]
    )  # the restart is not logged

    skipper = GasPolicy(lambda frames: runner.skip() if frames == 5 else None)
    runner._policy = lambda ckpt: skipper
    runner.submit("ckpt.pt", [{"uid": "fake:oval", "name": "Oval"}], repeats=1)
    st = wait_idle(runner)
    assert st["results"][-1]["aborted"] == "skipped"


def test_stop_clears_the_queue(fake_cfg: Config, tmp_path: Path):
    runner = None
    policy = GasPolicy(lambda frames: runner.stop() if frames == 3 else None)
    runner = make_runner(fake_cfg, tmp_path, policy)
    runner.submit("ckpt.pt", [{"uid": "fake:oval", "name": "Oval"}], repeats=3)
    st = wait_idle(runner)
    assert [r.get("aborted") for r in st["results"]] == ["stopped"]
