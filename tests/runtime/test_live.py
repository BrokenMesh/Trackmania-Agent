from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from tmagent.config import Config, save_config
from tmagent.runtime import live

from .test_helpers import DummyPolicy, FakeRealtimeGame


@pytest.fixture
def stubs(monkeypatch, tmp_path):
    """Replace the lazily imported game / model / experiment modules written elsewhere."""
    calls: dict = {}
    game = FakeRealtimeGame(fps=60.0)

    def make_game(game_cfg, data_cfg):
        calls["make_game"] = (game_cfg, data_cfg)
        game.start_frames()
        return game

    def load_streaming_policy(ckpt, device):
        calls["policy"] = (ckpt, device)
        return DummyPolicy()

    def create_run(name, cfg):
        calls["run"] = name
        d = tmp_path / f"2026-01-01-{name}"
        d.mkdir()
        return d

    for modname, attr, fn in [
        ("tmagent.game", "make_game", make_game),
        ("tmagent.model.streaming", "load_streaming_policy", load_streaming_policy),
        ("tmagent.experiment", "create_run", create_run),
    ]:
        mod = types.ModuleType(modname)
        setattr(mod, attr, fn)
        monkeypatch.setitem(sys.modules, modname, mod)
    calls["game"] = game
    return calls


def write_cfg(tmp_path: Path) -> Path:
    path = tmp_path / "cfg.yaml"
    save_config(Config(), path)
    return path


def test_live_main_writes_stats_and_latency(stubs, tmp_path):
    out = tmp_path / "out"
    args = [
        *("--config", str(write_cfg(tmp_path)), "--ckpt", "model.pt"),
        *("--map", "maps/A01.Challenge.Gbx", "--duration-s", "0.8"),
        *("--out", str(out), "--set", "runtime.hold_s=0.5"),
    ]
    rc = live.main(args)
    game = stubs["game"]
    assert rc == 0
    assert stubs["policy"] == ("model.pt", "auto")
    assert game.loaded == "maps/A01.Challenge.Gbx" and game.restarts == 1 and game.closed
    stats = json.loads((out / "live_stats.json").read_text())
    assert stats["stats"]["ticks"] > 20 and stats["config"]["hold_s"] == 0.5
    assert stats["finished"] is False
    assert "p99 ms" in (out / "latency.md").read_text()
    assert "control_jitter" in json.loads((out / "latency.json").read_text())["sections"]
    assert game.actions[-1][1].steer == 0.0  # neutral at the end


def test_live_stops_early_when_finished_and_uses_create_run(stubs, tmp_path, capsys):
    game = stubs["game"]
    threading.Timer(0.4, lambda: setattr(game, "finished", True)).start()
    t0 = time.perf_counter()
    args = [
        *("--config", str(write_cfg(tmp_path)), "--ckpt", "m.pt"),
        *("--map", "maps/My Map.Gbx", "--duration-s", "30"),
    ]
    rc = live.main(args)
    assert rc == 0 and time.perf_counter() - t0 < 5.0
    assert stubs["run"] == "live-My_Map"
    out = tmp_path / "2026-01-01-live-My_Map"
    assert json.loads((out / "live_stats.json").read_text())["finished"] is True
    assert "miss_pct=" in capsys.readouterr().out
