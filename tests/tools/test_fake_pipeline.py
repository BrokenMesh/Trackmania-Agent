"""tools/fake_pipeline.py: helpers and a tiny end-to-end run (render, train, eval, live)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tmagent.data.split import split_for_map
from tools import fake_pipeline as fp


def test_check_finite():
    assert fp.check_finite("x", 3) == 3.0
    for bad in (None, float("nan"), float("inf")):
        with pytest.raises(fp.PipelineError, match="x is not a finite number"):
            fp.check_finite("x", bad)


def test_make_config_is_a_small_cpu_setup(tmp_path: Path):
    cfg = fp.make_config(fp.DEFAULT_CONFIG, tmp_path, steps=40, overrides=["train.batch_size=2"])
    assert cfg.data.root == (tmp_path / "data").as_posix()
    assert cfg.data.resolution == [64, 48] and cfg.data.history_s == 0.5
    assert cfg.train.steps == 40 and cfg.train.eval_every == 10 and cfg.train.ckpt_every == 40
    assert cfg.train.device == cfg.runtime.device == "cpu" and cfg.game.backend == "fake"
    assert cfg.train.batch_size == 2  # explicit overrides win


def test_heldout_maps_prefer_test_then_val(tmp_path: Path):
    cfg = fp.make_config(fp.DEFAULT_CONFIG, tmp_path, steps=10)
    maps = [f"fake:random:{s}" for s in range(24)]
    held = fp.heldout_maps(cfg, maps)
    assert len(held) == fp.EVAL_MAPS
    assert all(split_for_map(m, cfg.data) == "test" for m in held)
    assert fp.heldout_maps(cfg, ["fake:random:10"]) == ["fake:random:10"]  # val fallback
    with pytest.raises(fp.PipelineError):
        fp.heldout_maps(cfg, ["fake:random:0"])  # train only


def test_metric_curve(tmp_path: Path):
    (tmp_path / "metrics.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
                {"step": 5, "train/loss": 2.0},
                {"step": 5, "val/loss": 3.0},
                {"step": 10, "train/loss": 1.0},
            )
        )
        + "\n"
    )
    assert fp.metric_curve(tmp_path, "train/loss") == [(5, 2.0), (10, 1.0)]
    assert fp.metric_curve(tmp_path, "val/loss") == [(5, 3.0)]
    assert fp.metric_curve(tmp_path, "nothing") == []


def test_pipeline_end_to_end_tiny(tmp_path: Path, capsys):
    out = tmp_path / "pipe"
    argv = ["--out", str(out), "--steps", "6", "--episodes", "6", "--live-s", "1.0"]
    argv += ["--set", "train.eval_every=6", "train.log_every=3"]
    assert fp.main(argv) == 0
    stdout = capsys.readouterr().out
    assert "fake pipeline OK" in stdout and str(out / "report.md") in stdout

    s = json.loads((out / "summary.json").read_text())
    report = (out / "report.md").read_text()
    for heading in ("## Data", "## Training", "## Closed-loop eval", "## Live", "## Stage timings"):
        assert heading in report
    # every stage ran and produced finite numbers
    assert set(s["timings_s"]) == {"render", "quality", "train", "eval", "live"}
    assert s["data"]["episodes"] == 6 and s["data"]["quality_ok"] and s["data"]["desyncs"] == 0
    assert s["data"]["train_windows"] > 0 and s["data"]["val_windows"] > 0
    t = s["train"]
    assert t["steps"] == 6 and Path(t["ckpt"]).is_file() and t["params_m"] > 0
    for key in ("train_first", "train_last", "val_first", "val_last", "s_per_step"):
        assert math.isfinite(t[key])
    e = s["eval"]
    assert e["n_episodes"] == len(e["maps"]) * 1 and math.isfinite(e["median_progress"])
    assert all(
        split_for_map(m, fp.load_config(fp.DEFAULT_CONFIG).data) == "test" for m in e["maps"]
    )
    assert (out / "eval" / "summary.json").is_file() and (out / "eval" / "episodes.jsonl").is_file()
    lv = s["live"]
    assert lv["ticks"] > 30 and lv["inference"]["frames_observed"] > 3
    assert lv["inference"]["policy_errors"] == 0 and math.isfinite(lv["miss_pct"])
    for path in s["paths"].values():
        assert Path(path).exists()
    assert (out / "data" / "render_report.json").is_file()


def test_pipeline_reports_failures(tmp_path: Path, capsys):
    # too few episodes to cover the splits: a message, not a traceback
    assert fp.main(["--out", str(tmp_path / "x"), "--episodes", "2", "--steps", "2"]) == 1
    assert "FAILED: need at least 3 episodes" in capsys.readouterr().out
