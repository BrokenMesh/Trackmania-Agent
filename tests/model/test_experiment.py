"""experiment.create_run / log_metrics."""

from __future__ import annotations

import datetime
import json
import subprocess

from tmagent import experiment
from tmagent.config import Config, load_config
from tmagent.experiment import create_run, log_metrics


def test_create_run_layout_and_suffix(tmp_path):
    cfg = Config()
    cfg.train.seed = 7
    a = create_run("demo", cfg, base=tmp_path / "exp")
    b = create_run("demo", cfg, base=tmp_path / "exp")
    c = create_run("demo", cfg, base=tmp_path / "exp")
    today = datetime.date.today().isoformat()
    assert [p.name for p in (a, b, c)] == [f"{today}-demo", f"{today}-demo-2", f"{today}-demo-3"]
    for run in (a, b, c):
        assert (run / "checkpoints").is_dir()
        assert (run / "metrics.jsonl").read_text() == ""
        assert (run / "seed.txt").read_text().strip() == "7"
        assert load_config(run / "config.yaml") == cfg
        assert (run / "git.txt").read_text().strip()


def test_git_txt_commit_or_unknown(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(experiment.subprocess, "run", boom)
    run = create_run("nogit", Config(), base=tmp_path)
    assert (run / "git.txt").read_text().strip() == "unknown"

    def fake(args, **kwargs):
        out = "abc123\n" if args[1] == "rev-parse" else " M file.py\n"
        return subprocess.CompletedProcess(args, 0, stdout=out, stderr="")

    monkeypatch.setattr(experiment.subprocess, "run", fake)
    assert experiment.git_info() == "commit: abc123\ndirty: true"


def test_log_metrics_appends_jsonl(tmp_path):
    run = create_run("m", Config(), base=tmp_path)
    log_metrics(run, 1, {"loss": 0.5, "split": "train"})
    log_metrics(run, 2, {"loss": 0.25})
    rows = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2]
    assert rows[0]["loss"] == 0.5 and rows[0]["split"] == "train"
    assert rows[1]["wall_time"] >= rows[0]["wall_time"] > 0
