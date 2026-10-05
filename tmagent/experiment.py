"""Experiment run directories: experiments/<YYYY-MM-DD>-<name>/ (see docs/ARCHITECTURE.md)."""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from tmagent.config import Config, save_config

_REPO_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str:
    out = subprocess.run(
        ["git", *args], cwd=_REPO_DIR, capture_output=True, text=True, timeout=10, check=True
    )
    return out.stdout.strip()


def git_info() -> str:
    """'commit: <hash>\\ndirty: <true|false>' or 'unknown' when not in a git repo."""
    try:
        commit = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"commit: {commit}\ndirty: {str(dirty).lower()}"


def create_run(name: str, cfg: Config, base: str | Path = "experiments") -> Path:
    """Create `<base>/<date>-<name>` (suffix -2, -3, ... if taken) and return its path.

    Contents: config.yaml, git.txt, seed.txt, empty metrics.jsonl, checkpoints/.
    """
    stem = f"{datetime.date.today().isoformat()}-{name}"
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    run_dir, n = base / stem, 1
    while True:
        try:
            run_dir.mkdir()
            break
        except FileExistsError:
            n += 1
            run_dir = base / f"{stem}-{n}"
    (run_dir / "checkpoints").mkdir()
    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git.txt").write_text(git_info() + "\n")
    (run_dir / "seed.txt").write_text(f"{cfg.train.seed}\n")
    (run_dir / "metrics.jsonl").touch()
    return run_dir


def log_metrics(run_dir: str | Path, step: int, metrics: dict[str, Any]) -> None:
    """Append one JSON line {step, wall_time, **metrics} to run_dir/metrics.jsonl."""
    row = {"step": int(step), "wall_time": time.time()}
    row.update({k: float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()})
    with open(Path(run_dir) / "metrics.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
