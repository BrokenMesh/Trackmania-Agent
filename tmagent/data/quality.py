"""Dataset QA: timestamp grids, sync, ranges, desync/respawn and duplicate checks.

CLI: python -m tmagent.data.quality <root> [--config path]  (exit 1 on problems;
issues prefixed "warn:" are reported but do not fail)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import DataConfig, load_config
from tmagent.data.episode_io import load_episode, read_index
from tmagent.data.split import SPLITS, filter_index
from tmagent.data.timeline import control_times_ms, frame_times_ms
from tmagent.interfaces import BRAKE, EPISODE_META_KEYS, GAS, STEER, Episode

MAX_JUMP = 20.0  # world units between consecutive control steps (respawn / desync)
MAX_DUP_FRAC = 0.5  # fraction of identical consecutive frames


def _nonbinary_frac(x: np.ndarray, allowed: tuple[float, ...]) -> float:
    if x.size == 0:
        return 0.0
    ok = np.zeros(x.shape, dtype=bool)
    for v in allowed:
        ok |= x == v
    return float(1.0 - ok.mean())


def episode_stats(ep: Episode) -> dict[str, float]:
    """Fraction of non-binary values (steer: not in {-1, 0, 1}, gas/brake: not in {0, 1})."""
    a = ep.actions
    return {
        "steer_nonbinary": _nonbinary_frac(a[:, STEER], (-1.0, 0.0, 1.0)),
        "gas_nonbinary": _nonbinary_frac(a[:, GAS], (0.0, 1.0)),
        "brake_nonbinary": _nonbinary_frac(a[:, BRAKE], (0.0, 1.0)),
    }


def _identical_frame_frac(frames: np.ndarray, chunk: int = 64) -> float:
    if len(frames) < 2:
        return 0.0
    same = 0
    for s in range(0, len(frames) - 1, chunk):
        a, b = frames[s : s + chunk], frames[s + 1 : s + 1 + chunk]
        same += int((a[: len(b)] == b).reshape(len(b), -1).all(axis=1).sum())
    return same / (len(frames) - 1)


def _grid_issue(name: str, times: np.ndarray, expected: np.ndarray, hz: int) -> str | None:
    if np.array_equal(times, expected):
        return None
    bad = np.flatnonzero(times != expected)
    return (
        f"grid: {name} differs from round(i*1000/{hz}) at {len(bad)} entries (first index {bad[0]})"
    )


def check_episode(
    ep: Episode, cfg: DataConfig, max_jump: float = MAX_JUMP, require_binary: bool = False
) -> list[str]:
    """Return a list of issues (empty = OK); warning-level ones start with "warn:".

    max_jump: largest plausible position change between consecutive control
    steps (respawn / desync detector). require_binary: also flag non-binary
    gas/brake (analog gamepad data) as a problem.
    """
    issues: list[str] = []
    missing = [k for k in EPISODE_META_KEYS if k not in ep.meta]
    if missing:
        issues.append(f"meta: missing keys {missing}")
    t_f, t_a = len(ep.frames), len(ep.actions)
    if t_f == 0 or t_a == 0:
        return [*issues, f"shape: empty episode (frames={t_f}, actions={t_a})"]
    shape_bad = []
    if ep.frames.ndim != 4 or ep.frames.dtype != np.uint8:
        shape_bad.append(f"frames {ep.frames.shape} {ep.frames.dtype} (want uint8 [T,H,W,C])")
    if ep.actions.shape != (t_a, 3):
        shape_bad.append(f"actions {ep.actions.shape}")
    if ep.positions.shape != (t_a, 3):
        shape_bad.append(f"positions {ep.positions.shape}")
    if ep.speeds_kmh.shape != (t_a,):
        shape_bad.append(f"speeds_kmh {ep.speeds_kmh.shape}")
    if ep.frame_times_ms.shape != (t_f,):
        shape_bad.append(f"frame_times_ms {ep.frame_times_ms.shape}")
    if ep.action_times_ms.shape != (t_a,):
        shape_bad.append(f"action_times_ms {ep.action_times_ms.shape}")
    if shape_bad:
        return [*issues, "shape: " + "; ".join(shape_bad)]

    for key, want in (("frame_hz", cfg.frame_hz), ("control_hz", cfg.control_hz)):
        if key in ep.meta and ep.meta[key] != want:
            issues.append(f"meta: {key}={ep.meta[key]} but config has {want}")
    w, h = cfg.resolution
    if ep.frames.shape[1:] != (h, w, cfg.channels):
        issues.append(f"shape: frames {ep.frames.shape[1:]} != (H,W,C)={(h, w, cfg.channels)}")

    for name, times in (
        ("frame_times_ms", ep.frame_times_ms),
        ("action_times_ms", ep.action_times_ms),
    ):
        if np.any(np.diff(times) <= 0):
            issues.append(f"monotonic: {name} not strictly increasing")
    for issue in (
        _grid_issue(
            "frame_times_ms", ep.frame_times_ms, frame_times_ms(t_f, cfg.frame_hz), cfg.frame_hz
        ),
        _grid_issue(
            "action_times_ms",
            ep.action_times_ms,
            control_times_ms(t_a, cfg.control_hz),
            cfg.control_hz,
        ),
    ):
        if issue:
            issues.append(issue)

    r = cfg.actions_per_frame
    if not (t_f * r - r <= t_a <= t_f * r + r):
        issues.append(
            f"length: T_a={t_a} outside [{t_f * r - r}, {t_f * r + r}] for T_f={t_f}, R={r}"
        )

    for name, arr in (
        ("actions", ep.actions),
        ("positions", ep.positions),
        ("speeds_kmh", ep.speeds_kmh),
    ):
        if not np.isfinite(arr).all():
            issues.append(f"nonfinite: {name} contains NaN/inf")
    finite = np.isfinite(ep.actions).all()
    if finite:
        a = ep.actions
        if a[:, STEER].min() < -1 or a[:, STEER].max() > 1:
            issues.append("range: steer outside [-1, 1]")
        for ch, name in ((GAS, "gas"), (BRAKE, "brake")):
            if a[:, ch].min() < 0 or a[:, ch].max() > 1:
                issues.append(f"range: {name} outside [0, 1]")
        if require_binary:
            stats = episode_stats(ep)
            for name in ("gas", "brake"):
                if stats[f"{name}_nonbinary"] > 0:
                    issues.append(
                        f"binary: {name} has {stats[f'{name}_nonbinary']:.1%} non-binary values"
                    )

    if t_a > 1 and np.isfinite(ep.positions).all():
        jumps = np.linalg.norm(np.diff(ep.positions.astype(np.float64), axis=0), axis=1)
        n_jump = int((jumps > max_jump).sum())
        if n_jump:
            at = int(ep.action_times_ms[int(jumps.argmax()) + 1])
            issues.append(
                f"jump: {n_jump} position jumps > {max_jump} (max {jumps.max():.1f} at {at} ms)"
            )

    dup = _identical_frame_frac(ep.frames)
    if dup > MAX_DUP_FRAC:
        issues.append(f"duplicate_frames: {dup:.0%} of consecutive frames identical")
    if ep.meta.get("desync"):
        issues.append("desync: meta['desync'] is True (re-drive did not reproduce the replay time)")
    if ep.meta.get("frame_time_mismatch", 0) > 0:
        n = ep.meta["frame_time_mismatch"]
        issues.append(f"sync: {n} frames had race_time_ms != tick time at capture")
    if ep.meta.get("resized_frames", 0) > 0:
        issues.append(
            f"warn: {ep.meta['resized_frames']} frames were resized to the data resolution"
        )
    return issues


def split_issues(issues: list[str]) -> tuple[list[str], list[str]]:
    """(problems, warnings) of a check_episode result."""
    warns = [i for i in issues if i.startswith("warn:")]
    return [i for i in issues if not i.startswith("warn:")], warns


def check_dataset(root: str | Path, cfg: DataConfig, **kwargs: Any) -> dict[str, Any]:
    """Check every indexed episode and summarize the dataset.

    Keys: ok, num_episodes, maps, episodes_per_split, hours (total + per split),
    finished_frac, nonbinary (mean fractions), problems {path: [issues]},
    warnings {path: ["warn: ..."]} (do not affect ok), index_issues. kwargs go to
    check_episode.
    """
    entries = read_index(root)
    split_of_path = {e["path"]: s for s in SPLITS for e in filter_index(entries, s, cfg, root)}
    problems: dict[str, list[str]] = {}
    warns: dict[str, list[str]] = {}
    index_issues: list[str] = []
    stats: list[dict[str, float]] = []
    frames_per_split = dict.fromkeys(SPLITS, 0)
    seen_ids: set[str] = set()
    for e in entries:
        path = e["path"]
        eid = e.get("episode_id", path)
        if eid in seen_ids:
            index_issues.append(f"duplicate episode_id {eid}")
        seen_ids.add(eid)
        try:
            ep = load_episode(Path(root) / path)
        except Exception as exc:  # missing/corrupt file is a dataset problem, not a crash
            problems[path] = [f"load: {type(exc).__name__}: {exc}"]
            continue
        if len(ep.frames) != e.get("num_frames"):
            index_issues.append(
                f"{path}: index num_frames={e.get('num_frames')} != {len(ep.frames)}"
            )
        frames_per_split[split_of_path[path]] += len(ep.frames)
        stats.append(episode_stats(ep))
        bad, warn = split_issues(check_episode(ep, cfg, **kwargs))
        if bad:
            problems[path] = bad
        if warn:
            warns[path] = warn
    per_split = {s: sum(1 for v in split_of_path.values() if v == s) for s in SPLITS}
    hours = {s: n / cfg.frame_hz / 3600 for s, n in frames_per_split.items()}
    hours["total"] = sum(hours.values())
    return {
        "root": str(root),
        "ok": not problems and not index_issues,
        "num_episodes": len(entries),
        "maps": sorted({e["map_uid"] for e in entries}),
        "episodes_per_split": per_split,
        "hours": hours,
        "finished_frac": float(np.mean([bool(e.get("finished")) for e in entries]))
        if entries
        else 0.0,
        "nonbinary": {
            k: float(np.mean([s[k] for s in stats])) if stats else 0.0
            for k in ("steer_nonbinary", "gas_nonbinary", "brake_nonbinary")
        },
        "problems": problems,
        "warnings": warns,
        "index_issues": index_issues,
    }


def format_report(summary: dict[str, Any]) -> str:
    """Human-readable report of check_dataset output."""
    hours = summary["hours"]
    lines = [
        f"dataset: {summary['root']}",
        f"episodes: {summary['num_episodes']}  maps: {len(summary['maps'])}  "
        f"finished: {summary['finished_frac']:.0%}",
        "splits: " + "  ".join(f"{s}={n}" for s, n in summary["episodes_per_split"].items()),
        f"hours of data: total {hours['total']:.3f} ("
        + ", ".join(f"{s} {hours[s]:.3f}" for s in SPLITS)
        + ")",
        "non-binary fraction: "
        + "  ".join(f"{k}={v:.3f}" for k, v in summary["nonbinary"].items()),
    ]
    lines += [f"INDEX: {msg}" for msg in summary["index_issues"]]
    for path, issues in summary["problems"].items():
        lines.append(f"PROBLEM {path}")
        lines += [f"  - {msg}" for msg in issues]
    for path, issues in summary["warnings"].items():
        lines.append(f"WARN {path}")
        lines += [f"  - {msg}" for msg in issues]
    n_bad, n_idx = len(summary["problems"]), len(summary["index_issues"])
    lines.append(
        "OK" if summary["ok"] else f"FAILED: {n_bad} problem episodes, {n_idx} index issues"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tmagent.data.quality", description=__doc__)
    ap.add_argument("root", help="dataset root (episodes/ + index.jsonl)")
    ap.add_argument("--config", default=None, help="YAML config (data section is used)")
    ap.add_argument("--max-jump", type=float, default=MAX_JUMP)
    ap.add_argument("--require-binary", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config).data
    summary = check_dataset(
        args.root, cfg, max_jump=args.max_jump, require_binary=args.require_binary
    )
    print(format_report(summary))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
