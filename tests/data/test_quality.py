from __future__ import annotations

import copy
import json
import shutil

import numpy as np
import pytest

from tmagent.data.episode_io import append_index, load_episode, read_index, save_episode
from tmagent.data.quality import (
    check_dataset,
    check_episode,
    format_report,
    main,
    split_issues,
)
from tmagent.interfaces import EPISODE_META_KEYS

from .test_helpers import StubGame, make_episode, small_cfg, write_dataset

CFG = small_cfg()


@pytest.fixture
def ep():
    return make_episode(CFG, 300)


def has(issues: list[str], prefix: str) -> bool:
    return any(i.startswith(prefix) for i in issues)


def test_clean_episode_has_no_issues(ep):
    assert check_episode(ep, CFG) == []
    assert all(k in ep.meta for k in EPISODE_META_KEYS)


def test_missing_meta_keys(ep):
    del ep.meta["map_uid"], ep.meta["renderer"]
    issues = check_episode(ep, CFG)
    assert has(issues, "meta") and "map_uid" in issues[0] and "renderer" in issues[0]


def test_grid_violations(ep):
    bad = copy.deepcopy(ep)
    bad.frame_times_ms[5:] += 1
    assert has(check_episode(bad, CFG), "grid: frame_times_ms")
    bad = copy.deepcopy(ep)
    bad.action_times_ms[10] += 3
    assert has(check_episode(bad, CFG), "grid: action_times_ms")
    assert has(check_episode(ep, small_cfg(control_hz=50)), "grid: action_times_ms")


def test_non_monotonic_times(ep):
    bad = copy.deepcopy(ep)
    bad.frame_times_ms[7], bad.frame_times_ms[8] = bad.frame_times_ms[8], bad.frame_times_ms[7]
    issues = check_episode(bad, CFG)
    assert has(issues, "monotonic: frame_times_ms") and has(issues, "grid")
    bad = copy.deepcopy(ep)
    bad.action_times_ms[20] = bad.action_times_ms[19]
    assert has(check_episode(bad, CFG), "monotonic: action_times_ms")


def test_length_consistency(ep):
    r = CFG.actions_per_frame
    bad = copy.deepcopy(ep)
    n = len(bad.frames) * r - r - 1  # one too few
    bad.actions, bad.action_times_ms = bad.actions[:n], bad.action_times_ms[:n]
    bad.positions, bad.speeds_kmh = bad.positions[:n], bad.speeds_kmh[:n]
    assert has(check_episode(bad, CFG), "length")
    ok = copy.deepcopy(ep)
    n = len(ok.frames) * r - r  # boundary is allowed
    ok.actions, ok.action_times_ms = ok.actions[:n], ok.action_times_ms[:n]
    ok.positions, ok.speeds_kmh = ok.positions[:n], ok.speeds_kmh[:n]
    assert not has(check_episode(ok, CFG), "length")


def test_shape_mismatch_and_empty(ep):
    bad = copy.deepcopy(ep)
    bad.positions = bad.positions[:-1]
    assert has(check_episode(bad, CFG), "shape")
    bad = copy.deepcopy(ep)
    bad.frames = bad.frames.astype(np.float32)
    assert has(check_episode(bad, CFG), "shape")
    empty = make_episode(CFG, 0)
    assert has(check_episode(empty, CFG), "shape: empty")
    assert has(check_episode(ep, small_cfg(resolution=[64, 48])), "shape: frames")
    assert has(check_episode(ep, small_cfg(channels=1)), "shape: frames")
    assert has(check_episode(ep, small_cfg(frame_hz=30)), "meta: frame_hz")


def test_nonfinite_values(ep):
    for name in ("actions", "positions", "speeds_kmh"):
        bad = copy.deepcopy(ep)
        getattr(bad, name).flat[3] = np.nan
        assert has(check_episode(bad, CFG), "nonfinite"), name
    bad = copy.deepcopy(ep)
    bad.positions[4, 0] = np.inf
    assert has(check_episode(bad, CFG), "nonfinite")


def test_action_ranges(ep):
    for ch, val, label in (
        (0, 1.5, "steer"),
        (0, -1.01, "steer"),
        (1, 1.2, "gas"),
        (2, -0.1, "brake"),
    ):
        bad = copy.deepcopy(ep)
        bad.actions[10, ch] = val
        assert any(i.startswith("range") and label in i for i in check_episode(bad, CFG)), (ch, val)


def test_binary_report_only_when_required(ep):
    bad = copy.deepcopy(ep)
    bad.actions[5:9, 1] = 0.5  # analog throttle
    assert check_episode(bad, CFG) == []
    issues = check_episode(bad, CFG, require_binary=True)
    assert has(issues, "binary: gas") and not any("brake" in i for i in issues)
    assert check_episode(ep, CFG, require_binary=True) == []  # steer may be analog


def test_position_jump_detection(ep):
    assert check_episode(ep, CFG, max_jump=5.0) == []  # the stub car moves < 1.5 per step
    bad = copy.deepcopy(ep)
    bad.positions[60:, 0] += 500.0  # respawn / desync teleport
    issues = check_episode(bad, CFG)
    assert has(issues, "jump") and "1 position jumps" in issues[0]
    assert "ms" in issues[0] and str(int(bad.action_times_ms[60])) in issues[0]
    assert not has(check_episode(bad, CFG, max_jump=1000.0), "jump")
    assert has(check_episode(ep, CFG, max_jump=0.01), "jump")  # threshold is configurable


def test_duplicate_frames(ep):
    bad = copy.deepcopy(ep)
    bad.frames[:] = bad.frames[0]
    assert has(check_episode(bad, CFG), "duplicate_frames")
    half = copy.deepcopy(ep)
    half.frames[: len(half.frames) // 2] = half.frames[0]  # ~48% identical pairs: still fine
    assert not has(check_episode(half, CFG), "duplicate_frames")
    mostly = copy.deepcopy(ep)
    mostly.frames[: len(mostly.frames) * 3 // 4] = mostly.frames[0]
    assert has(check_episode(mostly, CFG), "duplicate_frames")


def test_desync_flag(ep):
    assert not has(check_episode(ep, CFG), "desync")
    ep.meta["desync"] = True
    assert has(check_episode(ep, CFG), "desync")


def test_frame_time_mismatch_is_a_problem(ep):
    assert ep.meta["frame_time_mismatch"] == 0 and check_episode(ep, CFG) == []
    ep.meta["frame_time_mismatch"] = 3
    issues = check_episode(ep, CFG)
    assert has(issues, "sync") and "3 frames" in issues[0]
    lagging = make_episode(CFG, 100, game=StubGame(lag_ticks=1))
    assert lagging.meta["frame_time_mismatch"] > 0 and has(check_episode(lagging, CFG), "sync")


def test_resized_frames_is_only_a_warning(ep):
    ep.meta["resized_frames"] = 7
    assert check_episode(ep, CFG) == ["warn: 7 frames were resized to the data resolution"]
    resized = make_episode(CFG, 100, game=StubGame(size=(64, 48)))
    assert resized.meta["resized_frames"] == len(resized.frames)
    assert check_episode(resized, CFG)[0].startswith("warn:")
    assert split_issues(["warn: x", "grid: y"]) == (["grid: y"], ["warn: x"])


def test_multiple_faults_are_all_reported(ep):
    bad = copy.deepcopy(ep)
    bad.actions[3, 0] = np.nan
    bad.meta["desync"] = True
    del bad.meta["camera"]
    issues = check_episode(bad, CFG)
    assert has(issues, "meta") and has(issues, "nonfinite") and has(issues, "desync")


def build_root(tmp_path, bad_one: bool = False):
    write_dataset(tmp_path, CFG, {"mapA": 300, "mapB": 200, "mapC": 150})
    (tmp_path / "splits.json").write_text(
        json.dumps({"mapA": "train", "mapB": "train", "mapC": "val"})
    )
    if bad_one:
        entry = read_index(tmp_path)[1]
        ep = load_episode(tmp_path / entry["path"])
        ep.meta["desync"] = True
        save_episode(ep, tmp_path)
    return tmp_path


def test_check_dataset_summary(tmp_path):
    root = build_root(tmp_path)
    s = check_dataset(root, CFG)
    assert s["ok"] and s["problems"] == {} and s["index_issues"] == []
    assert s["num_episodes"] == 3 and s["maps"] == ["mapA", "mapB", "mapC"]
    assert s["episodes_per_split"] == {"train": 2, "val": 1, "test": 0}
    frames = {e["map_uid"]: e["num_frames"] for e in read_index(root)}
    assert s["hours"]["train"] == pytest.approx((frames["mapA"] + frames["mapB"]) / 20 / 3600)
    assert s["hours"]["val"] == pytest.approx(frames["mapC"] / 20 / 3600)
    assert s["hours"]["test"] == 0
    assert s["hours"]["total"] == pytest.approx(sum(frames.values()) / 20 / 3600)
    assert s["finished_frac"] == 0.0
    assert 0 <= s["nonbinary"]["steer_nonbinary"] <= 1
    assert s["nonbinary"]["gas_nonbinary"] == 0 and s["nonbinary"]["brake_nonbinary"] == 0
    text = format_report(s)
    assert "episodes: 3" in text and "train=2" in text and text.rstrip().endswith("OK")


def test_check_dataset_reports_problem_episodes(tmp_path):
    root = build_root(tmp_path, bad_one=True)
    s = check_dataset(root, CFG)
    assert not s["ok"] and len(s["problems"]) == 1
    ((path, issues),) = s["problems"].items()
    assert "mapB" in path and has(issues, "desync")
    assert "FAILED" in format_report(s) and path in format_report(s)


def test_check_dataset_missing_file_and_index_mismatch(tmp_path):
    root = build_root(tmp_path)
    entries = read_index(root)
    shutil.rmtree(root / entries[0]["path"])
    ep = load_episode(root / entries[1]["path"])
    append_index(root, ep, root / entries[1]["path"])  # re-index is deduped, no issue
    lines = (root / "index.jsonl").read_text().splitlines()
    lines[2] = json.dumps({**json.loads(lines[2]), "num_frames": 999})
    (root / "index.jsonl").write_text("\n".join(lines) + "\n")
    s = check_dataset(root, CFG)
    assert not s["ok"]
    assert has(s["problems"][entries[0]["path"]], "load:")
    assert any("num_frames" in m for m in s["index_issues"])


def test_check_dataset_warnings_do_not_fail(tmp_path, capsys):
    root = tmp_path
    for i, game in enumerate([None, StubGame(size=(64, 48))]):
        ep = make_episode(CFG, 150, map_uid=f"map{i}", source=f"s{i}", game=game)
        append_index(root, ep, save_episode(ep, root))
    s = check_dataset(root, CFG)
    assert s["ok"] and s["problems"] == {}
    ((path, warns),) = s["warnings"].items()
    assert "map1" in path and warns[0].startswith("warn:")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("data:\n  resolution: [32, 24]\n")
    assert main([str(root), "--config", str(cfg_file)]) == 0  # warn-only: exit 0
    out = capsys.readouterr().out
    assert "WARN" in out and "resized" in out and out.rstrip().endswith("OK")


def test_check_dataset_empty_root(tmp_path):
    s = check_dataset(tmp_path, CFG)
    assert s["ok"] and s["num_episodes"] == 0 and s["hours"]["total"] == 0


def test_cli_exit_codes_and_report(tmp_path, capsys):
    good = build_root(tmp_path / "good")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("data:\n  resolution: [32, 24]\n  val_frac: 0.3\n")
    assert main([str(good), "--config", str(cfg_file)]) == 0
    out = capsys.readouterr().out
    assert "episodes: 3" in out and "OK" in out

    bad = build_root(tmp_path / "bad", bad_one=True)
    assert main([str(bad), "--config", str(cfg_file)]) == 1
    out = capsys.readouterr().out
    assert "PROBLEM" in out and "desync" in out and "FAILED" in out

    # default config has 128x96 frames: every episode is flagged
    assert main([str(good)]) == 1
    capsys.readouterr()
    assert main([str(good), "--config", str(cfg_file), "--max-jump", "0.001"]) == 1
    assert "jump" in capsys.readouterr().out


def test_cli_runs_as_module(tmp_path):
    import subprocess
    import sys

    root = build_root(tmp_path / "ds")
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text("data:\n  resolution: [32, 24]\n")
    proc = subprocess.run(
        [
            sys.executable,
            "-W",
            "error",
            "-m",
            "tmagent.data.quality",
            str(root),
            "--config",
            str(cfg_file),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
