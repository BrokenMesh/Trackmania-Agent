"""tools/render_replays.py: fake runs, replay skip rules, TMI scripts, references, resumability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tmagent.config import Config, load_config
from tmagent.data.episode_io import load_episode, read_index
from tmagent.data.quality import check_dataset
from tmagent.data.split import split_of
from tmagent.data.timeline import timeline_from_events
from tmagent.eval.progress import ReferencePath, load_reference
from tmagent.game import make_game
from tmagent.game.fake import scripted_driver
from tmagent.interfaces import InputTimeline
from tools import render_replays as rr

MAPS = ["fake:oval", "fake:s_curve", "fake:random:3"]
REPO = Path(__file__).resolve().parents[2]
FAKE_YAML = str(REPO / "configs" / "fake.yaml")
TMNF_YAML = str(REPO / "configs" / "tmnf.yaml")


def test_pick_fake_maps_covers_all_splits():
    cfg = load_config(None)
    maps = rr.pick_fake_maps(cfg.data)
    assert {"fake:oval", "fake:s_curve"} <= set(maps) and len(set(maps)) == len(maps) >= 24
    first = [split_of(m, 0, 0.1, 0.1) for m in maps[:3]]
    assert first == ["train", "val", "test"]  # a few runs already cover every split
    count = {"train": 0, "val": 0, "test": 0}
    for m in maps:
        count[split_of(m, cfg.data.split_seed, cfg.data.val_frac, cfg.data.test_frac)] += 1
    assert count["val"] >= 2 and count["test"] >= 2 and count["train"] >= 10
    assert rr.pick_fake_maps(cfg.data) == maps  # deterministic
    # a split seed that starves val/test with 6 maps gets extra seeds
    cfg.data.split_seed = 5
    small = rr.pick_fake_maps(cfg.data, n_maps=6)
    assert {"val", "test"} <= {split_of(m, 5, 0.1, 0.1) for m in small[:3]}
    cfg.data.val_frac = cfg.data.test_frac = 0.0  # no quota to fill
    assert len(rr.pick_fake_maps(cfg.data, n_maps=6)) == 6


def test_fake_run_params_are_deterministic_and_diverse():
    params = [rr.fake_run_params(i) for i in range(60)]
    assert params == [rr.fake_run_params(i) for i in range(60)]  # independent of any run count
    assert rr.fake_run_params(3, seed0=1) != rr.fake_run_params(3, seed0=0)
    assert all(0.6 <= p["skill"] <= 1.0 and 0.0 <= p["noise"] <= 0.3 for p in params)
    assert {p["keyboard"] for p in params} == {True, False}
    assert any(p["noise"] == 0.0 for p in params) and any(p["noise"] > 0.1 for p in params)
    assert len({p["seed"] for p in params}) == 60


def test_render_fake_writes_episodes_refs_and_report(fake_cfg: Config):
    run = rr.render_fake(fake_cfg, 6, maps=MAPS)
    assert run["rendered"] == 6 and run["resumed"] == 0 and run["skipped"] == {}
    root = Path(fake_cfg.data.root)
    entries = read_index(root)
    assert len(entries) == 6 and {e["map_uid"] for e in entries} == set(MAPS)
    assert all(e["finished"] and not e["desync"] for e in entries)
    assert all("resample_loss" in e and e["renderer"].startswith("render_replays") for e in entries)
    assert len({e["episode_id"] for e in entries}) == 6

    rep = rr.finalize(fake_cfg, run)
    assert (root / rr.REPORT_NAME).is_file()
    saved = json.loads((root / rr.REPORT_NAME).read_text())
    assert saved["episodes"] == 6 and saved["maps"] == 3 and saved["desyncs"] == 0
    assert saved["hours"]["total"] > 0 and sum(saved["episodes_per_split"].values()) == 6
    assert saved["frame_time_mismatch"] == {"episodes": 0, "frames": 0}
    assert saved["resample_loss"]["mean_mismatch_frac"] == 0.0  # 100 ms holds survive 60 Hz
    assert sorted(rep["references"]) == sorted(MAPS)

    # quality check passes on the whole dataset
    assert check_dataset(root, fake_cfg.data)["ok"]

    # references: fastest finished run per map, loadable by the eval module
    for uid in MAPS:
        slug = rr.slug(uid)
        positions = np.load(root / "refs" / f"{slug}.npy")
        info = json.loads((root / "refs" / f"{slug}.json").read_text())
        best = min((e for e in entries if e["map_uid"] == uid), key=lambda e: e["race_time_ms"])
        assert info["race_time_ms"] == best["race_time_ms"] and info["source"] == best["source"]
        assert positions.ndim == 2 and positions.shape[1] == 3 and len(positions) > 100
        ref = ReferencePath(positions, time_ms=info["race_time_ms"])
        assert ref.length > 100


def test_references_use_the_fastest_non_desynced_run_and_load_reference(fake_cfg: Config):
    """Two runs of one 'real' map uid: the faster desynced one must not become the reference."""
    root = Path(fake_cfg.data.root)
    game = make_game(fake_cfg.game, fake_cfg.data)
    tl, info = scripted_driver("fake:s_curve", fake_cfg.data)
    sink = rr.Sink(fake_cfg, game, require_finished=False, keep_bad=True)
    base = {"map_uid": "UID_abc-123", "player": "p", "race_time_ms": info["race_time_ms"]}
    for eid, expected in (
        ("slow-ok", info["race_time_ms"]),
        ("desynced", info["race_time_ms"] - 5000),
    ):
        job = rr.Job(eid, tl, "fake:s_curve", {**base, "source": f"file:{eid}"}, expected)
        assert sink.render(job)
    entries = {e["episode_id"]: e for e in read_index(root)}
    assert entries["desynced"]["desync"] and not entries["slow-ok"]["desync"]
    refs = rr.write_references(root)
    assert refs["UID_abc-123"]["episode_id"] == "slow-ok"
    # load_reference resolves refs/<uid>.npy for a non-fake map ref (uid or file name)
    ref = load_reference("UID_abc-123.Challenge.Gbx", root)
    assert ref.time_ms == info["race_time_ms"] and ref.length > 100


def test_render_fake_is_resumable_and_extendable(fake_cfg: Config):
    root = Path(fake_cfg.data.root)
    first = rr.render_fake(fake_cfg, 3, maps=MAPS)
    assert first["rendered"] == 3
    again = rr.render_fake(fake_cfg, 3, maps=MAPS)
    assert again["rendered"] == 0 and again["resumed"] == 3
    more = rr.render_fake(fake_cfg, 5, maps=MAPS)  # only runs 3 and 4 are new
    assert more["rendered"] == 2 and more["resumed"] == 3
    assert len(read_index(root)) == 5
    # --limit stops after N new episodes
    limited = rr.render_fake(fake_cfg, 9, maps=MAPS, limit=1)
    assert limited["rendered"] == 1 and len(read_index(root)) == 6
    # --overwrite re-renders everything requested (index keeps one entry per path)
    over = rr.render_fake(fake_cfg, 2, maps=MAPS, overwrite=True)
    assert over["rendered"] == 2 and len(read_index(root)) == 6


def test_render_fake_hold_ticks_shows_resample_loss(fake_cfg: Config):
    run = rr.render_fake(fake_cfg, 3, maps=MAPS, hold_ticks=2)
    rep = rr.build_report(fake_cfg, run)
    assert rep["resample_loss"]["mean_mismatch_frac"] > 0.0
    assert rep["resample_loss"]["lost_changes"] > 0
    assert "resample loss at 60 Hz" in rr.format_report(rep)


def test_cli_fake(fake_cfg: Config, tmp_path: Path, capsys):
    root = tmp_path / "cli_data"
    argv = ["--config", FAKE_YAML, "--set", f"data.root={root.as_posix()}"]
    argv += ["data.resolution=[32, 24]", "--fake", "3", "--fake-maps", "3"]
    assert rr.main(argv) == 0
    out = capsys.readouterr().out
    assert "3 episodes" in out and "references for" in out
    assert (root / rr.REPORT_NAME).is_file() and len(read_index(root)) == 3
    assert rr.main([*argv, "--limit", "1"]) == 0  # resumes, nothing new
    # --fake needs the fake backend
    assert rr.main(["--config", TMNF_YAML, "--fake", "1"]) == 2


# ----------------------------------------------------------------- TMNF replays (stand-in game)


def actions_to_events(actions: np.ndarray) -> list[tuple[int, str, float]]:
    """Tick actions -> (ms, name, value) events at every change (analog steer, binary rest)."""
    events: list[tuple[int, str, float]] = []
    prev = np.full(3, np.nan)
    for i, a in enumerate(actions):
        for ch, name in enumerate(("steer", "accelerate", "brake")):
            if a[ch] != prev[ch]:
                events.append((i * 10, name, float(a[ch])))
        prev = a
    return events


class ReplayFixture:
    """Fake replay files on disk plus patched loader hooks (no pygbx, fake maps)."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.dir = tmp_path / "replays"
        self.dir.mkdir()
        self.meta: dict[str, dict] = {}
        self.events: dict[str, list] = {}
        self.missing_maps: set[str] = set()
        self.broken: set[str] = set()
        self.loaded: list[str] = []
        monkeypatch.setattr(rr, "_load_replay", self._load)
        monkeypatch.setattr(rr, "_resolve_map", self._resolve)
        monkeypatch.setattr(rr, "_replay_to_timeline", self._timeline)
        self._cache: dict[str, tuple[InputTimeline, dict]] = {}

    def add(self, name: str, map_ref: str = "fake:s_curve", **overrides) -> Path:
        if map_ref not in self._cache:
            self._cache[map_ref] = scripted_driver(map_ref, load_config(None).data, keyboard=True)
        tl, info = self._cache[map_ref]
        self.events[name] = actions_to_events(tl.actions)
        self.meta[name] = {
            "map_uid": map_ref,
            "map_name": map_ref,
            "player": f"p-{name}",
            "race_time_ms": info["race_time_ms"],
            "num_respawns": 0,
            "source": f"file:{name}",
            **overrides,
        }
        path = self.dir / name
        path.write_bytes(name.encode())  # content only feeds the episode id digest
        return path

    def _load(self, path: Path):
        self.loaded.append(path.name)
        if path.name in self.broken:
            raise ValueError("not a replay")
        return list(self.events[path.name]), dict(self.meta[path.name])

    def _resolve(self, map_ref: str, map_dir: str) -> Path:
        if map_ref in self.missing_maps:
            raise FileNotFoundError(f"no map {map_ref}")
        return Path(map_ref)

    @staticmethod
    def _timeline(events, meta) -> InputTimeline:
        rt = meta.get("race_time_ms")
        return timeline_from_events(
            events, duration_ms=-(-rt // 10) * 10 if rt else None, meta=meta
        )


@pytest.fixture
def replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ReplayFixture:
    return ReplayFixture(tmp_path, monkeypatch)


def test_replay_skip_rules(fake_cfg: Config, replays: ReplayFixture):
    good = replays.add("good.Replay.Gbx")
    resp = replays.add("respawns.Replay.Gbx", num_respawns=2, respawns=2)
    unfinished = replays.add("unfinished.Replay.Gbx", race_time_ms=0)
    nomap = replays.add("nomap.Replay.Gbx", map_ref="fake:oval")
    replays.missing_maps.add("fake:oval")
    nouid = replays.add("nouid.Replay.Gbx", map_uid="")
    broken = replays.add("broken.Replay.Gbx")
    replays.broken.add(broken.name)
    paths = [good, resp, unfinished, nomap, nouid, broken]

    run = rr.render_tmnf_replays(fake_cfg, paths)
    assert run["rendered"] == 1
    assert run["skipped"] == {
        "respawns": 1,
        "unfinished": 1,
        "missing_map": 2,
        "load_error": 1,
    }
    entries = read_index(Path(fake_cfg.data.root))
    assert len(entries) == 1
    e = entries[0]
    assert e["map_uid"] == "fake:s_curve" and e["player"] == "p-good.Replay.Gbx"
    assert e["finished"] and not e["desync"] and e["respawns"] == 0
    assert e["expected_time_ms"] == e["race_time_ms"] == e["finish_time_ms"]
    assert e["camera"] == fake_cfg.game.camera and e["resample_loss"]["ticks"] > 0
    ep = load_episode(Path(fake_cfg.data.root) / e["path"])
    assert ep.frames.shape[1:] == (24, 32, 3)

    # --allow-respawns keeps the replay with respawns (flagged in the meta)
    run = rr.render_tmnf_replays(fake_cfg, [resp], allow_respawns=True)
    assert run["rendered"] == 1
    kept = [x for x in read_index(Path(fake_cfg.data.root)) if x["respawns"] == 2]
    assert len(kept) == 1
    # references never use a run with respawns
    assert rr.write_references(fake_cfg.data.root)["fake:s_curve"]["episode_id"] == e["episode_id"]


def test_replay_resumability_and_desync_memory(fake_cfg: Config, replays: ReplayFixture):
    root = Path(fake_cfg.data.root)
    a = replays.add("a.Replay.Gbx", map_ref="fake:s_curve")
    b = replays.add("b.Replay.Gbx", map_ref="fake:oval")
    bad = replays.add("bad.Replay.Gbx", map_ref="fake:s_curve")
    replays.meta["bad.Replay.Gbx"]["race_time_ms"] += 4000  # the replay claims a slower time
    first = rr.render_tmnf_replays(fake_cfg, [a, b, bad])
    assert first["rendered"] == 2 and first["skipped"] == {"desync": 1}
    assert "desync" in first["problems"][next(iter(first["problems"]))][0]
    assert len(read_index(root)) == 2
    assert (root / rr.SKIPS_NAME).is_file()

    # second invocation: nothing is rendered again, the desynced run is remembered
    second = rr.render_tmnf_replays(fake_cfg, [a, b, bad])
    assert second["rendered"] == 0 and second["resumed"] == 3 and second["skipped"] == {}
    # --keep-bad stores the desynced run (flagged), --overwrite re-renders the others
    third = rr.render_tmnf_replays(fake_cfg, [bad], keep_bad=True, overwrite=True)
    assert third["rendered"] == 1
    entries = read_index(root)
    assert len(entries) == 3 and sum(bool(e["desync"]) for e in entries) == 1
    assert rr.build_report(fake_cfg)["desyncs"] == 1
    # the desynced run is excluded from the references
    refs = rr.write_references(root)
    assert (
        refs["fake:s_curve"]["episode_id"]
        != entries[[e["desync"] for e in entries].index(True)]["episode_id"]
    )


def test_replay_limit_and_map_ref_is_the_uid(
    fake_cfg: Config, replays: ReplayFixture, monkeypatch: pytest.MonkeyPatch
):
    paths = [replays.add(f"r{i}.Replay.Gbx", map_ref=m) for i, m in enumerate(MAPS)]
    seen: list[str] = []
    real_render = rr.render_episode

    def spy(game, timeline, map_ref, *a, **kw):
        seen.append(map_ref)
        return real_render(game, timeline, map_ref, *a, **kw)

    monkeypatch.setattr(rr, "render_episode", spy)
    run = rr.render_tmnf_replays(fake_cfg, paths, limit=2)
    assert run["rendered"] == 2 and seen == MAPS[:2]  # the game gets the uid as map_ref


def test_find_replays(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    for name in ("a.Replay.Gbx", "sub/b.replay.gbx", "c.Challenge.Gbx", "sub/d.txt"):
        (tmp_path / name).write_bytes(b"x")
    found = rr.find_replays(str(tmp_path))
    assert [p.name for p in found] == ["a.Replay.Gbx", "b.replay.gbx"]
    assert [p.name for p in rr.find_replays(str(tmp_path / "**" / "*.Replay.Gbx"))] == [
        "a.Replay.Gbx"
    ]
    assert rr.find_replays(str(tmp_path / "a.Replay.Gbx")) == [tmp_path / "a.Replay.Gbx"]
    assert rr.find_replays(str(tmp_path / "nothing*")) == []


# ----------------------------------------------------------------- TMI input scripts


def timeline_to_script(actions: np.ndarray) -> str:
    """Keyboard timeline -> TMI input script text (`<ms> press|rel <key>`)."""
    lines = []
    prev = {"up": False, "down": False, "left": False, "right": False}
    for i, a in enumerate(actions):
        now = {"up": a[1] > 0.5, "down": a[2] > 0.5, "left": a[0] < -0.5, "right": a[0] > 0.5}
        for key, on in now.items():
            if on != prev[key]:
                lines.append(f"{i * 10} {'press' if on else 'rel'} {key}")
        prev = now
    return "# generated\n" + "\n".join(lines) + "\n"


def test_render_tmi_scripts_from_manifest(fake_cfg: Config, tmp_path: Path, monkeypatch):
    pytest.importorskip("tmagent.game.tmnf.replay")
    monkeypatch.setattr(rr, "_resolve_map", lambda ref, d: Path(ref))
    tl, info = scripted_driver("fake:s_curve", fake_cfg.data, keyboard=True)
    sdir = tmp_path / "scripts"
    sdir.mkdir()
    (sdir / "run1.txt").write_text(timeline_to_script(tl.actions))
    (sdir / "bad.txt").write_text("this is not a script\n")
    manifest = sdir / "manifest.jsonl"
    rec = {"map_ref": "fake:s_curve", "map_uid": "UID1", "player": "ana"}
    rec["race_time_ms"] = info["race_time_ms"]
    lines = [
        {"script": "run1.txt", **rec},
        {"script": "missing.txt", **rec},
        {"script": "bad.txt", **rec},
        {"script": "run1.txt", **{**rec, "map_ref": ""}, "map_uid": ""},
    ]
    manifest.write_text("# comment\n" + "\n".join(json.dumps(x) for x in lines) + "\n")

    run = rr.render_tmi_scripts(fake_cfg, manifest)
    assert run["rendered"] == 1
    assert run["skipped"] == {"missing_script": 1, "load_error": 1, "missing_map": 1}
    (e,) = read_index(Path(fake_cfg.data.root))
    assert e["map_uid"] == "UID1" and e["player"] == "ana" and e["source"] == "script:run1.txt"
    assert e["finished"] and not e["desync"]
    assert e["finish_time_ms"] == info["race_time_ms"]
    # resumable
    assert rr.render_tmi_scripts(fake_cfg, manifest)["rendered"] == 0

    with pytest.raises(ValueError, match="invalid JSON"):
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json}\n")
        rr.read_manifest(bad)
    with pytest.raises(ValueError, match="'script'"):
        bad.write_text('{"map_ref": "x"}\n')
        rr.read_manifest(bad)


def test_cli_errors(tmp_path: Path, capsys):
    argv = ["--config", FAKE_YAML, "--set", f"data.root={tmp_path.as_posix()}/d"]
    assert rr.main([*argv, "--replays", str(tmp_path / "none")]) == 2
    assert "no .Replay.Gbx files" in capsys.readouterr().out
    assert rr.main([*argv, "--tmi-scripts", str(tmp_path)]) == 2
    assert "manifest" in capsys.readouterr().out
