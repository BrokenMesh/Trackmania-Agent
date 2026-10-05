"""Phase 1 data generation: replays, TMI input scripts or scripted fake drivers -> episodes.

    python tools/render_replays.py --config configs/tmnf.yaml --replays data/tmnf/replays
    python tools/render_replays.py --config configs/tmnf.yaml \
        --tmi-scripts DIR --manifest DIR/manifest.jsonl
    python tools/render_replays.py --config configs/fake.yaml --fake 200

Every run is re-driven on the game (tmagent.data.render.render_episode), checked with
tmagent.data.quality.check_episode and stored under cfg.data.root (episodes/, index.jsonl).
Afterwards refs/<map_uid>.npy|json (the fastest finished, non-desynced run per map) and
render_report.json are written. Resumable: episode ids already in the index are skipped
unless --overwrite; runs rejected after rendering (desync, quality problems) are remembered
in render_skips.jsonl so they are not re-rendered either.

TMI manifest (JSON lines): {"script": path, "map_ref": ..., "map_uid": ..., "player": ...,
"race_time_ms": ...}; a relative script path is relative to --tmi-scripts.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import Config, DataConfig, load_config
from tmagent.data.episode_io import EpisodeReader, append_index, read_index, save_episode
from tmagent.data.quality import check_episode, split_issues
from tmagent.data.render import render_episode
from tmagent.data.split import SPLITS, load_overrides, split_for_map, split_of
from tmagent.data.timeline import resample_loss
from tmagent.experiment import git_info
from tmagent.game import make_game
from tmagent.game.fake import scripted_driver
from tmagent.interfaces import InputTimeline, SyncGame

SKIPS_NAME = "render_skips.jsonl"
REPORT_NAME = "render_report.json"
MAX_CONSECUTIVE_ERRORS = 3  # stop when the game keeps failing (it is probably gone)
FAKE_DEFAULT_MAPS = 24  # fake maps used by --fake (oval, s_curve and random tracks)
FAKE_MIN_SPLIT_MAPS = 2  # at least this many val and test maps among the fake maps


# ---------------------------------------------------------------- lazy TMNF hooks
# Thin wrappers so that the optional replay stack is imported only when needed (and so
# that tests can replace them).


def _load_replay(path: Path) -> tuple[list[tuple[int, str, float]], dict[str, Any]]:
    from tmagent.game.tmnf.replay import load_replay

    return load_replay(path)


def _parse_script(text: str) -> list[tuple[int, str, float]]:
    from tmagent.game.tmnf.replay import parse_tmi_input_script

    return parse_tmi_input_script(text)


def _replay_to_timeline(events: Any, meta: dict[str, Any]) -> InputTimeline:
    from tmagent.game.tmnf.replay import replay_to_timeline

    return replay_to_timeline(events, meta)


def _resolve_map(map_ref: str, map_dir: str) -> Path:
    from tmagent.game.tmnf.game import resolve_map

    return resolve_map(map_ref, map_dir)


# ---------------------------------------------------------------- helpers


def slug(name: str) -> str:
    """File-system safe version of a map uid / id."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name)).lstrip(".") or "_"


def _digest(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:10]


def _renderer_tag() -> str:
    for line in git_info().splitlines():
        if line.startswith("commit: "):
            return f"render_replays@{line[8:15]}"
    return "render_replays"


@dataclass
class Job:
    """One run to render."""

    episode_id: str
    timeline: InputTimeline
    map_ref: str
    meta: dict[str, Any]
    expected_time_ms: int | None = None


@dataclass
class Sink:
    """Renders jobs into cfg.data.root and keeps the bookkeeping of one invocation."""

    cfg: Config
    game: SyncGame
    overwrite: bool = False
    keep_bad: bool = False  # keep episodes with quality problems / desync / unfinished
    require_finished: bool = True
    limit: int | None = None  # stop after this many new episodes
    desync_retries: int = 1  # re-render a desynced run this often before rejecting it
    rendered: list[str] = field(default_factory=list)
    resumed: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)
    renderer: str = field(default_factory=_renderer_tag)
    errors: int = 0

    def __post_init__(self) -> None:
        self.root = Path(self.cfg.data.root)
        self.known = {e["episode_id"] for e in read_index(self.root) if "episode_id" in e}
        self.rejected = self._read_rejected()
        self.t0 = time.perf_counter()

    def _read_rejected(self) -> dict[str, str]:
        path = self.root / SKIPS_NAME
        if self.overwrite or not path.exists():
            return {}
        out = {}
        for line in path.read_text().splitlines():
            try:
                rec = json.loads(line)
                out[rec["episode_id"]] = rec["reason"]
            except (ValueError, KeyError, TypeError):
                continue
        return out

    def is_done(self, episode_id: str) -> bool:
        """Already stored (or rejected earlier) and not to be re-rendered; counts as resumed."""
        done = not self.overwrite and (episode_id in self.known or episode_id in self.rejected)
        self.resumed += done
        return done

    @property
    def full(self) -> bool:
        return self.limit is not None and len(self.rendered) >= self.limit

    def skip(self, name: str, reason: str, persist_id: str | None = None) -> None:
        """Log a skipped run; with persist_id it is remembered across invocations."""
        print(f"  skip {name}: {reason}")
        self.skipped.append({"name": name, "reason": reason})
        if persist_id is not None:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(self.root / SKIPS_NAME, "a") as f:
                f.write(json.dumps({"episode_id": persist_id, "reason": reason}) + "\n")

    def render(self, job: Job) -> bool:
        """Render, check and store one job; True if an episode was written."""
        cfg, eid = self.cfg, job.episode_id
        if self.is_done(eid):
            return False
        meta = {
            **job.meta,
            "episode_id": eid,
            "camera": job.meta.get("camera", cfg.game.camera),
            "renderer": self.renderer,
            "expected_time_ms": job.expected_time_ms,
            "resample_loss": resample_loss(job.timeline, cfg.data.control_hz),
        }
        t0 = time.perf_counter()
        try:
            ep = self._render_with_retries(job, meta)
        except Exception as exc:
            self.errors += 1
            self.skip(eid, f"render_error: {type(exc).__name__}: {exc}")
            if self.errors >= MAX_CONSECUTIVE_ERRORS:
                raise RuntimeError(f"{self.errors} render errors in a row, giving up") from exc
            return False
        self.errors = 0
        problems, _ = split_issues(check_episode(ep, cfg.data))
        if self.require_finished and not ep.meta["finished"]:
            problems.append("unfinished: the re-driven run never reported finished")
        dt = time.perf_counter() - t0
        flags = f"finished={ep.meta['finished']} desync={ep.meta['desync']}"
        print(
            f"  {eid}: {len(ep.frames)} frames, {ep.meta['race_time_ms'] / 1000:.2f} s, "
            f"{flags}, {dt:.1f} s"
        )
        if problems:
            self.problems[eid] = problems
            if not self.keep_bad:
                self.skip(eid, "; ".join(problems), persist_id=eid)
                return False
        append_index(self.root, ep, save_episode(ep, self.root))
        self.known.add(eid)
        self.rendered.append(eid)
        return True

    def _render_with_retries(self, job: Job, meta: dict[str, Any]):
        """VERIFIED need (TMNF): runs that desynced in a long render re-drove exactly on later
        attempts after a fresh map load, so a desync is retried (with a map reload when the
        game supports it) before the run is rejected."""
        for attempt in range(self.desync_retries + 1):
            ep = render_episode(
                self.game,
                job.timeline,
                job.map_ref,
                self.cfg.data,
                meta,
                expected_time_ms=job.expected_time_ms,
            )
            if not ep.meta["desync"]:
                return ep
            if attempt < self.desync_retries:
                print(f"  {job.episode_id}: desync, retrying ({attempt + 1}/{self.desync_retries})")
                forget_loaded_map = getattr(self.game, "forget_loaded_map", None)
                if forget_loaded_map is not None:
                    forget_loaded_map()
        return ep

    def summary(self) -> dict[str, Any]:
        reasons = Counter(s["reason"].split(":")[0] for s in self.skipped)
        return {
            "rendered": len(self.rendered),
            "resumed": self.resumed,
            "skipped": dict(reasons),
            "skip_list": self.skipped,
            "problems": self.problems,
            "elapsed_s": time.perf_counter() - self.t0,
        }


def _game_for(cfg: Config, game: SyncGame | None) -> tuple[SyncGame, bool]:
    return (game, False) if game is not None else (make_game(cfg.game, cfg.data), True)  # type: ignore[return-value]


# ---------------------------------------------------------------- fake drivers


def pick_fake_maps(
    data: DataConfig, n_maps: int = FAKE_DEFAULT_MAPS, min_split: int = FAKE_MIN_SPLIT_MAPS
) -> list[str]:
    """fake:oval, fake:s_curve and random tracks (ascending seeds from 0), split-balanced order.

    Seeds are added beyond n_maps until the hash split of the config has at least
    `min_split` val and test maps (when val_frac / test_frac are > 0), so every split
    exists. The list starts with one train, one val and one test map (so that even a few
    runs cover all splits), the others follow in discovery order. Deterministic for a
    given config.
    """
    maps = ["fake:oval", "fake:s_curve"]

    def split(m: str) -> str:
        return split_of(m, data.split_seed, data.val_frac, data.test_frac)

    need = {"val": data.val_frac > 0, "test": data.test_frac > 0}
    count = Counter(split(m) for m in maps)
    seed = 0
    while seed < 20 * max(n_maps, 1):
        if len(maps) >= n_maps and all(count[s] >= min_split for s in need if need[s]):
            break
        m = f"fake:random:{seed}"
        s = split(m)
        # past n_maps only maps that fill a missing val/test quota are added
        if len(maps) < n_maps or (need.get(s) and count[s] < min_split):
            maps.append(m)
            count[s] += 1
        seed += 1
    head = [next((m for m in maps if split(m) == s), None) for s in SPLITS]
    head = [m for m in head if m is not None]
    return head + [m for m in maps if m not in head]


def fake_run_params(index: int, seed0: int = 0) -> dict[str, Any]:
    """Driver parameters of fake run `index`: skill 0.6..1.0, noise 0..0.3, keyboard or
    analog steering. Deterministic in (index, seed0) and independent of the run count."""
    rng = np.random.default_rng([int(seed0), int(index)])
    noise = 0.0 if rng.random() < 0.25 else float(rng.uniform(0.02, 0.3))
    return {
        "skill": round(float(rng.uniform(0.6, 1.0)), 3),
        "noise": round(noise, 3),
        "keyboard": bool(rng.integers(2)),
        "seed": int(index) + 100_003 * int(seed0),
    }


def fake_episode_id(map_ref: str, index: int, seed0: int = 0) -> str:
    return f"{slug(map_ref)}-{index + 100_000 * seed0:05d}"


def fake_job(cfg: Config, index: int, map_ref: str, seed0: int = 0, hold_ticks: int = 10) -> Job:
    """Scripted driver run `index` on `map_ref` (parameters from fake_run_params)."""
    timeline, info = scripted_driver(
        map_ref, cfg.data, hold_ticks=hold_ticks, **fake_run_params(index, seed0)
    )
    meta = {
        **timeline.meta,
        "camera": "fake",
        "map_name": map_ref,
        "race_time_ms": info["race_time_ms"],
    }
    expected = info["race_time_ms"] if info["finished"] else None
    return Job(fake_episode_id(map_ref, index, seed0), timeline, map_ref, meta, expected)


def render_fake(
    cfg: Config,
    n: int,
    maps: list[str] | None = None,
    seed0: int = 0,
    hold_ticks: int = 10,
    game: SyncGame | None = None,
    **sink_kw: Any,
) -> dict[str, Any]:
    """Render n scripted fake runs into cfg.data.root; returns the Sink summary.

    Run i drives maps[i % len(maps)] (default pick_fake_maps) with the parameters of
    fake_run_params(i), so a larger n later only adds runs. Unfinished runs are kept (they
    are legitimate low-quality drivers). sink_kw: overwrite, keep_bad, limit.
    """
    maps = maps or pick_fake_maps(cfg.data)
    game, own = _game_for(cfg, game)
    sink = Sink(cfg, game, require_finished=False, **sink_kw)
    try:
        for i in range(n):
            if sink.full:
                break
            map_ref = maps[i % len(maps)]
            print(f"[{i + 1}/{n}] {map_ref}")
            if not sink.is_done(fake_episode_id(map_ref, i, seed0)):
                sink.render(fake_job(cfg, i, map_ref, seed0, hold_ticks))
    finally:
        if own:
            game.close()
    return sink.summary()


# ---------------------------------------------------------------- TMNF replays


def find_replays(spec: str) -> list[Path]:
    """*.Replay.Gbx files of a directory (recursive), a glob pattern or a single file."""
    p = Path(spec).expanduser()
    if p.is_dir():
        found = [f for f in p.rglob("*") if f.name.lower().endswith(".replay.gbx")]
    elif p.is_file():
        found = [p]
    else:
        found = [Path(f) for f in glob.glob(str(p), recursive=True) if Path(f).is_file()]
    return sorted(found)


def replay_job(
    path: Path, cfg: Config, allow_respawns: bool = False
) -> tuple[Job | None, str | None]:
    """(job, None) or (None, skip reason) for one .Replay.Gbx file."""
    try:
        events, meta = _load_replay(path)
    except ImportError:
        raise
    except Exception as exc:
        return None, f"load_error: {type(exc).__name__}: {exc}"
    uid = str(meta.get("map_uid") or "")
    respawns = max(int(meta.get("num_respawns") or 0), int(meta.get("respawns") or 0))
    race_time = int(meta.get("race_time_ms") or 0)
    if respawns > 0 and not allow_respawns:
        return None, f"respawns: {respawns} respawn(s) in the replay"
    if race_time <= 0:
        return None, "unfinished: replay has no finish time"
    if not uid:
        return None, "missing_map: replay carries no map uid"
    try:
        _resolve_map(uid, cfg.game.map_dir)
    except (FileNotFoundError, ValueError) as exc:
        return None, f"missing_map: {exc}"
    meta = {**meta, "respawns": respawns, "race_time_ms": race_time}
    timeline = _replay_to_timeline(events, meta)
    eid = f"{slug(uid)}-{_digest(path.read_bytes())}"
    return Job(eid, timeline, uid, meta, expected_time_ms=race_time), None


def render_tmnf_replays(
    cfg: Config,
    paths: Iterable[Path],
    allow_respawns: bool = False,
    game: SyncGame | None = None,
    **sink_kw: Any,
) -> dict[str, Any]:
    """Render .Replay.Gbx files into cfg.data.root; returns the Sink summary.

    Skipped (and logged): respawns > 0 unless allow_respawns, unfinished replays, replays
    whose map file cannot be found in game.map_dir, unreadable files. The map is passed to
    the game as its uid (TMNFGame resolves it through game.map_dir).
    """
    paths = list(paths)
    game, own = _game_for(cfg, game)
    sink = Sink(cfg, game, **sink_kw)
    try:
        for i, path in enumerate(paths, 1):
            if sink.full:
                break
            print(f"[{i}/{len(paths)}] {path.name}")
            job, reason = replay_job(path, cfg, allow_respawns)
            if job is None:
                sink.skip(path.name, reason or "unknown")
                continue
            sink.render(job)
    finally:
        if own:
            game.close()
    return sink.summary()


# ---------------------------------------------------------------- TMI input scripts


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """JSON-lines manifest (blank lines and # comments ignored)."""
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{n}: invalid JSON ({exc})") from exc
        if "script" not in rec:
            raise ValueError(f"{path}:{n}: manifest line needs a 'script' key")
        out.append(rec)
    return out


def script_job(
    rec: dict[str, Any], scripts_dir: Path, cfg: Config
) -> tuple[Job | None, str | None]:
    """(job, None) or (None, skip reason) for one manifest record."""
    script = Path(rec["script"])
    if not script.is_absolute():
        script = scripts_dir / script
    if not script.is_file():
        return None, f"missing_script: {script}"
    text = script.read_text(encoding="utf-8")
    try:
        events = _parse_script(text)
    except ValueError as exc:
        return None, f"load_error: {exc}"
    map_ref = str(rec.get("map_ref") or rec.get("map_uid") or "")
    uid = str(rec.get("map_uid") or map_ref)
    if not map_ref:
        return None, "missing_map: manifest line has no map_ref / map_uid"
    try:
        _resolve_map(map_ref, cfg.game.map_dir)
    except (FileNotFoundError, ValueError) as exc:
        return None, f"missing_map: {exc}"
    race_time = int(rec["race_time_ms"]) if rec.get("race_time_ms") else None
    meta = {
        "map_uid": uid,
        "map_name": str(rec.get("map_name") or uid),
        "player": str(rec.get("player") or ""),
        "source": f"script:{script.name}",
        "race_time_ms": race_time,
    }
    timeline = _replay_to_timeline(events, meta)
    eid = f"{slug(uid)}-{_digest(text.encode())}"
    return Job(eid, timeline, map_ref, meta, expected_time_ms=race_time), None


def render_tmi_scripts(
    cfg: Config,
    manifest: Path,
    scripts_dir: Path | None = None,
    game: SyncGame | None = None,
    **sink_kw: Any,
) -> dict[str, Any]:
    """Render the TMI input scripts listed in a manifest; returns the Sink summary."""
    recs = read_manifest(manifest)
    scripts_dir = scripts_dir if scripts_dir is not None else manifest.parent
    game, own = _game_for(cfg, game)
    sink = Sink(cfg, game, **sink_kw)
    try:
        for i, rec in enumerate(recs, 1):
            if sink.full:
                break
            print(f"[{i}/{len(recs)}] {rec['script']}")
            job, reason = script_job(rec, scripts_dir, cfg)
            if job is None:
                sink.skip(str(rec["script"]), reason or "unknown")
                continue
            sink.render(job)
    finally:
        if own:
            game.close()
    return sink.summary()


# ---------------------------------------------------------------- references and report


def write_references(root: str | Path) -> dict[str, dict[str, Any]]:
    """refs/<map_uid>.npy (+ .json) from the fastest finished, non-desynced run of each map.

    Reads the index (so earlier invocations count), skips runs with respawns. The .npy holds
    the control-rate car positions (N, 3); the .json {"race_time_ms", "source", ...} matches
    tmagent.eval.progress.load_reference. Returns {map_uid: json content}.
    """
    root = Path(root)
    best: dict[str, dict[str, Any]] = {}
    for e in read_index(root):
        if not e.get("finished") or e.get("desync") or e.get("respawns", 0) > 0:
            continue
        uid = e["map_uid"]
        if uid not in best or e["race_time_ms"] < best[uid]["race_time_ms"]:
            best[uid] = e
    refs = root / "refs"
    out: dict[str, dict[str, Any]] = {}
    for uid, e in sorted(best.items()):
        positions = EpisodeReader(root / e["path"]).positions
        if len(positions) < 2:
            continue
        refs.mkdir(parents=True, exist_ok=True)
        np.save(refs / f"{slug(uid)}.npy", positions.astype(np.float32))
        info = {
            "race_time_ms": int(e["race_time_ms"]),
            "source": e.get("source", ""),
            "episode_id": e.get("episode_id", ""),
            "player": e.get("player", ""),
        }
        (refs / f"{slug(uid)}.json").write_text(json.dumps(info))
        out[uid] = info
    return out


def build_report(cfg: Config, run: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dataset-wide summary from the index (no frame decoding) plus this invocation's `run`."""
    root = Path(cfg.data.root)
    entries = read_index(root)
    overrides = load_overrides(root)
    split_of_map = {e["map_uid"]: split_for_map(e["map_uid"], cfg.data, overrides) for e in entries}
    frames = dict.fromkeys(SPLITS, 0)
    episodes = dict.fromkeys(SPLITS, 0)
    for e in entries:
        s = split_of_map[e["map_uid"]]
        frames[s] += int(e.get("num_frames", 0))
        episodes[s] += 1
    hours = {s: frames[s] / cfg.data.frame_hz / 3600 for s in SPLITS}
    hours["total"] = sum(hours.values())
    losses = [e["resample_loss"] for e in entries if isinstance(e.get("resample_loss"), dict)]
    mm = [e.get("frame_time_mismatch", 0) for e in entries]
    return {
        "root": str(root),
        "episodes": len(entries),
        "episodes_per_split": episodes,
        "hours": hours,
        "maps": len(split_of_map),
        "maps_per_split": {s: sum(v == s for v in split_of_map.values()) for s in SPLITS},
        "finished": sum(bool(e.get("finished")) for e in entries),
        "desyncs": sum(bool(e.get("desync")) for e in entries),
        "frame_time_mismatch": {"episodes": sum(m > 0 for m in mm), "frames": int(sum(mm))},
        "resample_loss": {
            "control_hz": cfg.data.control_hz,
            "mean_mismatch_frac": float(np.mean([r["mismatch_frac"] for r in losses]))
            if losses
            else 0.0,
            "mean_lost_changes_frac": float(np.mean([r["lost_changes_frac"] for r in losses]))
            if losses
            else 0.0,
            "lost_changes": int(sum(r["lost_changes"] for r in losses)),
            "changes": int(sum(r["changes"] for r in losses)),
        },
        "this_run": run or {},
    }


def format_report(rep: dict[str, Any]) -> str:
    h, rl, run = rep["hours"], rep["resample_loss"], rep["this_run"]
    lines = [
        f"dataset {rep['root']}: {rep['episodes']} episodes, {h['total']:.3f} h, "
        f"{rep['maps']} maps, {rep['finished']} finished",
        "  episodes per split: " + ", ".join(f"{s}={rep['episodes_per_split'][s]}" for s in SPLITS),
        "  maps per split:     " + ", ".join(f"{s}={rep['maps_per_split'][s]}" for s in SPLITS),
        "  hours per split:    " + ", ".join(f"{s}={h[s]:.3f}" for s in SPLITS),
        f"  desyncs: {rep['desyncs']}   frame_time_mismatch: "
        f"{rep['frame_time_mismatch']['episodes']} episodes "
        f"({rep['frame_time_mismatch']['frames']} frames)",
        f"  resample loss at {rl['control_hz']} Hz: mean tick mismatch "
        f"{rl['mean_mismatch_frac']:.4f}, lost input changes {rl['lost_changes']}/{rl['changes']}",
    ]
    if run:
        skipped = ", ".join(f"{k}={v}" for k, v in run["skipped"].items()) or "none"
        lines.append(
            f"  this run: rendered {run['rendered']}, already done {run['resumed']}, "
            f"skipped: {skipped}, {run['elapsed_s']:.1f} s"
        )
        for eid, issues in run["problems"].items():
            lines.append(f"  quality {eid}: " + "; ".join(issues))
    return "\n".join(lines)


def finalize(cfg: Config, run: dict[str, Any]) -> dict[str, Any]:
    """Write references and render_report.json, print the summary; returns the report."""
    refs = write_references(cfg.data.root)
    rep = build_report(cfg, run)
    rep["references"] = sorted(refs)
    root = Path(cfg.data.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / REPORT_NAME).write_text(json.dumps(rep, indent=2))
    print(format_report(rep))
    print(f"references for {len(refs)} maps in {root / 'refs'}; report in {root / REPORT_NAME}")
    return rep


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config (game, data sections)")
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="k=v"
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replays", metavar="DIR_OR_GLOB", help="*.Replay.Gbx directory or glob")
    src.add_argument("--tmi-scripts", metavar="DIR", help="directory of TMI input scripts")
    src.add_argument("--fake", type=int, metavar="N", help="N scripted runs on the fake game")
    ap.add_argument("--manifest", default=None, help="manifest.jsonl (default DIR/manifest.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N new episodes")
    ap.add_argument("--overwrite", action="store_true", help="re-render episodes already stored")
    ap.add_argument("--allow-respawns", action="store_true", help="keep replays with respawns")
    ap.add_argument(
        "--desync-retries", type=int, default=1, help="re-render a desynced run N times first"
    )
    ap.add_argument(
        "--keep-bad", action="store_true", help="keep episodes failing the quality check"
    )
    ap.add_argument("--seed", type=int, default=0, help="--fake: driver seed offset")
    ap.add_argument("--fake-maps", type=int, default=FAKE_DEFAULT_MAPS, help="--fake: map count")
    ap.add_argument("--hold-ticks", type=int, default=10, help="--fake: driver hold (10 ms ticks)")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    kw: dict[str, Any] = {
        "overwrite": args.overwrite,
        "keep_bad": args.keep_bad,
        "limit": args.limit,
        "desync_retries": args.desync_retries,
    }
    if args.fake is not None:
        if cfg.game.backend != "fake":
            print("error: --fake needs game.backend fake (use configs/fake.yaml)")
            return 2
        maps = pick_fake_maps(cfg.data, args.fake_maps)
        run = render_fake(
            cfg, args.fake, maps=maps, seed0=args.seed, hold_ticks=args.hold_ticks, **kw
        )
    elif args.replays is not None:
        paths = find_replays(args.replays)
        if not paths:
            print(f"error: no .Replay.Gbx files found for {args.replays!r}")
            return 2
        try:
            run = render_tmnf_replays(cfg, paths, allow_respawns=args.allow_respawns, **kw)
        except ImportError as exc:  # pygbx missing
            print(f"error: {exc}")
            return 2
    else:
        scripts = Path(args.tmi_scripts)
        manifest = Path(args.manifest) if args.manifest else scripts / "manifest.jsonl"
        if not manifest.is_file():
            print(f"error: manifest {manifest} not found (see --manifest)")
            return 2
        run = render_tmi_scripts(cfg, manifest, scripts, **kw)
    finalize(cfg, run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
