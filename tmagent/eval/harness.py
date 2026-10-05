"""Closed-loop evaluation: a policy drives maps x seeds, metrics go to JSONL + summary.

CLI: python -m tmagent.eval.harness --config X.yaml --ckpt path.pt [--set k=v ...] [--out dir]
"""

from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_left
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from tmagent.config import Config, DataConfig, load_config
from tmagent.eval.progress import ProgressTracker, ReferencePath, load_reference
from tmagent.game import make_game
from tmagent.interfaces import PHYSICS_TICK_MS, Action, ChunkPolicy, RealtimeGame, SyncGame

POLL_HZ = 20.0  # realtime mode: how often the harness reads the game state


@lru_cache(maxsize=32)
def _fake_reference_time_ms(map_ref: str) -> int | None:
    from tmagent.game.fake import scripted_driver

    _, info = scripted_driver(map_ref, DataConfig(), noise=0.0)
    return info["race_time_ms"] if info["finished"] else None


def reference_time_ms(map_ref: str, ref: ReferencePath) -> int | None:
    """Reference run time of a map if known: `ref.time_ms`, or the noise-free scripted
    driver on fake maps (cached)."""
    if ref.time_ms is not None:
        return ref.time_ms
    return _fake_reference_time_ms(map_ref) if map_ref.startswith("fake:") else None


def _episode_result(
    map_ref: str, seed: int, mode: str, tracker: ProgressTracker, ref: ReferencePath
) -> dict:
    res = tracker.result()
    ref_time = reference_time_ms(map_ref, ref)
    ratio = None
    if res["finished"] and ref_time:
        ratio = res["finish_time_ms"] / ref_time
    return {"map": map_ref, "seed": seed, "mode": mode, **res, "time_ratio": ratio}


def control_row_for_tick(tick_time_ms: int, frame_index: int, control_hz: int, r: int) -> int:
    """Chunk row (0..r-1) applied during the physics tick that starts at `tick_time_ms`.

    Control step i = frame_index * r + j sits at c_i = round(i * 1000 / control_hz) and its
    action is the timeline input of tick c_i // PHYSICS_TICK_MS (sample-and-hold, see
    tmagent.data.timeline.resample_to_control). The exact inverse: tick [t, t + 10) uses the
    latest row whose c_i < t + 10, so that tick c_i // 10 always runs row j. At 60 Hz the
    ticks of a 20 Hz frame (t = 0, 10, 20, 30, 40) use rows 0, 1, 1, 2, 2.
    """
    times = [round((frame_index * r + j) * 1000 / control_hz) for j in range(r)]
    return min(max(bisect_left(times, tick_time_ms + PHYSICS_TICK_MS) - 1, 0), r - 1)


def run_episode_sync(
    game: SyncGame,
    policy: ChunkPolicy,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
) -> dict:
    """One deterministic episode on a SyncGame; resets the policy first.

    Frame k is grabbed at race time round(k * 1000 / frame_hz). The policy then predicts
    a chunk whose row j is the input of control step i = k * R + j (time
    round(i * 1000 / control_hz)); each physics tick runs the row picked by
    `control_row_for_tick`; rows past the chunk hold the last row.
    The actions executed in a frame period (with eval-time steer noise, if configured)
    are handed to the next observe().
    """
    d, ev = cfg.data, cfg.eval
    r = d.actions_per_frame
    rng = np.random.default_rng(seed)
    policy.reset()
    game.load_map(map_ref)
    state = game.start_race()
    tracker = ProgressTracker(ref, ev)
    tracker.update(state)
    past = np.zeros((r, 3), np.float32)
    k = 0
    while not tracker.done:
        frame = game.grab_frame()
        policy.observe(frame.image, past)
        chunk = np.asarray(policy.predict(), dtype=np.float32).reshape(-1, 3)
        rows = chunk[np.minimum(np.arange(r), len(chunk) - 1)].copy()
        if ev.action_noise > 0.0:
            rows[:, 0] += rng.normal(0.0, ev.action_noise, r).astype(np.float32)
        rows[:, 0] = np.clip(rows[:, 0], -1.0, 1.0)
        rows[:, 1:] = np.clip(rows[:, 1:], 0.0, 1.0)
        t_next = round((k + 1) * 1000 / d.frame_hz)
        while state.race_time_ms < t_next and not tracker.done:
            j = control_row_for_tick(state.race_time_ms, k, d.control_hz, r)
            prev_t = state.race_time_ms
            state = game.step(Action.from_array(rows[j]), 1)
            if state.race_time_ms <= prev_t and not state.finished:
                raise RuntimeError("game did not advance the race time")
            tracker.update(state)
        past = rows
        k += 1
    return {**_episode_result(map_ref, seed, "sync", tracker, ref), "frames": k}


def run_episode_realtime(
    game: RealtimeGame,
    policy: ChunkPolicy,
    map_ref: str,
    cfg: Config,
    seed: int,
    ref: ReferencePath,
) -> dict:
    """One live episode: LiveSession plays while the harness polls the state at 20 Hz.

    The result carries LiveSession.stats() (deadline misses etc.) under "stats".
    """
    from tmagent.runtime.session import LiveSession  # lazy: optional runtime dependency

    ev = cfg.eval
    policy.reset()
    game.load_map(map_ref)
    tracker = ProgressTracker(ref, ev)
    session = LiveSession(game, policy, cfg)
    game.restart()
    speed = max(float(getattr(cfg.game, "game_speed", 1.0)), 1e-3)
    wall_limit = time.perf_counter() + ev.timeout_s / speed + 10.0
    session.start()
    try:
        while not tracker.done:
            tracker.update(game.get_state())
            if time.perf_counter() > wall_limit:
                tracker.finish("timeout")
            elif not tracker.done:
                time.sleep(1.0 / POLL_HZ)
    finally:
        session.stop()
    return {**_episode_result(map_ref, seed, "realtime", tracker, ref), "stats": session.stats()}


def _summarize(episodes: list[dict]) -> dict:
    def med(xs: list[float]) -> float | None:
        return float(np.median(xs)) if xs else None

    def agg(eps: list[dict]) -> dict:
        prog = [e["progress"] for e in eps]
        ratios = [e["time_ratio"] for e in eps if e.get("time_ratio") is not None]
        return {
            "n_episodes": len(eps),
            "finish_rate": float(np.mean([e["finished"] for e in eps])) if eps else 0.0,
            "median_progress": med(prog),
            "mean_progress": float(np.mean(prog)) if prog else None,
            "median_time_ratio": med(ratios),
            "mean_dist": float(np.mean([e["mean_dist"] for e in eps])) if eps else None,
            "max_dist": float(max((e["max_dist"] for e in eps), default=0.0)),
        }

    out = agg(episodes)
    out["reasons"] = {
        r: sum(e["reason"] == r for e in episodes)
        for r in dict.fromkeys(e["reason"] for e in episodes)
    }
    stats = [e["stats"] for e in episodes if "stats" in e]
    misses = sum(s.get("deadline_misses", 0) for s in stats)
    slots = sum(s.get("slots", 0) for s in stats)
    if slots:
        out["deadline_miss_pct"] = 100.0 * misses / slots
    elif stats:
        out["deadline_miss_pct"] = float(np.mean([s.get("miss_pct", 0.0) for s in stats]))
    else:
        out["deadline_miss_pct"] = None
    out["per_map"] = {
        m: agg([e for e in episodes if e["map"] == m])
        for m in dict.fromkeys(e["map"] for e in episodes)
    }
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def evaluate(
    policy_factory: Callable[[], ChunkPolicy],
    cfg: Config,
    game: SyncGame | RealtimeGame | None = None,
    out_dir: str | Path | None = None,
) -> dict:
    """Run cfg.eval.maps x cfg.eval.seeds in cfg.eval.mode and summarize.

    The policy is built once and reset per episode. If `game` is None one is made from
    cfg.game (and closed afterwards). With `out_dir`, writes episodes.jsonl (one line per
    episode, as they finish) and summary.json. Returns the summary plus an "episodes" list.
    """
    ev = cfg.eval
    if ev.mode not in ("sync", "realtime"):
        raise ValueError(f"unknown eval.mode {ev.mode!r} (expected 'sync' or 'realtime')")
    maps = list(ev.maps)
    if not maps:
        if cfg.game.backend != "fake":
            raise ValueError("eval.maps is empty")
        maps = [f"fake:{cfg.game.fake_track}"]
    policy = policy_factory()
    own_game = game is None
    if game is None:
        game = make_game(cfg.game, cfg.data)
    run = run_episode_sync if ev.mode == "sync" else run_episode_realtime
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    try:
        log = open(out / "episodes.jsonl", "w") if out is not None else None
        try:
            for map_ref in maps:
                ref = load_reference(map_ref, cfg.data.root)
                for seed in ev.seeds:
                    ep = run(game, policy, map_ref, cfg, seed, ref)  # type: ignore[arg-type]
                    episodes.append(ep)
                    if log is not None:
                        log.write(json.dumps(ep, default=_json_default) + "\n")
                        log.flush()
        finally:
            if log is not None:
                log.close()
    finally:
        if own_game:
            game.close()
    summary = {"name": cfg.name, "mode": ev.mode, **_summarize(episodes)}
    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    return {**summary, "episodes": episodes}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Closed-loop evaluation of a checkpoint.")
    ap.add_argument("--config", required=True, help="YAML config")
    ap.add_argument("--ckpt", required=True, help="checkpoint (.pt) to evaluate")
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="K=V"
    )
    ap.add_argument("--out", default=None, help="output dir (default: a new experiments/ run)")
    args = ap.parse_args(argv)

    from tmagent.model.streaming import load_streaming_policy

    cfg = load_config(args.config, args.overrides)
    if args.out is None:
        from tmagent.experiment import create_run

        out_dir = create_run(f"eval-{cfg.name}", cfg)
    else:
        out_dir = Path(args.out)
    policy = load_streaming_policy(args.ckpt, cfg.runtime.device)
    summary = evaluate(lambda: policy, cfg, out_dir=out_dir)
    summary.pop("episodes")
    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
