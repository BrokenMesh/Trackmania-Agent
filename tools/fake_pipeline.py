"""End-to-end proof of the code path on FakeGame (CPU, no game, a few minutes).

    python tools/fake_pipeline.py [--out DIR] [--steps 200] [--episodes 40]

Stages: render scripted fake episodes -> dataset quality check -> train_bc on the train/val
map splits -> closed-loop eval (sync) of the final checkpoint on held-out (test-split) fake
maps -> a few seconds of realtime LiveSession with the StreamingPolicy. It proves that every
stage runs and produces finite numbers, not that the model is any good. Writes report.md and
summary.json into --out (default: a temp directory) and prints where everything went.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # run as a script: make the repo root importable
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from tmagent.config import Config, load_config
from tmagent.data.episode_io import read_index
from tmagent.data.quality import check_dataset, format_report
from tmagent.data.split import split_for_map
from tools import render_replays

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fake.yaml"
MIN_EPISODES = 3  # one episode per split: train, val and test map
EVAL_MAPS = 2  # held-out maps evaluated
EVAL_TIMEOUT_S = 30.0  # race seconds per eval episode (keeps the stage short)


class PipelineError(RuntimeError):
    pass


def check_finite(name: str, value: Any) -> float:
    """float(value), or PipelineError if it is missing or not finite."""
    if value is None or not math.isfinite(float(value)):
        raise PipelineError(f"{name} is not a finite number: {value!r}")
    return float(value)


@contextmanager
def stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    print(f"\n=== {name} ===", flush=True)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = time.perf_counter() - t0
        print(f"[{name}] {timings[name]:.1f} s", flush=True)


def make_config(
    config: str | Path, out: Path, steps: int, overrides: list[str] | None = None
) -> Config:
    """configs/fake.yaml with a small CPU setup and the data root under `out`."""
    ev = max(steps // 4, 1)
    base = [
        f"data.root={(out / 'data').as_posix()}",
        "data.resolution=[64, 48]",
        "data.history_s=0.5",
        "train.device=cpu",
        "train.precision=fp32",
        "train.num_workers=0",
        "train.batch_size=8",
        f"train.steps={steps}",
        f"train.warmup_steps={min(20, max(steps // 10, 1))}",
        f"train.eval_every={ev}",
        f"train.ckpt_every={steps}",
        f"train.log_every={min(25, ev)}",
        "runtime.device=cpu",
        "runtime.precision=fp32",
        "eval.mode=sync",
        "eval.seeds=[0]",
        f"eval.timeout_s={EVAL_TIMEOUT_S}",
        "game.backend=fake",
    ]
    return load_config(config, base + list(overrides or []))


def heldout_maps(cfg: Config, maps: list[str], n: int = EVAL_MAPS) -> list[str]:
    """Rendered maps of the test split (val split if there are none), at most n."""
    for split in ("test", "val"):
        found = [m for m in maps if split_for_map(m, cfg.data) == split]
        if found:
            return found[:n]
    raise PipelineError("no rendered map falls into the test or val split")


def metric_curve(run_dir: Path, key: str) -> list[tuple[int, float]]:
    """(step, value) of `key` from metrics.jsonl."""
    out = []
    for line in (run_dir / "metrics.jsonl").read_text().splitlines():
        row = json.loads(line)
        if key in row:
            out.append((int(row["step"]), float(row[key])))
    return out


def mean_value(curve: list[tuple[int, float]]) -> float:
    return sum(v for _, v in curve) / len(curve) if curve else float("nan")


def run_live(cfg: Config, ckpt: str, map_ref: str, seconds: float) -> dict[str, Any]:
    """Play `map_ref` in realtime with the checkpoint's StreamingPolicy; session stats."""
    from tmagent.game import make_game
    from tmagent.model.streaming import load_streaming_policy
    from tmagent.runtime.session import LiveSession

    policy = load_streaming_policy(ckpt, "cpu")
    game = make_game(cfg.game, cfg.data)
    try:
        game.load_map(map_ref)
        game.restart()
        session = LiveSession(game, policy, cfg)  # type: ignore[arg-type]
        with session:
            time.sleep(seconds)
        stats = session.stats()
        state = game.get_state()
    finally:
        game.close()
    stats["final"] = {
        "race_time_ms": state.race_time_ms,
        "speed_kmh": state.speed_kmh,
        "checkpoint": state.checkpoint,
        "finished": state.finished,
    }
    return stats


def render_markdown(s: dict[str, Any]) -> str:
    d, t, e, lv = s["data"], s["train"], s["eval"], s["live"]
    prof = lv["profiler"]["sections"]
    hours = ", ".join(f"{k} {d['hours'][k]:.3f}" for k in ("train", "val", "test"))

    def ms(name: str, key: str = "p99_ms") -> str:
        return f"{prof[name][key]:.2f}" if name in prof else "n/a"

    lines = [
        "# Fake pipeline report",
        "",
        "Proves the code path on FakeGame (CPU). The numbers say nothing about model quality.",
        "",
        "## Data",
        f"- {d['episodes']} episodes on {d['maps']} maps, {d['hours']['total']:.3f} h ({hours})",
        f"- episodes per split: {d['episodes_per_split']}, maps per split: {d['maps_per_split']}",
        f"- finished: {d['finished']}/{d['episodes']}, desyncs: {d['desyncs']}, "
        f"quality check: {'OK' if d['quality_ok'] else 'FAILED'}",
        f"- windows: train {d['train_windows']}, val {d['val_windows']}",
        "",
        "## Training",
        f"- {t['steps']} steps, {t['params_m']:.3f} M parameters, "
        f"{t['s_per_step']:.3f} s/step, checkpoint `{t['ckpt']}`",
        f"- train loss: {t['train_first']:.4f} (step {t['train_first_step']}) -> "
        f"{t['train_last']:.4f} (step {t['train_last_step']})",
        f"- val loss: {t['val_first']:.4f} (step {t['val_first_step']}) -> "
        f"{t['val_last']:.4f} (step {t['val_last_step']})",
        "",
        "## Closed-loop eval (sync, held-out maps)",
        f"- maps {e['maps']}, {e['n_episodes']} episodes",
        f"- finish rate {e['finish_rate']:.2f}, median progress {e['median_progress']:.3f}, "
        f"median time ratio {e['median_time_ratio']}",
        f"- end reasons: {e['reasons']}",
        "",
        "## Live (realtime LiveSession, FakeGame)",
        f"- {lv['seconds']:.1f} s on `{lv['map']}`: {lv['ticks']} control ticks, "
        f"deadline misses {lv['deadline_misses']} ({lv['miss_pct']:.2f} %)",
        f"- control jitter p99 {lv['jitter_p99_ms']:.2f} ms, set_action p99 "
        f"{lv['set_action_p99_ms']:.3f} ms",
        f"- inference: {lv['inference']['frames_observed']} frames observed, "
        f"{lv['inference']['policy_errors']} policy errors; observe p99 {ms('observe')} ms, "
        f"predict p99 {ms('predict')} ms, frame age p99 {ms('frame_age')} ms",
        f"- final state: {lv['final']}",
        "",
        "## Stage timings",
        *[f"- {k}: {v:.1f} s" for k, v in s["timings_s"].items()],
        "",
        "## Where things are",
        *[f"- {k}: `{v}`" for k, v in s["paths"].items()],
        "",
    ]
    return "\n".join(lines)


def run_pipeline(
    out: Path,
    steps: int = 200,
    episodes: int = 40,
    config: str | Path = DEFAULT_CONFIG,
    overrides: list[str] | None = None,
    live_s: float = 5.0,
) -> dict[str, Any]:
    """Run all stages into `out`; returns the summary (also written as summary.json)."""
    from tmagent.data.dataset import WindowDataset
    from tmagent.eval.harness import evaluate
    from tmagent.experiment import create_run
    from tmagent.model.policy import TMPolicy, count_parameters
    from tmagent.model.streaming import load_streaming_policy
    from tmagent.train.train_bc import train

    if episodes < MIN_EPISODES:
        raise PipelineError(f"need at least {MIN_EPISODES} episodes (train, val and test map)")
    out.mkdir(parents=True, exist_ok=True)
    cfg = make_config(config, out, steps, overrides)
    timings: dict[str, float] = {}
    root = Path(cfg.data.root)
    summary: dict[str, Any] = {"config": cfg.name, "steps": steps, "episodes": episodes}

    with stage("render", timings):
        maps = render_replays.pick_fake_maps(cfg.data)
        run = render_replays.render_fake(cfg, episodes, maps=maps)
        rep = render_replays.finalize(cfg, run)
        if rep["episodes"] == 0:
            raise PipelineError("no episodes were rendered")

    with stage("quality", timings):
        check = check_dataset(root, cfg.data)
        print(format_report(check))
        if not check["ok"]:
            raise PipelineError("dataset quality check failed")
        used = sorted({e["map_uid"] for e in read_index(root)})

    with stage("train", timings):
        train_ds = WindowDataset(str(root), cfg.data, "train", True)
        val_ds = WindowDataset(str(root), cfg.data, "val", False, stride=5)
        print(f"windows: train {len(train_ds)}, val {len(val_ds)}")
        run_dir = create_run("fake_pipeline", cfg, base=out / "runs")
        res = train(cfg, train_ds, val_ds, run_dir)
        tl, vl = metric_curve(run_dir, "train/loss"), metric_curve(run_dir, "val/loss")
        if not tl or not vl:
            raise PipelineError("training logged no train/val loss")
        ckpt = res["last_ckpt"]
        n_params = count_parameters(TMPolicy(cfg.model, cfg.data))["total"]
        summary["train"] = {
            "steps": res["step"],
            "params_m": n_params / 1e6,
            "s_per_step": check_finite(
                "s_per_step", mean_value(metric_curve(run_dir, "train/s_per_step"))
            ),
            "ckpt": ckpt,
            "train_first_step": tl[0][0],
            "train_first": check_finite("train loss", tl[0][1]),
            "train_last_step": tl[-1][0],
            "train_last": check_finite("train loss", tl[-1][1]),
            "val_first_step": vl[0][0],
            "val_first": check_finite("val loss", vl[0][1]),
            "val_last_step": vl[-1][0],
            "val_last": check_finite("val loss", vl[-1][1]),
        }

    with stage("eval", timings):
        cfg.eval.maps = heldout_maps(cfg, used)
        res_eval = evaluate(lambda: load_streaming_policy(ckpt, "cpu"), cfg, out_dir=out / "eval")
        for ep in res_eval["episodes"]:
            check_finite("eval progress", ep["progress"])
        summary["eval"] = {
            "maps": cfg.eval.maps,
            "n_episodes": res_eval["n_episodes"],
            "finish_rate": check_finite("finish_rate", res_eval["finish_rate"]),
            "median_progress": check_finite("median_progress", res_eval["median_progress"]),
            "median_time_ratio": res_eval["median_time_ratio"],
            "reasons": res_eval["reasons"],
        }
        print(json.dumps(summary["eval"], indent=2))

    with stage("live", timings):
        stats = run_live(cfg, ckpt, cfg.eval.maps[0], live_s)
        if stats["ticks"] <= 0 or stats["inference"]["frames_observed"] <= 0:
            raise PipelineError(f"live session did nothing: {stats}")
        summary["live"] = {
            "map": cfg.eval.maps[0],
            "seconds": live_s,
            "ticks": stats["ticks"],
            "deadline_misses": stats["deadline_misses"],
            "miss_pct": check_finite("miss_pct", stats["miss_pct"]),
            "jitter_p99_ms": check_finite("jitter_p99_ms", stats["jitter_p99_ms"]),
            "set_action_p99_ms": check_finite("set_action_p99_ms", stats["set_action_p99_ms"]),
            "inference": stats["inference"],
            "profiler": stats["profiler"],
            "final": stats["final"],
        }
        print(
            f"live: {stats['ticks']} ticks, miss {stats['miss_pct']:.2f} %, "
            f"{stats['inference']['frames_observed']} frames observed"
        )

    summary["data"] = {
        "episodes": rep["episodes"],
        "maps": rep["maps"],
        "hours": rep["hours"],
        "episodes_per_split": rep["episodes_per_split"],
        "maps_per_split": rep["maps_per_split"],
        "finished": rep["finished"],
        "desyncs": rep["desyncs"],
        "quality_ok": check["ok"],
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
    }
    summary["timings_s"] = timings
    summary["paths"] = {
        "output": str(out),
        "dataset": str(root),
        "render report": str(root / render_replays.REPORT_NAME),
        "run (config, metrics, checkpoints)": str(run_dir),
        "checkpoint": str(ckpt),
        "eval results": str(out / "eval"),
        "report": str(out / "report.md"),
        "summary": str(out / "summary.json"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(render_markdown(summary))
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", default=None, help="output directory (default: a new temp dir)")
    ap.add_argument("--steps", type=int, default=200, help="training steps")
    ap.add_argument("--episodes", type=int, default=40, help="fake episodes to render")
    ap.add_argument("--live-s", type=float, default=5.0, help="seconds of realtime live play")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="k=v"
    )
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="tm_fake_pipeline_"))
    t0 = time.perf_counter()
    try:
        s = run_pipeline(out, args.steps, args.episodes, args.config, args.overrides, args.live_s)
    except PipelineError as exc:
        print(f"\nFAILED: {exc}")
        return 1
    print(f"\nfake pipeline OK in {time.perf_counter() - t0:.0f} s")
    for k, v in s["paths"].items():
        print(f"  {k}: {v}")
    print(f"  miss {s['live']['miss_pct']:.2f} %, eval progress {s['eval']['median_progress']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
