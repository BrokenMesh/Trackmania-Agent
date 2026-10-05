"""CLI: play a map live with a trained policy.

    python -m tmagent.runtime.live --config X.yaml --ckpt path.pt --map MAPREF \
        [--duration-s 60] [--out DIR] [--set key=value ...]

Writes live_stats.json and latency.md into --out (default: a new
experiments/<date>-live-<map>/ run created with tmagent.experiment.create_run).

The policy is loaded first: its checkpoint's data settings (frame_hz, control_hz,
resolution, channels, history_s, chunk_len) override the config's, so the game captures
at the resolution the model was trained on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import time
from pathlib import Path
from typing import Any

from tmagent.config import Config, load_config
from tmagent.runtime.session import LiveSession


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m tmagent.runtime.live", description=__doc__)
    p.add_argument("--config", required=True, help="YAML config")
    p.add_argument("--ckpt", required=True, help="policy checkpoint (.pt)")
    p.add_argument("--map", required=True, help="map reference (path or uid)")
    p.add_argument("--duration-s", type=float, default=60.0)
    p.add_argument(
        "--out", default=None, help="output dir (default: experiments/<date>-live-<map>)"
    )
    p.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE", help="config override"
    )
    return p.parse_args(argv)


def _run_dir(map_ref: str, cfg: Config) -> Path:
    from tmagent.experiment import create_run

    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(map_ref).stem) or "map"
    return Path(create_run(f"live-{stem}", cfg))


CKPT_DATA_FIELDS = ("frame_hz", "control_hz", "resolution", "channels", "history_s", "chunk_len")


def adopt_checkpoint_data(cfg: Config, policy: Any) -> list[str]:
    """Replace cfg.data by the checkpoint's data section (if the policy carries one).

    Keeps cfg.runtime / cfg.game, sets runtime.control_hz = data.control_hz. Returns the
    names of the CKPT_DATA_FIELDS whose values differed.
    """
    ckpt_cfg = getattr(policy, "cfg", None)
    if ckpt_cfg is None:
        return []
    diff = [f for f in CKPT_DATA_FIELDS if getattr(cfg.data, f) != getattr(ckpt_cfg.data, f)]
    cfg.data = dataclasses.replace(ckpt_cfg.data)
    cfg.runtime.control_hz = cfg.data.control_hz
    cfg.validate()
    return diff


def _state_dict(game: Any) -> dict[str, Any]:
    try:
        st = game.get_state()
    except Exception:  # the game may already be gone; the stats are still worth writing
        return {}
    return {
        "race_time_ms": st.race_time_ms,
        "speed_kmh": st.speed_kmh,
        "checkpoint": st.checkpoint,
        "num_checkpoints": st.num_checkpoints,
        "finished": st.finished,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config(args.config, args.set)

    from tmagent.game import make_game
    from tmagent.model.streaming import load_streaming_policy

    policy = load_streaming_policy(args.ckpt, cfg.runtime.device)  # before make_game
    if hasattr(policy, "precision"):
        policy.precision = cfg.runtime.precision  # live precision comes from the CLI config
    old = dataclasses.replace(cfg.data)
    diff = adopt_checkpoint_data(cfg, policy)
    if diff:
        changes = ", ".join(f"{f}: {getattr(old, f)} -> {getattr(cfg.data, f)}" for f in diff)
        print(
            f"WARNING: config.data differs from the checkpoint, using the checkpoint's: {changes}"
        )

    out = Path(args.out) if args.out else _run_dir(args.map, cfg)
    out.mkdir(parents=True, exist_ok=True)

    game = make_game(cfg.game, cfg.data)
    wall0 = time.perf_counter()
    try:
        game.load_map(args.map)
        game.restart()
        session = LiveSession(game, policy, cfg)
        finished = False
        try:
            with session:
                deadline = time.perf_counter() + args.duration_s
                while time.perf_counter() < deadline:
                    time.sleep(0.05)
                    if game.get_state().finished:
                        finished = True
                        break
        except KeyboardInterrupt:
            print("interrupted, stopping")
        report = {
            "map": args.map,
            "ckpt": args.ckpt,
            "duration_s": time.perf_counter() - wall0,
            "finished": finished,
            "final_state": _state_dict(game),
            "config": dataclasses.asdict(cfg.runtime),
            "stats": session.stats(),
        }
    finally:
        game.close()

    (out / "live_stats.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "latency.md").write_text(session.profiler.to_markdown())
    session.profiler.dump(out / "latency.json")
    st = report["stats"]
    print(
        f"ticks={st['ticks']} miss_pct={st['miss_pct']:.2f} "
        f"jitter_p50={st['jitter_p50_ms']:.2f}ms jitter_p99={st['jitter_p99_ms']:.2f}ms "
        f"finished={finished} -> {out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
