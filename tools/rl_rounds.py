"""Reward-weighted RL test: rounds of collect -> fine-tune -> evaluate on A01-A10.

    python tools/rl_rounds.py --config configs/tmnf.yaml \
        --ckpt experiments/2026-10-07-ctx2_dagger1/checkpoints/step_010000.pt [--rounds 3] \
        [--attempts 6] [--steps 2500] [--share 0.25] [--eval-start]

Round n: tools/collect_rl.py drives the maps with the current checkpoint (tag r<n>), then
train_bc fine-tunes it on human data + dagger:d1 (as for dagger1) + all rl:* episodes, the rl
share of the sampled windows set to --share, then the eval harness drives the maps 3 times
each (steer noise 0.05, the DAgger protocol). One summary line per round goes to
<data.root>/rl_rounds.jsonl. Needs the game running with the plugin listening; runs for
hours, one heavy job at a time.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tmagent.config import load_config
from tmagent.data.dataset import source_weight
from tmagent.data.episode_io import read_index
from tmagent.data.split import split_of

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_rl import resolve_maps  # noqa: E402

BASE_WEIGHTS = {"dagger": 0.0, "dagger:d1": 10.0, "takeover": 0.0}  # the dagger1 recipe


def rl_source_weights(index: list[dict], cfg, share: float) -> dict[str, float]:
    """BASE_WEIGHTS plus an "rl" weight that makes rl:* episodes `share` of the sampled
    train windows (windows are proportional to frames)."""
    d = cfg.data
    rl_frames, other = 0.0, 0.0
    for e in index:
        if split_of(e["map_uid"], d.split_seed, d.val_frac, d.test_frac) != "train":
            continue
        source = str(e.get("source", ""))
        if source.startswith("rl:"):
            rl_frames += e["num_frames"]
        else:
            other += source_weight(source, BASE_WEIGHTS) * e["num_frames"]

    weights = dict(BASE_WEIGHTS)
    weights["rl"] = share * other / ((1.0 - share) * rl_frames) if rl_frames else 1.0
    return weights


def run(cmd: list[str]) -> list[str]:
    """Run a command, echo its output, return the lines; raise on failure."""
    print(">", " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace"
    )
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))
    if proc.wait() != 0:
        raise RuntimeError(f"failed ({proc.returncode}): {' '.join(cmd)}")
    return lines


def evaluate(config: str, ckpt: str, uids: list[str], out: Path, extra: list[str]) -> dict:
    overrides = [f"eval.maps={json.dumps(uids)}", "eval.seeds=[0, 1, 2]", "eval.action_noise=0.05"]
    harness = [sys.executable, "-m", "tmagent.eval.harness", "--config", config, "--ckpt", ckpt]
    run([*harness, "--out", str(out), "--set", *extra, *overrides])

    episodes = [
        json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines() if line
    ]
    progress = [float(e["progress"]) for e in episodes]
    return {
        "eval_dir": str(out),
        "median_progress": float(np.median(progress)),
        "mean_progress": float(np.mean(progress)),
        "finishes": int(sum(bool(e["finished"]) for e in episodes)),
        "episodes": len(episodes),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reward-weighted RL rounds on A01-A10.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="starting checkpoint")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--first", type=int, default=1, help="number of the first round (resume)")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2500, help="fine-tuning steps per round")
    ap.add_argument("--share", type=float, default=0.25, help="rl share of the train windows")
    ap.add_argument(
        "--eval-start", action="store_true", help="evaluate the starting checkpoint first"
    )
    ap.add_argument("--maps", nargs="*", default=[], help="map names or uids (default A01-A10)")
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="K=V"
    )
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg.data.root)
    maps = resolve_maps(read_index(root), args.maps, cfg)
    uids = [uid for uid, _ in maps]
    log = root / "rl_rounds.jsonl"
    stamp = time.strftime("%Y-%m-%d-%H%M")
    extra = list(args.overrides)

    def record(entry: dict) -> None:
        with open(log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    ckpt = args.ckpt
    if args.eval_start:
        out = Path("experiments") / f"{stamp}-eval-rl_start"
        record({"round": 0, "ckpt": ckpt, **evaluate(args.config, ckpt, uids, out, extra)})

    for n in range(args.first, args.first + args.rounds):
        tag = f"r{n}"
        collect = [sys.executable, str(Path(__file__).with_name("collect_rl.py"))]
        run([*collect, "--config", args.config, "--ckpt", ckpt, "--tag", tag, "--maps", *uids,
             "--attempts", str(args.attempts), "--temperature", str(args.temperature),
             *(["--set", *extra] if extra else [])])  # fmt: skip

        weights = rl_source_weights(read_index(root), cfg, args.share)
        train = [sys.executable, "-m", "tmagent.train.train_bc", "--config", args.config]
        overrides = [
            *extra,
            f"name=rl_{tag}",
            f"train.init_from={ckpt}",
            f"train.steps={args.steps}",
            "train.lr=0.0001",
            f"train.warmup_steps={min(200, args.steps // 5)}",
            f"train.ckpt_every={args.steps}",
            f"data.source_weights={json.dumps(weights)}",
        ]
        lines = run([*train, "--set", *overrides])
        run_dir = next(line.split("] ", 1)[1] for line in lines if line.startswith("[run] "))
        ckpt = str(Path(run_dir) / "checkpoints" / "last.pt")

        out = Path("experiments") / f"{stamp}-eval-rl_{tag}"
        res = evaluate(args.config, ckpt, uids, out, extra)
        record({"round": n, "tag": tag, "ckpt": ckpt, "source_weights": weights, **res})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
