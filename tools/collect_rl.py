"""Collect reward-weighted RL episodes: a checkpoint drives with sampled actions.

    python tools/collect_rl.py --config configs/tmnf.yaml --ckpt <run>/checkpoints/x.pt \
        --tag r1 [--maps A01-Race A02-Race ...] [--attempts 6] [--temperature 1.0]

Every map is driven `--attempts` times; the per-step weights compare the attempts with each
other (tmagent/eval/rl.py), so they are computed per map once all its attempts are in. Episodes
go to data.root with source "rl:<tag>" (episode id <uid>-rl-<tag>-<attempt>); maps already
collected for the tag are skipped. Only train-split maps with a reference run are used.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from tmagent.config import load_config
from tmagent.data.episode_io import append_index, read_index, save_episode
from tmagent.data.quality import check_episode, split_issues
from tmagent.data.split import split_of
from tmagent.eval.progress import load_reference
from tmagent.eval.rl import RLConfig, advantage_weights, collect_attempt
from tmagent.game import make_game
from tmagent.model.streaming import load_streaming_policy

DEFAULT_MAPS = r"A(0[1-9]|10)-"  # the first ten campaign maps


def resolve_maps(index: list[dict], names: list[str], cfg) -> list[tuple[str, str]]:
    """(uid, name) of the train-split maps with a reference run, by name or uid; default the
    A01-A10 campaign maps."""
    d = cfg.data
    known = {e["map_uid"]: e.get("map_name", e["map_uid"]) for e in index}
    pattern = re.compile(DEFAULT_MAPS)

    maps = []
    for uid, name in sorted(known.items(), key=lambda kv: kv[1]):
        wanted = (uid in names or name in names) if names else bool(pattern.match(name))
        if not wanted:
            continue
        if split_of(uid, d.split_seed, d.val_frac, d.test_frac) != "train":
            continue
        if has_reference(uid, d.root):
            maps.append((uid, name))
    return maps


def has_reference(uid: str, root: str) -> bool:
    try:
        load_reference(uid, root)
    except FileNotFoundError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reward-weighted RL collection.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="policy that drives")
    ap.add_argument("--tag", required=True, help="round tag, e.g. r1")
    ap.add_argument("--maps", nargs="*", default=[], help="map names or uids (default A01-A10)")
    ap.add_argument("--attempts", type=int, default=6, help="attempts per map")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="K=V"
    )
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")
    cfg = load_config(args.config, args.overrides)
    root = Path(cfg.data.root)
    index = read_index(root)
    done = {e["episode_id"] for e in index}
    maps = resolve_maps(index, args.maps, cfg)
    rcfg = RLConfig(temperature=args.temperature)

    policy = load_streaming_policy(args.ckpt, cfg.runtime.device)
    game = make_game(cfg.game, cfg.data)
    summary = {"maps": len(maps), "attempts": 0, "finishes": 0, "game_s": 0.0, "progress": []}
    t0 = time.perf_counter()
    try:
        for n, (uid, name) in enumerate(maps, 1):
            ids = [f"{uid}-rl-{args.tag}-{i}" for i in range(args.attempts)]
            if all(eid in done for eid in ids):
                continue

            ref = load_reference(uid, root)
            attempts = []
            for i in range(args.attempts):
                seed = zlib.crc32(f"{args.tag}:{uid}:{i}".encode())
                a = collect_attempt(game, policy, uid, cfg, seed, ref, rcfg)
                attempts.append(a)
                line = f"{name[:24]:24} #{i} {a.reason:9} prog {a.progress:.2f} {a.seconds:5.1f}s"
                print(f"[{n}/{len(maps)}] {line}", flush=True)

            weights = advantage_weights(attempts, cfg.data.control_hz, rcfg)
            for i, (a, w) in enumerate(zip(attempts, weights, strict=True)):
                ep = a.episode
                ep.weights = w
                ep.meta.update(
                    episode_id=ids[i],
                    map_uid=uid,
                    map_name=name,
                    source=f"rl:{args.tag}",
                    player="agent",
                    camera=cfg.game.camera,
                    renderer="rl",
                    agent_ckpt=args.ckpt,
                    rl_reason=a.reason,
                    rl_progress=a.progress,
                    rl_temperature=args.temperature,
                )
                problems, _ = split_issues(check_episode(ep, cfg.data))
                if problems:
                    print(f"    skip #{i}: {'; '.join(problems)}", flush=True)
                    continue
                append_index(root, ep, save_episode(ep, root))

                summary["attempts"] += 1
                summary["finishes"] += a.reason == "finished"
                summary["game_s"] += a.seconds
                summary["progress"].append(a.progress)
    finally:
        game.close()

    prog = summary.pop("progress")
    summary["mean_progress"] = float(np.mean(prog)) if prog else None
    summary["wall_s"] = time.perf_counter() - t0
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
