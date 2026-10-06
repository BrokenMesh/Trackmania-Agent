"""Collect DAgger episodes: a checkpoint drives, the line expert labels and rescues.

    python tools/collect_dagger.py --config configs/tmnf.yaml --ckpt <run>/checkpoints/x.pt \
        --tag d1 [--split train] [--runs 1] [--limit N]

Episodes go to data.root with source "dagger:<tag>" (episode id <uid>-dagger-<tag>-<run>);
maps already collected for the tag are skipped, so the tool resumes after a crash. Only maps
with a reference run (refs/<uid>.json) are used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tmagent.config import load_config
from tmagent.data.episode_io import append_index, read_index, save_episode
from tmagent.data.quality import check_episode, split_issues
from tmagent.data.split import split_of
from tmagent.eval.dagger import DaggerConfig, collect_episode
from tmagent.eval.expert import LineExpert
from tmagent.eval.progress import load_reference
from tmagent.game import make_game
from tmagent.model.streaming import load_streaming_policy


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="DAgger collection with the line expert.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True, help="policy that drives")
    ap.add_argument("--tag", required=True, help="iteration tag, e.g. d1")
    ap.add_argument("--split", default="train", help="maps of this split (never collect on test)")
    ap.add_argument("--runs", type=int, default=1, help="episodes per map")
    ap.add_argument("--limit", type=int, default=None, help="stop after N maps")
    ap.add_argument("--out", default=None, help="write episodes here (default: data.root)")
    ap.add_argument(
        "--set", dest="overrides", nargs="+", action="extend", default=[], metavar="K=V"
    )
    args = ap.parse_args(argv)

    sys.stdout.reconfigure(errors="replace")  # map names carry odd characters (BOM, $-codes)
    cfg = load_config(args.config, args.overrides)
    d = cfg.data
    root = Path(d.root)
    index = read_index(root)
    names = {e["map_uid"]: e.get("map_name", e["map_uid"]) for e in index}
    out = Path(args.out) if args.out else root
    done = {e["episode_id"] for e in read_index(out)}
    maps = sorted(
        uid for uid in names
        if (root / "refs" / f"{uid}.json").exists()
        and split_of(uid, d.split_seed, d.val_frac, d.test_frac) == args.split
    )  # fmt: skip
    if args.limit:
        maps = maps[: args.limit]
    policy = load_streaming_policy(args.ckpt, cfg.runtime.device)
    game = make_game(cfg.game, d)
    total_s, t0 = 0.0, time.perf_counter()
    try:
        for n, uid in enumerate(maps, 1):
            for run in range(args.runs):
                eid = f"{uid}-dagger-{args.tag}-{run}"
                if eid in done:
                    continue
                expert = LineExpert.from_reference(root, uid)
                ref = load_reference(uid, root)
                ep, stats = collect_episode(
                    game, policy, expert, uid, cfg, run, ref, DaggerConfig()
                )
                ep.meta.update(
                    episode_id=eid,
                    map_uid=uid,
                    map_name=names[uid],
                    source=f"dagger:{args.tag}",
                    player="expert",
                    camera=cfg.game.camera,
                    renderer="dagger",
                    agent_ckpt=args.ckpt,
                    expert_share=stats["expert_share"],
                )
                problems, _ = split_issues(check_episode(ep, d))
                status = "skip: " + "; ".join(problems) if problems else "saved"
                if not problems:
                    append_index(out, ep, save_episode(ep, out))
                    total_s += stats["seconds"]
                print(
                    f"[{n}/{len(maps)}] {names[uid][:28]:28} {stats['reason']:9} prog "
                    f"{stats['progress']:.2f} expert {stats['expert_share']:.0%} "
                    f"{stats['seconds']:.1f}s {status}",
                    flush=True,
                )
    finally:
        game.close()
    wall = time.perf_counter() - t0
    print(json.dumps({"maps": len(maps), "game_hours": total_s / 3600, "wall_hours": wall / 3600}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
