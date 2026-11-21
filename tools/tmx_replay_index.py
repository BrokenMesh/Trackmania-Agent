"""Index of every TMX replay listed for the maps in tracks.jsonl: downloaded, rejected or new.

    python tools/tmx_replay_index.py --offline
    python tools/tmx_replay_index.py --i-have-read-the-tmx-terms \
        --user-agent "tmagent-research/0.1 (contact: you@example.org)"

Sources are tmx_download.py out dirs (tracks.jsonl, state.json, cache/), by default
data/tmnf/tmx and data/tmnf/tmx_community. The replay lists come from the downloader's cache;
pages not cached yet are fetched at 1 request per second, or skipped with --offline (the map is
then reported as not listed). Status per replay:
  downloaded  a <track id>-<replay id>.Replay.Gbx exists in any --replays-glob dir
  rejected    the downloader dropped it (analog input or respawns), not worth fetching again
  new         neither

Writes <out> (one row per replay, with its download URL) and <out stem>_maps.csv (counts per
map), both UTF-8 with BOM so Excel shows driver names correctly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import tmx_download as tmx  # noqa: E402

DEFAULT_SOURCES = ("data/tmnf/tmx", "data/tmnf/tmx_community")
DEFAULT_REPLAYS_GLOB = "data/tmnf/replays*"
DEFAULT_OUT = "data/tmnf/replay_index.csv"
WHOLE_LIST = 1_000_000
REPLAY_COLUMNS = (
    "map", "track_id", "map_uid", "rank", "of", "replay_id", "driver", "time", "time_ms",
    "status", "source", "download_url",
)  # fmt: skip
MAP_COLUMNS = ("map", "track_id", "map_uid", "listed", "downloaded", "rejected", "new", "source")


@dataclass
class MapCounts:
    name: str
    track_id: str
    uid: str
    source: str
    listed: int | None = None
    status: dict[str, int] = field(
        default_factory=lambda: {"downloaded": 0, "rejected": 0, "new": 0}
    )


def format_time(ms: Any) -> str:
    if not isinstance(ms, int | float):
        return ""
    cs = int(ms) // 10
    return f"{cs // 6000}:{cs // 100 % 60:02d}.{cs % 100:02d}"


def downloaded_replay_ids(replays_glob: str) -> set[str]:
    ids = set()
    for folder in Path().glob(replays_glob):
        if not folder.is_dir():
            continue
        for path in folder.glob(f"*{tmx.REPLAY_SUFFIX}"):
            stem = path.name[: -len(tmx.REPLAY_SUFFIX)]
            ids.add(stem.rpartition("-")[2])
    return ids


def rejected_replay_ids(source: Path) -> set[str]:
    state_file = source / "state.json"
    if not state_file.is_file():
        return set()
    return set(json.loads(state_file.read_text(encoding="utf-8")).get("rejected", {}))


def replay_status(rid: str, downloaded: set[str], rejected: set[str]) -> str:
    if rid in downloaded:
        return "downloaded"
    if rid in rejected:
        return "rejected"
    return "new"


def index_source(
    client: tmx.TMXClient, source: Path, downloaded: set[str], writer: Any
) -> list[MapCounts]:
    rejected = rejected_replay_ids(source)
    tracks = tmx.read_tracks_file(source / "tracks.jsonl")
    maps = []

    for i, rec in enumerate(tracks, 1):
        tid = str(tmx.get_path(rec, tmx.F_TRACK_ID))
        counts = MapCounts(
            str(tmx.get_path(rec, tmx.F_TRACK_NAME, "")),
            tid,
            str(tmx.get_path(rec, tmx.F_TRACK_UID, "")),
            source.name,
        )
        maps.append(counts)
        prefix = f"[{source.name} {i}/{len(tracks)}] {counts.name}:"

        try:
            listed = tmx.list_replays(client, tid, WHOLE_LIST)
        except tmx.TMXError as exc:
            print(f"{prefix} failed ({exc})")
            continue
        if listed is None:
            print(f"{prefix} not cached, skipped (--offline)")
            continue

        counts.listed = len(listed)
        for rank, rep in enumerate(listed, 1):
            rid = str(tmx.get_path(rep, tmx.F_REPLAY_ID))
            status = replay_status(rid, downloaded, rejected)
            counts.status[status] += 1
            writer.writerow(replay_row(client, counts, rep, rid, rank, len(listed), status))

        print(f"{prefix} {len(listed)} listed, {counts.status['new']} new")

    return maps


def replay_row(
    client: tmx.TMXClient,
    m: MapCounts,
    rep: dict[str, Any],
    rid: str,
    rank: int,
    of: int,
    status: str,
) -> list[Any]:
    time_ms = tmx.get_path(rep, tmx.F_REPLAY_TIME)
    driver = tmx.get_path(rep, tmx.F_REPLAY_USER, "")
    url = client.url(tmx.EP_REPLAY_FILE.format(id=rid))
    return [
        m.name,
        m.track_id,
        m.uid,
        rank,
        of,
        rid,
        driver,
        format_time(time_ms),
        time_ms,
        status,
        m.source,
        url,
    ]


def write_map_summary(path: Path, maps: list[MapCounts]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(MAP_COLUMNS)
        for m in maps:
            listed = "" if m.listed is None else m.listed
            s = m.status
            writer.writerow(
                [
                    m.name,
                    m.track_id,
                    m.uid,
                    listed,
                    s["downloaded"],
                    s["rejected"],
                    s["new"],
                    m.source,
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        tmx.TERMS_FLAG,
        dest="terms_read",
        action="store_true",
        help=f"required unless --offline ({tmx.TERMS_URL})",
    )
    ap.add_argument(
        "--user-agent",
        default=tmx.DEFAULT_USER_AGENT,
        help="must contain your contact (unless --offline)",
    )
    ap.add_argument(
        "--offline", action="store_true", help="use cached replay lists only, send no request"
    )
    ap.add_argument(
        "--source",
        action="append",
        default=None,
        help=f"downloader out dir, repeatable (default {', '.join(DEFAULT_SOURCES)})",
    )
    ap.add_argument(
        "--replays-glob", default=DEFAULT_REPLAYS_GLOB, help="dirs holding downloaded replays"
    )
    ap.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--out", default=DEFAULT_OUT)
    return ap


def main(argv: Sequence[str] | None = None, opener: Any | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.offline:
        print(f"TMX terms / replay rules: {tmx.TERMS_URL}")
        if not args.terms_read:
            print(
                f"refusing to run: pass {tmx.TERMS_FLAG} after reading the terms, or use --offline"
            )
            return 2
        if tmx.UA_PLACEHOLDER in args.user_agent:
            print("refusing to run: put your contact into --user-agent, or use --offline")
            return 2

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    downloaded = downloaded_replay_ids(args.replays_glob)
    maps: list[MapCounts] = []
    requests = 0

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REPLAY_COLUMNS)
        for source in map(Path, args.source or DEFAULT_SOURCES):
            client = tmx.TMXClient(
                args.user_agent,
                min_interval=args.min_interval,
                opener=opener,
                cache_dir=source / "cache",
                dry_run=args.offline,
                log=lambda _msg: None,
            )
            maps.extend(index_source(client, source, downloaded, writer))
            requests += client.requests

    summary = out.with_name(f"{out.stem}_maps.csv")
    write_map_summary(summary, maps)

    listed = [m for m in maps if m.listed is not None]
    t = {k: sum(m.status[k] for m in listed) for k in ("downloaded", "rejected", "new")}
    print(
        f"{len(listed)}/{len(maps)} maps listed, {sum(t.values())} replays: {t['downloaded']} "
        f"downloaded, {t['rejected']} rejected, {t['new']} new ({requests} requests)"
    )
    print(f"index: {out}\nper map: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
