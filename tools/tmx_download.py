"""TMNF-Exchange downloader for maps and replays (Phase 1 input), gated by the terms rule.

Project rule (docs/PLAN.md, Phase 0.3): no large downloads before the terms were checked.
This tool therefore refuses to run unless `--i-have-read-the-tmx-terms` is given and it
prints the terms URL first. Rate limit: 1 request per second by default, exponential
backoff on 429 / 5xx, a User-Agent that names the project and a contact you must fill in.

    python tools/tmx_download.py --i-have-read-the-tmx-terms \
        --user-agent "tmagent-research/0.1 (contact: you@example.org)" \
        tracks --param environment=... --max-tracks 50
    python tools/tmx_download.py --i-have-read-the-tmx-terms --user-agent "..." \
        replays --per-track 3
    python tools/tmx_download.py --i-have-read-the-tmx-terms --dry-run tracks --param ...

Modes:
  tracks    search tracks (query parameters pass through via --param key=value) and append them
            to <out>/tracks.jsonl, at most --max-tracks.
  replays   for the tracks of --track-ids or --tracks-file: list the replays, keep
            --per-track of them spread over the ranks (best, median, tail: mixed driver
            quality), download the map to game.map_dir (<map_uid or track id>.Challenge.Gbx
            plus map_dir/index.json {uid: file}) and the replays to --replays-dir
            (<track id>-<replay id>.Replay.Gbx).
Resumable: <out>/state.json records finished downloads and the tracks cursor, API responses
are cached in <out>/cache/. --dry-run prints the requests without sending any (cached
responses are used to expand them, otherwise only the first-level requests are shown).

All endpoint paths and JSON field names below are UNVERIFIED (docs/research.md: the API docs
were not reachable); check them once with --dry-run and a single real request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------- UNVERIFIED API constants
# docs/research.md: paths from the TMX-Downloader README ([V]) and search snippets ([S]);
# the API field names are from memory of the TMX API v2 and NOT verified. Change them here.
BASE_URL = "https://tmnf.exchange"  # UNVERIFIED
EP_TRACKS = "/api/tracks"  # [V] TMX-Downloader README
EP_REPLAYS = "/api/replays"  # [S] needs `fields=`
EP_TRACK_FILE = "/trackgbx/{id}"  # [V] TMX-Downloader README
EP_REPLAY_FILE = "/recordgbx/{id}"  # [V] TMX-Downloader README
P_FIELDS = "fields"  # UNVERIFIED query parameter names ([V]: count, after)
P_COUNT = "count"
P_AFTER = "after"
P_REPLAY_TRACK = "trackId"  # UNVERIFIED: replay list filter by track
F_RESULTS = "Results"  # UNVERIFIED: list wrapper
F_MORE = "More"  # UNVERIFIED: "another page exists" flag
F_TRACK_ID = "TrackId"  # UNVERIFIED
F_TRACK_UID = "UId"  # UNVERIFIED (map uid)
F_TRACK_NAME = "TrackName"  # UNVERIFIED
F_REPLAY_ID = "ReplayId"  # UNVERIFIED
F_REPLAY_TIME = "ReplayTime"  # UNVERIFIED (ms, lower = better)
F_REPLAY_USER = "User.Name"  # UNVERIFIED (dotted = nested)
TRACK_FIELDS = (F_TRACK_ID, F_TRACK_UID, F_TRACK_NAME)
REPLAY_FIELDS = (F_REPLAY_ID, F_REPLAY_TIME, F_REPLAY_USER)
# Challenge .Gbx files carry an XML header with <ident uid="..." .../> (UNVERIFIED layout).
GBX_UID_RE = re.compile(rb'<ident\s+uid="([^"]+)"')
GBX_HEADER_BYTES = 1 << 16

TERMS_URL = "https://tmnf.exchange/threadshow/10428863"  # "Replay Rules", docs/research.md
TERMS_FLAG = "--i-have-read-the-tmx-terms"
UA_PLACEHOLDER = "CHANGE-ME"
DEFAULT_USER_AGENT = f"tmagent-research/0.1 (contact: {UA_PLACEHOLDER}@example.invalid)"
MAP_SUFFIX = ".Challenge.Gbx"
REPLAY_SUFFIX = ".Replay.Gbx"
DEFAULT_OUT = "data/tmnf/tmx"
DEFAULT_REPLAYS_DIR = "data/tmnf/replays"
MAX_RETRY_AFTER_S = 300.0


class TMXError(RuntimeError):
    """A request failed for good (non-retryable status or retries exhausted)."""


# ---------------------------------------------------------------- rate limiting and HTTP


class RateLimiter:
    """At least `min_interval` seconds between the starts of consecutive requests."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = max(float(min_interval), 0.0)
        self._clock, self._sleep = clock, sleep
        self._next: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._next is not None and now < self._next:
            self._sleep(self._next - now)
            now = self._clock()
        self._next = now + self.min_interval


def get_path(obj: Any, dotted: str, default: Any = None) -> Any:
    """obj["A"]["B"] for "A.B"; default when a level is missing."""
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return default
        obj = obj[part]
    return obj


def retry_after_s(err: urllib.error.HTTPError) -> float | None:
    """Seconds from a numeric Retry-After header, capped; None if absent or not numeric."""
    value = err.headers.get("Retry-After") if err.headers is not None else None
    try:
        return min(max(float(value), 0.0), MAX_RETRY_AFTER_S) if value is not None else None
    except ValueError:
        return None


class TMXClient:
    """Rate-limited, caching, retrying GET client. `opener` needs `.open(request, timeout=)`."""

    def __init__(
        self,
        user_agent: str,
        base_url: str = BASE_URL,
        min_interval: float = 1.0,
        opener: Any | None = None,
        cache_dir: Path | None = None,
        dry_run: bool = False,
        max_retries: int = 5,
        backoff_s: float = 2.0,
        timeout_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = print,
    ) -> None:
        self.user_agent, self.base_url = user_agent, base_url.rstrip("/")
        self.opener = opener if opener is not None else urllib.request.build_opener()
        self.cache_dir, self.dry_run = cache_dir, dry_run
        self.max_retries, self.backoff_s, self.timeout_s = max_retries, backoff_s, timeout_s
        self._sleep, self._log = sleep, log
        self.limiter = RateLimiter(min_interval, clock, sleep)
        self.requests = 0  # requests actually sent

    def url(self, path: str, params: Sequence[tuple[str, Any]] | None = None) -> str:
        url = self.base_url + path
        return f"{url}?{urllib.parse.urlencode(list(params))}" if params else url

    def _fetch(self, url: str) -> bytes:
        req = urllib.request.Request(
            url, headers={"User-Agent": self.user_agent, "Accept": "*/*"}, method="GET"
        )
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            self.requests += 1
            try:
                with self.opener.open(req, timeout=self.timeout_s) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and not 500 <= exc.code < 600:
                    raise TMXError(f"HTTP {exc.code} for {url}") from exc
                last, hint = exc, retry_after_s(exc)
                why = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last, hint, why = exc, None, f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                delay = hint if hint is not None else self.backoff_s * 2**attempt
                self._log(f"  {why}, retry {attempt + 1}/{self.max_retries} in {delay:.1f} s")
                self._sleep(delay)
        raise TMXError(f"giving up on {url} after {self.max_retries} retries ({last})") from last

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{hashlib.sha1(url.encode()).hexdigest()[:20]}.json"

    def get_json(self, path: str, params: Sequence[tuple[str, Any]] | None = None) -> Any | None:
        """Parsed JSON of a GET (cached on disk). None in --dry-run when not cached."""
        url = self.url(path, params)
        cache = self._cache_path(url)
        if cache is not None and cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        if self.dry_run:
            self._log(f"[dry-run] GET {url}")
            return None
        body = self._fetch(url)
        try:
            data = json.loads(body.decode("utf-8"))
        except ValueError as exc:
            raise TMXError(f"response of {url} is not JSON: {body[:80]!r}") from exc
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(data), encoding="utf-8")
        return data

    def download(self, path: str, dest: Path) -> Path | None:
        """GET a .Gbx file to `dest` unless it exists; None in --dry-run.

        The body must start with the GBX magic (a TMX error page would not).
        """
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        url = self.url(path)
        if self.dry_run:
            self._log(f"[dry-run] GET {url} -> {dest}")
            return None
        body = self._fetch(url)
        if body[:3] != b"GBX":
            raise TMXError(f"{url}: response is not a GBX file ({body[:40]!r})")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        return dest


# ---------------------------------------------------------------- state


class State:
    """Resume state in <out>/state.json (written atomically after every change)."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"tracks": {}, "maps": {}, "replays": {}, "failed": {}}
        if path.is_file():
            self.data.update(json.loads(path.read_text(encoding="utf-8")))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.data, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def fail(self, key: str, message: str) -> None:
        self.data["failed"][key] = message
        self.save()


# ---------------------------------------------------------------- tracks


def parse_params(items: Sequence[str]) -> list[tuple[str, str]]:
    """--param key=value (repeatable; a repeated key is sent repeatedly)."""
    out = []
    for item in items:
        key, eq, value = item.partition("=")
        if not eq or not key.strip():
            raise ValueError(f"--param must be key=value, got {item!r}")
        out.append((key.strip(), value))
    return out


def fetch_tracks(
    client: TMXClient,
    state: State,
    out: Path,
    params: list[tuple[str, str]],
    max_tracks: int,
    page_size: int,
) -> int:
    """Page through the track search into <out>/tracks.jsonl; returns the tracks stored.

    Resumes from the saved cursor when the query (params) is unchanged.
    """
    query = {"params": params, "fields": list(TRACK_FIELDS)}
    st = state.data["tracks"]
    if st.get("query") != json.loads(json.dumps(query)):
        st.clear()
        st.update(query=query, after=None, count=0, done=False)
    tracks_file = out / "tracks.jsonl"
    while st["count"] < max_tracks and not st["done"]:
        page_params = [(P_FIELDS, ",".join(TRACK_FIELDS)), *params]
        page_params.append((P_COUNT, min(page_size, max_tracks - st["count"])))
        if st["after"] is not None:
            page_params.append((P_AFTER, st["after"]))
        page = client.get_json(EP_TRACKS, page_params)
        if page is None:  # dry run: the next page depends on this response
            break
        results = page.get(F_RESULTS) or []
        take = results[: max_tracks - st["count"]]
        tracks_file.parent.mkdir(parents=True, exist_ok=True)
        with open(tracks_file, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(t) + "\n" for t in take)
        st["count"] += len(take)
        if take:
            st["after"] = get_path(take[-1], F_TRACK_ID)
        st["done"] = not results or not page.get(F_MORE) or st["after"] is None
        state.save()
        print(f"tracks: {st['count']} stored")
    return int(st.get("count", 0))


def read_tracks_file(path: Path) -> list[dict[str, Any]]:
    """Track records of a tracks.jsonl (unique by track id, first wins)."""
    seen: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            tid = get_path(rec, F_TRACK_ID)
            if tid is not None:
                seen.setdefault(str(tid), rec)
    return list(seen.values())


# ---------------------------------------------------------------- replays


def select_spread(items: Sequence[Any], n: int) -> list[Any]:
    """n items spread evenly over `items` (best first): first, median-ish ..., last."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    if n == 1:
        return [items[0]]
    idx = sorted({round(i * (len(items) - 1) / (n - 1)) for i in range(n)})
    return [items[i] for i in idx]


def list_replays(client: TMXClient, track_id: str, max_listed: int) -> list[dict[str, Any]] | None:
    """Replays of a track sorted best (lowest time) first; None in --dry-run when uncached."""
    found: list[dict[str, Any]] = []
    after = None
    while len(found) < max_listed:
        params: list[tuple[str, Any]] = [
            (P_REPLAY_TRACK, track_id),
            (P_FIELDS, ",".join(REPLAY_FIELDS)),
            (P_COUNT, min(1000, max_listed - len(found))),
        ]
        if after is not None:
            params.append((P_AFTER, after))
        page = client.get_json(EP_REPLAYS, params)
        if page is None:
            return None
        results = page.get(F_RESULTS) or []
        found.extend(results)
        after = get_path(results[-1], F_REPLAY_ID) if results else None
        if not results or not page.get(F_MORE) or after is None:
            break
    valid = [
        r
        for r in found
        if get_path(r, F_REPLAY_ID) is not None
        and isinstance(get_path(r, F_REPLAY_TIME), int | float)
    ]
    return sorted(valid, key=lambda r: get_path(r, F_REPLAY_TIME))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(name))


def uid_from_gbx_header(path: Path) -> str | None:
    """Map uid from the XML header of a .Challenge.Gbx (best effort, UNVERIFIED layout)."""
    try:
        with open(path, "rb") as f:
            head = f.read(GBX_HEADER_BYTES)
    except OSError:
        return None
    m = GBX_UID_RE.search(head)
    return m.group(1).decode("ascii", "replace") if m else None


def update_map_index(map_dir: Path, uid: str, filename: str) -> None:
    """Set map_dir/index.json[uid] = filename (atomic write)."""
    index = map_dir / "index.json"
    data = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
    if data.get(uid) == filename:
        return
    data[uid] = filename
    tmp = index.with_name("index.json.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(index)


def ensure_map(
    client: TMXClient, state: State, track_id: str, uid: str | None, map_dir: Path
) -> Path | None:
    """Download the map of a track into map_dir and register it in index.json.

    File name: <uid>.Challenge.Gbx with the uid from the track record, else from the file's
    XML header, else the track id.
    """
    known = state.data["maps"].get(track_id)
    if known and (map_dir / known).is_file():
        return map_dir / known
    if uid:
        dest = map_dir / f"{_safe_name(uid)}{MAP_SUFFIX}"
        if client.download(EP_TRACK_FILE.format(id=track_id), dest) is None:
            return None
    else:
        tmp = map_dir / f"{track_id}{MAP_SUFFIX}"
        if client.download(EP_TRACK_FILE.format(id=track_id), tmp) is None:
            return None
        uid = uid_from_gbx_header(tmp)
        dest = map_dir / f"{_safe_name(uid)}{MAP_SUFFIX}" if uid else tmp
        if dest != tmp:
            tmp.replace(dest)
    update_map_index(map_dir, uid or track_id, dest.name)
    state.data["maps"][track_id] = dest.name
    state.save()
    return dest


def fetch_replays(
    client: TMXClient,
    state: State,
    out: Path,
    tracks: list[dict[str, Any]],
    per_track: int,
    map_dir: Path,
    replays_dir: Path,
    max_listed: int,
    with_maps: bool = True,
) -> int:
    """Maps and rank-spread replays of `tracks`; returns the number of replays downloaded."""
    total = 0
    for i, rec in enumerate(tracks, 1):
        tid = str(get_path(rec, F_TRACK_ID))
        print(f"[{i}/{len(tracks)}] track {tid} {get_path(rec, F_TRACK_NAME, '')}")
        try:
            if with_maps:
                ensure_map(client, state, tid, get_path(rec, F_TRACK_UID), map_dir)
            listed = list_replays(client, tid, max_listed)
        except TMXError as exc:
            print(f"  failed: {exc}")
            state.fail(f"track:{tid}", str(exc))
            continue
        if listed is None:
            continue
        chosen = select_spread(listed, per_track)
        print(f"  {len(listed)} replays listed, {len(chosen)} chosen")
        for rec_r in chosen:
            rid = str(get_path(rec_r, F_REPLAY_ID))
            if (
                rid in state.data["replays"]
                and (replays_dir / state.data["replays"][rid]["file"]).is_file()
            ):
                continue
            dest = replays_dir / f"{tid}-{rid}{REPLAY_SUFFIX}"
            try:
                got = client.download(EP_REPLAY_FILE.format(id=rid), dest)
            except TMXError as exc:
                print(f"  replay {rid} failed: {exc}")
                state.fail(f"replay:{rid}", str(exc))
                continue
            if got is None:
                continue
            rank = listed.index(rec_r) + 1
            state.data["replays"][rid] = {
                "file": dest.name,
                "track_id": tid,
                "rank": rank,
                "of": len(listed),
                "time_ms": get_path(rec_r, F_REPLAY_TIME),
                "user": get_path(rec_r, F_REPLAY_USER),
            }
            state.save()
            total += 1
            print(f"  replay {rid} rank {rank}/{len(listed)} -> {dest.name}")
    return total


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument(
        TERMS_FLAG,
        dest="terms_read",
        action="store_true",
        help=f"required: confirm you read the TMX terms / replay rules ({TERMS_URL})",
    )
    ap.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="must contain your contact")
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--min-interval", type=float, default=1.0, help="seconds between requests")
    ap.add_argument("--out", default=DEFAULT_OUT, help="state, cache and tracks.jsonl directory")
    ap.add_argument("--config", default=None, help="YAML config: game.map_dir is the map dir")
    ap.add_argument("--map-dir", default=None, help="override game.map_dir")
    ap.add_argument("--replays-dir", default=DEFAULT_REPLAYS_DIR)
    ap.add_argument("--dry-run", action="store_true", help="print the requests, send nothing")
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("tracks", help="search tracks into tracks.jsonl")
    t.add_argument("--param", action="append", default=[], metavar="key=value")
    t.add_argument("--max-tracks", type=int, default=100)
    t.add_argument("--page-size", type=int, default=100)
    r = sub.add_parser("replays", help="download maps and rank-spread replays")
    r.add_argument("--track-ids", default=None, help="comma-separated TMX track ids")
    r.add_argument("--tracks-file", default=None, help="tracks.jsonl (default <out>/tracks.jsonl)")
    r.add_argument("--per-track", type=int, default=3, help="replays per track (best..tail)")
    r.add_argument("--max-listed", type=int, default=1000, help="replays listed per track")
    r.add_argument("--max-tracks", type=int, default=None, help="process at most N tracks")
    r.add_argument("--no-maps", action="store_true", help="skip the map downloads")
    return ap


def main(
    argv: Sequence[str] | None = None,
    opener: Any | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    args = build_parser().parse_args(argv)
    print(f"TMX terms / replay rules (read them before downloading): {TERMS_URL}")
    if not args.terms_read:
        print(
            f"refusing to run: pass {TERMS_FLAG} after reading the terms above "
            "(project rule: no large downloads before the terms were checked)."
        )
        return 2
    if UA_PLACEHOLDER in args.user_agent and not args.dry_run:
        print(f"refusing to run: put your contact into --user-agent (still has {UA_PLACEHOLDER!r})")
        return 2
    out = Path(args.out)
    map_dir = args.map_dir
    if map_dir is None:
        from tmagent.config import load_config

        map_dir = load_config(args.config or _default_config()).game.map_dir
    client = TMXClient(
        args.user_agent,
        base_url=args.base_url,
        min_interval=args.min_interval,
        opener=opener,
        cache_dir=out / "cache",
        dry_run=args.dry_run,
        clock=clock,
        sleep=sleep,
    )
    state = State(out / "state.json")
    try:
        if args.mode == "tracks":
            n = fetch_tracks(
                client, state, out, parse_params(args.param), args.max_tracks, args.page_size
            )
            print(f"{n} tracks in {out / 'tracks.jsonl'} ({client.requests} requests)")
        else:
            tracks = _replay_tracks(args, out)
            n = fetch_replays(
                client,
                state,
                out,
                tracks,
                args.per_track,
                Path(map_dir),
                Path(args.replays_dir),
                args.max_listed,
                with_maps=not args.no_maps,
            )
            print(f"{n} replays downloaded ({client.requests} requests), maps in {map_dir}")
    except (TMXError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    if args.dry_run:
        print("dry run: no request was sent, nothing was downloaded")
    return 0


def _default_config() -> str:
    return str(Path(__file__).resolve().parents[1] / "configs" / "tmnf.yaml")


def _replay_tracks(args: argparse.Namespace, out: Path) -> list[dict[str, Any]]:
    if args.track_ids:
        ids = [x.strip() for x in args.track_ids.split(",") if x.strip()]
        tracks: list[dict[str, Any]] = [{F_TRACK_ID: int(x) if x.isdigit() else x} for x in ids]
    else:
        path = Path(args.tracks_file) if args.tracks_file else out / "tracks.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"no tracks: give --track-ids or run `tracks` first ({path})")
        tracks = read_tracks_file(path)
    return tracks[: args.max_tracks] if args.max_tracks else tracks


if __name__ == "__main__":
    sys.exit(main())
