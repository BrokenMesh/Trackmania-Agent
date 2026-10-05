"""tools/tmx_download.py with a mocked opener (no network): gate, dry-run, rate limit, resume."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.parse
from email.message import Message
from pathlib import Path

import pytest

from tools import tmx_download as tmx

REPO = Path(__file__).resolve().parents[2]
FLAG = tmx.TERMS_FLAG
UA = "tmagent-test/0 (contact: test@example.org)"


class FakeClock:
    """Monotonic clock advanced only by sleep()."""

    def __init__(self) -> None:
        self.t = 100.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


def http_error(url: str, code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError(url, code, f"HTTP {code}", hdrs, None)


class MockTMX:
    """Opener stand-in serving a tiny fake TMX: 5 tracks, 10 replays per track, gbx files."""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.clock = clock
        self.requests: list[dict] = []
        self.errors: dict[str, list] = {}  # url substring -> queue of exceptions to raise
        self.html: set[str] = set()  # url substrings answered with an HTML page
        self.tracks = [
            {"TrackId": 100 + i, "UId": f"UID{i}", "TrackName": f"Track {i}"} for i in range(5)
        ]
        self.replays = {
            t["TrackId"]: [
                {
                    "ReplayId": t["TrackId"] * 100 + j,
                    "ReplayTime": 20000 + 1000 * ((j * 7) % 10),  # unsorted, times 20..29 s
                    "User": {"Name": f"u{j}"},
                }
                for j in range(10)
            ]
            for t in self.tracks
        }

    def open(self, req, timeout=None):
        url = req.full_url
        self.requests.append(
            {"url": url, "ua": req.get_header("User-agent"), "t": self.clock() if self.clock else 0}
        )
        for key, queue in self.errors.items():
            if key in url and queue:
                raise queue.pop(0)
        parts = urllib.parse.urlsplit(url)
        q = dict(urllib.parse.parse_qsl(parts.query))
        if any(key in url for key in self.html):
            return io.BytesIO(b"<html>error</html>")
        if parts.path == "/api/tracks":
            return self._page(self.tracks, "TrackId", q)
        if parts.path == "/api/replays":
            if int(q["trackId"]) not in self.replays:
                raise http_error(url, 404)
            return self._page(self.replays[int(q["trackId"])], "ReplayId", q)
        if parts.path.startswith("/trackgbx/"):
            tid = int(parts.path.rsplit("/", 1)[1])
            uid = next((t["UId"] for t in self.tracks if t["TrackId"] == tid), f"HDR{tid}")
            return io.BytesIO(b'GBX\x00<header><ident uid="' + uid.encode() + b'" name="n"/>')
        if parts.path.startswith("/recordgbx/"):
            return io.BytesIO(b"GBX replay " + parts.path.rsplit("/", 1)[1].encode())
        raise http_error(url, 404)

    @staticmethod
    def _page(items: list[dict], id_key: str, q: dict):
        items = sorted(items, key=lambda r: r[id_key])
        if "after" in q:
            items = [r for r in items if r[id_key] > int(q["after"])]
        count = int(q.get("count", 100))
        body = {"Results": items[:count], "More": len(items) > count}
        return io.BytesIO(json.dumps(body).encode())

    def urls(self, fragment: str = "") -> list[str]:
        return [r["url"] for r in self.requests if fragment in r["url"]]


class Boom:
    def open(self, req, timeout=None):
        raise AssertionError(f"unexpected request {req.full_url}")


@pytest.fixture
def env(tmp_path: Path):
    clock = FakeClock()
    mock = MockTMX(clock)

    def run(*args: str, opener=None) -> int:
        base = [FLAG, "--user-agent", UA, "--out", str(tmp_path / "out")]
        base += ["--map-dir", str(tmp_path / "maps"), "--replays-dir", str(tmp_path / "replays")]
        return tmx.main([*base, *args], opener=opener or mock, clock=clock, sleep=clock.sleep)

    return tmp_path, clock, mock, run


# ----------------------------------------------------------------- gate


def test_refuses_without_the_terms_flag(capsys):
    assert tmx.main(["tracks"], opener=Boom()) == 2
    out = capsys.readouterr().out
    assert tmx.TERMS_URL in out and "refusing to run" in out and FLAG in out
    assert tmx.main(["--dry-run", "replays"], opener=Boom()) == 2  # also for dry runs


def test_runs_when_the_flag_is_given(tmp_path: Path):
    argv = [FLAG, "--user-agent", UA, "--out", str(tmp_path), "tracks", "--max-tracks", "0"]
    assert tmx.main(argv, opener=Boom()) == 0  # gate open, nothing to fetch


def test_terms_url_comes_from_the_research_notes():
    assert tmx.TERMS_URL in (REPO / "docs" / "research.md").read_text(encoding="utf-8")


def test_placeholder_user_agent_is_refused_except_for_dry_runs(tmp_path: Path, capsys):
    argv = [FLAG, "--out", str(tmp_path), "--map-dir", str(tmp_path / "m")]
    assert tmx.main([*argv, "tracks"], opener=Boom()) == 2
    assert "contact" in capsys.readouterr().out
    assert tmx.main([*argv, "--dry-run", "tracks"], opener=Boom()) == 0
    assert (
        "tmagent-research" in tmx.DEFAULT_USER_AGENT
        and tmx.UA_PLACEHOLDER in tmx.DEFAULT_USER_AGENT
    )


# ----------------------------------------------------------------- dry run


def test_dry_run_prints_the_requests_and_sends_nothing(tmp_path: Path, capsys):
    out = tmp_path / "out"
    argv = [FLAG, "--dry-run", "--out", str(out), "--map-dir", str(tmp_path / "maps")]
    argv += ["--replays-dir", str(tmp_path / "replays")]
    assert tmx.main([*argv, "tracks", "--param", "environment=1", "--param", "difficulty=2",
                     "--max-tracks", "20"], opener=Boom()) == 0  # fmt: skip
    text = capsys.readouterr().out
    assert (
        "[dry-run] GET https://tmnf.exchange/api/tracks?fields=TrackId%2CUId%2CTrackName"
        "&environment=1&difficulty=2&count=20" in text
    )
    assert tmx.main([*argv, "replays", "--track-ids", "11,22"], opener=Boom()) == 0
    text = capsys.readouterr().out
    assert "[dry-run] GET https://tmnf.exchange/trackgbx/11 ->" in text
    assert "[dry-run] GET https://tmnf.exchange/trackgbx/22 ->" in text
    assert (
        "[dry-run] GET https://tmnf.exchange/api/replays?trackId=11"
        "&fields=ReplayId%2CReplayTime%2CUser.Name&count=1000" in text
    )
    assert not (tmp_path / "maps").exists() and not (tmp_path / "replays").exists()
    assert not (out / "tracks.jsonl").exists()


def test_dry_run_expands_downloads_from_cached_listings(env, capsys):
    tmp_path, clock, mock, run = env
    assert run("replays", "--track-ids", "100", "--per-track", "3") == 0
    capsys.readouterr()
    n = len(mock.requests)
    # another destination, same cache: the chosen replays and the map are shown, none sent
    argv = [FLAG, "--dry-run", "--out", str(tmp_path / "out"), "--map-dir", str(tmp_path / "m2")]
    argv += ["--replays-dir", str(tmp_path / "r2"), "replays", "--track-ids", "100"]
    argv += ["--per-track", "3"]
    assert tmx.main(argv, opener=Boom()) == 0
    text = capsys.readouterr().out
    assert text.count("/recordgbx/") == 3 and "/trackgbx/100" in text
    assert len(mock.requests) == n


# ----------------------------------------------------------------- rate limit, backoff


def test_rate_limiter_spaces_request_starts_with_a_fake_clock():
    clock = FakeClock()
    lim = tmx.RateLimiter(1.0, clock, clock.sleep)
    starts = []
    for work in (0.0, 0.0, 0.3, 2.5, 0.0):
        lim.wait()
        starts.append(clock.t)
        clock.t += work  # the request itself takes `work` seconds
    gaps = [b - a for a, b in zip(starts, starts[1:], strict=False)]
    # slow requests count towards the interval: only the missing 0.7 s is slept after the
    # 0.3 s one, nothing after the 2.5 s one
    assert gaps == pytest.approx([1.0, 1.0, 1.0, 2.5])
    assert clock.sleeps == pytest.approx([1.0, 1.0, 0.7])


def test_rate_limiter_first_call_does_not_sleep_and_zero_interval_disables():
    clock = FakeClock()
    lim = tmx.RateLimiter(1.0, clock, clock.sleep)
    lim.wait()
    assert clock.sleeps == []
    off = tmx.RateLimiter(0.0, clock, clock.sleep)
    for _ in range(3):
        off.wait()
    assert clock.sleeps == []


def client_for(opener, clock: FakeClock, **kw) -> tmx.TMXClient:
    return tmx.TMXClient(
        UA, opener=opener, clock=clock, sleep=clock.sleep, log=lambda m: None, **kw
    )


def test_backoff_on_429_and_5xx_and_retry_after():
    clock = FakeClock()
    mock = MockTMX(clock)
    mock.errors["/api/tracks"] = [
        http_error("u", 429, "7"),
        http_error("u", 503),
        http_error("u", 500),
    ]
    c = client_for(mock, clock, min_interval=0.0, backoff_s=2.0)
    page = c.get_json("/api/tracks", [("count", 2)])
    assert len(page["Results"]) == 2
    assert clock.sleeps == [7.0, 4.0, 8.0]  # Retry-After first, then backoff 2 * 2**attempt
    assert c.requests == 4


def test_backoff_on_network_errors_and_exhaustion():
    clock = FakeClock()
    mock = MockTMX(clock)
    mock.errors["/api/tracks"] = [urllib.error.URLError("down"), TimeoutError("slow")]
    c = client_for(mock, clock, min_interval=0.0, backoff_s=1.0)
    assert c.get_json("/api/tracks")["Results"]
    assert clock.sleeps == [1.0, 2.0]
    mock.errors["/api/tracks"] = [http_error("u", 502)] * 10
    c2 = client_for(mock, clock, min_interval=0.0, backoff_s=0.5, max_retries=2)
    with pytest.raises(tmx.TMXError, match="giving up"):
        c2.get_json("/api/tracks", [("count", 1)])
    assert c2.requests == 3


def test_client_errors_are_not_retried():
    clock = FakeClock()
    mock = MockTMX(clock)
    c = client_for(mock, clock, min_interval=0.0)
    with pytest.raises(tmx.TMXError, match="HTTP 404"):
        c.get_json("/api/nothing")
    assert c.requests == 1 and clock.sleeps == []


def test_retry_after_parsing():
    assert tmx.retry_after_s(http_error("u", 429, "12")) == 12.0
    assert tmx.retry_after_s(http_error("u", 429, "99999")) == tmx.MAX_RETRY_AFTER_S
    assert tmx.retry_after_s(http_error("u", 429, "Wed, 21 Oct 2026 07:28:00 GMT")) is None
    assert tmx.retry_after_s(http_error("u", 429)) is None


# ----------------------------------------------------------------- selection


def test_select_spread_best_median_tail():
    ranks = list(range(1, 11))
    assert tmx.select_spread(ranks, 3) == [1, 5, 10]
    assert tmx.select_spread(ranks, 2) == [1, 10]
    assert tmx.select_spread(ranks, 1) == [1]
    assert tmx.select_spread(ranks, 0) == []
    assert tmx.select_spread(ranks, 10) == ranks and tmx.select_spread(ranks, 99) == ranks
    assert tmx.select_spread([], 3) == []
    for n_items in range(1, 40):
        for n in range(1, n_items + 1):
            picked = tmx.select_spread(list(range(n_items)), n)
            assert len(picked) == len(set(picked)) == n  # never fewer than asked, no repeats
            assert picked == sorted(picked) and picked[0] == 0
            if n > 1:
                assert picked[-1] == n_items - 1


def test_param_and_path_helpers():
    assert tmx.parse_params(["a=1", "b=x=y", "a=2"]) == [("a", "1"), ("b", "x=y"), ("a", "2")]
    for bad in ("novalue", "=1"):
        with pytest.raises(ValueError):
            tmx.parse_params([bad])
    rec = {"User": {"Name": "ann"}, "Id": 3}
    assert tmx.get_path(rec, "User.Name") == "ann" and tmx.get_path(rec, "Id") == 3
    assert tmx.get_path(rec, "User.Nope", "d") == "d" and tmx.get_path(rec, "X.Y") is None


# ----------------------------------------------------------------- full flow with resume


def test_tracks_mode_pages_caches_and_resumes(env):
    tmp_path, clock, mock, run = env
    assert run("tracks", "--param", "environment=1", "--max-tracks", "3", "--page-size", "2") == 0
    out = tmp_path / "out"
    stored = [json.loads(x) for x in (out / "tracks.jsonl").read_text().splitlines()]
    assert [t["TrackId"] for t in stored] == [100, 101, 102]
    urls = mock.urls("/api/tracks")
    assert len(urls) == 2 and "after" not in urls[0] and "after=101" in urls[1]
    assert "environment=1" in urls[0] and "count=2" in urls[0] and "count=1" in urls[1]
    assert all(r["ua"] == UA for r in mock.requests)
    times = [r["t"] for r in mock.requests]
    assert all(b - a >= 1.0 - 1e-9 for a, b in zip(times, times[1:], strict=False))  # 1 request/s

    # same query, higher limit: continues from the cursor, only the missing page is requested
    assert run("tracks", "--param", "environment=1", "--max-tracks", "5", "--page-size", "2") == 0
    assert len(mock.urls("/api/tracks")) == 3 and "after=102" in mock.urls("/api/tracks")[2]
    assert [json.loads(x)["TrackId"] for x in (out / "tracks.jsonl").read_text().splitlines()] == [
        100, 101, 102, 103, 104
    ]  # fmt: skip
    # finished: nothing more to request
    n = len(mock.requests)
    assert run("tracks", "--param", "environment=1", "--max-tracks", "50") == 0
    assert len(mock.requests) == n
    state = json.loads((out / "state.json").read_text())
    assert state["tracks"]["done"] and state["tracks"]["count"] == 5


def test_replays_mode_downloads_a_rank_spread_and_the_maps(env):
    tmp_path, clock, mock, run = env
    assert run("tracks", "--max-tracks", "2") == 0
    assert run("replays", "--per-track", "3") == 0  # tracks from <out>/tracks.jsonl
    maps, replays = tmp_path / "maps", tmp_path / "replays"
    assert sorted(p.name for p in maps.glob("*.Challenge.Gbx")) == [
        "UID0.Challenge.Gbx",
        "UID1.Challenge.Gbx",
    ]
    assert json.loads((maps / "index.json").read_text()) == {
        "UID0": "UID0.Challenge.Gbx",
        "UID1": "UID1.Challenge.Gbx",
    }
    files = sorted(p.name for p in replays.glob("*.Replay.Gbx"))
    assert len(files) == 6 and files[0].startswith("100-") and files[-1].startswith("101-")
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    chosen = [v for v in state["replays"].values() if v["track_id"] == "100"]
    assert sorted(v["rank"] for v in chosen) == [1, 5, 10]  # best, median, tail of 10
    best = min(chosen, key=lambda v: v["rank"])
    assert best["time_ms"] == 20000 and best["of"] == 10
    assert next(replays.glob("100-*")).read_bytes().startswith(b"GBX")
    assert all(not r["url"].endswith(".part") for r in mock.requests)
    assert not list(replays.glob("*.part")) and not list(maps.glob("*.part"))

    # a rerun is free: listings are cached, files and state exist
    n = len(mock.requests)
    assert run("replays", "--per-track", "3") == 0
    assert len(mock.requests) == n


def test_replays_mode_resumes_after_failures(env):
    tmp_path, clock, mock, run = env
    # one replay download fails for good (404), one map download answers with an HTML page
    mock.errors["/recordgbx/10000"] = [http_error("u", 404)]
    mock.html.add("/trackgbx/101")
    assert run("replays", "--track-ids", "100,101", "--per-track", "3") == 0
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    assert set(state["failed"]) == {"replay:10000", "track:101"}
    assert "not a GBX file" in state["failed"]["track:101"]
    got = sorted(p.name for p in (tmp_path / "replays").glob("*.Replay.Gbx"))
    assert len(got) == 2 and all(x.startswith("100-") for x in got)  # track 101 was abandoned
    assert not (tmp_path / "maps" / "HDR101.Challenge.Gbx").exists()

    # the problems are fixed: only the missing pieces are requested
    mock.html.clear()
    before = len(mock.requests)
    assert run("replays", "--track-ids", "100,101", "--per-track", "3") == 0
    new = [r["url"] for r in mock.requests[before:]]
    assert sum("/recordgbx/10000" in u for u in new) == 1
    assert sum("/api/replays" in u for u in new) == 1 and sum("/trackgbx/" in u for u in new) == 1
    assert sum("/recordgbx/" in u for u in new) == 1 + 3  # the failed one + track 101's three
    assert len(list((tmp_path / "replays").glob("*.Replay.Gbx"))) == 6


def test_map_name_falls_back_to_the_gbx_header_uid(env):
    tmp_path, clock, mock, run = env
    # track 777 is not in any tracks file: no uid from the API, the uid comes from the header
    assert run("replays", "--track-ids", "777", "--per-track", "1") == 0  # its replay list 404s
    maps = tmp_path / "maps"
    assert (maps / "HDR777.Challenge.Gbx").is_file()
    assert json.loads((maps / "index.json").read_text()) == {"HDR777": "HDR777.Challenge.Gbx"}
    state = json.loads((tmp_path / "out" / "state.json").read_text())
    assert state["maps"] == {"777": "HDR777.Challenge.Gbx"}
    assert "track:777" in state["failed"]  # the failed listing is recorded, the run goes on


def test_no_maps_flag_and_max_tracks(env):
    tmp_path, clock, mock, run = env
    assert run("tracks", "--max-tracks", "5") == 0
    assert run("replays", "--per-track", "1", "--no-maps", "--max-tracks", "2") == 0
    assert not (tmp_path / "maps").exists()
    assert len(list((tmp_path / "replays").glob("*.Replay.Gbx"))) == 2
    assert run("replays", "--tracks-file", str(tmp_path / "missing.jsonl")) == 1


def test_uid_from_gbx_header_and_map_index(tmp_path: Path):
    f = tmp_path / "x.Challenge.Gbx"
    f.write_bytes(b'GBX\x00\x01<header type="challenge"><ident uid="AbC_12-x" name="n"/></header>')
    assert tmx.uid_from_gbx_header(f) == "AbC_12-x"
    g = tmp_path / "y.Challenge.Gbx"
    g.write_bytes(b"GBX\x00no header here")
    assert tmx.uid_from_gbx_header(g) is None and tmx.uid_from_gbx_header(tmp_path / "no") is None
    tmx.update_map_index(tmp_path, "u1", "a.Challenge.Gbx")
    tmx.update_map_index(tmp_path, "u2", "b.Challenge.Gbx")
    tmx.update_map_index(tmp_path, "u1", "c.Challenge.Gbx")  # overwrite
    assert json.loads((tmp_path / "index.json").read_text()) == {
        "u1": "c.Challenge.Gbx",
        "u2": "b.Challenge.Gbx",
    }


def test_read_tracks_file_dedupes(tmp_path: Path):
    p = tmp_path / "tracks.jsonl"
    rows = [{"TrackId": 1, "UId": "a"}, {"TrackId": 2}, {"TrackId": 1, "UId": "dup"}, {"x": 1}]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert [r["TrackId"] for r in tmx.read_tracks_file(p)] == [1, 2]
    assert tmx.read_tracks_file(p)[0]["UId"] == "a"
