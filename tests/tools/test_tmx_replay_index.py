"""tools/tmx_replay_index.py on a fake downloader dir (cached lists only, no network)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tools import tmx_download as tmx
from tools import tmx_replay_index as idx


def make_source(root: Path) -> Path:
    source = root / "tmx"
    tracks = [
        {"TrackId": 1, "UId": "U1", "TrackName": "Ä-Map"},
        {"TrackId": 2, "UId": "U2", "TrackName": "Uncached"},
    ]
    (source / "cache").mkdir(parents=True)
    (source / "tracks.jsonl").write_text(
        "".join(json.dumps(t) + "\n" for t in tracks), encoding="utf-8"
    )
    (source / "state.json").write_text(json.dumps({"rejected": {"12": "1"}}), encoding="utf-8")

    client = tmx.TMXClient("test", cache_dir=source / "cache", dry_run=True)
    params = [
        (tmx.P_REPLAY_TRACK, "1"),
        (tmx.P_FIELDS, ",".join(tmx.REPLAY_FIELDS)),
        (tmx.P_COUNT, 1000),
    ]
    page = {
        "More": False,
        "Results": [
            {"ReplayId": 13, "ReplayTime": 61230, "User": {"Name": "c"}},
            {"ReplayId": 11, "ReplayTime": 30000, "User": {"Name": "a"}},
            {"ReplayId": 12, "ReplayTime": 45000, "User": {"Name": "b"}},
        ],
    }
    client._cache_path(client.url(tmx.EP_REPLAYS, params)).write_text(
        json.dumps(page), encoding="utf-8"
    )
    return source


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_offline_index_marks_status(tmp_path: Path, monkeypatch) -> None:
    source = make_source(tmp_path)
    (tmp_path / "replays_a").mkdir()
    (tmp_path / "replays_a" / "1-11.Replay.Gbx").write_bytes(b"GBX")
    monkeypatch.chdir(tmp_path)

    out = tmp_path / "index.csv"
    assert (
        idx.main(
            ["--offline", "--source", str(source), "--replays-glob", "replays_*", "--out", str(out)]
        )
        == 0
    )

    rows = read_csv(out)
    assert [(r["replay_id"], r["rank"], r["status"]) for r in rows] == [
        ("11", "1", "downloaded"),
        ("12", "2", "rejected"),
        ("13", "3", "new"),
    ]
    assert rows[0]["map"] == "Ä-Map"
    assert rows[2]["time"] == "1:01.23"
    assert rows[2]["download_url"] == "https://tmnf.exchange/recordgbx/13"

    maps = {m["track_id"]: m for m in read_csv(tmp_path / "index_maps.csv")}
    assert (maps["1"]["listed"], maps["1"]["new"]) == ("3", "1")
    assert maps["2"]["listed"] == ""


def test_online_requires_terms_and_contact(tmp_path: Path) -> None:
    out = str(tmp_path / "i.csv")
    assert idx.main(["--out", out]) == 2
    assert idx.main([tmx.TERMS_FLAG, "--out", out]) == 2
