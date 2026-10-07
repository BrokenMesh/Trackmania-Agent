"""Save the maps embedded in .Replay.Gbx files, so replays of unknown maps can be rendered.

    python tools/extract_replay_maps.py --replays data/tmnf/replays_brokenmesh --config configs/tmnf.yaml

Every TMNF replay carries its map (chunk 0x03093002). The map is written to game.map_dir as
<uid>.Challenge.Gbx (uid from the replay's XML header); existing files are never touched.
VERIFIED quirk: the game only knows maps that existed when it started -> restart it after.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pygbx.gbx as pg

from tmagent.config import load_config
from tmagent.game.tmnf.replay import map_uid_from_replay_header


def embedded_map_bytes(replay: Path) -> bytes | None:
    """Raw GBX bytes of the map inside a replay, or None. pygbx parses the embedded map with
    the module-level Gbx class and drops the bytes; a capturing subclass keeps them."""
    original = pg.Gbx
    captured: list[bytes] = []

    class Capture(original):  # type: ignore[misc, valid-type]
        def __init__(self, obj):
            if isinstance(obj, (bytes, bytearray)):
                captured.append(bytes(obj))
            super().__init__(obj)

    pg.Gbx = Capture
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original(replay.read_bytes())
    except Exception:
        return None
    finally:
        pg.Gbx = original
    return captured[0] if captured and captured[0][:3] == b"GBX" else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Extract maps embedded in replays.")
    ap.add_argument("--replays", required=True, help="directory of .Replay.Gbx files")
    ap.add_argument("--config", required=True, help="config with game.map_dir")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(errors="replace")

    map_dir = Path(load_config(args.config, []).game.map_dir)
    have = {p.name.split(".")[0] for p in map_dir.glob("*.Challenge.Gbx")}
    added = failed = 0
    for replay in sorted(Path(args.replays).iterdir()):
        uid = map_uid_from_replay_header(replay)
        if not uid or uid in have:
            continue
        data = embedded_map_bytes(replay)
        if data is None:
            failed += 1
            print(f"no embedded map: {replay.name}")
            continue
        (map_dir / f"{uid}.Challenge.Gbx").write_bytes(data)
        have.add(uid)
        added += 1
    print(f"added {added} maps to {map_dir}, {failed} replays without a readable map")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
