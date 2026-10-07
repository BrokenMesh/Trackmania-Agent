"""tools/tmnf_smoke.py runs end to end against the fake plugin and reports failures."""

from __future__ import annotations

import importlib.util
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest

from tmagent.game.tmnf.fake_server import reference_finish_time

TOOL = Path(__file__).resolve().parents[2] / "tools" / "tmnf_smoke.py"


@pytest.fixture(scope="module")
def smoke():
    spec = importlib.util.spec_from_file_location("tmnf_smoke", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolves string annotations through sys.modules
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(spec.name, None)


def read_png(path: Path) -> np.ndarray:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, ihdr = 8, b"", None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag, body = data[pos + 4 : pos + 8], data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        assert crc == zlib.crc32(tag + body)
        if tag == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", body)
        elif tag == b"IDAT":
            idat += body
        pos += 12 + length
    w, h, depth, ctype, *_ = ihdr
    assert depth == 8 and ctype in (0, 2)
    c = 1 if ctype == 0 else 3
    raw = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + w * c)
    assert (raw[:, 0] == 0).all()  # filter type none
    return raw[:, 1:].reshape(h, w, c)


@pytest.mark.parametrize("channels", [1, 3])
def test_write_png_roundtrip(smoke, tmp_path, channels):
    img = np.random.default_rng(0).integers(0, 256, (7, 5, channels), dtype=np.uint8)
    smoke.write_png(tmp_path / "a.png", img)
    assert np.array_equal(read_png(tmp_path / "a.png"), img)


def test_smoke_fake_end_to_end(smoke, tmp_path, capsys):
    rc = smoke.main(
        ["--fake", "--out", str(tmp_path), "--realtime-seconds", "0.6", "--grabs", "20"]
    )
    out = capsys.readouterr().out
    assert rc == 0, out
    for needle in (
        "connect + handshake",
        "step additivity",
        "grab frames",
        "realtime mode",
        "determinism",
        "replay re-drive",
    ):
        assert f"[PASS  ] {needle}" in out, needle
    assert "[FAIL  ]" not in out
    report = (tmp_path / "tmnf_smoke_report.md").read_text()
    assert "FAKE protocol server" in report and "| PASS | replay re-drive |" in report
    assert "grab_frame latency" in report
    pngs = sorted(tmp_path.glob("frame_t*.png"))
    assert [p.name for p in pngs] == [f"frame_t{t:05d}.png" for t in (0, 250, 500, 1000, 1500)]
    img = read_png(pngs[3])
    assert img.shape == (96, 128, 3)
    assert (int(img[48, 64, 0]) << 16 | int(img[48, 64, 1]) << 8 | int(img[48, 64, 2])) == 1000


def test_smoke_reports_failures_and_nonzero_exit(smoke, tmp_path, capsys):
    rc = smoke.main(
        [
            "--map",
            "x",
            "--out",
            str(tmp_path),
            "--set",
            "game.tmi_port=1",
            "--set",
            "game.connect_timeout_s=0.3",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL  ] connect + handshake" in out and "[SKIP  ] remaining steps" in out
    assert "| FAIL | connect + handshake |" in (tmp_path / "tmnf_smoke_report.md").read_text()


def test_smoke_replay_check_compares_the_finish_time(smoke, tmp_path, capsys):
    script = tmp_path / "run.txt"
    script.write_text("0-4500 press up\n")
    ref = reference_finish_time(np.tile([[0.0, 1.0, 0.0]], (450, 1)))
    common = ["--fake", "--realtime-seconds", "0.3", "--grabs", "5", "--replay", str(script)]
    assert smoke.main([*common, "--out", str(tmp_path / "a"), "--expect-time-ms", str(ref)]) == 0
    out = capsys.readouterr().out
    assert f"finish {ref} ms vs replay {ref} ms (diff +0 ms)" in out
    # a wrong expectation fails and the report carries the hint
    assert (
        smoke.main([*common, "--out", str(tmp_path / "b"), "--expect-time-ms", str(ref + 10)]) == 1
    )
    assert "[FAIL  ] replay re-drive" in capsys.readouterr().out
    assert "keyboard-only" in (tmp_path / "b" / "tmnf_smoke_report.md").read_text()
    # shifting the inputs by 3 ticks delays the finish, so it no longer matches the unshifted time
    assert (
        smoke.main(
            [
                *common,
                "--out",
                str(tmp_path / "c"),
                "--expect-time-ms",
                str(ref),
                "--tick-offset",
                "3",
            ]
        )
        == 1
    )
    assert "diff +30 ms" in capsys.readouterr().out


def test_smoke_missing_replay_file_is_a_failed_step(smoke, tmp_path, capsys):
    rc = smoke.main(["--fake", "--out", str(tmp_path), "--realtime-seconds", "0.3", "--grabs", "5",
                     "--replay", str(tmp_path / "nope.txt")])  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 1 and "[FAIL  ] replay re-drive" in out
