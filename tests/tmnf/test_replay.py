"""replay.py: ghost control entries, TMI input scripts, timelines. pygbx is not needed."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
import pytest

from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf import replay as R

SIGN = R.ANALOG_STEER_SIGN


@dataclass
class Entry:
    """Stand-in for pygbx ControlEntry."""

    time: int
    event_name: str
    enabled: int = 0
    flags: int = 0


# ------------------------------------------------------------------- analog


@pytest.mark.parametrize(
    ("enabled", "flags", "raw"),
    [
        (0, 0, 0),
        (1000, 0, -1000),  # dir 0: -val
        (65535, 0, -65535),
        (1000, 0xFF, 65536 - 1000),  # dir 0xFF: 65536 - val
        (0, 0xFF, 65536),  # full right
        (65535, 0xFF, 1),
        (0, 1, -65536),  # dir 1: full left whatever val is
        (1234, 1, -65536),
        (100, 2, -300),  # dir 2: -val * 3 (out of range values are clipped later)
        (100, 0x1FF, 65536 - 100),  # only the low byte of the high word is the direction
    ],
)
def test_decode_analog_table(enabled, flags, raw):
    assert R.decode_analog(enabled, flags) == raw


def test_analog_steer_normalized_and_clipped():
    entries = [
        Entry(10, "Steer", 1000, 0xFF),  # 64536 / 65536 right
        Entry(20, "Steer", 0, 1),  # full left
        Entry(30, "Steer", 40000, 2),  # -120000 -> clipped to -1
        Entry(40, "Steer", 0, 0),
    ]
    ev = R.events_from_control_entries(entries)
    assert [e[:2] for e in ev] == [(10, "steer"), (20, "steer"), (30, "steer"), (40, "steer")]
    vals = [e[2] for e in ev]
    assert vals == pytest.approx([SIGN * 64536 / 65536, -SIGN * 1.0, -SIGN * 1.0, 0.0])
    assert all(-1.0 <= v <= 1.0 for v in vals)


def test_steer_sign_constant_matches_protocol():
    assert (SIGN == 1) == P.STEER_NEGATIVE_IS_LEFT


# ------------------------------------------------------------------ digital


def test_digital_events_and_names():
    entries = [
        Entry(10, "Accelerate", 0x80),
        Entry(20, "SteerLeft", 0x80),
        Entry(30, "SteerLeft", 0),
        Entry(40, "SteerRight", 0, 0x0001),  # any non-zero data counts as pressed
        Entry(50, "Brake", 1),
        Entry(60, "Brake", 0),
        Entry(70, "Accelerate", 0),
    ]
    assert R.events_from_control_entries(entries) == [
        (10, "accelerate", 1.0),
        (20, "steer_left", 1.0),
        (30, "steer_left", 0.0),
        (40, "steer_right", 1.0),
        (50, "brake", 1.0),
        (60, "brake", 0.0),
        (70, "accelerate", 0.0),
    ]


def test_gas_threshold_and_brake_from_negative_gas():
    full = Entry(10, "Gas", 0, 0xFF)  # +65536
    mid = Entry(20, "Gas", 25536 + 0, 0xFF)  # 65536 - 25536 = 40000 -> 0.61 -> on
    low = Entry(30, "Gas", 50000, 0xFF)  # 15536 -> 0.237 < threshold -> off
    neg = Entry(40, "Gas", 30000, 0)  # -30000 -> brake
    off = Entry(50, "Gas", 0, 0)
    ev = R.events_from_control_entries([full, mid, low, neg, off])
    assert ev == [
        (10, "gas", 1.0),
        (20, "gas", 1.0),
        (30, "gas", 0.0),
        (40, "gas", 0.0),
        (40, "brake", 1.0),
        (50, "gas", 0.0),
        (50, "brake", 0.0),
    ]


def test_respawn_is_counted_and_flagged_not_dropped():
    meta: dict = {}
    entries = [Entry(500, "Respawn", 1), Entry(510, "Respawn", 0), Entry(2000, "Respawn", 1)]
    ev = R.events_from_control_entries(entries, meta=meta)
    assert meta["respawns"] == 2
    assert [e[1] for e in ev] == ["_respawn"] * 3
    R.events_from_control_entries(entries)  # meta is optional


def test_fake_and_misc_events_pass_through_with_underscore():
    entries = [
        Entry(0, "_FakeIsRaceRunning", 1),
        Entry(5000, "_FakeFinishLine", 1),
        Entry(100, "Horn", 1),
        Entry(110, "AccelerateReal", 1),
        Entry(120, "_FakeDontInverseAxis", 1),
    ]
    names = [e[1] for e in R.events_from_control_entries(entries)]
    assert names == [
        "_fake_is_race_running",
        "_fake_finish_line",
        "_horn",
        "_accelerate_real",
        "_fake_dont_inverse_axis",
    ]


def test_unknown_names_warn_once_and_skip():
    entries = [Entry(10, "Teleport", 1), Entry(20, "Teleport", 0), Entry(30, "Accelerate", 1)]
    with pytest.warns(RuntimeWarning, match="Teleport") as rec:
        ev = R.events_from_control_entries(entries)
    assert ev == [(30, "accelerate", 1.0)]
    assert len(rec) == 1


def test_control_names_fallback_for_indexed_entries():
    @dataclass
    class Indexed:
        time: int
        event_index: int
        enabled: int
        flags: int = 0

    ev = R.events_from_control_entries([Indexed(10, 1, 1)], control_names=["Brake", "Accelerate"])
    assert ev == [(10, "accelerate", 1.0)]


def test_events_feed_the_real_timeline():
    entries = [
        Entry(0, "_FakeIsRaceRunning", 1),
        Entry(10, "Accelerate", 0x80),
        Entry(30, "SteerRight", 0x80),
        Entry(50, "SteerRight", 0),
        Entry(60, "Brake", 1),
    ]
    tl = R.replay_to_timeline(
        R.events_from_control_entries(entries), {"race_time_ms": 85, "player": "x"}
    )
    assert tl.actions.shape == (9, 3)  # ceil(85 / 10) ticks
    assert tl.meta["player"] == "x"
    steer, gas, brake = tl.actions[:, 0], tl.actions[:, 1], tl.actions[:, 2]
    assert gas.tolist() == [0.0] + [1.0] * 8  # tick 0 is neutral, the first input comes at tick 1
    assert steer.tolist() == [0, 0, 0, 1, 1, 0, 0, 0, 0]
    assert brake.tolist() == [0] * 6 + [1, 1, 1]


# ------------------------------------------------------------------- script


SCRIPT = """
# comment line
0-100 press up
50 press left   # trailing comment
80 rel left
120 press down
150 rel up
150 rel down
200-300 press right
400 steer -32768
410 steer 65536
420 gas 65536
430 gas -30000
440 gas 0
"""


def test_parse_script_events_and_ranges():
    ev = R.parse_tmi_input_script(SCRIPT)
    assert ev[:4] == [
        (0, "accelerate", 1.0),
        (50, "steer_left", 1.0),
        (80, "steer_left", 0.0),
        (100, "accelerate", 0.0),  # range end sorted into place
    ]
    assert (200, "steer_right", 1.0) in ev and (300, "steer_right", 0.0) in ev
    assert (120, "brake", 1.0) in ev and (150, "brake", 0.0) in ev
    assert (400, "steer", SIGN * -0.5) in ev
    assert (410, "steer", SIGN * 1.0) in ev
    assert (420, "gas", 1.0) in ev
    assert (430, "gas", 0.0) in ev and (430, "brake", 1.0) in ev
    assert (440, "brake", 0.0) in ev
    assert [e[0] for e in ev] == sorted(e[0] for e in ev)


def test_parse_script_case_decimal_and_blank():
    assert R.parse_tmi_input_script("\n   \n10 PRESS UP\n1.5 press left\n") == [
        (10, "accelerate", 1.0),
        (1500, "steer_left", 1.0),
    ]
    assert R.parse_tmi_input_script("") == []


@pytest.mark.parametrize(
    ("text", "line", "fragment"),
    [
        ("0 press up\nfoo bar\n", 2, "cannot parse"),
        ("# c\n\n10 press jump\n", 3, "unknown key"),
        ("10 press\n", 1, "unknown key"),
        ("10 steer abc\n", 1, "integer"),
        ("10 steer\n", 1, "integer"),
        ("10 steer 70000\n", 1, "outside"),
        ("300-100 press up\n", 1, "range end"),
        ("0-100 rel up\n", 1, "range"),
        ("0-100 steer 5\n", 1, "range"),
        ("10 explode now\n", 1, "unknown command"),
        ("-5 press up\n", 1, "cannot parse"),
    ],
)
def test_parse_script_errors_name_the_line(text, line, fragment):
    with pytest.raises(ValueError, match=rf"line {line}: .*{fragment}"):
        R.parse_tmi_input_script(text)


def test_script_to_timeline_via_file(tmp_path):
    f = tmp_path / "run.txt"
    f.write_text("0-50 press up\n20-40 press right\n")
    tl = R.replay_to_timeline(f, {"race_time_ms": 100})
    assert tl.actions.shape == (10, 3)
    assert tl.actions[:5, 1].tolist() == [1.0] * 5 and tl.actions[5:, 1].sum() == 0
    assert tl.actions[:, 0].tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert tl.meta["source"] == "script:run.txt" and tl.meta["race_time_ms"] == 100
    assert tl.actions.dtype == np.float32


def test_timeline_without_duration_uses_last_event():
    tl = R.replay_to_timeline(R.parse_tmi_input_script("0 press up\n95 rel up\n"))
    assert len(tl.actions) == 10


# --------------------------------------------------------------------- gbx


def test_load_replay_without_pygbx_explains_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pygbx", None)  # makes `import pygbx` raise ImportError
    with pytest.raises(
        ImportError, match=r"pip install git\+https://github.com/donadigo/pygbx.*python-lzo"
    ):
        R.load_replay("x.Replay.Gbx")


def test_load_replay_meta_with_a_fake_pygbx(monkeypatch, tmp_path):
    """Exercise load_replay's meta extraction against a minimal pygbx look-alike."""
    import types

    ghost = types.SimpleNamespace(
        race_time=1230, num_respawns=0, cp_times=[500, 1230], login="driver", uid="GHOSTUID",
        control_names=["Accelerate"], game_version="TMNF",
        control_entries=[Entry(10, "Accelerate", 0x80), Entry(1230, "_FakeFinishLine", 1)],
    )  # fmt: skip
    challenge = types.SimpleNamespace(map_uid="MAPUID", map_name="Map", map_author="auth")

    class FakeGbxInner:
        def get_class_by_id(self, t):
            return challenge

    class FakeGbx:
        def __init__(self, path):
            self.path = path

        def get_class_by_id(self, t):
            if t == "GHOST":
                return ghost
            return types.SimpleNamespace(track=FakeGbxInner())

    mod = types.SimpleNamespace(
        Gbx=FakeGbx,
        GbxType=types.SimpleNamespace(CTN_GHOST="GHOST", REPLAY_RECORD="REC", CHALLENGE="CH"),
    )
    monkeypatch.setitem(sys.modules, "pygbx", mod)
    (tmp_path / "a.Replay.Gbx").write_bytes(b"GBX fake")
    ev, meta = R.load_replay(tmp_path / "a.Replay.Gbx")
    assert ev == [(0, "accelerate", 1.0), (1220, "_fake_finish_line", 1.0)]  # GBX_INPUT_LEAD_MS
    assert meta == {
        "map_uid": "MAPUID", "map_name": "Map", "map_author": "auth", "player": "driver",
        "race_time_ms": 1230, "num_respawns": 0, "cp_times": [500, 1230], "game_version": "TMNF",
        "source": "file:a.Replay.Gbx", "respawns": 0,
    }  # fmt: skip
    # a missing embedded challenge falls back to the ghost uid
    challenge.map_uid = ""
    _, meta2 = R.load_replay(tmp_path / "a.Replay.Gbx")
    assert meta2["map_uid"] == "GHOSTUID"


def test_align_gbx_events_measures_from_race_start_and_leads_one_tick():
    offset = [
        (65535, "_fake_is_race_running", 1.0),
        (65545, "accelerate", 1.0),
        (66356, "steer_right", 1.0),
    ]
    assert R.align_gbx_events(offset) == [
        (-10, "_fake_is_race_running", 1.0), (0, "accelerate", 1.0), (811, "steer_right", 1.0),
    ]  # fmt: skip
    no_start = [(10, "accelerate", 1.0)]
    assert R.align_gbx_events(no_start) == [(0, "accelerate", 1.0)]


def test_map_uid_from_replay_header(tmp_path):
    p = tmp_path / "a.Replay.Gbx"
    p.write_bytes(
        b'GBX\x06\x00BUCR<header type="replay" version="TMr.7" exever="2.11.16">'
        b'<challenge uid="BeySZdnfuSh4nHY5xztiXLmlrXe"/><times best="36020"/></header>\x00\x01'
    )
    assert R.map_uid_from_replay_header(p) == "BeySZdnfuSh4nHY5xztiXLmlrXe"
    (tmp_path / "b.Replay.Gbx").write_bytes(b"GBX no xml header")
    assert R.map_uid_from_replay_header(tmp_path / "b.Replay.Gbx") is None
    assert R.map_uid_from_replay_header(tmp_path / "missing.Replay.Gbx") is None


def test_steer_left_wins_while_both_keys_are_held():
    events = [
        (0, "accelerate", 1.0),
        (100, "steer_right", 1.0),
        (150, "steer_left", 1.0),  # right is held: left takes over
        (200, "steer_left", 0.0),  # right still held: right is back
        (300, "steer_right", 0.0),
        (400, "steer_left", 1.0),
        (450, "steer_right", 1.0),  # no effect while left is held
        (500, "steer_right", 0.0),
        (600, "steer_left", 0.0),
    ]
    assert R.resolve_steer_overlap(events) == [
        (0, "accelerate", 1.0),
        (100, "steer_right", 1.0),
        (150, "steer_left", 1.0),
        (150, "steer_right", 0.0),
        (200, "steer_left", 0.0),
        (200, "steer_right", 1.0),
        (300, "steer_right", 0.0),
        (400, "steer_left", 1.0),
        (600, "steer_left", 0.0),
    ]
    tl = R.replay_to_timeline(R.resolve_steer_overlap(events), {"race_time_ms": 700})
    steer = tl.actions[:, 0]
    assert steer[17] == -1.0 and steer[25] == 1.0 and steer[47] == -1.0 and steer[65] == 0.0
