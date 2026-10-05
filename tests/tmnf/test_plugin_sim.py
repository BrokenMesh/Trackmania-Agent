"""Opt-in: run the real TMAgentLink.as in a simulated TMI engine and drive it with the client.

The plugin is AngelScript and cannot run in CI. tests/tmnf/plugin_sim/ holds a small C++
host (AngelScript 2.35) with stubs of the TMI API and a toy engine; the real Python client
talks to the real plugin over TCP. This checks syntax/type errors and the state machine
(pause, STEP accounting, restart, finish, realtime, reconnect) under several engine
behaviours (variant bits in stubs.as). It does NOT prove anything about the real engine.

Known limitation (xfail): if RaceTime reads stale right after RewindToState (bit 16) AND
SetSpeed(0) is not immediate AND the callback precedes the tick (bit 1), the paused guard
can make a STEP lose a tick. tools/tmnf_smoke.py ("step additivity", "plugin diagnostics")
detects this on a real install; `game.render_speed: 1` reduces the exposure.

Enable with TMAGENT_PLUGIN_SIM=1 (needs g++ and `apt install angelscript-dev
libangelscript-addon2.35.1t64`); TMAGENT_PLUGIN_SIM=all runs all realistic variants.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tmagent.game.tmnf import protocol as P
from tmagent.game.tmnf.client import TMAgentClient, convert_pixels

HERE = Path(__file__).resolve().parent / "plugin_sim"
PLUGIN = (
    Path(__file__).resolve().parents[2] / "tmagent" / "game" / "tmnf" / "plugin" / "TMAgentLink.as"
)
INC = os.environ.get("ANGELSCRIPT_INC", "/usr/include/angelscript")
ADDON = os.environ.get(
    "ANGELSCRIPT_ADDON", "/usr/lib/x86_64-linux-gnu/libangelscript-addon.so.2.35.1"
)
MODE = os.environ.get("TMAGENT_PLUGIN_SIM", "")

# bits: 1 callback before tick, 2 SetSpeed immediate, 4 no state events, 8 no countdown
# callbacks (only realistic together with 1), 16 RaceTime lags after a rewind
ALL = [v for v in range(32) if not (v & 8 and not v & 1)]
VARIANTS = ALL if MODE == "all" else [0, 1, 3, 16 | 1]


def _unavailable() -> str:
    if not MODE:
        return "opt-in: set TMAGENT_PLUGIN_SIM=1"
    if (
        shutil.which("g++") is None
        or not Path(INC, "scriptarray.h").exists()
        or not Path(ADDON).exists()
    ):
        return "needs g++ and AngelScript dev files (angelscript-dev, libangelscript-addon)"
    return ""


pytestmark = pytest.mark.skipif(bool(_unavailable()), reason=_unavailable() or "ok")

GAS = P.InputCmd(accelerate=True)


@pytest.fixture(scope="module")
def host_binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("assim") / "host"
    cmd = [
        "g++", "-std=c++17", "-O1", "-DAS_USE_NAMESPACE", str(HERE / "host.cpp"), f"-I{INC}",
        ADDON, "-langelscript", "-lpthread", f"-Wl,-rpath,{Path(ADDON).parent}", "-o", str(out),
    ]  # fmt: skip
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        pytest.skip(f"cannot build the AngelScript host: {res.stderr[-400:]}")
    return out


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Host:
    def __init__(self, binary: Path, variant: int):
        self.port = free_port()
        args = [str(binary), str(HERE / "stubs.as"), str(PLUGIN), str(self.port), "120"]
        self.proc = subprocess.Popen(
            [*args, f"variant={variant}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self.lines: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        end = time.monotonic() + 10
        while not any("listening on" in line or "BUILD FAILED" in line for line in self.lines):
            assert time.monotonic() < end and self.proc.poll() is None, "\n".join(self.lines)
            time.sleep(0.02)
        assert not any("BUILD FAILED" in line for line in self.lines), "\n".join(self.lines)

    def _pump(self) -> None:
        for line in self.proc.stdout:
            self.lines.append(line.rstrip())

    def stop(self) -> None:
        self.proc.terminate()
        try:
            self.proc.wait(5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _param(v: int):
    if v & 16 and v & 1:  # known limitation, see the module docstring below
        mark = pytest.mark.xfail(strict=False, reason="stale RaceTime after a rewind loses a tick")
        return pytest.param(v, marks=mark)
    return v


@pytest.mark.parametrize("variant", [_param(v) for v in VARIANTS])
def test_plugin_scenario(host_binary, variant):
    host = Host(host_binary, variant)
    try:
        run_scenario(host, variant)
    except AssertionError:
        print("\n".join(host.lines[-30:]))
        raise
    finally:
        host.stop()
    assert not any("EXCEPTION" in line for line in host.lines), "\n".join(host.lines)


def run_scenario(host: Host, variant: int) -> None:
    c = TMAgentClient(port=host.port, timeout=10, request_timeout=5)
    assert c.connect().protocol_version == P.PROTOCOL_VERSION
    c.set_mode(P.Mode.SYNC)
    c.set_speed(10.0)
    assert c.load_map("/maps/Test.Challenge.Gbx", timeout=15).startswith("FAKEUID")
    st = c.restart()
    assert st.race_time_ms == 0 and st.paused
    st100 = c.step(100, GAS)
    assert st100.race_time_ms == 1000 and st100.speed_kmh > 0 and st100.paused
    fm = c.request_frame(8, 6, 1)  # the simulated screenshot encodes the rendered race time
    img = convert_pixels(fm, 3)
    assert (
        fm.race_time_ms == 1000
        and (int(img[3, 4, 0]) << 16 | int(img[3, 4, 1]) << 8 | int(img[3, 4, 2])) == 1000
    )
    c.restart()
    for _ in range(100):  # additivity: no tick lost or added by pausing
        last = c.step(1, GAS)
    assert last.race_time_ms == 1000
    assert last.position == pytest.approx(st100.position, abs=1e-4)
    assert last.speed_kmh == pytest.approx(st100.speed_kmh, abs=1e-3)
    time.sleep(0.15)
    assert c.get_state().race_time_ms == 1000  # paused state is stable
    fin = c.step(2000, GAS)  # the finish ends a STEP early
    assert fin.finished and fin.cp_count == fin.cp_target == 3 and fin.race_time_ms < 4000
    assert c.step(5, GAS).race_time_ms == fin.race_time_ms
    st = c.restart()
    assert (st.race_time_ms, st.finished, st.cp_count) == (0, False, 0)
    st = c.step(10, P.InputCmd(accelerate=True, right=True))
    assert st.race_time_ms == 100 and st.velocity[2] != 0  # steering reaches the car
    c.set_speed(2.0)  # realtime
    c.set_mode(P.Mode.REALTIME)
    c.stream_frames(True, 16, 12, 30)
    c.stream_state(1)
    c.set_input(GAS)
    time.sleep(1.0)
    ls = c.latest_state()
    assert ls is not None and ls[0].race_time_ms > 100 and not ls[0].paused and ls[0].seq > 50
    assert c.frames_received >= 5
    assert c.restart().race_time_ms == 0
    c.set_input(P.InputCmd())
    c.set_mode(P.Mode.SYNC)  # back to sync: paused again and stays paused
    t1 = c.get_state().race_time_ms
    time.sleep(0.2)
    assert c.get_state().paused and c.get_state().race_time_ms == t1
    assert c.step(10, GAS).race_time_ms == t1 + 100
    c._sock.close()  # abrupt client death; a new client replaces it and the game resumes
    time.sleep(0.2)
    c2 = TMAgentClient(port=host.port, timeout=5, request_timeout=5)
    c2.connect()
    c2.set_mode(P.Mode.SYNC)
    c2.set_speed(10.0)
    st = c2.restart()
    assert st.race_time_ms == 0 and st.paused
    assert c2.load_map("/maps/Other.Challenge.Gbx", timeout=15).startswith("FAKEUID")
    assert c2.restart().race_time_ms == 0
    assert c2.step(50, GAS).race_time_ms == 500
    st = c2.restart(P.RestartMethod.GIVE_UP, timeout=15)  # the game's own restart path
    assert st.race_time_ms == 0 and st.paused
    assert c2.step(50, GAS).race_time_ms == 500
    c2.close()
