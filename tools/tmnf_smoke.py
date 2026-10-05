#!/usr/bin/env python
"""Phase-0 checklist runner for the TMNF bridge (TMInterface + TMAgentLink plugin).

    python tools/tmnf_smoke.py --config configs/tmnf.yaml --map <path.Challenge.Gbx> \
        [--replay <Run.Replay.Gbx | inputs.txt>] [--fake] [--out DIR]

Runs the steps below against the real game (start TMNF with the plugin first, see
docs/setup_windows.md) or, with --fake, against the in-process protocol fake. Every
step prints a PASS/FAIL/WARN/MANUAL/SKIP line; a markdown report with the latency
numbers and saved frames is written to the output directory. Exit code 1 if any step
failed. Offline use only: never submit runs made with TMInterface to online
leaderboards (docs/research.md, "Terms").

If the config file does not exist, defaults plus --set overrides are used, e.g.
`--set game.tmi_port=8477 --set data.resolution=[128,96]`.
"""

from __future__ import annotations

import argparse
import contextlib
import platform
import statistics
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmagent.config import Config, load_config  # noqa: E402
from tmagent.game.tmnf.fake_server import FakePluginServer, reference_finish_time  # noqa: E402
from tmagent.game.tmnf.game import TMNFGame  # noqa: E402
from tmagent.interfaces import PHYSICS_TICK_MS, Action  # noqa: E402

GAS = Action(gas=1.0)
FAKE_SCRIPT = """# synthetic TMI input script used with --fake
0-4500 press up
500-900 press right
1200-1500 press left
"""
HINTS = {
    "step additivity": "Ticks are lost or duplicated around pausing: see PROTOCOL.md 'Pausing'. "
    "Send tmagent_status output and plugin log lines (paused guard / drift) to the developer.",
    "plugin diagnostics": "The plugin had to rewind to hold the pause (SetSpeed(0) is not "
    "immediate). Try --set game.render_speed=1 (fewer ticks per frame) and rerun.",
    "replay re-drive": "Try --tick-offset 1 (input timing is UNVERIFIED), check the map uid line, "
    "and that the replay has no respawns (meta respawns).",
}


@dataclass
class Result:
    status: str
    name: str
    detail: str = ""


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)
    metrics: list[tuple[str, str]] = field(default_factory=list)
    frames: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.results.append(Result(status, name, detail))
        print(f"[{status:6}] {name}" + (f": {detail}" if detail else ""), flush=True)

    def metric(self, name: str, value: str) -> None:
        self.metrics.append((name, value))
        print(f"         {name}: {value}", flush=True)

    @property
    def failed(self) -> bool:
        return any(r.status == "FAIL" for r in self.results)


class Fatal(Exception):
    """Abort the run: later steps depend on this one."""


@contextlib.contextmanager
def step(rep: Report, name: str, fatal: bool = False):
    """Run a step; an exception becomes a FAIL line (and aborts when fatal)."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 - the whole point is to report any failure
        rep.add("FAIL", name, f"{type(e).__name__}: {e}")
        if name in HINTS:
            rep.notes.append(f"{name}: {HINTS[name]}")
        if fatal:
            raise Fatal(name) from e


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q)) if values else float("nan")


def lat_str(values_ms: list[float]) -> str:
    return f"p50 {pct(values_ms, 50):.2f} ms, p99 {pct(values_ms, 99):.2f} ms, max {max(values_ms):.2f} ms (n={len(values_ms)})"  # noqa: E501


def write_png(path: Path, img: np.ndarray) -> None:
    """Write uint8 (H, W, 1|3) as PNG with zlib + struct (no PIL)."""
    h, w, c = img.shape
    raw = np.concatenate([np.zeros((h, 1), np.uint8), img.reshape(h, w * c)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0 if c == 1 else 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


# ------------------------------------------------------------------- steps


def check_frames(game: TMNFGame, rep: Report, out: Path) -> None:
    """Save frames at known race times and check shape, dtype, race-time stamp, content."""
    w, h = game.data_cfg.resolution
    c = game.data_cfg.channels
    game.start_race()
    race_ms, first = 0, None
    for target in (0, 250, 500, 1000, 1500):
        if target > race_ms:
            game.step(GAS, (target - race_ms) // PHYSICS_TICK_MS)
            race_ms = target
        fr = game.grab_frame()
        assert fr.image.shape == (h, w, c) and fr.image.dtype == np.uint8, (
            fr.image.shape,
            fr.image.dtype,
        )
        assert fr.race_time_ms == race_ms, f"frame says {fr.race_time_ms} ms, expected {race_ms}"
        assert int(fr.image.max()) > int(fr.image.min()), "frame is a single flat color"
        first = fr.image if first is None else first
        name = f"frame_t{race_ms:05d}.png"
        write_png(out / name, fr.image)
        rep.frames.append(name)
    assert not np.array_equal(first, fr.image), "frames at t=0 and t=1500 ms are identical"
    rep.add(
        "PASS",
        "grab frames",
        f"{len(rep.frames)} PNGs ({w}x{h}x{c}) with matching race time -> {out}",
    )
    rep.add(
        "MANUAL", "frame content",
        f"open {out / 'frame_t01000.png'}: in-game timer must read 0:01.00, the car must be "
        "visible behind/near the camera and the image upright (else set "
        "client.CAPTURE_FLIP_VERTICAL or --flip-vertical); HUD/opponent ghosts hidden",
    )  # fmt: skip


def check_latency(game: TMNFGame, rep: Report, n_grabs: int) -> None:
    game.start_race()
    game.step(GAS, 10)
    rtt = [game.client.ping() * 1000 for _ in range(50)]
    rep.metric("ping round trip", lat_str(rtt))
    grabs = []
    for _ in range(n_grabs):
        t0 = time.perf_counter()
        game.grab_frame()
        grabs.append((time.perf_counter() - t0) * 1000)
    steps = []
    for _ in range(n_grabs):
        t0 = time.perf_counter()
        game.step(GAS, 1)
        steps.append((time.perf_counter() - t0) * 1000)
    loop = []
    for _ in range(min(n_grabs, 100)):  # one rendered frame at 20 fps / 100 Hz physics = 5 ticks
        t0 = time.perf_counter()
        game.step(GAS, 5)
        game.grab_frame()
        loop.append((time.perf_counter() - t0) * 1000)
    rep.metric("grab_frame latency", lat_str(grabs))
    rep.metric("step(1 tick) latency", lat_str(steps))
    rep.metric("step(5 ticks) + grab_frame", lat_str(loop))
    speedup = 5 * PHYSICS_TICK_MS / statistics.median(loop)
    rep.metric("render loop speed", f"{speedup:.1f}x real time at 20 fps (median)")
    rep.add("PASS", "latency", f"grab p50 {pct(grabs, 50):.2f} ms / p99 {pct(grabs, 99):.2f} ms; "
            f"step p50 {pct(steps, 50):.2f} ms / p99 {pct(steps, 99):.2f} ms")  # fmt: skip


def check_realtime(game: TMNFGame, rep: Report, seconds: float) -> None:
    cfg = game.cfg
    game.restart()
    c = game.client
    game.set_action(GAS)
    calls = []
    for i in range(500):  # set_action must be cheap even when the input changes
        t0 = time.perf_counter()
        game.set_action(Action(steer=1.0 if i % 2 else -1.0, gas=1.0))
        calls.append((time.perf_counter() - t0) * 1000)
    game.set_action(GAS)
    f0, s0 = c.frames_received, c.latest_state()
    t0 = time.perf_counter()
    time.sleep(seconds)
    dt = time.perf_counter() - t0
    f1, s1 = c.frames_received, c.latest_state()
    assert s0 is not None and s1 is not None, "no pushed state arrived in realtime mode"
    fps = (f1 - f0) / dt
    ticks = (s1[0].seq - s0[0].seq) / dt
    speed = (
        ticks / 100
    )  # plugin ticks per wall second / physics rate (race time freezes at a finish)
    expected = 100 * cfg.game_speed
    fr = game.latest_frame()
    rep.metric("set_action latency", lat_str(calls))
    rep.metric("realtime frames", f"{fps:.1f} frames/s received ({seconds:g} s window)")
    rep.metric(
        "realtime ticks",
        f"{ticks:.1f} ticks/s (expected {expected:.0f} at game_speed {cfg.game_speed:g})",
    )
    rep.metric("realtime game speed", f"{speed:.2f}x (ticks per second / 100)")
    assert fr is not None, "no frame pushed"
    assert fps > 0 and ticks > 0
    rep.add(
        "PASS",
        "realtime mode",
        f"{fps:.1f} frames/s, {ticks:.0f} ticks/s, speed {game.get_state().speed_kmh:.0f} km/h",
    )
    if pct(calls, 99) > 1.0:
        rep.add(
            "WARN", "set_action latency", f"p99 {pct(calls, 99):.2f} ms exceeds the 1 ms budget"
        )
    if ticks < 0.8 * expected:
        rep.add("WARN", "realtime tick rate", f"{ticks:.0f} ticks/s is below 80% of {expected:.0f}")


def drive(game: TMNFGame, actions: np.ndarray, tail_ticks: int = 200):
    """Run per-tick actions in sync mode (runs of identical actions in one STEP)."""
    st = game.start_race()
    i, n, steps = 0, len(actions), 0
    while i < n and not st.finished:
        j = i + 1
        while j < n and np.array_equal(actions[j], actions[i]):
            j += 1
        st = game.step(Action(*(float(v) for v in actions[i])), j - i)
        steps += 1
        i = j
    left = tail_ticks
    while not st.finished and left > 0:  # inputs ended (e.g. last tick rounding): coast a little
        st = game.step(Action(), 10)
        left -= 10
    return st, steps


def check_replay(game: TMNFGame, rep: Report, args: argparse.Namespace, workdir: Path) -> None:
    from tmagent.game.tmnf.replay import replay_to_timeline

    src = args.replay
    if src is None:  # --fake: exercise the same path with a synthetic script
        script = workdir / "synthetic_inputs.txt"
        script.write_text(FAKE_SCRIPT)
        src = str(script)
    tl = replay_to_timeline(src)
    meta = tl.meta
    expected = args.expect_time_ms or meta.get("race_time_ms")
    actions = np.concatenate([np.zeros((args.tick_offset, 3), np.float32), tl.actions])
    if expected is None and args.fake:  # independent reference simulation of the fake car
        expected = reference_finish_time(actions)
    rep.add(
        "PASS",
        "replay inputs",
        f"{len(tl.actions)} ticks from {meta.get('source', src)}, "
        f"respawns {meta.get('respawns', 0)}",
    )
    if meta.get("respawns"):
        rep.add(
            "WARN",
            "replay respawns",
            f"{meta['respawns']} respawn(s): re-driving cannot reproduce them",
        )
    uid = meta.get("map_uid")
    if uid and game.map_info and game.map_info[0]:
        same = uid == game.map_info[0]
        rep.add("PASS" if same else "FAIL", "map uid", f"replay {uid} vs loaded {game.map_info[0]}")
    results = []
    for k in range(2):
        t0 = time.perf_counter()
        st, n_steps = drive(game, actions)
        results.append(st)
        wall = time.perf_counter() - t0
        rep.metric(
            f"re-drive run {k + 1}",
            f"finished={st.finished} time={st.race_time_ms} ms, {n_steps} STEP calls, "
            f"{wall:.2f} s wall",
        )
    a, b = results
    assert a.finished and b.finished, "the replayed inputs did not reach the finish line"
    same = a.race_time_ms == b.race_time_ms and np.allclose(a.position, b.position, atol=1e-4)
    rep.add("PASS" if same else "FAIL", "determinism",
            f"run 1 {a.race_time_ms} ms, run 2 {b.race_time_ms} ms, final position diff "
            f"{float(np.abs(a.position - b.position).max()):.2e}")  # fmt: skip
    if expected:
        diff = a.race_time_ms - int(expected)
        detail = f"finish {a.race_time_ms} ms vs replay {int(expected)} ms (diff {diff:+d} ms)"
        rep.add("PASS" if diff == 0 else "FAIL", "replay re-drive", detail)
        if diff != 0:
            rep.notes.append(f"replay re-drive: {HINTS['replay re-drive']}")
    else:
        rep.add(
            "SKIP",
            "replay re-drive",
            "no expected finish time known (synthetic script or --expect-time-ms missing)",
        )


def check_diagnostics(game: TMNFGame, rep: Report) -> None:
    """Counters from the plugin; corrective rewinds mean SetSpeed(0) did not stop ticks at once."""
    diag = game.client.diagnostics()
    rep.metric("plugin counters", ", ".join(f"{k}={v}" for k, v in diag.items()))
    n = diag.get("guard_rewinds", 0)
    if n > 0:
        rep.add(
            "WARN", "plugin diagnostics", f"{n} corrective rewinds were needed to hold the pause"
        )
        rep.notes.append(f"plugin diagnostics: {HINTS['plugin diagnostics']}")
    else:
        rep.add("PASS", "plugin diagnostics", "no corrective rewinds were needed to hold the pause")


# -------------------------------------------------------------------- main


def write_report(
    rep: Report, path: Path, cfg: Config, args: argparse.Namespace, build: str
) -> None:
    g, d = cfg.game, cfg.data
    lines = [
        "# TMNF bridge smoke report",
        "",
        f"- date: {datetime.now().isoformat(timespec='seconds')}",
        f"- backend: {'FAKE protocol server (not the real game)' if args.fake else 'real game'}",
        f"- plugin build: {build or 'n/a'}",
        f"- platform: {platform.platform()}, python {platform.python_version()}",
        f"- map: {args.map or '(fake)'}; replay: {args.replay or '(none)'}",
        f"- game: port {g.tmi_port}, game_speed {g.game_speed:g}, render_speed {g.render_speed:g}, "
        f"steer_mode {g.steer_mode}, camera {g.camera}",
        f"- data: resolution {d.resolution}, channels {d.channels}, frame_hz {d.frame_hz}",
        "",
        "## Steps",
        "",
        "| status | step | detail |",
        "|--------|------|--------|",
    ]
    lines += [f"| {r.status} | {r.name} | {r.detail.replace('|', '/')} |" for r in rep.results]
    if rep.metrics:
        lines += ["", "## Measurements", "", "| metric | value |", "|--------|-------|"]
        lines += [f"| {k} | {v} |" for k, v in rep.metrics]
    if rep.frames:
        lines += ["", "## Frames", ""] + [f"- {f}" for f in rep.frames]
    if rep.notes:
        lines += ["", "## Notes", ""] + [f"- {n}" for n in rep.notes]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--config", default="configs/tmnf.yaml", help="YAML config (optional if missing)"
    )
    ap.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        metavar="KEY=VALUE",
        help="config override, repeatable",
    )
    ap.add_argument("--map", help="map file/uid/name (resolved like game.load_map)")
    ap.add_argument(
        "--replay", help=".Replay.Gbx (needs pygbx) or TMI input script .txt to re-drive"
    )
    ap.add_argument("--expect-time-ms", type=int, help="expected finish time for --replay scripts")
    ap.add_argument(
        "--fake", action="store_true", help="use the in-process fake plugin instead of the game"
    )
    ap.add_argument("--out", help="output dir (default experiments/<date>-tmnf-smoke)")
    ap.add_argument("--steps", type=int, default=100, help="ticks for the gas step test")
    ap.add_argument("--grabs", type=int, default=100, help="grab_frame/step latency samples")
    ap.add_argument("--realtime-seconds", type=float, default=5.0)
    ap.add_argument("--restart-method", choices=["rewind", "give_up"], default="rewind")
    ap.add_argument("--map-path-style", choices=["auto", "absolute", "relative"], default="auto")
    ap.add_argument(
        "--settle", type=int, default=1, help="Render() calls before a capture (UNVERIFIED need)"
    )
    ap.add_argument(
        "--flip-vertical", action="store_true", help="flip captured frames (UNVERIFIED orientation)"
    )
    ap.add_argument(
        "--tick-offset", type=int, default=0, help="neutral ticks prepended to replay inputs"
    )
    args = ap.parse_args(argv)

    cfg_path = Path(args.config) if args.config else None
    overrides = list(args.set)
    if cfg_path is not None and not cfg_path.is_file():
        print(f"note: {cfg_path} not found, using defaults + --set (sample: docs/setup_windows.md)")
        cfg_path = None
    out = (
        Path(args.out)
        if args.out
        else ROOT / "experiments" / f"{datetime.now():%Y-%m-%d-%H%M}-tmnf-smoke"
    )
    out.mkdir(parents=True, exist_ok=True)

    server = None
    if args.fake:
        server = FakePluginServer().start()
        overrides += [f"game.tmi_port={server.port}", "game.connect_timeout_s=5"]
        dummy = out / "FakeMap.Challenge.Gbx"
        dummy.write_bytes(b"GBX fake map")
        args.map = args.map or str(dummy)
    if not args.map:
        ap.error("--map is required (or use --fake)")
    cfg = load_config(cfg_path, overrides)
    print(
        "TMNF smoke test. Offline use only: never submit TMInterface runs to online leaderboards."
    )

    rep = Report()
    game = TMNFGame(
        cfg.game, cfg.data, restart_method=args.restart_method, map_path_style=args.map_path_style,
        frame_settle_renders=args.settle, flip_vertical=args.flip_vertical,
    )  # fmt: skip
    steps = [
        ("connect + handshake", True, lambda: do_connect(game, rep)),
        ("load map", True, lambda: do_load(game, rep, args)),
        ("start race", True, lambda: do_start(game, rep)),
        ("step with gas", False, lambda: do_step(game, rep, args.steps)),
        ("step additivity", False, lambda: do_additivity(game, rep, args.steps)),
        ("grab frames", False, lambda: check_frames(game, rep, out)),
        ("latency", False, lambda: check_latency(game, rep, args.grabs)),
        ("realtime mode", False, lambda: check_realtime(game, rep, args.realtime_seconds)),
    ]
    if args.replay or args.fake:
        steps.append(("replay re-drive", False, lambda: check_replay(game, rep, args, out)))
    steps.append(("plugin diagnostics", False, lambda: check_diagnostics(game, rep)))
    try:
        for name, fatal, fn in steps:
            with step(rep, name, fatal):
                fn()
    except Fatal:
        rep.add("SKIP", "remaining steps", "a required step failed")
    except KeyboardInterrupt:
        rep.add("FAIL", "interrupted", "Ctrl+C")
    finally:
        with contextlib.suppress(Exception):
            game.close()
        if server is not None:
            server.stop()
    write_report(rep, out / "tmnf_smoke_report.md", cfg, args, game.plugin_build or "")
    counts = {
        s: sum(r.status == s for r in rep.results)
        for s in ("PASS", "FAIL", "WARN", "MANUAL", "SKIP")
    }
    print(
        "\n"
        + ", ".join(f"{v} {k}" for k, v in counts.items())
        + f"\nreport: {out / 'tmnf_smoke_report.md'}"
    )
    return 1 if rep.failed else 0


def do_connect(game: TMNFGame, rep: Report) -> None:
    game.connect()
    info = game.client.hello
    rep.add(
        "PASS", "connect + handshake", f"plugin '{info.build}', protocol v{info.protocol_version}"
    )


def do_load(game: TMNFGame, rep: Report, args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    game.load_map(args.map)
    uid, name = game.map_info or ("", "")
    rep.add(
        "PASS",
        "load map",
        f"uid {uid or '?'}, name {name or '?'}, {time.perf_counter() - t0:.1f} s",
    )


def do_start(game: TMNFGame, rep: Report) -> None:
    st = game.start_race()
    assert st.race_time_ms == 0, f"race time {st.race_time_ms} ms at start"
    detail = f"race time 0, speed {st.speed_kmh:.1f} km/h, in_race={st.extra['in_race']}, "
    rep.add("PASS", "start race", detail + f"checkpoints {st.checkpoint}/{st.num_checkpoints}")


def do_step(game: TMNFGame, rep: Report, n: int) -> None:
    game.start_race()
    t0 = time.perf_counter()
    st = game.step(GAS, n)
    dt = (time.perf_counter() - t0) * 1000
    assert st.race_time_ms == n * PHYSICS_TICK_MS, (
        f"race time {st.race_time_ms} ms, expected {n * PHYSICS_TICK_MS}"
    )
    assert st.speed_kmh > 0, "speed is 0 after full gas (is the input reaching the game?)"
    detail = f"{n} ticks -> race time {st.race_time_ms} ms, speed {st.speed_kmh:.1f} km/h, "
    rep.add("PASS", "step with gas", detail + f"{dt:.1f} ms wall")


def do_additivity(game: TMNFGame, rep: Report, n: int) -> None:
    """n single-tick steps must end exactly where one n-tick step ends (no tick lost or added)."""
    game.start_race()
    one = game.step(GAS, n)
    game.start_race()
    for _ in range(n):
        many = game.step(GAS, 1)
    dpos = float(np.abs(one.position - many.position).max())
    ok = (
        one.race_time_ms == many.race_time_ms
        and dpos < 1e-3
        and abs(one.speed_kmh - many.speed_kmh) < 1e-2
    )
    detail = (
        f"{n} x step(1) vs step({n}): time {many.race_time_ms}/{one.race_time_ms} ms, "
        f"position diff {dpos:.2e}"
    )
    if not ok:
        raise AssertionError(detail)
    rep.add("PASS", "step additivity", detail)


if __name__ == "__main__":
    sys.exit(main())
