# Windows setup: TMNF + TMInterface + tmagent

Goal: run `tools/tmnf_smoke.py` against the real game (Phase 0 of `docs/PLAN.md`).
Facts and sources: `docs/research.md`. **UNVERIFIED** marks anything that could not be
checked from the cloud session (web pages that were blocked, key bindings, exact menu
names). Check those on your machine and fix this file.

## 0. Before you start: terms (Phase 0.3, do this first)

Read these yourself; stop and ask if anything forbids automation or third-party tools.

- [ ] TMNF EULA / terms on the Steam page (app 11020) or the Ubisoft/Nadeo legal pages
      (not readable from the cloud session, UNVERIFIED).
- [ ] TMX / TMNF-Exchange terms and its replay rules
      (https://tmnf.exchange/threadshow/10428863, UNVERIFIED) before any bulk download.
- [ ] Rule for everything below: TMInterface runs count as TAS. **Never submit runs
      made with TMInterface (including agent runs) to online leaderboards or play online
      with TMInterface loaded.** Use a separate offline profile if you also play normally.

## 1. TMNF

Install TrackMania Nations Forever: Steam (app 11020) or the official installer. Start it
once without mods, create a profile, and check that a race runs.

## 2. TMLoader + TMInterface 2.x

1. Install TrackMania ModLoader (TMLoader): https://tomashu.dev/software/tmloader/
2. In TMLoader create a profile (e.g. `default`) and install the **TMInterface** mod
   (2.x; the Linesight docs require 2.1.0, others use 2.2.1; the newest version is
   UNVERIFIED, take the current one from the TMInterface site).
3. Always start the game through TMLoader, not through Steam.
4. The `tminterface` PyPI package is for TMI 1.x only and is **not** used by tmagent.

## 3. Optional: small window with TmForeverResFix

A small window makes frames cheap. TmForeverResFix (MIT,
https://github.com/snightshade/TmForeverResFix) is a proxy `d3d9.dll` placed next to
`TmForever.exe`, configured by `TmWindow.ini` with `Width`, `Height`, `FullscreenMode`
(read its README for the exact syntax). Example intent: windowed, 256x192 (4:3, same
aspect as `data.resolution` 128x96). Whether `CaptureScreenshot(w, h)` rescales or crops
the window is UNVERIFIED: compare a saved smoke frame with the game window.

## 4. Install the plugin

1. Copy `tmagent/game/tmnf/plugin/TMAgentLink.as` to
   `Documents\TMInterface\Plugins\TMAgentLink.as`.
2. Pick a port (default 8477, must equal `game.tmi_port`). Launch with
   `TMLoader.exe run TmForever "default" /configstring="set tmagent_port 8477"` (quoting
   differs between cmd.exe and PowerShell, UNVERIFIED; or any other way of passing a
   TMInterface command), or type `set tmagent_port 8477` and
   `tmagent_listen` in the TMInterface console.
3. Enable the plugin in the TMInterface window (Settings, Plugins tab; menu names
   UNVERIFIED). The console log must say `TMAgentLink listening on 127.0.0.1:8477`.
   Compile errors also show up there: send them to the developer, the plugin was
   compile-checked against AngelScript 2.35 with stubs but never against the real game.
4. Console command `tmagent_status` prints the connection and pause state.

## 5. Game settings for clean frames

- Windowed, lowest quality/resolution (in-game video options).
- Camera: `game.camera: cam1` is sent as the console command `cam 1` after connecting
  (cam 2 default chase, cam 3 near-first-person, from search snippets, UNVERIFIED).
- Hide HUD, ghosts and opponents: key bindings UNVERIFIED. Try the race-UI toggle keys,
  and remove your own personal-best ghost for the map (move it out of
  `Documents\TrackMania\Tracks\Replays\`) so no ghost car is drawn. Judge by the
  `frame_t*.png` files the smoke test writes. Keep the in-game timer visible during
  Phase 0: it is how you verify frame/race-time sync by eye.
- `TMNFGame` sends these TMInterface console commands on connect (from the Linesight bridge,
  names for TMI 2.1.x): `skip_map_load_screens`, `unfocused_fps_limit false`,
  `autorewind false`, `auto_reload_plugins false`, `disable_forced_camera`,
  `countdown_speed`. Keep the game window visible while rendering (do not minimize).
- Maps: copy `.Challenge.Gbx` files to
  `Documents\TrackMania\Tracks\Challenges\tmagent\` and set `game.map_dir` to that folder.
  Then both interpretations of TMI's `map` argument (absolute path or path relative to
  `Tracks\Challenges`, UNVERIFIED) point at the same file. Use paths without spaces if
  `load map` times out. tmagent resolves refs by file path, by `index.json`
  (`{uid: filename}`) or by (part of) the file name in `map_dir`.

## 6. Python environment

```powershell
py -3.11 -m venv .venv ; .venv\Scripts\Activate.ps1
# CUDA build of torch for your GPU, see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e .[dev,game]
# pygbx on PyPI (0.3) is outdated: use git master (0.3.1). It needs python-lzo.
pip install --upgrade git+https://github.com/donadigo/pygbx
pip install python-lzo        # needs a prebuilt wheel or MSVC build tools (UNVERIFIED)
```

pygbx is GPL-3 and optional (decisions D-008): without it you can still re-drive TMI
input-script text files (`--replay inputs.txt`). Run `python -m pytest -q tests/tmnf`
to check the install (no game needed).

## 7. Config (`configs/tmnf.yaml`)

`configs/tmnf.yaml` holds the TMNF settings (resolution 128x96 RGB, 20 fps
frames, 60 Hz control, binary steering, port, `map_dir`, ...). Edit at least
`game.map_dir` and `game.tmi_port`. First-run knobs for unverified TMI behaviour
are also there: `game.restart_method` (`rewind` | `give_up`),
`game.map_path_style`, `game.frame_settle_renders`, `game.capture_flip_vertical`.
Any value can be overridden per command with `--set key=value`.

Without a config file the smoke tool uses defaults plus `--set` overrides, e.g.
`--set game.tmi_port=8477 --set game.map_dir=D:/maps`.

## 8. Run the smoke test

```powershell
python tools/tmnf_smoke.py --fake                      # sanity check of the Python side
# start TMNF via TMLoader, plugin enabled, then:
python tools/tmnf_smoke.py --config configs/tmnf.yaml --map MyMap
python tools/tmnf_smoke.py --config configs/tmnf.yaml --map MyMap --replay D:\replays\run.Replay.Gbx
```

Steps (PASS/FAIL line each, report `experiments/<date>-tmnf-smoke/tmnf_smoke_report.md`):

| step | what it proves | on FAIL |
|------|----------------|---------|
| connect + handshake | plugin loaded, port, protocol version | plugin not enabled / port mismatch / copy the current `.as` |
| load map | `map` command works, countdown detection, map uid | check `map_dir`, `--map-path-style absolute\|relative`, path without spaces |
| start race | rewind to the saved t=0 state | `--restart-method give_up` |
| step with gas | 100 ticks == race time 1000 ms and speed > 0 (input reaches the game) | input API or pause logic wrong |
| step additivity | 100 x step(1) ends where step(100) ends: no tick lost/added by pausing | send plugin log lines (paused guard / drift) |
| grab frames | PNGs `frame_t*.png`, race-time stamp, not flat | `--settle 2`, check window visible |
| latency | grab / step p50, p99, render-loop speed | informational |
| realtime mode | frames/s, ticks/s, `set_action` latency < 1 ms | WARN lines |
| plugin diagnostics | `guard_rewinds` = 0: pausing needed no corrective rewinds | try `--set game.render_speed=1` |
| replay re-drive | inputs of a replay reproduce its finish time twice (determinism / desync) | `--tick-offset 1`, check map uid and respawns |

The `MANUAL` line asks you to open `frame_t01000.png`: the in-game timer must read
0:01.00, the image must be upright (else `game.capture_flip_vertical: true` /
`--flip-vertical`), HUD and ghosts hidden.

## 9. Phase 0 checklist (docs/PLAN.md) and the tools

| Phase 0 item | tool | output |
|--------------|------|--------|
| 0.1 GPU, VRAM, CUDA | `python tools/system_report.py` | `docs/system.md` |
| 0.2 game version, replay format, play/render, capture, input | `tools/tmnf_smoke.py` (this page) | smoke report, PNGs |
| 0.3 terms of game and replay sources | manual, section 0 | tick the boxes, note findings in `docs/decisions.md` |
| 0.4 latency of the chain (capture, preprocess, model, input) | `python tools/measure_latency.py` (+ smoke latency numbers) | `docs/latency.md` |
| 0.5 latency budget, chunk length, max model size | derive from 0.4 | `docs/latency.md`, `docs/PLAN.md` |

## 10. Wine / Linux (brief, unsupported)

TMNF + TMInterface also run under Wine (Linesight's troubleshooting notes): winehq-staging,
Steam TMNF, the TMLoader zip, `winetricks dxvk`, then
`wine TMLoader.exe run TmForever "default" /configstring="set tmagent_port 8477"`.
Linesight reports that it needs an online TMNF account login there. Frame capture and
timing under Wine are untested with tmagent (UNVERIFIED). Run Python natively and connect
to the Wine process through `127.0.0.1`.

## Troubleshooting

- `cannot connect ... within N s`: game not started through TMLoader, plugin not
  enabled, or port mismatch (`set tmagent_port` vs `game.tmi_port`).
- `ProtocolMismatch`: the plugin file in `Documents\TMInterface\Plugins` is older/newer
  than this checkout; copy it again and restart the game.
- `load map` timeout (default 120 s): see the `map` line in the TMInterface console. If it
  says "The map was successfully queued" but the game stays in the menu, the map file was
  added after the game started: **restart the game after copying or downloading maps**
  (VERIFIED twice; the game scans `Tracks\` only at startup). "The map file does not exist"
  means a wrong path: `map tmagent/<file>` (relative to `Tracks\Challenges`) works.
- Car does not move (speed 0 after full gas): the game window had no focus while the map
  intro played, so the start state was saved with the car still locked. `game.focus_window:
  true` focuses the window before every map load; do not click into other windows during
  a run.
- Replay re-drive does not finish: analog replays (pad/wheel) drift off with binary
  steering; download with `tools/tmx_download.py ... --keyboard-only`.
- `restart returned race time 10 ms, expected 0`: TMI did not call `OnRunStep` at race
  time 0, the saved start state is one tick late. Report it (PROTOCOL.md "LOAD_MAP").
- Frames upside down: `--set game.capture_flip_vertical=true` (or in `configs/tmnf.yaml`).
- A crashed Python client never leaves the game frozen: the plugin resets to speed 1.0
  when the connection drops, and a new client replaces the old connection.
