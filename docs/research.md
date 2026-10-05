# Research: TMNF tooling, replays, sources, terms

Collected 2026-10-05 by web research from the cloud session. Markers:
**[V]** primary source read (code/README/docs), **[S]** search snippet only,
**UNVERIFIED** no source. The cloud proxy blocked tmnf.exchange,
api.mania.exchange, store.steampowered.com, legal.ubi.com, donadigo.com,
huggingface.co, kaggle.com, tmnf.miraheze.org; everything depending on them
must be re-checked from the user's machine (Phase 0.3).

## Replay files (.Replay.Gbx)

- **pygbx** (donadigo/pygbx) parses TMNF ghosts incl. inputs [V].
  `Gbx(path).get_class_by_id(GbxType.CTN_GHOST)` -> `CGameCtnGhost` with
  `race_time`, `num_respawns`, `cp_times`, `login`, `uid`, `control_names`,
  `control_entries` (list of `ControlEntry(time, event_name, enabled, flags)`),
  `events_duration`, `game_version`.
  - Raw: `time = uint32 - 100000` (ms relative to race start); `enabled` =
    low 16 bits, `flags` = high 16 bits of the 32-bit data word (not a bool).
  - Embedded map: `g.get_class_by_id(GbxType.REPLAY_RECORD).track` is a nested
    Gbx; `.get_class_by_id(GbxType.CHALLENGE)` has `map_uid`, `map_author`,
    `map_name`. Prefer this over `ghost.uid`.
  - Packaging: PyPI 0.3 (2021) is outdated; install from git master (0.3.1),
    needs `python-lzo`. License **GPL-3** -> kept as an optional dependency
    used only in `tmagent/game/tmnf/replay.py` (see decisions D-008).
  - Sources: https://github.com/donadigo/pygbx (gbx.py `read_ghost_events`, headers.py)
- **GBX.NET** (C#, MIT core, LZO add-on GPL-3) reads the same inputs [V]:
  https://github.com/BigBang1112/gbx-net
- **TMInterface** can validate a replay and `dump_inputs` as a script
  (`<ms> press up`, `<ms> steer 20000`) [V]:
  https://github.com/FelicianPutz/TMInterfacePublic/blob/master/docs.md
- Other: TM-Gbx-input-visualizer (raw input txt + mp4, no license found),
  Clip Input 2 (AGPL, MediaTracker overlay).

### Input event semantics [V unless marked]
- Names: `Accelerate`, `Brake`, `SteerLeft`, `SteerRight` (keyboard, digital),
  `Steer` (analog, gamepad/wheel), `Gas`, `Respawn`, `Horn`,
  `_FakeIsRaceRunning` (t = 0), `_FakeFinishLine`; also `_FakeDontInverseAxis`,
  `AccelerateReal`, `BrakeReal` (GBX.NET).
- Digital: any non-zero data = pressed (0x80 typical [S]).
- Analog steer (GBX.NET decoding): `data = enabled | flags << 16`,
  `dir = (data >> 16) & 0xFF`, `val = data & 0xFFFF`;
  `dir == 0xFF -> 65536 - val`, `dir == 1 -> -65536`, else `-val * (dir + 1)`.
  Range [-65536, 65536], normalized `/65536`. Negative = left (TMI docs).
  gbxtools negates a sign-extended 24-bit value and skips entries with
  `time % 10 == 5` (cause UNVERIFIED). `_FakeDontInverseAxis` effect UNVERIFIED.
- Gas: TMNF has no analog acceleration strength; analog gas > ~19661 = on,
  < -19661 = brake (TMI docs, `gas` command).
- Physics tick 10 ms: TMI docs say first player events come at tick 10 after
  `_FakeIsRaceRunning` at 0 (consistent with 100 Hz).

## Replay and map sources
- TMNF-Exchange API (docs at https://api.mania.exchange/, blocked here) [S]:
  `GET https://tmnf.exchange/api/replays?...` (needs `fields=`), requires a
  User-Agent. Download paths per TMX-Downloader README [V]
  (https://github.com/cheatoskar/TMX-Downloader): `/api/tracks?...&count=1000`
  with `after=` paging, `/trackgbx/{trackId}`, `/recordgbx/{replayId}`.
  Rate limit undocumented; README warns about throttling.
- TMX terms / "Replay Rules" (https://tmnf.exchange/threadshow/10428863):
  **UNVERIFIED**, must be read before any bulk download.
- Existing datasets [S]: Kaggle `catalystgma/trackmania-replays` (1000+ TMNF
  replays, 100+ maps, CSV inputs); Hugging Face
  `jeanmidev/trackmania-community-tracks-and-telemetry` (~30 GB parquet,
  material subject to Nadeo/Ubisoft terms). Inputs only, no frames; still
  need maps to render.

## Terms
- TMNF EULA text: **UNVERIFIED** (Steam app 11020 page and legal.ubi.com
  blocked). User must read it before automating the game.
- TMInterface: runs count as TAS; uploading them as legit runs gets banned
  [S]. Offline agent use only, never submit runs to leaderboards.

## Game settings for rendering
- Low-res windowed mode: **TmForeverResFix** (MIT, proxy d3d9.dll +
  `TmWindow.ini` with Width/Height/FullscreenMode) [V]:
  https://github.com/snightshade/TmForeverResFix
- Config dir `%USERPROFILE%\Documents\TrackMania\Config\` [S].
- Command line (TMF, may be inaccurate) [V gist]: `/windowless /singleinst
  /nodaemon /nologs /useexedir /config= /ini= /userdir= /file= ...`
  https://gist.github.com/BigBang1112/7d4c58829aaf1313e7cac35b4bcd8481
- Cameras [S]: cam 2 default chase, cam 1 higher chase, cam 3 near-FPV.
- HUD/ghost toggle keys for TMNF: UNVERIFIED.
- TMI helpers: `skip_map_load_screens`, `sim_speed` [V].

## TMInterface (TMI)

- 1.x is legacy (last 1.4.3); **2.x** uses AngelScript plugins and supports
  TMNF. Installed via **TrackMania ModLoader (TMLoader)**
  (https://tomashu.dev/software/tmloader/), mod "TMInterface". Linesight docs
  require TMI 2.1.0 [V]
  (https://github.com/Linesight-RL/linesight/blob/main/docs/source/installation.rst);
  others use 2.2.1. Latest version UNVERIFIED (2.2.x per wiki snippets).
- pip `tminterface` 1.0.2 (GPL-3) targets TMI < 2.0 only, unmaintained [V].
  **Not used** by tmagent.
- TMI 2.x external control = a plugin hosting a TCP server via `Net::Socket`
  (`Listen`, `Accept`, `ReadInt32`, `Write`) [V]
  (gist https://gist.github.com/donadigo/c010cd682c9b8eec0dbe3f23dab4188b,
  Linesight `trackmania_rl/tmi_interaction/Python_Link.as`).
- Plugin API seen in Linesight's Python_Link.as [V]: callbacks
  `OnRunStep(SimulationManager@)`, `OnCheckpointCountChanged`,
  `OnLapCountChanged`, `Render()`; `simManager.SetInputState(InputType::Left/
  Right/Up/Down, 0|1)`, `SetSpeed(float)`, `SaveState()`,
  `RewindToState(state)`, `GiveUp()`, `PreventSimulationFinish()`,
  `RaceTime`, `TickTime`, `InRace`, `PlayerInfo.RaceFinished`,
  `get_InputEvents().ToCommandsText()`, `ExecuteCommand(str)`,
  `Graphics::CaptureScreenshot(vec2(W, H))` (inside `Render()`, returns raw
  BGRA). `InputType::Steer` [S]; `InputType::Gas` UNVERIFIED.
- Inputs apply at the next tick. Analog gas has no effect in TMNF; binary
  threshold +-19661 [V, 1.x docs mirror].
- `set_speed` > ~100 may skip input reads [V, 1.x docstring].
- Replay inputs: GUI Replays -> Validate -> Input Editor; commands `load
  file.txt`, `dump_inputs` (1.x docs; `load` also on 2.x per DeltaZero). Input
  script format `<ms> press up`, `<t0>-<t1> press left`, `<ms> steer N`.
- Determinism of re-driving inputs: claimed by TMNF-C (finish times
  reproduced) [V README]. tmagent verifies it per run (finish time vs replay,
  `meta.desync`).
- Tick: 10 ms confirmed (Linesight `ms_per_tm_engine_step = 10`,
  tmuf_physics `TMUF_TICK_MS 10`).
- Rule: never compete on public leaderboards with TMI injected (docs mirror).
- Wine: works (Linesight troubleshooting: winehq-staging, Steam TMNF, TMLoader
  zip, `wine TMLoader.exe run TmForever "default" /configstring="set
  custom_port 8483"`, `winetricks dxvk`). Linesight needs an online TMNF
  account login.

## Linesight (closest existing tool)

- https://github.com/Linesight-RL/linesight. License **ambiguous**: setup.py
  says MIT, no LICENSE file. tmagent does not copy its code; it only uses the
  documented TMI plugin API pattern (decisions D-009).
- Frames: 160x120, plugin `Graphics::CaptureScreenshot(vec2(W,H))` in
  `Render()`, raw BGRA over TCP, converted to gray. Game windowed at lowest
  resolution, low quality; paused with `SetSpeed(0)` while the network
  decides, then frame requested.
- Protocol: int message types (SCRunStepSync, SCRequestedFrameSync, ...,
  CSetSpeed, CSetInputState, CRequestFrame, CExecuteCommand, ...).
- Actions: 12 binary combinations of left/right/accelerate/brake, held 5
  ticks; no analog input.
- Progress: "virtual checkpoints" every 0.5 m from a reference ghost
  (`scripts/tools/gbx_to_vcp.py`, pygbx).

## Other related tools (none renders replay -> frame+input datasets)

| Tool | License | Relevance |
|------|---------|-----------|
| Teero888/tmuf_physics | AGPL-3 | bit-exact TMNF physics, reads replay inputs; no renderer |
| adonis-singh/TMNF-C, juanpagp/tmnf-physics | MIT | C physics + gym envs, Wine + TMI capture |
| cheatoskar/DeltaZero | MIT | BC on TMX replay positions (no pixels), `fetch-tmx` |
| dersiwi/trackmania-gym | GPL-3 | vision TMNF gym |
| trackmania-rl/tmrl | MIT | TM2020, vgamepad |
| PedroM2626/Imitation-player | MIT | generic dxcam + vgamepad BC |

Conclusion: no existing tool produces low-res (frame, input) datasets from
TMNF replays. tmagent builds it on TMI 2.x with its own plugin, following the
proven Linesight capture approach.

## Windows capture / virtual gamepad (fallback path)
- vgamepad (MIT): `VX360Gamepad().left_joystick_float(x, y)`,
  `right_trigger_float(v)`, `update()`; needs ViGEmBus.
- dxcam (MIT, Windows): ~240 fps capture claimed; mss (MIT) ~76 fps;
  windows-capture (MIT, WGC API).

## TMI 2.x plugin API used by TMAgentLink.as
The plugin header (`tmagent/game/tmnf/plugin/TMAgentLink.as`) lists every API
name with VERIFIED/UNVERIFIED status and sources (Linesight API names only,
TMNF `as.predefined` declarations in github.com/sashi0034/angel-lsp,
Archmetrus/TMNf-RLAgent RealtimeDataPublisher.as, XD1674 and Sai-Moen plugins).
UNVERIFIED items are each isolated in one `Api*` helper.
