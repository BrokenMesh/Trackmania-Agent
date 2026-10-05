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
