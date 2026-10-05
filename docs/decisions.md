# Decisions

Format: ID, date, decision, reason, status. Record what was tried, measured,
and rejected. Numbers only from own measurements.

## D-001 (2026-10-05) Game = TrackMania Nations Forever + TMInterface
- Decision: target TMNF (free) driven through TMInterface (TMI) instead of
  TM2020/Openplanet.
- Reason: user request; TMNF physics is deterministic at fixed 10 ms ticks, and
  TMI exposes tick-level input injection, state reads and frame capture, so
  replay inputs can be re-driven and frames rendered in exact sync. Details
  and sources: `docs/research.md`.
- Status: accepted.

## D-002 (2026-10-05) Build-only session, no training or data generation here
- Decision: this cloud session builds and verifies code only (CPU tests,
  FakeGame end-to-end). No replay downloads, no clip rendering, no real
  training. Those run on the user's Windows machine.
- Status: accepted (user instruction).

## D-003 (2026-10-05) Control rate 60 Hz vs 100 Hz physics
- Fact: TMNF simulates at 100 Hz (10 ms ticks). 60 Hz does not divide 100 Hz,
  so a 60 Hz control loop changes inputs on a 10/20/20 ms tick pattern.
- Decision: keep the plan's 60 Hz as default (`data.control_hz =
  runtime.control_hz = 60`), all code takes the rate from config.
- Open: 50 Hz (every 2nd tick) aligns exactly with physics and is the
  recommended alternative. Decide by closed-loop metrics in Phase 2.
- Measured on FakeGame (eval agent): labels at any control rate < 100 Hz
  drop input changes that last less than one control period. A per-tick
  scripted driver replayed from 60 Hz labels drifts up to 4.2 m on the oval
  and leaves the road on `fake:random:3`; a driver that holds inputs for
  100 ms replays exactly at 60 and 50 Hz. Real keyboard replays contain short
  taps, so expect label loss; `tmagent.data.quality` should report the share
  of input changes lost by resampling (todo).
- Execution mapping (tick i uses the latest control row with time
  < 10(i+1)) lives in `tmagent.eval.harness.control_row_for_tick`.
- Status: open.

## D-004 (2026-10-05) One action token per frame step
- Decision: interleave one action token per frame step (embedding the
  `control_hz / frame_hz` control actions since the previous frame), instead
  of one token per control step. Keeps the sequence length at
  `K * (tokens_per_frame + 1)`.
- Status: accepted.

## D-005 (2026-10-05) Dense causal supervision
- Decision: predict a chunk at every frame step of a window, loss over all
  steps (masked). Equivalent to K samples per window at the cost of one.
- Status: accepted.

## D-006 (2026-10-05) No KV-cache in the first streaming policy
- Decision: `StreamingPolicy` caches per-frame encoder outputs and recomputes
  the small temporal transformer per call. KV-cache streaming only if the
  Phase 0/4 latency measurement shows the transformer is the bottleneck.
- Status: accepted.

## D-007 (2026-10-05) Episode storage = one directory per run + index.jsonl
- First version: one `np.savez_compressed` file per episode. Measured by the
  data agent: one random window read decompresses the whole episode, ~0.34 s
  for a 60 s 128x96 RGB episode, so shuffled training would be I/O bound.
  Rejected.
- Decision (format 2): per-episode directory with per-frame zlib blobs +
  offsets (`frames.bin`), small arrays in `arrays.npz`, `meta.json`. A window
  decodes only its K frames. No tar shards, no extra codec dependency.
- Status: accepted.

## D-008 (2026-10-05) GPL tools stay optional
- pygbx (GPL-3) and the `tminterface` pip package (GPL-3, TMI 1.x only) are
  not required dependencies. pygbx is an optional extra used only inside
  `tmagent/game/tmnf/replay.py`; the TMI input-script path (`dump_inputs`
  text) needs no GPL code. The repository has no license yet; the user decides.
- Status: accepted.

## D-009 (2026-10-05) Own TMI 2.x plugin instead of reusing Linesight code
- Linesight has no LICENSE file (MIT only declared in setup.py). tmagent
  writes its own AngelScript plugin (`TMAgentLink.as`) and Python client with
  its own protocol, using only the TMI plugin API calls listed in
  research.md. API names marked UNVERIFIED are collected in one place in the
  plugin and must be checked against the installed TMI version (Phase 0).
- Status: accepted.

## D-010 (2026-10-05) Binary steering by default on TMNF
- Most TMX replays come from keyboard drivers (binary steer events), and
  `SetInputState(InputType::Left/Right)` is verified, while analog steer
  injection (`InputType::Steer`) is only snippet-verified and in TMI 1.x
  needed a detected gamepad. `game.steer_mode = binary` (threshold 0.5);
  `analog` available once verified on the user's machine.
- Status: accepted, revisit in Phase 0.

## D-011 (2026-10-05) Relative step-distance attention bias instead of absolute time embedding
- Tried: learned absolute time embedding indexed by distance to the window's
  last step. Problem found in review: with dense supervision (D-005) step k
  sees itself at distance K-1-k during training but always at distance 0 at
  inference, so only the last step's loss matched inference.
- Decision: learned per-head relative bias over step distance, plus key
  masking of invalid steps. Output at step k is invariant to window start and
  padding (tested).
- Status: accepted.

## D-012 (2026-10-05) TMAgentLink protocol: non-blocking polling, pause by SetSpeed(0)
- Decision: the plugin never blocks the game thread on socket reads; it polls
  `Net::Socket.Available` in `OnRunStep` and `Render` (confirmed in three
  independent TMI 2.x plugins). Sync mode pauses with `SetSpeed(0)` plus a saved
  state, so `Render()` keeps running and screenshots show the paused tick. STEP
  targets are race-time based. Every command carries a `req_id`.
- Rejected: lockstep (plugin waits for a reply every tick) — would stall the
  game whenever Python hiccups; kept documented as fallback in PROTOCOL.md.
- Verification so far: Python client vs fake server (115 tests), and the real
  `.as` compiled with AngelScript 2.35.1 against stubbed TMI API in a
  simulated engine (opt-in `TMAGENT_PLUGIN_SIM=1`, 23/24 variants; the stale
  RaceTime-after-rewind variant can lose a tick, xfail). Never run in real TMI.
- First-run knobs in `GameConfig`: restart_method, map_path_style,
  frame_settle_renders, capture_flip_vertical. `tools/tmnf_smoke.py` checks
  step additivity, frame/race-time sync, determinism of replay re-drive.
- Status: accepted, pending Phase 0 on the user's machine.

## D-013 (2026-10-05) Copycat confirmed on FakeGame; action history off by default
- Setup (CPU, code proving, not a benchmark): 80 scripted fake runs on 24
  maps (16 train / 4 val / 4 test), 64x48 RGB, history 0.5 s (K=11), 0.45 M
  params, batch 32. Closed-loop sync eval, 1 seed, 3 train maps / 3 test maps.
- Run A, with action history, defaults (dropout 0.3, token dropout 0.1),
  3000 steps: val steer MAE 0.154, gas acc 0.84, but median progress **0.0**
  on train and 0.09 on test maps. Diagnosis: on the same standstill frame,
  P(gas) = 0.15 with history gas=0 and 0.90 with history gas=1; the first
  prediction (no history) is 0.42 -> no gas -> history says no gas -> the car
  never starts. With a forced 1 s start it stops again at the next gas-off.
- Run B, no action history, 1500 steps: gas acc only 0.58, but median
  progress **0.40** on train maps, 0.09 on test maps (car drives).
- Run C, history with action_dropout 0.7, token dropout 0.3, gas/brake flip
  0.1, 1500 steps: median progress **0.0** on both. Stronger dropout alone
  does not fix it.
- Decision: `model.use_action_history = false` is the default (fake and tmnf
  configs; Phase 2 baseline is single frame without history anyway). Action
  history stays available for the Phase 3 ablation (`configs/context_2s.yaml`).
- Training loss/accuracy did not reveal the failure; only closed-loop eval
  did (PLAN rule confirmed). Low train-map progress for B also shows BC at
  this scale is weak; more data, noise-injected (DAgger-style) runs
  (Phase 5) and longer training are the expected levers.
- Ideas not tried yet: feed history delayed by one frame step, predict
  gas/brake from images only, condition on speed instead of past actions,
  start-of-race oversampling.
- Status: accepted for now; revisit in Phase 3 with TMNF data.
