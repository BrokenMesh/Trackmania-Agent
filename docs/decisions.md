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

## D-014 (2026-10-06) Verified TMNF bridge behaviour (first run on the real game)
- Setup: TrackMania United Forever 2.12 (contains Nations) via TMLoader 1.0.1,
  TMInterface 2.2.1, RTX 4060, Python 3.13.
- The map intro only finishes while the game window has focus; without focus the
  t=0 state is saved with the car locked. `game.focus_window: true` focuses the
  window before each map load (`tmagent/game/tmnf/window.py`).
- Maps copied or downloaded after the game started cannot be loaded (`map` says
  "queued", nothing happens): restart the game after adding maps.
- Replay input timing: an input at replay time T must be set in OnRunStep at
  T - 10 ms (`GBX_INPUT_LEAD_MS`); with it the A01 author replay re-drives to
  exactly 24540 ms. pygbx reports some replays with a 65535 ms offset; times are
  measured from the race-start event.
- Left + right held together: left wins (only rule that re-drove 3 A01 replays).
- The finish is reported one step after the last input tick: replay renders hold
  the last input for 0.6 s more.
- The plugin saves the t=0 state once per map load; that state was bad once for a
  whole map (D11, 10/10 desync); a fresh load fixed it. Render retries a desync
  once with a map reload. Maps are no longer reloaded between runs (saves the
  ~12 s intro per run).
- Status: accepted, measured.

## D-015 (2026-10-06) First TMNF dataset: keyboard-only TMX replays, no respawns
- Source: TMX tracks 2233-2455 (the 65 Nations campaign maps, author Nadeo, map
  uids equal to the installed ones). 10 replays per map: 3 fastest, 3 slowest
  under 2x the best time (AFK runs of up to 16 h exist), 4 random middle.
- Keyboard only (`tmx_download.py --keyboard-only`): analog pad/wheel replays drift
  off with binary steering (5/5 failed on A01; 5/5 keyboard runs exact). About 3
  of 4 TMX runs are analog. B01 has 7 replays (no keyboard run among its 100
  fastest).
- Replays with respawns are excluded (a respawn teleports the car; the agent only
  gets full restarts). Map uid comes from the replay XML header (the pygbx ghost
  uid fallback was wrong for most TMX replays).
- Result: 545 episodes, 7.8 h, 64 maps (E05 skipped: about 1 h to render);
  split train 469 / val 43 / test 33 episodes; 3 runs (0.5 %) still desync.
- Frames must not contain the author-medal ghost: use a game profile without
  records (the in-game toggle did not hide it).
- Status: accepted.

## D-016 (2026-10-06) Steering as 3 classes + PWM execution; Huber regression rejected
- Baseline (tiny_cnn, single frame, regression head, Huber, 20k steps, 469 train
  episodes): closed loop 0/8 finished, median progress 0.126 (4 test + 4 train maps).
  Offline on val: predicted |steer| > 0.5 in 0.3 % of frames vs 42 % steering labels,
  corr(pred, label) 0.05. Cause: keyboard labels are -1/0/1 with 60 % "none"; Huber
  (L1-like) regresses to the conditional median, which is 0, so the model never steers.
- Fix: `model.head: discrete`, `model.steer_bins: 3` (cross-entropy over left/none/right,
  decoded as P(right) - P(left)) and `game.steer_mode: pwm` (sigma-delta: steer 0.4 =
  right key in 40 % of ticks; -1/0/1 unchanged, so replays drive as before).
  Offline val after 5k steps: corr 0.50, steer sign accuracy 0.79 (was 0.05 / 0.53).
- Closed loop, 20k steps, same 8 maps: median progress 0.145, A02 0.92, A05 0.49
  (train maps); test maps A14 0.34, B10 0.17, D05 0.12, D09 0.11. Still 0 finishes.
- `eval.steer_gain` 1.5 / 2.0: worse (0.157 / 0.120 median, more off-track) -> keep 1.0.
- Failure mode seen on A01 (GIF): turn-in too late and too weak at 170 km/h, then the
  car stays against the wall with full gas (the model never brakes or reverses, brake
  labels are 3 %). Single frames carry no speed / approach-rate information: next lever
  is the 2 s context (PLAN Phase 3), then more data (render of ~1270 more replays runs).
- Status: accepted (discrete3 + pwm is the new default for TMNF experiments).

## D-017 (2026-10-07) DAgger with a line-following expert: one round, ~20 % share
- Closed-loop failures are compounding error, not a pipeline bug (eval frames equal training
  frames, live vs offline predictions corr 0.992). More human data (7.8 h -> 51.6 h) did not
  help in closed loop.
- Expert: `tmagent/eval/expert.py` LineExpert (copy the fastest human run's input when within
  1.5 m and aligned, else pure pursuit 0.45 s ahead). Collection: `tools/collect_dagger.py`,
  random agent / expert blocks plus intervention at 2.5 m, stop after 4 s without progress;
  labels always from the expert. Training: `data.source_weights={dagger: w}`,
  `train.init_from=<ckpt>`, lr 0.0001, 8-10k steps.
- Results (3 attempts per map, steer noise 0.05): round 1 (1.24 h DAgger, 23 % of samples)
  A01-A10 median 0.44 -> 0.60; rounds 2-3 (36-45 % of samples) fell back to 0.43, and all
  three rounds at 23 % gave 0.45 (10 maps x 3 attempts: gaps of ~0.1 are partly noise). Unseen
  test maps unchanged in every round (mean 0.07-0.08).
- Decision: keep round 1 (`experiments/2026-10-07-ctx2_dagger1/checkpoints/step_010000.pt`)
  as the current best; keep the expert share near 20 %; do not iterate DAgger with this
  expert further. Next levers: better recovery labels (human takeovers via
  `tools/eval_ui.py`, or a smoother analog expert), and recovery data on many more maps for
  transfer.
- Evaluation protocol from now on: >= 3 attempts per map with eval.action_noise 0.05; one
  deterministic attempt per map is a point sample.
- Status: accepted.


## D-018 (2026-11-20) RL test on A01-A10: reward-weighted BC (AWR style), user request
- PLAN.md keeps RL out of scope unless the user asks; the user asked for a small RL test on
  A01-A10 (stuck check and completion speed as reward).
- Method (`tmagent/eval/rl.py`, `tools/collect_rl.py`, `tools/rl_rounds.py`): the policy drives
  with sampled actions (steer class from the softmax, gas / brake Bernoulli, temperature 1),
  the executed actions are the labels. Reward per control step: progress along the reference
  line (clipped to 5 m per step, so projection jumps on stunt maps earn nothing), -50 m when
  the run ends stuck / offtrack / stalled, +50 m at the finish; speed is rewarded through
  progress per time. Advantage = discounted return (3 s horizon) minus the mean return of all
  attempts of the round at the same 5 m stretch of line; weight exp(A / std(A)), capped at
  20, mean 1 per map. No value network: the attempts on the same map are the baseline.
- Training: fine-tune from the previous round's checkpoint on human data + dagger:d1 (weight
  10, the dagger1 recipe) + all rl:* episodes at 25 % of the sampled windows, lr 0.0001,
  2500 steps; per-step weights multiply the BC loss mask. Human data keeps the policy near BC.
- Chosen over PPO for the first test: it reuses the BC training loop and needs no value head
  or on-policy optimizer. PPO is the next step if reward weighting moves A01-A10.
- Evaluation: the D-017 protocol (3 attempts per map, steer noise 0.05) after every round,
  logged to data/tmnf/rl_rounds.jsonl. Start: dagger1 10k (A01-A10 median 0.60).
- Budget: one round ~70 min on the user PC (6 attempts x 10 maps ~30 min, training ~23 min,
  eval ~15 min). One game instance.
- Results (2026-11-21, 3 rounds, 60 sampled attempts each; eval 10 maps x 3 attempts):

  | | median | mean | finishes |
  |---|---|---|---|
  | dagger1 10k (start) | 0.56 | 0.56 | 4 |
  | rl r1 | 0.47 | 0.59 | 5 |
  | rl r2 | 0.58 | 0.60 | 4 |
  | rl r3 | 0.65 | 0.61 | 4 |

  Per-map mean progress vs start: r3 +0.054 (bootstrap 95 % CI over maps -0.014 .. +0.124),
  5 maps better / 2 worse. Consistent gains on A02 (0.73 -> 0.97-0.99), A05 (0.41 ->
  0.60-0.65), A08 (0.22 -> 0.33); A04 (stunt map, unreliable progress) lower, A07 stuck at
  0.07 in every round (all attempts crash at the same spot, so the same-place baseline gives
  no advantage signal there). Finish count and finish times did not improve.
- Confound: every RL round also trained on the 696 user replays that dagger1 never saw.
- Verdict: a small upward trend across rounds, not significant with this eval size; no
  finish-rate gain. Not worth more rounds at ~1.2 h each. Throughput is the limit: the next
  step is faster sync stepping (fewer round trips per frame) before larger RL runs.
- Incident: round 1 training hung at the step-2000 validation (GPU memory at 7.6 / 8.2 GB,
  main process spinning) for 50 min; restarted with train.eval_every=0 (validation only at
  the end), no further hangs.
- Status: done, inconclusive.
