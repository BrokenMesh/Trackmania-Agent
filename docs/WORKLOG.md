# Work log

Read this first if you are an agent picking up the project. Newest entry on top.
Companion files: `docs/PLAN.md` (requirements, German), `docs/ARCHITECTURE.md`
(spec), `docs/decisions.md` (decision record), `docs/research.md` (verified
facts about TMNF tooling with sources).

## Status board

| Area | State | Notes |
|------|-------|-------|
| Research (TMI, linesight, replay format, TMX, ToS) | done | `docs/research.md`; ToS items UNVERIFIED (proxy-blocked) |
| Shared contracts (`interfaces.py`, `config.py`) | done | |
| Data pipeline (`tmagent/data`) | done, reviewed | storage format 2 (D-007) |
| Model + training (`tmagent/model`, `tmagent/train`) | done, reviewed | relative step bias (D-011) |
| Runtime (`tmagent/runtime`) | done, reviewed | 0 % misses with a 100 ms policy |
| FakeGame + eval (`tmagent/game/fake.py`, `tmagent/eval`) | done, reviewed | |
| TMNF bridge (`tmagent/game/tmnf`, plugin, smoke tool, setup doc) | done, **verified on the real game** | D-012, D-014 |
| Tools (render, fake pipeline, system/latency reports, TMX download) | done, reviewed | |
| Fake end-to-end learning probe | done | copycat found, history off by default (D-013) |
| Phase 0 on user machine | done | `docs/system.md`, `docs/latency.md`, smoke 12/12 PASS |
| First TMNF dataset | done | 545 episodes, 7.8 h, 64 maps (D-015) |
| Baseline training (single frame) | done, evaluated | 0/8 finished, median progress 0.126; 3-class steering + PWM: 0.145, A02 0.92 (D-016) |
| Context model (2 s) | done, evaluated | full data (51.6 h): 0/26 test maps finished, median progress 0.051; first finish on A03 (train map) |
| Dataset (campaign + community) | done | 3,585 episodes, 51.6 h, 259 maps (214 train / 19 val / 26 test) |
| User replays (brokenmesh3) | rendered | 696 episodes on 632 own maps; dataset 64.7 h, not trained on yet |
| DAgger with line expert | done, evaluated | round 1 best: A01-A10 median 0.60 (was 0.44); unseen maps unchanged (D-017) |
| Test drive UI | done | `python tools/eval_ui.py --config configs/tmnf.yaml` -> http://127.0.0.1:8600 |
| Takeover mode (human recoveries) | verified live | 15 segments, 2.8 min on A01-A10 (2026-11-20); cut at respawns, restart = new segment |
| RL test, reward-weighted BC (D-018) | built, fake-tested | `python tools/rl_rounds.py --config configs/tmnf.yaml --ckpt <dagger1 10k> --eval-start`; not run on TMNF yet |
| Stage 2 plan (memory agent) | written | `docs/PLAN_PHASE2.md` |

Test suite: `python -m pytest -q` -> 502 passed, 3 failed, 4 skipped on the Windows PC (2026-11-20,
~2 min). The 3 failures are platform-only and fail without the session-5 changes too (no fork,
`\` in paths, case-insensitive glob): test_reader_survives_pickle_and_fork,
test_map_command_path_styles, test_find_replays.
Opt-in plugin simulation: `TMAGENT_PLUGIN_SIM=1` (builds an AngelScript host).

### Next steps (user machine)
1. Read TMNF EULA and TMX terms (Phase 0.3), note result in decisions.md.
2. Install per `docs/setup_windows.md`; `python tools/system_report.py`.
3. `python tools/tmnf_smoke.py --config configs/tmnf.yaml --map <map> --replay <replay>`;
   fix first-run knobs in `configs/tmnf.yaml` until all PASS (most likely:
   restart_method, map_path_style, frame_settle_renders, flip, tick offset).
4. `python tools/measure_latency.py --config configs/tmnf.yaml --map <map>`
   -> docs/latency.md; fill the plan's success criteria (TBD values).
5. Small data run (a few maps, tens of replays) with render_replays + quality,
   then baseline training (single frame, no history), closed-loop eval.

## 2026-11-20 Session 5 (user machine): takeover mode live

- First live takeover session (plugin 0.2.0, checkpoint dagger1 10k), A01-A10, 1 run each.
  All 10 finished; A03 and A05 without any help. 15 player segments saved (2.8 min after
  the cuts below), all on train maps.
- Data check: passthrough driving, labels and frames are sound (sampling jitter only, some
  repeated frames). Two problems found and fixed:
  - Checkpoint respawns (4 segments, jumps of 280-370 m): before a respawn the player gives
    up (no keys), and the model has no respawn action. Takeover segments now keep only the
    part after the last respawn (`PlayerSegment.after_last_respawn`). The 4 saved
    segments were cut the same way (originals in `data/tmnf/takeover_backup_2026-11-20`).
  - A full restart (race time jumps back) did not end the segment: A04's segment had a
    negative duration and was lost. Now it ends the segment, resets policy and progress,
    and the player keeps driving in a new segment.
  - Also fixed: a jump counted as a respawn on single-sample position glitches (A01, A06,
    A07 had false counts); respawns now need the car to stay away 200 ms. After a restart
    the newest streamed frame was still the pre-restart one and blocked all newer frames;
    frames ahead of the state's race time are now ignored.
- Next: short corrections ("fix it and hand back") on many different train maps, not
  just A01-A10; then fine-tune with takeover data as a weighted source.
- RL test built on user request (D-018): sampled driving, progress reward, advantage weights
  per step stored with the episode (`Episode.weights`, optional `weights` array in
  arrays.npz, `target_weight` in training windows, multiplies the loss mask), exact-source
  sampling weights (`dagger:d1` before `dagger`). End-to-end round verified on the fake game
  (collect -> train -> eval -> rl_rounds.jsonl); the dagger1 checkpoint samples correctly
  offline. Not run on TMNF yet: the game was in use for takeovers.

## 2026-10-07 Session 4 (overnight, autonomous): DAgger

- Morning additions: recording mode in `tools/eval_ui.py` (`tmagent/data/player.py`, every
  attempt saved, source "player:<session>"); 703 of the user's own keyboard replays copied
  to `data/tmnf/replays_brokenmesh` (profile elkor3 = account brokenmesh3; originals not
  touched) and their 636 embedded maps extracted with `tools/extract_replay_maps.py`.
- User replays rendered (single instance, 10:26-16:20): 696 of 703 exact (finish time
  reproduced), skipped 1 desync, 1 map that hangs the loader (bad_map list), 2 respawn runs.
  Dataset now 64.7 h (train 51.7 / val 8.6 / test 4.5); the 26 original test maps keep their
  split (split by map hash). Not trained on yet.

- Pipeline check (VERIFIED): a human A01 replay re-driven through the eval loop gives the
  training frames (mean |diff| 0.7/255) and the same predictions (corr 0.992). The
  closed-loop gap is compounding error, not a bug.
- Takeover mode built (plugin 0.2.0 SET_PASSTHROUGH, `tmagent/eval/takeover.py`, UI
  buttons); live test pending a game restart (plugin reload).
- LineExpert (`tmagent/eval/expert.py`) + DAgger collector (`tools/collect_dagger.py`),
  weighted source sampling (`data.source_weights`) and `train.init_from`.
- Round d1: ctx2_full drives, expert labels; 214 train maps, 1.24 h, expert drove 79 %.
  Fine-tune from ctx2_full, dagger weight 10 (23 % of samples), lr 1e-4.
- Step 2.5k, 30-map eval: train maps A01 0.05 -> 0.53 (first-corner crash gone),
  A02 0.46 -> 0.91, A05 0.19 -> 0.49; 26 unseen test maps unchanged (mean 0.067 -> 0.055,
  noise). DAgger fixes recoveries where it collected; no transfer to new maps yet.
- Round d1 final (10k steps), 36 maps x 1 attempt:

  | | ctx2_full | dagger1 10k |
  |---|---|---|
  | A01-A10 median progress | 0.21 | 0.55 |
  | A01-A10 finishes | 1 (A03) | 2 (A02, A03) |
  | 26 test maps median / mean | 0.051 / 0.067 | 0.042 / 0.079 |

  A01 0.05 -> 0.64, A08 0.03 -> 0.28; test maps unchanged within noise.
- Round d2 (driver: dagger1 2.5k): 214 maps, 1.09 h, expert 76 %, mean progress 0.22.
  Round 2 training: init from dagger1 10k, human + d1 + d2, dagger weight 10.
- Round d3 (driver: dagger1 10k): 214 maps, 1.14 h. Round 3 training: init dagger2,
  human + d1..d3, dagger weight 7.
- Single deterministic attempts are point samples (the sim is deterministic, tiny weight
  changes flip a run). New protocol: 3 attempts per map, eval.action_noise=0.05:

  | Model | A01-A10 median / mean (finishes of 30) | 26 test maps median / mean |
  |---|---|---|
  | ctx2_full | 0.443 / 0.470 (4) | 0.053 / 0.072 |
  | dagger1 10k | **0.596 / 0.591** (4) | 0.048 / 0.076 |
  | dagger3 8k | 0.434 / 0.476 (3) | 0.050 / 0.079 |
  | d1+d2+d3 at 23 % (from ctx2_full, 10k) | 0.445 / 0.520 (4) | 0.048 / 0.082 |

  Round 1 is the best model (A01 0.05 -> 0.95, A08 0.03 -> 0.38). More DAgger data at the
  same 23 % share (all three rounds) did not beat it either. Rounds 2-3 did not add:
  the expert share of the samples grew to ~36-45 % and its labels (bang-bang pursuit,
  ~1.5x human time) pull the policy towards the worse driver. No transfer to unseen maps
  in any round (D-017).
- YAML gotcha: `train.lr=1e-4` parses as a string (YAML 1.1); write 0.0001.

## 2026-10-07 Session 3 (user machine, overnight)

- Community maps: 197 top-awarded TMX race maps (author time 15-60 s), 1,959
  keyboard replays; rendered with 2-3 game instances (`--shard I/N`, plugin
  port fallback). 3 instances were no faster than 2 (CPU bound, 4 cores).
- Render fixes: retry a briefly locked episode rename (Windows PermissionError);
  a map whose load hangs the game is stored as `bad_map` in render_skips.jsonl
  and its runs are skipped from then on.
- Training on the full set (40k steps, ~6.5 h, GPU). Val loss bottomed at ~0.79
  (step 9-12k), ended at 0.84; train loss 0.56. Episodes are 78 GB on an HDD,
  which slows data loading (0.55 s/step vs 0.43 s on the small set).
- Closed-loop eval, 30 maps x 1 seed (26 test maps + 4 train maps of the old
  8-map set):

  | Model | test26 median / mean progress | old 8 median / mean | finished |
  |---|---|---|---|
  | ctx2_full step 40k (51.6 h) | 0.051 / 0.067 | 0.160 / 0.198 | 0/34 |
  | ctx2 step 20k (7.8 h) | 0.046 / 0.058 | 0.166 / 0.204 | 0/34 |
  | discrete3 single frame | 0.047 / 0.058 | 0.093 / 0.141 | 0/34 |

  6.6x more data did not move closed-loop progress beyond noise. Offline on val
  maps the model is good (steer sign accuracy 0.88, corr 0.63, gas 0.96), so
  the gap is closed-loop: compounding errors / missing recoveries (or a
  remaining timing mismatch, not yet ruled out).
- A01-A10 with ctx2_full: A03 finished (25.86 s, first finish); A02 60 %,
  A04 69 %; A10 shows 100 % while offtrack at 11 s (progress projection is
  unreliable on acrobatic maps with overlapping path sections).
- `tools/eval_ui.py`: local browser UI to pick a checkpoint and maps (presets
  A01-A10, campaign, test split), queue drives, restart / skip / stop;
  results in `data/tmnf/eval_ui_results.jsonl`.
- Running 2-3 game instances plus GPU training made the PC stutter: run one
  heavy job at a time.

## 2026-10-06 Session 2 (user machine, real game)

- Setup done: TMUF 2.12 + TMLoader + TMI 2.2.1, venv (Python 3.13, torch cu124),
  python-lzo replaced by a local `lzo.py` shim over `lzallright` inside `.venv`
  (no Windows wheels; recreate it with a new venv).
- Smoke test 12/12 PASS after four bridge fixes (D-014). Latency: chain p99
  18-24 ms, budget 133 ms (`docs/latency.md`).
- Data: TMX downloader extended (best/worst/random groups, time cap,
  keyboard-only with replacement); 651 replays downloaded, 545 episodes rendered
  (D-015). `tools/episode_gif.py` turns episodes into GIFs with an input overlay.
- Rendering runs at ~2.2x real time (two round trips per frame); speed-ups are
  listed in `docs/PLAN_PHASE2.md` M0.
- Training speed: ~0.3-0.4 s/step at batch 32 (data loading bound).
- Next: baseline training -> closed-loop eval (needs the game, focused, plugin
  listening), then the 2 s context model and fractional steering execution.

## 2026-10-05 Session 1 (cloud, build only)

Process: the planner session writes specs/contracts and reviews; sonnet
sub-agents write module code against `docs/ARCHITECTURE.md`. Every module was
reviewed, tests run, then committed. Review findings that changed code:
- model: absolute time embedding mismatched dense supervision -> relative
  step bias (D-011); live CLI now adopts the checkpoint's data settings.
- data: compressed .npz made random window reads ~0.34 s -> storage format 2
  (D-007); render now checks frame race time == tick time (sync).
- eval: control-row/tick mapping was off by one tick at 60 Hz -> exact
  inverse of the dataset resampling (`control_row_for_tick`); finding about
  label loss at < 100 Hz recorded in D-003.
- `.gitignore` `data/` hid `tmagent/data/` -> `/data/`.
- bridge: first-run knobs moved into `GameConfig`; pygbx from git, not PyPI.
- Fake end-to-end probe (planner): BC with action history never starts the
  car in closed loop (copycat) although val metrics look fine; without
  history it drives. Default switched off (D-013).
- Not done here on purpose (user instruction): real training, replay
  downloads, clip rendering, docs/system.md, docs/latency.md.

- Repository was empty. Added SessionStart hook
  (`.claude/hooks/session-start.sh`, registered in `.claude/settings.json`):
  installs `pip -e .[dev]` + ruff/pytest in cloud sessions when manifests exist.
  Validated on a scratch project: hook exit 0 (twice, idempotent), ruff and
  a pytest test passed. Removed a `pip install --upgrade pip` step that fails
  on the Debian system pip.
- User constraints: no training and no clip generation in the cloud; build the
  finished setup here, test on the user's machine. Game = TMNF. Low-res clips.
  Look for existing tools first. Planner/reviewer delegates coding to cheaper
  agents. Keep this log current.
- Wrote `docs/ARCHITECTURE.md`, `tmagent/interfaces.py`, `tmagent/config.py`,
  `docs/decisions.md` (D-001..D-007).
- Cloud container: no GPU, `download.pytorch.org` blocked by proxy; torch is
  installed from PyPI instead.
