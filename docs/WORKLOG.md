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
| Test drive UI | done | `python tools/eval_ui.py --config configs/tmnf.yaml` -> http://127.0.0.1:8600 |
| Stage 2 plan (memory agent) | written | `docs/PLAN_PHASE2.md` |

Test suite: `python -m pytest -q` -> 463 passed, 4 skipped (CPU, ~60 s).
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

## 2026-10-07 Session 4 (overnight, autonomous): DAgger

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
- Round d2 collection started with the 2.5k checkpoint (DAgger iteration 2).
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
