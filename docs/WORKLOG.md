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
| Baseline training (single frame) | running | then closed-loop eval on test maps |
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
