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
| Data pipeline (`tmagent/data`) | done, reviewed | storage format 2 (D-007), 125 tests |
| Model + training (`tmagent/model`, `tmagent/train`) | done, reviewed | relative step bias (D-011), 56+ tests |
| Runtime (`tmagent/runtime`) | done, reviewed | 37 tests, 0 % misses with 100 ms policy |
| FakeGame + eval (`tmagent/game/fake.py`, `tmagent/eval`) | done, reviewed | 66 tests |
| TMNF bridge (`tmagent/game/tmnf`, plugin, smoke tool, setup doc) | in progress (agent) | |
| Tools (render, fake pipeline, system/latency reports, TMX download) | in progress (agent) | |
| Phase 0 on user machine | not started | `docs/setup_windows.md` once written |

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
