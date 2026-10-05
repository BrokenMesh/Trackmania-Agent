# Work log

Read this first if you are an agent picking up the project. Newest entry on top.
Companion files: `docs/PLAN.md` (requirements, German), `docs/ARCHITECTURE.md`
(spec), `docs/decisions.md` (decision record), `docs/research.md` (verified
facts about TMNF tooling with sources).

## Status board

| Area | State | Owner notes |
|------|-------|-------------|
| Research (TMI, linesight, replay format, TMX, ToS) | in progress | |
| Shared contracts (`interfaces.py`, `config.py`) | done | |
| Data pipeline (`tmagent/data`) | todo | |
| Model + training (`tmagent/model`, `tmagent/train`) | todo | |
| Runtime + eval + FakeGame | todo | |
| TMNF bridge + render tool | todo | needs research |
| End-to-end fake pipeline | todo | |
| Phase 0 on user machine | not started | must run on the user's Windows PC |

## 2026-10-05 Session 1 (cloud, build only)

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
