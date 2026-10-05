#!/bin/bash
# Installs project dependencies for Claude Code cloud sessions.
# Detects manifests at the repo root; no-ops for anything not present.
set -euo pipefail

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(dirname "$0")/../..}"

# Python
if [ -f pyproject.toml ] || ls requirements*.txt >/dev/null 2>&1; then
  if [ -f pyproject.toml ]; then
    python3 -m pip install --quiet -e ".[dev]" 2>/dev/null || python3 -m pip install --quiet -e .
  fi
  for req in requirements*.txt; do
    [ -f "$req" ] && python3 -m pip install --quiet -r "$req"
  done
  # Linter and test runner
  python3 -m pip install --quiet ruff pytest
  echo 'export PYTHONPATH="${CLAUDE_PROJECT_DIR:-.}${PYTHONPATH:+:$PYTHONPATH}"' >> "${CLAUDE_ENV_FILE:-/dev/null}"
fi

# Node
if [ -f package.json ]; then
  npm install --no-audit --no-fund
fi

echo "session-start: dependency setup complete"
