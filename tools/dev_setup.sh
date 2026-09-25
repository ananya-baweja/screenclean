#!/usr/bin/env bash
# Developer setup: install the package with dev extras, then run the fast tests.
# Usage: bash tools/dev_setup.sh
# Uses the active virtual environment if there is one, otherwise the system Python.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install -q uv
if [ -n "${VIRTUAL_ENV:-}" ]; then
  uv pip install -q -e ".[dev]"
else
  uv pip install -q --system -e ".[dev]"
fi

python -m pytest -q -m "not slow"
