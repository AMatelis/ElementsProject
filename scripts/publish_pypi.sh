#!/usr/bin/env bash
set -euo pipefail

if ! command -v twine >/dev/null 2>&1; then
  echo 'twine not installed; installing...'
  python -m pip install --upgrade twine
fi

python -m build

if [[ -z "${PYPI_TOKEN-}" ]]; then
  echo "PYPI_TOKEN not set; aborting upload. Build artifacts are in dist/"
  exit 1
fi

python -m twine upload -u __token__ -p "$PYPI_TOKEN" dist/*
