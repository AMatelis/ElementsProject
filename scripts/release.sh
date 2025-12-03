#!/usr/bin/env bash
set -eu

echo "Running tests..."
python -m pytest -q

echo "Building distribution..."
python -m build

echo "Generating docs demo..."
python scripts/export_demo.py

echo "Done. You can now push a tag or run 'gh release' or upload dist/* to PyPI via twine." 
