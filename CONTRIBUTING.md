# Contributing to ElementsProject

Thank you for your interest in contributing! This is a small guide to get started.

## Running tests

Make a virtualenv and install developer requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Run pytest locally:

```powershell
python -m pytest -q
```

## Adding a demo / exporting demo HTML

A demo and export script is available in `scripts/export_demo.py`. To export a Plotly HTML demo, make sure `plotly` is installed and then run:

```powershell
python scripts\export_demo.py
``` 

The HTML is written to `docs/demo_sim.html` and `docs/index.html` is created if not present.

## Creating a reproducible demo build (publish)

- Run the demo script and commit the generated files in `docs/` if you intend to host via GitHub Pages.
- For CI, the workflow will build and run the tests automatically and you can extend it to upload artifacts.

## Code Style and Formatting

Please use `flake8` / `mypy` and optionally `black` to keep style consistent. We will add formatting hooks gradually.

