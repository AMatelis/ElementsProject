# ElementsProject — Molecular Reaction Simulation

This project simulates a simple 2D particle-based reaction system with heuristic rules for bond formation and breakage, a small physics engine, and facilities for logging and ML dataset generation.

## New Publishable Feature — Interactive HTML Export

A new feature exports a recorded simulation as a single interactive HTML animation using Plotly. The HTML can be shared on GitHub Pages, uploaded to a blog, or hosted as a standalone file.

How to use (CLI):

```powershell
# Run a reaction simulation and export the interactive animation to 'sim.html' (deterministic mode example)
python cli.py --formulas H2,O --frames 600 --export-html sim.html --export-frames 500 --export-fps 12 --deterministic
# Show detected products and explain how bonds were formed
python cli.py --formulas H2,O --frames 600 --deterministic --show-products --explain-products
```

Notes:
- Plotly is optional and will be used only if installed. Install dependencies with:

```powershell
pip install -r requirements.txt
```

- For best results, run a simulation that records histories in the `sim` object so Plotly can render positions across frames.

Publishing the generated HTML (GitHub Pages / Static hosting):

1. Copy the generated HTML (e.g., `demo_sim.html`) into a `docs/` directory at the repository root.
2. Commit and push the `docs/` folder to GitHub.
3. In your GitHub repository Settings → Pages, select the `docs/` folder as the source to host the HTML as a static site.

Alternatively, add the HTML as a release artifact in GitHub Releases or serve it via a static host (Netlify, Vercel, or a Cloud Storage).

## Example Usage (Python)

```python
from simulation_viewer import SimulationManager, export_simulation_to_plotly_html
# Setup a simple water simulation
sim = SimulationManager([{"H":2, "O":1}], temperature=300)
sim.run_steps(200, vis_interval=10)
export_simulation_to_plotly_html(sim, 'water_sim.html', n_frames=200, fps=10)
```

### Dev: Tests & CI
We provide a minimal CI workflow that runs tests and generates a demo. To run tests locally:
```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
``` 

If you'd like, I can implement any of these next steps.

## Quick start

1. Create virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Run demo and export an interactive HTML animation:

```powershell
# Using the provided demo script
python scripts\export_demo.py

# Or run the CLI simulation and export (deterministic example)
python cli.py --formulas H2,O --frames 600 --export-html sim.html --export-frames 500 --export-fps 12 --deterministic

# If you installed via pip, you may run the console script:
elements-sim --formulas H2,O --frames 600 --export-html sim.html --export-frames 500 --export-fps 12 --deterministic
```

3. View the exported HTML locally or upload to GitHub Pages.

## GUI
The project contains a simple Tkinter GUI with a formula input screen. Launching the script without CLI arguments will open the GUI. The formula screen includes:
- Input entry for formulas (comma-separated), frames, and a deterministic checkbox
- Run & Show / Stop simulation controls
- Embedded Matplotlib visualization (live rendering)
- Product list and an "Explain Product" button that prints a timeline of bond events that led to the detected product

To open the GUI run:
```powershell
python simulation_viewer.py
```
