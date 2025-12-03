import os
import tempfile
import numpy as np
from simulation_viewer import SimulationManager, compute_reactivity_heatmap, export_simulation_to_plotly_html, PLOTLY_AVAILABLE


def test_heatmap_nonzero():
    sim = SimulationManager([{"H":2, "O":1}, {"Na":1, "Cl":1}])
    # Ensure atoms are positioned close to produce non-zero reactivity scores
    for i, a in enumerate(sim.atoms):
        a.pos = np.array([0.5 + 0.01 * (i % 4), 0.5 + 0.01 * (i // 4)])
    sim.run_steps(n_steps=1, vis_interval=1)
    xx, yy, heat = compute_reactivity_heatmap(sim, grid_size=40, radius=0.2)
    # heatmap should be normalized and contain non-zero contributions
    assert heat.max() > 0.0
    assert heat.min() >= 0.0


def test_plotly_export(tmp_path):
    if not PLOTLY_AVAILABLE:
        import pytest
        pytest.skip("plotly not installed")
    sim = SimulationManager([{"H":2, "O":1}], deterministic_mode=True)
    # Ensure we have histories
    sim.run_steps(n_steps=20, vis_interval=1)
    out = tmp_path / "test_export.html"
    export_simulation_to_plotly_html(sim, str(out), n_frames=20, fps=5)
    assert out.exists()
