import os
import tempfile

from engine.simulation_manager import SimulationManager


def test_simulation_basic_step():
    sim = SimulationManager(formula_list=[{"H":2, "O":1}], deterministic_mode=True, seed=42)
    # run a few steps
    sim.run_steps(n_steps=10, vis_interval=5)
    assert sim.frame == 10
    assert len(sim.energy_history) == 10


def test_frame_export_writes_file(tmp_path):
    outdir = tmp_path / "exports"
    outdir.mkdir()
    sim = SimulationManager(formula_list=[{"H":2, "O":1}], deterministic_mode=True, seed=123)
    sim.start_frame_export(str(outdir))
    sim.run_steps(n_steps=5, vis_interval=1)
    # expect at least one frames_*.jsonl file
    files = [p for p in outdir.iterdir() if p.name.startswith('frames_') and p.suffix == '.jsonl']
    assert len(files) >= 1
