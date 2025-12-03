import pytest
import numpy as np
from simulation_viewer import SimulationManager, parse_formula, extract_connected_components, graph_to_formula


def test_import_and_manager():
    # Quick smoke test: import and run small steps
    sim = SimulationManager([{"H":2, "O":1}], temperature=300)
    sim.run_steps(n_steps=2, vis_interval=1)
    assert len(sim.atoms) >= 3


def test_parse_custom_formula():
    eldata = {"H": {"symbol":"H", "covalent_radius":0.3}}
    f = parse_formula("Xy2H2", eldata)
    assert f.get("XY") == 2
    assert f.get("H") == 2


def test_parse_arbitrary_formula_feo():
    from simulation_viewer import parse_formula, ELEMENT_DATA
    fdict = parse_formula('Fe2O3', ELEMENT_DATA)
    assert fdict.get('FE') == 2
    assert fdict.get('O') == 3


def test_parse_unknown_element_fallback():
    from simulation_viewer import parse_formula, ELEMENT_DATA
    # use a fake element symbol 'Xx' that won't exist in ELEMENT_DATA
    fdict = parse_formula('Xx2H', ELEMENT_DATA)
    assert fdict.get('XX') == 2
    # fallback added into ELEMENT_DATA keys
    assert 'XX' in ELEMENT_DATA


def test_deterministic_reaction():
    # Deterministic mode: H2 + O should produce H2O deterministic bonds
    sim = SimulationManager([{"H":2, "O":1}], temperature=300, deterministic_mode=True)
    # run a single step to let deterministic engine apply
    sim.step()
    comps = extract_connected_components(sim.atoms, sim.bonds)
    # expect single component of H2O
    formulas = [graph_to_formula(a,b) for a,b in comps]
    assert any(f.get("H",0)==2 and f.get("O",0)==1 for f in formulas)


def test_ch4_deterministic_formation():
    # CH4 deterministic check: C should form bonds with 4 H atoms
    sim = SimulationManager([{"C":1, "H":4}], temperature=300, deterministic_mode=True)
    # place atoms close to ensure energy heuristics allow bonding
    for i, a in enumerate(sim.atoms):
        a.pos = np.array([0.5 + 0.001*(i%4), 0.5 + 0.001*(i//4)])
    # step once to trigger deterministic pairing
    sim.step()
    # detect bonds on carbon
    c_atoms = [a for a in sim.atoms if a.symbol == 'C']
    assert len(c_atoms) == 1
    carbon = c_atoms[0]
    assert len(carbon.bonds) == 4
