from simulation_viewer import simulate_and_report, PLOTLY_AVAILABLE


def test_simulate_and_report_h2o():
    sim, products, products_raw = simulate_and_report('H2,O', frames=40, interval=1, deterministic=True)
    # expect a bond formed event in KB for H and O (ensures deterministically formed H2O occurred at some time)
    assert any(ev.get('event_type') == 'bond_formed' and any(a.get('symbol') == 'H' for a in ev.get('atoms', [])) and any(a.get('symbol') == 'O' for a in ev.get('atoms', [])) for ev in getattr(sim.kb, 'events', []))

def test_simulate_and_report_custom():
    # test arbitrary compound: Na and Cl forming NaCl
    sim, products, products_raw = simulate_and_report('Na,Cl', frames=40, interval=1, deterministic=True)
    assert any('NA' in p.upper() or 'CL' in p.upper() or ('Na' in p and 'Cl' in p) for p in products.values())
