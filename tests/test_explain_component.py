from simulation_viewer import simulate_and_report, explain_component


def test_explain_component_h2o():
    sim, pretty, products = simulate_and_report('H2,O', frames=30, interval=1, deterministic=True)
    # pick a component that contains H and O
    comp_keys = [k for k, (atoms, bonds) in products.items()]
    assert comp_keys, 'no components generated'
    # call explain_component (prints to stdout); ensure it does not raise
    explain_component(sim, comp_keys[0], products)
