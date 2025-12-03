from simulation_viewer import SimulationManager, extract_connected_components, graph_to_formula

sim = SimulationManager([{"H":2, "O":1}], temperature=300, deterministic_mode=True)
sim.step()
comps = extract_connected_components(sim.atoms, sim.bonds)
for a,b in comps:
    print(graph_to_formula(a,b))
