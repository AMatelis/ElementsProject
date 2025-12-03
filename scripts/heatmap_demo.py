import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation_viewer import SimulationManager, plot_reactivity_heatmap


def main(out_png: str = 'docs/reac_heatmap.png'):
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    formulas = [{'H': 2, 'O':1}, {'Na':1, 'Cl':1}, {'C':6, 'H':6}]
    sim = SimulationManager(formulas)
    sim.run_steps(n_steps=100, vis_interval=1)
    try:
        plot_reactivity_heatmap(sim, grid_size=160, radius=0.15, cmap='inferno')
        print('Displayed heatmap (interactive). To save as PNG invoke save_frame_png or modify plot_reactivity_heatmap)')
    except Exception:
        print('Heatmap generation failed')

if __name__ == '__main__':
    main()
