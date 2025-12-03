import os, sys
# Ensure repo root is on sys.path so `simulation_viewer` can be imported when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation_viewer import SimulationManager, export_simulation_to_plotly_html, PLOTLY_AVAILABLE

def main(out_html: str = 'docs/demo_sim.html'):
    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    formulas = [{'H': 2, 'O':1}, {'Cl':1}, {'C':6, 'H':6}]
    sim = SimulationManager(formulas)
    sim.run_steps(n_steps=300, vis_interval=1)
    if PLOTLY_AVAILABLE:
        export_simulation_to_plotly_html(sim, out_html, n_frames=300, fps=10)
        print(f'Exported {out_html}')
    else:
        print('Plotly not installed; demo GUI export skipped. To export HTML install plotly via pip install plotly')
    # Create a simple index.html if it doesn't exist
    try:
        idx_path = os.path.join(os.path.dirname(out_html), 'index.html')
        if not os.path.exists(idx_path):
            with open(idx_path, 'w', encoding='utf-8') as fh:
                fh.write(f"<html><body><h1>Simulation Demo</h1><iframe src=\"{os.path.basename(out_html)}\" width=100% height=900></iframe></body></html>")
            print(f'Created {idx_path}')
    except Exception:
        print('Failed to create docs/index.html')

if __name__ == '__main__':
    main()
