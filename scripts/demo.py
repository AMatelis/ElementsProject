import os
import sys
import argparse
from engine.formula_parser import parse_formula
from engine.simulation_manager import SimulationManager
from visual.export_tools import export_simulation_to_plotly_html, export_simulation_to_gif, export_simulation_to_mp4
from engine.products import generate_reaction_story
from engine.elements_data import ELEMENT_DATA
from engine.products import graph_to_formula

# -----------------------------
# Command-line interface
# -----------------------------
def run_demo_cli():
    parser = argparse.ArgumentParser(description="Molecular Reaction Simulation Demo")
    parser.add_argument("--formulas", type=str, required=False, default="H2,O", help="Comma-separated molecular formulas (e.g., H2,O)")
    parser.add_argument("--frames", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--deterministic", action="store_true", help="Run deterministic reaction engine")
    parser.add_argument("--temperature", type=float, default=None, help="Simulation temperature")
    parser.add_argument("--export-html", type=str, default=None, help="Path to export interactive HTML")
    parser.add_argument("--export-gif", type=str, default=None, help="Path to export animated GIF")
    parser.add_argument("--export-mp4", type=str, default=None, help="Path to export MP4 video")
    parser.add_argument("--show-products", action="store_true", help="Print detected products")
    parser.add_argument("--explain-products", action="store_true", help="Print reaction timeline per product")
    parser.add_argument("--gui", action="store_true", help="Launch GUI after running simulation")
    parser.add_argument("--elements", type=str, default=None, help="Optional JSON file to override elements")
    args = parser.parse_args()

    # Load optional custom elements
    if args.elements:
        import json
        with open(args.elements, 'r', encoding='utf-8') as fh:
            raw = json.load(fh)
        if isinstance(raw, dict) and 'elements' in raw:
            for el in raw['elements']:
                ELEMENT_DATA[el['symbol'].upper()] = el
        elif isinstance(raw, dict):
            for k, v in raw.items():
                ELEMENT_DATA[k.upper()] = v

    # Parse formulas
    formula_strings = [f.strip() for f in args.formulas.split(",") if f.strip()]
    formula_dicts = [parse_formula(f) for f in formula_strings]

    sim_temp = float(args.temperature) if args.temperature else (50.0 if args.deterministic else 300.0)

    sim = SimulationManager(formula_dicts, deterministic_mode=args.deterministic, temperature=sim_temp)
    print(f"Running simulation with formulas: {args.formulas}")
    sim.run_steps(n_steps=args.frames, vis_interval=1)
    print("Simulation complete.")

    # Show detected products
    products = sim.detect_products()
    if args.show_products:
        print("Detected products:")
        for k, (atoms, bonds) in products.items():
            formula = "".join([f"{el}{c if c>1 else ''}" for el, c in sorted(graph_to_formula(atoms, bonds).items())])
            print(f"- {k}: {formula}")

    # Explain reaction events
    if args.explain_products:
        for k in products.keys():
            print(f"\n--- Reaction timeline for {k} ---")
            story = generate_reaction_story(sim, k, products)
            print(story)

    # Export HTML
    if args.export_html:
        export_simulation_to_plotly_html(sim, args.export_html, n_frames=min(args.frames, sim.frame), fps=10)
        print(f"Exported interactive HTML to {args.export_html}")

    # Export GIF
    if args.export_gif:
        export_simulation_to_gif(sim, args.export_gif, n_frames=min(args.frames, sim.frame), fps=10)
        print(f"Exported GIF to {args.export_gif}")

    # Export MP4
    if args.export_mp4:
        export_simulation_to_mp4(sim, args.export_mp4, n_frames=min(args.frames, sim.frame), fps=10)
        print(f"Exported MP4 to {args.export_mp4}")

    # Optionally launch GUI
    if args.gui:
        try:
            from gui.simulation_gui import SimulationGUIAdvanced
            gui = SimulationGUIAdvanced(sim=sim)
            gui.main_menu()
            gui.start()
        except Exception as e:
            print(f"Failed to launch GUI: {e}")


# -----------------------------
# Optional quick demo function
# -----------------------------
def quick_demo():
    """Run a default demo for publication or testing purposes."""
    formulas = ["H2,O", "C6,H6", "Na,Cl", "C,O2"]
    for f in formulas:
        formula_dicts = [parse_formula(f)]
        sim = SimulationManager(formula_dicts, deterministic_mode=True, temperature=300.0)
        sim.run_steps(n_steps=200, vis_interval=1)
        products = sim.detect_products()
        print(f"\nDemo run: {f}")
        for k, (atoms, bonds) in products.items():
            formula = "".join([f"{el}{c if c>1 else ''}" for el, c in sorted(graph_to_formula(atoms, bonds).items())])
            print(f"- {k}: {formula}")


# -----------------------------
# Main entry
# -----------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_demo_cli()
    else:
        print("No arguments provided. Running quick demo...")
        quick_demo()