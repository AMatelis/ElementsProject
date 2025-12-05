import os
import sys
import json
import argparse
from typing import Optional

# Engine imports
from engine.simulation_manager import SimulationManager
from engine.formula_parser import parse_formula
from engine.elements_data import ELEMENT_DATA, load_elements  # <-- fixed import

# Visualization exports
from visual.export_tools import (
    export_simulation_to_plotly_html,
    ensure_dir,
    save_frame_png
)

# ML and KB
from ml.knowledge_base import ReactionKnowledgeBase
from ml.gnn_model import BondPredictorGNN


def run_simulation_cli():
    """
    Run a simulation from CLI with options:
    --formulas, --frames, --deterministic, --temperature, --export-html
    """
    parser = argparse.ArgumentParser(description="Run molecular reaction simulation.")
    parser.add_argument("--formulas", type=str, required=True, help="Comma-separated molecule formulas")
    parser.add_argument("--frames", type=int, default=600, help="Number of frames")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic rule-based reactions")
    parser.add_argument("--temperature", type=float, default=None, help="Simulation temperature override")
    parser.add_argument("--elements", type=str, default=None, help="Custom elements JSON file")
    parser.add_argument("--export-html", type=str, default=None, help="Path to export interactive HTML")
    parser.add_argument("--export-product-json", type=str, default=None, help="Directory to export product timeline JSON")
    parser.add_argument("--export-product-story", type=str, default=None, help="Directory to export product story text")
    args = parser.parse_args()

    # Load global elements JSON
    load_elements()  # ensure ELEMENT_DATA is populated

    # Load custom elements if provided
    if args.elements:
        try:
            with open(args.elements, 'r', encoding='utf-8') as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and 'elements' in raw:
                for el in raw['elements']:
                    ELEMENT_DATA[el['symbol'].upper()] = el
            elif isinstance(raw, dict):
                for k, v in raw.items():
                    ELEMENT_DATA[k.upper()] = v
            print(f"[INFO] Loaded custom elements from {args.elements}")
        except Exception as e:
            print(f"[WARN] Failed to load elements.json: {e}")

    # Parse formulas
    formula_strings = [f.strip() for f in args.formulas.split(",") if f.strip()]
    formula_dicts = []
    for f in formula_strings:
        try:
            formula_dicts.append(parse_formula(f))
        except Exception as e:
            print(f"[ERROR] Invalid formula '{f}': {e}")
            sys.exit(1)

    # Determine temperature
    sim_temp = float(args.temperature) if args.temperature is not None else (50.0 if args.deterministic else 300.0)

    # Initialize SimulationManager
    sim = SimulationManager(formula_dicts, deterministic_mode=args.deterministic, temperature=sim_temp)
    sim.run_steps(n_steps=args.frames, vis_interval=1)

    # Export events
    output_events = getattr(sim.kb, 'events', [])
    events_file = "simulation_events.json"
    try:
        with open(events_file, "w", encoding='utf-8') as f:
            json.dump(output_events, f, indent=2)
        print(f"[INFO] Simulation completed. Events saved to {events_file}")
    except Exception as e:
        print(f"[WARN] Failed to save events: {e}")

    # Optional HTML export
    if args.export_html:
        try:
            n_frames = min(args.frames, sim.frame)
            export_simulation_to_plotly_html(sim, args.export_html, n_frames=n_frames, fps=10)
            print(f"[INFO] Interactive HTML exported to {args.export_html}")
        except Exception as e:
            print(f"[WARN] HTML export failed: {e}")

    # Optional product JSON export
    if args.export_product_json:
        try:
            ensure_dir(args.export_product_json)
            products = sim.detect_products()
            from visual.export_tools import export_product_timeline
            for k in products.keys():
                fname = os.path.join(args.export_product_json, f"{k}.json")
                export_product_timeline(sim, k, fname)
            print(f"[INFO] Exported product timelines to {args.export_product_json}")
        except Exception as e:
            print(f"[WARN] Product timeline export failed: {e}")

    # Optional product story export
    if args.export_product_story:
        try:
            ensure_dir(args.export_product_story)
            products = sim.detect_products()
            from visual.export_tools import generate_reaction_story
            for k in products.keys():
                fname = os.path.join(args.export_product_story, f"{k}.txt")
                story = generate_reaction_story(sim, k, products)
                with open(fname, 'w', encoding='utf-8') as fh:
                    fh.write(story)
            print(f"[INFO] Exported product stories to {args.export_product_story}")
        except Exception as e:
            print(f"[WARN] Product story export failed: {e}")


def launch_gui():
    """
    Launch GUI mode using SimulationGUIAdvanced.
    """
    try:
        from gui.simulation_gui_advanced import SimulationGUIAdvanced
        gui = SimulationGUIAdvanced(title="Molecular Reaction Simulation - Advanced")
        gui.start()  # This calls mainloop()
    except Exception as e:
        print(f"[ERROR] GUI launch failed: {e}")  
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # If CLI args indicate formulas, elements, or export, run CLI mode
    cli_args = [arg for arg in sys.argv[1:] if arg.startswith("--formulas") or arg.startswith("--export-html") or arg.startswith("--elements")]
    if cli_args:
        run_simulation_cli()
    else:
        # Default: launch GUI
        launch_gui()