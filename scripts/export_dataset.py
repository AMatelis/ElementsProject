"""Command-line dataset exporter for ElementsProject.

Writes per-frame JSONL traces for ML training.

Example:
    python scripts/export_dataset.py --frames 500 --seed 42 --out data/exports
"""
import os
import sys
import argparse
import time

# ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.simulation_manager import SimulationManager


def main():
    p = argparse.ArgumentParser(description="Headless dataset exporter for ElementsProject")
    p.add_argument("--frames", type=int, default=500, help="Number of frames to simulate")
    p.add_argument("--seed", type=int, default=12345, help="Random seed for deterministic runs")
    p.add_argument("--out", type=str, default=os.path.join("data", "exports"), help="Output directory for per-frame JSONL")
    p.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature")
    p.add_argument("--deterministic", action="store_true", help="Enable ReactionEngine deterministic mode")
    p.add_argument("--formula", type=str, default=None, help="Formula as JSON-like e.g. '{\"H\":2, \"O\":1}' or a simple symbol like 'H:10' (not fully parsed)'")
    args = p.parse_args()

    # prepare output dir
    os.makedirs(args.out, exist_ok=True)

    # build a simple formula_list if none provided
    if args.formula:
        try:
            import json
            fdict = json.loads(args.formula)
            formula_list = [fdict]
        except Exception:
            # fallback simple parser 'H:10' or 'H=10'
            parts = args.formula.replace('=', ':').split(':')
            if len(parts) == 2:
                sym = parts[0].strip()
                cnt = int(parts[1].strip())
                formula_list = [{sym: cnt}]
            else:
                formula_list = [{"H": 10}]
    else:
        # default small cluster
        formula_list = [{"H": 2, "O": 1}]

    print(f"Starting export: frames={args.frames} seed={args.seed} out={args.out} formula={formula_list}")

    sim = SimulationManager(formula_list=formula_list, temperature=args.temperature, deterministic_mode=args.deterministic, seed=args.seed)
    sim.start_frame_export(args.out)

    # run simulation (synchronous)
    sim.run_steps(n_steps=args.frames, vis_interval=100)

    # finish and report
    time.sleep(0.05)
    files = [os.path.join(args.out, f) for f in os.listdir(args.out) if f.startswith("frames_") and f.endswith('.jsonl')]
    files = sorted(files, key=lambda p: os.path.getmtime(p))
    if files:
        print("Wrote frames to:", files[-1])
    else:
        print("No exports found in", args.out)

if __name__ == '__main__':
    main()
