"""Headless run for producing ML-ready per-frame JSONL export.

Usage: python scripts/headless_run.py
"""
import os
import sys
import time

# Ensure project root is on sys.path when running from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.simulation_manager import SimulationManager

OUT_DIR = os.path.join("data", "exports")
os.makedirs(OUT_DIR, exist_ok=True)

# create simulation: water-like cluster (H2O)
sim = SimulationManager(formula_list=[{"H":2, "O":1}], temperature=300.0, deterministic_mode=True, seed=12345)
# enable per-frame export into OUT_DIR
sim.start_frame_export(OUT_DIR)

# run a short deterministic simulation
print(f"Starting headless run: frames -> {OUT_DIR}")
sim.run_steps(n_steps=200, vis_interval=50)

# allow writer to flush
time.sleep(0.1)

# discover newest file in OUT_DIR
files = [os.path.join(OUT_DIR, f) for f in os.listdir(OUT_DIR) if f.startswith("frames_") and f.endswith('.jsonl')]
files = sorted(files, key=lambda p: os.path.getmtime(p))
if files:
    newest = files[-1]
    print("Wrote frames to:", newest)
    # print a sample of first 3 lines
    try:
        with open(newest, 'r', encoding='utf-8') as fh:
            for i, line in enumerate(fh):
                if i >= 3:
                    break
                print('LINE', i+1, line.strip())
    except Exception as e:
        print('Failed to read sample from export file:', e)
else:
    print('No export files found in', OUT_DIR)

print('Headless run complete, frame count =', sim.frame)
