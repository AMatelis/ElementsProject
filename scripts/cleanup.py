import os
import shutil
from datetime import datetime

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT_BASE = os.path.join(ROOT, 'outputs')
FILES_TO_MOVE = ['sim.html', 'sim_custom.html', 'out.html', 'demo_sim.html', 'simulation_events.json', 'simulation_events.csv']

def main():
    ts = datetime.now().strftime('%Y%m%dT%H%M%S')
    out_dir = os.path.join(OUT_BASE, f'run_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    for fname in FILES_TO_MOVE:
        src = os.path.join(ROOT, fname)
        if os.path.exists(src):
            dst = os.path.join(out_dir, fname)
            shutil.move(src, dst)
            print(f'Moved {fname} -> {dst}')
    print(f'Cleaned up generated files into {out_dir}')

if __name__ == '__main__':
    main()
