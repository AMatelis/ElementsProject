"""
Per-frame exporter for SimulationManager.
Writes one JSON object per line containing frame metadata and atom/bond states.
"""
from __future__ import annotations
from typing import Optional, Dict, Any
import os
import time
import json


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_frame_jsonl(sim, out_path: Optional[str] = None) -> str:
    """Append current simulation frame state to a JSONL file.

    If out_path is None, a timestamped file is created under `data/exports`.
    Returns the path written to.
    """
    out_dir = None
    if out_path:
        # if a directory provided, write into that directory with timestamped filename
        if os.path.isdir(out_path):
            out_dir = out_path
            os.makedirs(out_dir, exist_ok=True)
            filename = f"frames_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
            target = os.path.join(out_dir, filename)
        else:
            # treat out_path as file path
            ensure_parent_dir(out_path)
            target = out_path
    if out_dir is None:
        out_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(out_dir, exist_ok=True)
        filename = f"frames_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
        target = os.path.join(out_dir, filename)

    record = {
        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
        "frame": int(sim.frame),
        "dt": float(getattr(sim.physics, "dt", 0.0)),
        "seed": getattr(sim, "seed", None),
        "atoms": [],
        "bonds": []
    }

    # atoms
    for a in sim.atoms:
        try:
            record["atoms"].append({
                "uid": a.uid,
                "symbol": a.symbol,
                "pos": [float(a.pos[0]), float(a.pos[1])],
                "vel": [float(a.vel[0]), float(a.vel[1])],
                "mass": float(a.mass)
            })
        except Exception:
            continue

    # bonds
    for b in sim.bonds:
        try:
            record["bonds"].append((b.atom1.uid, b.atom2.uid, int(getattr(b, "order", 1))))
        except Exception:
            continue

    # append line
    try:
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # try to ensure directory exists and retry once
        ensure_parent_dir(target)
        with open(target, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return target
