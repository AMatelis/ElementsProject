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
        "bonds": [],
        # element_map provides per-symbol metadata from elements.json to avoid repeated lookups downstream
        "element_map": {}
    }

    # atoms
    # Gather element metadata map once
    try:
        from engine.elements_data import get_element
    except Exception:
        get_element = None

    for a in sim.atoms:
        try:
            symbol = a.symbol
            atom_entry = {
                "uid": a.uid,
                "symbol": symbol,
                "pos": [float(a.pos[0]), float(a.pos[1])],
                "vel": [float(a.vel[0]), float(a.vel[1])],
                "mass": float(a.mass),
                "radius": float(getattr(a, "radius", 0.0)),
                "electronegativity": float(getattr(a, "en", 0.0)),
                "charge": float(getattr(a, "charge", 0.0)),
                "category": getattr(a, "category", None),
            }
            record["atoms"].append(atom_entry)
            # populate element_map if available
            if get_element is not None and symbol not in record["element_map"]:
                try:
                    em = get_element(symbol)
                    # only include a small set of stable fields
                    record["element_map"][symbol] = {
                        "atomic_mass": float(em.get("atomic_mass", 0.0)),
                        "covalent_radius": float(em.get("covalent_radius", 0.0)),
                        "electronegativity_pauling": float(em.get("electronegativity_pauling", 0.0)),
                        "cpk-hex": em.get("cpk-hex", None),
                        "category": em.get("category", None)
                    }
                except Exception:
                    pass
        except Exception:
            continue

    # bonds
    # Deduplicate bonds (in case duplicates exist in sim.bonds)
    seen = set()
    for b in sim.bonds:
        try:
            u1, u2 = b.atom1.uid, b.atom2.uid
            key = tuple(sorted((u1, u2))) + (int(getattr(b, "order", 1)),)
            if key in seen:
                continue
            seen.add(key)
            record["bonds"].append([u1, u2, int(getattr(b, "order", 1))])
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
