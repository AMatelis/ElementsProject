from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import logging
import json
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Type aliases for clarity
AtomType = Any  # expected to have uid, symbol, pos, history
BondType = Any  # expected to have atom1, atom2, order


# -----------------------
# Graph ↔ Formula utilities
# -----------------------

def graph_to_formula(atoms: List[AtomType], bonds: List[BondType]) -> Dict[str, int]:
    """
    Convert a list of Atom objects into a formula-like count dictionary.
    Normalizes element symbols (first letter uppercase, rest lowercase).
    Missing or malformed symbols are represented as '?'.
    """
    counts: Dict[str, int] = defaultdict(int)
    for atom in atoms:
        sym = getattr(atom, "symbol", "?")
        if isinstance(sym, str) and sym:
            counts[sym.capitalize()] += 1
        else:
            counts["?"] += 1
            logger.debug("Atom with missing or invalid symbol: %s", getattr(atom, "uid", "unknown"))
    return dict(counts)


def formula_to_pretty_string(counts: Dict[str, int]) -> str:
    """
    Convert a formula dict to a human-readable string, sorted in Hill order.
    Unknown atoms '?' are placed last.
    """
    if not counts:
        return "Unknown"

    keys = sorted([k for k in counts.keys() if k != "?"])
    unknowns = ["?"] if "?" in counts else []

    if "C" in counts:
        others = [k for k in keys if k not in ("C", "H")]
        order = ["C", "H"] + others
    else:
        order = keys
    order += unknowns

    parts = [f"{k}{counts[k] if counts[k] != 1 else ''}" for k in order if counts.get(k, 0) > 0]
    return "".join(parts) if parts else "Unknown"


# -----------------------
# Connected components
# -----------------------

def extract_connected_components(atoms: List[AtomType], bonds: List[BondType]) -> List[Tuple[List[AtomType], List[BondType]]]:
    """
    Identify connected components (sub-molecules) from a list of atoms and bonds.
    Returns a list of tuples: (atoms_in_component, bonds_in_component)
    """
    uid_to_atom: Dict[str, AtomType] = {getattr(a, "uid"): a for a in atoms}
    adj: Dict[str, List[str]] = defaultdict(list)

    for bond in bonds:
        try:
            u1 = getattr(bond.atom1, "uid")
            u2 = getattr(bond.atom2, "uid")
        except AttributeError:
            try:
                u1, u2 = bond[0], bond[1]
            except Exception:
                logger.debug("Malformed bond skipped: %s", bond)
                continue
        adj[u1].append(u2)
        adj[u2].append(u1)

    visited = set()
    components: List[Tuple[List[AtomType], List[BondType]]] = []

    for atom in atoms:
        uid = getattr(atom, "uid")
        if uid in visited:
            continue
        stack = [uid]
        comp_uids = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            comp_uids.add(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited:
                    stack.append(neighbor)
        comp_atoms = [uid_to_atom[u] for u in comp_uids if u in uid_to_atom]
        comp_bonds = []
        for bond in bonds:
            try:
                a1 = getattr(bond.atom1, "uid") if hasattr(bond, "atom1") else bond[0]
                a2 = getattr(bond.atom2, "uid") if hasattr(bond, "atom2") else bond[1]
            except Exception:
                continue
            if a1 in comp_uids and a2 in comp_uids:
                comp_bonds.append(bond)
        components.append((comp_atoms, comp_bonds))

    return components


def detect_products_from_scene(atoms: List[AtomType], bonds: List[BondType]) -> Dict[str, Tuple[List[AtomType], List[BondType]]]:
    """
    Detect connected components and return a dict mapping:
    'comp_0', 'comp_1', ... -> (atoms, bonds)
    Components are sorted by number of atoms descending.
    """
    components = extract_connected_components(atoms, bonds)
    components_sorted = sorted(components, key=lambda cb: len(cb[0]), reverse=True)
    return {f"comp_{i}": (a_list, b_list) for i, (a_list, b_list) in enumerate(components_sorted)}


# -----------------------
# Event / timeline utilities
# -----------------------

def get_events_for_component(events: List[Dict[str, Any]], comp_atoms: List[AtomType]) -> List[Dict[str, Any]]:
    """
    Return all events referencing any UID in comp_atoms.
    Events must have 'atoms' field containing dicts with 'uid'.
    """
    uids = {getattr(a, "uid") for a in comp_atoms}
    matched_events = []
    for ev in events:
        atoms_in_event = ev.get("atoms", [])
        ev_uids = {a.get("uid") for a in atoms_in_event if isinstance(a, dict) and "uid" in a}
        if uids & ev_uids:
            matched_events.append(ev)
    return matched_events


def generate_reaction_story(events: List[Dict[str, Any]], comp_atoms: List[AtomType], max_sentences: int = 100) -> str:
    """
    Generate a textual reaction story for the given component atoms.
    """
    relevant_events = get_events_for_component(events, comp_atoms)
    if not relevant_events:
        return "No events recorded for this component."

    relevant_events_sorted = sorted(relevant_events, key=lambda e: (e.get("frame", 0), e.get("timestamp", "")))
    lines = []

    for ev in relevant_events_sorted[:max_sentences]:
        frame = ev.get("frame", "NA")
        typ = ev.get("event_type", "").lower()
        atoms_involved = ev.get("atoms", [])
        symbols = sorted({a.get("symbol") for a in atoms_involved if isinstance(a, dict) and "symbol" in a})
        symbol_str = ",".join(symbols) if symbols else "atoms"
        if typ in ("bond_formed", "bond_formation", "bondform"):
            lines.append(f"At frame {frame}, a bond formed involving {symbol_str}.")
        elif typ in ("bond_broken", "bond_break", "bondbreak"):
            lines.append(f"At frame {frame}, a bond broke involving {symbol_str}.")
        else:
            desc = ev.get("description") or ev.get("event_type") or "an event"
            lines.append(f"At frame {frame}, {desc} involved {symbol_str}.")
    return "\n".join(lines)


def export_product_timeline(events: List[Dict[str, Any]], comp_atoms: List[AtomType], out_path: str) -> None:
    """
    Export JSON timeline of events involving comp_atoms to out_path.
    """
    try:
        relevant_events = get_events_for_component(events, comp_atoms)
        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(relevant_events, fh, indent=2, ensure_ascii=False)
        logger.info("Exported %d events to %s", len(relevant_events), out_path)
    except Exception:
        logger.exception("Failed to export product timeline to %s", out_path)


# -----------------------
# Visualization helpers
# -----------------------

def annotate_products_on_axes(ax, components: Dict[str, Tuple[List[AtomType], List[BondType]]],
                              fontsize: int = 9, boxstyle: str = "round,pad=0.2", color: str = "#222222"):
    """
    Annotate matplotlib axes with product formulas at centroids.
    """
    try:
        import numpy as np
        for key, (atoms, bonds) in components.items():
            if not atoms:
                continue
            positions = np.array([getattr(a, "pos") for a in atoms], dtype=float)
            if positions.size == 0:
                continue
            centroid = positions.mean(axis=0)
            formula = graph_to_formula(atoms, bonds)
            label = formula_to_pretty_string(formula)
            ax.text(float(centroid[0]), float(centroid[1]) + 0.06, label,
                    fontsize=fontsize, color="white", ha="center", va="bottom",
                    bbox=dict(facecolor=color, alpha=0.8, boxstyle=boxstyle))
    except Exception:
        logger.exception("annotate_products_on_axes failed (matplotlib may be missing).")


# -----------------------
# Utilities
# -----------------------

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
def explain_component(
    atoms: List[AtomType],
    bonds: List[BondType],
    events: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Explain a single connected component in a GUI-friendly way.
    """
    formula = graph_to_formula(atoms, bonds)
    pretty = formula_to_pretty_string(formula)

    out = {
        "formula": dict(formula),
        "pretty": pretty,
        "num_atoms": len(atoms),
        "num_bonds": len(bonds),
        "atoms": [getattr(a, "uid", "?") for a in atoms],
    }

    if events is not None:
        out["story"] = generate_reaction_story(events, atoms)

    return out


# -----------------------
# Convenience module-level function
# -----------------------

def detect_and_describe(atoms: List[AtomType], bonds: List[BondType], kb_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Detect products and return a structured dictionary:
    {
        "components": { "comp_0": {...}, ... },
        "stories": { "comp_0": "...", ... }  # if kb_events provided
    }
    """
    components = detect_products_from_scene(atoms, bonds)
    out = {"components": {}, "stories": {}}

    for key, (a_list, b_list) in components.items():
        formula = graph_to_formula(a_list, b_list)
        pretty = formula_to_pretty_string(formula)
        bonds_list = []
        for b in b_list:
            try:
                a1 = getattr(b.atom1, "uid", None) if hasattr(b, "atom1") else b[0]
                a2 = getattr(b.atom2, "uid", None) if hasattr(b, "atom2") else b[1]
                order = getattr(b, "order", None)
                bonds_list.append((a1, a2, order))
            except Exception:
                bonds_list.append(("?", "?", None))

        out["components"][key] = {
            "formula": dict(formula),
            "pretty": pretty,
            "num_atoms": len(a_list),
            "num_bonds": len(b_list),
            "atoms": [getattr(a, "uid", "?") for a in a_list],
            "bonds": bonds_list
        }

        if kb_events is not None:
            out["stories"][key] = generate_reaction_story(kb_events, a_list)

    return out