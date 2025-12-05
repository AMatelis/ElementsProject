from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, deque
import logging
import json
import os

logger = logging.getLogger(__name__)

# Type aliases for clarity
AtomType = Any  # avoid import cycle; expected to have uid, symbol, pos, history
BondType = Any  # expected to have atom1, atom2, order


# -----------------------
# Graph ↔ Formula utilities
# -----------------------

def graph_to_formula(atoms: List[AtomType], bonds: List[BondType]) -> Dict[str, int]:
    """
    Convert a list of Atom objects into a formula-like count dict:
        {'C': 6, 'H': 6, ...}

    This function normalizes element symbols to the canonical capitalization
    used in elements.json (first letter uppercase, rest lowercase).
    """
    counts: Dict[str, int] = defaultdict(int)
    for a in atoms:
        sym = getattr(a, "symbol", "?")
        if not isinstance(sym, str) or not sym:
            logger.debug("Atom without symbol encountered; skipping in formula.")
            continue
        # Normalize symbol: e.g., 'CL' or 'cl' -> 'Cl'
        norm = sym.capitalize()
        counts[norm] += 1
    return dict(counts)


def formula_to_pretty_string(counts: Dict[str, int]) -> str:
    """
    Convert counts dict into a human-readable molecular formula string,
    sorting elements in Hill order (C, H, then alphabetical) when appropriate.
    """
    if not counts:
        return ""

    # Hill system: C then H then alphabetical for organic molecules. If C not present,
    # alphabetical ordering is used.
    keys = list(counts.keys())
    if "C" in counts:
        # C, H, then others alphabetically
        others = sorted(k for k in keys if k not in ("C", "H"))
        order = ["C", "H"] + others
    else:
        order = sorted(keys)

    parts = []
    for k in order:
        if k not in counts:
            continue
        v = counts[k]
        parts.append(f"{k}{v if v != 1 else ''}")
    return "".join(parts)


# -----------------------
# Connected components (product detection)
# -----------------------

def extract_connected_components(atoms: List[AtomType], bonds: List[BondType]) -> List[Tuple[List[AtomType], List[BondType]]]:
    """
    Return a list of connected components (sub-molecules) given atoms and bonds.

    Returns:
        List of tuples: (list_of_atoms_in_component, list_of_bonds_in_component)
    """
    uid_to_atom: Dict[str, AtomType] = {getattr(a, "uid"): a for a in atoms}
    # Build adjacency
    adj: Dict[str, List[str]] = defaultdict(list)
    for b in bonds:
        try:
            u1 = getattr(b.atom1, "uid")
            u2 = getattr(b.atom2, "uid")
            adj[u1].append(u2)
            adj[u2].append(u1)
        except Exception:
            # allow bonds that are represented as tuples (uid1, uid2, order)
            try:
                u1, u2 = b[0], b[1]
                adj[u1].append(u2)
                adj[u2].append(u1)
            except Exception:
                logger.debug("Skipping malformed bond entry in extract_connected_components.")

    visited = set()
    components: List[Tuple[List[AtomType], List[BondType]]] = []

    for a in atoms:
        uid = getattr(a, "uid")
        if uid in visited:
            continue
        # BFS/DFS
        stack = [uid]
        comp_uids = set()
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_uids.add(u)
            for v in adj.get(u, []):
                if v not in visited:
                    stack.append(v)
        # collect atoms and bonds for this component
        comp_atoms = [uid_to_atom[u] for u in comp_uids if u in uid_to_atom]
        comp_bonds = []
        for b in bonds:
            try:
                a1 = getattr(b.atom1, "uid")
                a2 = getattr(b.atom2, "uid")
            except Exception:
                try:
                    a1, a2 = b[0], b[1]
                except Exception:
                    continue
            if a1 in comp_uids and a2 in comp_uids:
                comp_bonds.append(b)
        components.append((comp_atoms, comp_bonds))

    return components


def components_to_products(components: List[Tuple[List[AtomType], List[BondType]]]) -> List[Dict[str, int]]:
    """
    Convert connected components into product formula dictionaries.
    """
    prods = []
    for atoms, bonds in components:
        prods.append(graph_to_formula(atoms, bonds))
    return prods


# -----------------------
# Product detection utilities (convenience)
# -----------------------

def detect_products_from_scene(atoms: List[AtomType], bonds: List[BondType]) -> Dict[str, Tuple[List[AtomType], List[BondType]]]:
    """
    Detect connected components (products) and return a dictionary mapping a generated
    component key to (atoms, bonds).

    Keys will be 'comp_0', 'comp_1', ... ordered by component size (descending).
    """
    comps = extract_connected_components(atoms, bonds)
    # sort components by size (number of atoms) descending to make output deterministic
    comps_sorted = sorted(comps, key=lambda cb: len(cb[0]), reverse=True)
    out: Dict[str, Tuple[List[AtomType], List[BondType]]] = {}
    for i, (a_list, b_list) in enumerate(comps_sorted):
        out[f"comp_{i}"] = (a_list, b_list)
    return out

def explain_component(sim: Any, component_key: str, detected_products: Dict[str, Any]) -> str:
    """
    Generate a text explanation for a specific product component.
    
    Args:
        sim: The SimulationManager instance
        component_key: The product key/formula to explain
        detected_products: The detected products dict from sim.detect_products()
    
    Returns:
        A text explanation of the component
    """
    if component_key not in detected_products:
        return f"Component '{component_key}' not found in products."
    
    info = detected_products[component_key]
    explanation = f"Component: {info.get('pretty', component_key)}\n"
    explanation += f"Formula: {info.get('formula', {})}\n"
    explanation += f"Atoms: {info.get('num_atoms', 0)}\n"
    explanation += f"Bonds: {info.get('num_bonds', 0)}\n"
    
    if 'stories' in info and component_key in info['stories']:
        explanation += f"\nReaction Story:\n{info['stories'][component_key]}"
    
    return explanation


# -----------------------
# Event / timeline helpers
# -----------------------

def get_events_for_component(events: List[Dict[str, Any]], comp_atoms: List[AtomType]) -> List[Dict[str, Any]]:
    """
    Given a KB-style list of events and a component's atoms list, return all events
    that reference any UID in the component. Events are expected to have an 'atoms'
    field which is a list of dicts with 'uid' keys.
    """
    uids = {getattr(a, "uid") for a in comp_atoms}
    out = []
    for ev in events:
        atom_objs = ev.get("atoms", []) or []
        # atom dicts may contain 'uid'
        ev_uids = {a.get("uid") for a in atom_objs if isinstance(a, dict) and "uid" in a}
        if uids & ev_uids:
            out.append(ev)
    return out


def export_product_timeline(events: List[Dict[str, Any]], comp_atoms: List[AtomType], out_path: str) -> None:
    """
    Export the timeline of events that involve any atom in comp_atoms to JSON at out_path.
    """
    try:
        related = get_events_for_component(events, comp_atoms)
        ensure_parent_dir(out_path)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(related, fh, indent=2, ensure_ascii=False)
        logger.info("Exported product timeline to %s (events=%d)", out_path, len(related))
    except Exception:
        logger.exception("Failed to export product timeline to %s", out_path)


def generate_reaction_story(events: List[Dict[str, Any]], comp_atoms: List[AtomType], max_sentences: int = 100) -> str:
    """
    Generate a short textual 'story' describing bond events that built the given product.
    Uses chronological events that reference the atoms in comp_atoms.

    Output: multi-line string of simple sentences.
    """
    evs = get_events_for_component(events, comp_atoms)
    if not evs:
        return "No events recorded for this component."

    # sort by frame if present else timestamp
    def _key(e: Dict[str, Any]):
        return (e.get("frame") if e.get("frame") is not None else 0, e.get("timestamp") or "")

    evs_sorted = sorted(evs, key=_key)
    lines = []
    for ev in evs_sorted[:max_sentences]:
        frame = ev.get("frame", "NA")
        typ = ev.get("event_type", "").lower()
        atoms_involved = ev.get("atoms", [])
        # build symbol summary
        syms = sorted({a.get("symbol") for a in atoms_involved if isinstance(a, dict) and "symbol" in a})
        sym_str = ",".join(syms) if syms else "atoms"
        if typ in ("bond_formed", "bond_formation", "bondform"):
            lines.append(f"At frame {frame}, a bond formed involving {sym_str}.")
        elif typ in ("bond_broken", "bond_break", "bondbreak"):
            lines.append(f"At frame {frame}, a bond broke involving {sym_str}.")
        else:
            # generic event
            desc = ev.get("description") or ev.get("event_type") or "an event"
            lines.append(f"At frame {frame}, {desc} involved {sym_str}.")

    return "\n".join(lines)


# -----------------------
# Visualization helpers (lightweight, optional)
# -----------------------

def annotate_products_on_axes(ax, components: Dict[str, Tuple[List[AtomType], List[BondType]]],
                              fontsize: int = 9, boxstyle: str = "round,pad=0.2", color: str = "#222222"):
    """
    Annotate detected components on a matplotlib Axes with labels and light boxes.
    Expects `ax` to be a matplotlib.axes.Axes instance.
    Each annotation shows the product formula and is positioned at the centroid of atoms.
    """
    try:
        import numpy as _np
        for key, (atoms, bonds) in components.items():
            if not atoms:
                continue
            pts = _np.array([getattr(a, "pos") for a in atoms], dtype=float)
            if pts.size == 0:
                continue
            centroid = pts.mean(axis=0)
            formula = graph_to_formula(atoms, bonds)
            label = formula_to_pretty_string(formula)
            ax.text(float(centroid[0]), float(centroid[1]) + 0.06, label,
                    fontsize=fontsize, color="white", ha="center", va="bottom",
                    bbox=dict(facecolor=color, alpha=0.8, boxstyle=boxstyle))
    except Exception:
        logger.exception("annotate_products_on_axes failed (matplotlib may be missing).")


# -----------------------
# Helper utilities
# -----------------------

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


# -----------------------
# Module-level convenience functions
# -----------------------

def detect_and_describe(atoms: List[AtomType], bonds: List[BondType], kb_events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Convenience function that detects products and returns a structured dictionary:
    {
        "components": { "comp_0": {"formula": {...}, "pretty": "H2O", "atoms": [...], "bonds":[...]}, ... },
        "stories": {"comp_0": "At frame ...", ...}  # only if kb_events provided
    }
    """
    comps = detect_products_from_scene(atoms, bonds)
    out = {"components": {}, "stories": {}}
    for k, (a_list, b_list) in comps.items():
        formula = graph_to_formula(a_list, b_list)
        out["components"][k] = {
            "formula": formula,
            "pretty": formula_to_pretty_string(formula),
            "num_atoms": len(a_list),
            "num_bonds": len(b_list),
            "atoms": [getattr(a, "uid") for a in a_list],
            "bonds": [(getattr(b, "atom1", None) and getattr(b.atom1, "uid", None),
                       getattr(b, "atom2", None) and getattr(b.atom2, "uid", None),
                       getattr(b, "order", None)) for b in b_list]
        }
        if kb_events is not None:
            out["stories"][k] = generate_reaction_story(kb_events, a_list)
    return out