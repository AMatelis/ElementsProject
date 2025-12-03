import os
import sys
import json
import math
import time
import random
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
from itertools import combinations, product
import threading
from collections import deque 
import datetime 
import argparse 
import csv 
from models.simulation_gui import SimulationGUIAdvanced


# Numeric & ML libs (optional gracefully)
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("NumPy is required. Install with `pip install numpy`.") from e

# Matplotlib (visualization)
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
    from matplotlib.lines import Line2D
except Exception:
    raise RuntimeError("matplotlib is required. Install with `pip install matplotlib`.")

# Optional: PyTorch & PyG for production GNN (if available)
USE_TORCH = False
USE_PYG = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USE_TORCH = True
    # Try to import PyTorch Geometric (pyg)
    try:
        import torch_geometric
        from torch_geometric.data import Data as PyGData, DataLoader as PyGDataLoader
        from torch_geometric.nn import MessagePassing, global_mean_pool, GraphConv, GINEConv, SAGEConv
        USE_PYG = True
    except Exception:
        # PyG not installed; we will provide a fallback GNN implemented in PyTorch (dense)
        USE_PYG = False
except Exception:
    USE_TORCH = False
    USE_PYG = False

# Joblib (model persistence)
try:
    import joblib
except Exception:
    joblib = None

# GUI (tkinter) - optional, fallback to CLI if missing
try:
    import tkinter as tk
    from tkinter import messagebox, filedialog, simpledialog
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# Optional RDKit-like functions? We will avoid RDKit dependency; implement light helpers.
# For graph drawing and layout, use networkx if available
try:
    import networkx as nx
    NX_AVAILABLE = True
except Exception:
    NX_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MolecularEngine")

# Optional Plotly for HTML export
try:
    import plotly.graph_objects as go  # type: ignore[reportMissingImports]
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# -----------------------
# Config & Constants
# -----------------------

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REACT_DATA_DIR = os.path.join(DATA_DIR, "reactions")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REACT_DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Element data file (user-supplied). We'll attempt to load; else provide minimal fallback.
ELEMENTS_FILE = os.path.join(DATA_DIR, "elements.json")

# Default parameters
VDW_SCALE = 0.05  # visualization scaling for van der Waals
DEFAULT_TEMPERATURE = 300.0  # Kelvin
BOLTZMANN = 1.380649e-23  # J/K (used with scaled units; we'll operate in reduced units)
TIME_STEP = 1e-3  # reduced timestep units
MAX_PARTITION_SIZE = 64  # for spatial hashing optimization

# Simulation physics constants (reduced / adjustable)
LJ_EPSILON = 0.1  # depth of Lennard-Jones well (reduced units)
LJ_SIGMA_FACTOR = 0.9  # factor to convert radius -> sigma
COULOMB_CONSTANT = 8.9875517923e9  # SI, but we'll scale in reduced units in code where needed

# Bonding rules thresholds
BOND_DISTANCE_FACTOR = 1.2  # times covalent radius sum to attempt bond
BOND_BREAK_DISTANCE_FACTOR = 2.2
BOND_ENERGY_SCALE = 2.5  # scale factor for bond energies in reduced units
BOND_BREAK_FORCE = 5.0  # force threshold to break bond (reduced units)

# GNN configuration
GNN_HIDDEN_DIM = 128
GNN_MESSAGE_PASSES = 3

# Training pipeline config
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 40
LEARNING_RATE = 1e-3

# Visualization config
NODE_MIN_RADIUS = 0.008
NODE_MAX_RADIUS = 0.06
CANVAS_SIZE = (9, 9)
VISUAL_BG = "#0b0f14"
VISUAL_ATOM_OUTLINE = "#111111"

# Random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if USE_TORCH:
    torch.manual_seed(SEED)

# -----------------------
# Utility helpers
# -----------------------

def safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def safe_save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_str() -> str:
    return time.strftime("%Y%m%dT%H%M%S")

# -----------------------
# Element Data Loading
# -----------------------
if not os.path.exists(ELEMENTS_FILE):
    # Provide a minimal fallback element dataset (subset) if file missing.
    logger.warning(f"{ELEMENTS_FILE} not found — creating minimal fallback element dataset in data/")
    minimal = {
        "elements": [
            {"name": "Hydrogen", "symbol": "H", "atomic_number": 1, "atomic_mass": 1.008, "electronegativity_pauling": 2.20, "group": 1, "period": 1, "cpk-hex": "FFFFFF", "category": "diatomic nonmetal", "covalent_radius": 0.31},
            {"name": "Carbon", "symbol": "C", "atomic_number": 6, "atomic_mass": 12.011, "electronegativity_pauling": 2.55, "group": 14, "period": 2, "cpk-hex": "909090", "category": "nonmetal", "covalent_radius": 0.76},
            {"name": "Nitrogen", "symbol": "N", "atomic_number": 7, "atomic_mass": 14.007, "electronegativity_pauling": 3.04, "group": 15, "period": 2, "cpk-hex": "3050F8", "category": "diatomic nonmetal", "covalent_radius": 0.71},
            {"name": "Oxygen", "symbol": "O", "atomic_number": 8, "atomic_mass": 15.999, "electronegativity_pauling": 3.44, "group": 16, "period": 2, "cpk-hex": "FF0D0D", "category": "diatomic nonmetal", "covalent_radius": 0.66},
            {"name": "Sodium", "symbol": "Na", "atomic_number": 11, "atomic_mass": 22.990, "electronegativity_pauling": 0.93, "group": 1, "period": 3, "cpk-hex": "AB5CF2", "category": "alkali metal", "covalent_radius": 1.66},
            {"name": "Chlorine", "symbol": "Cl", "atomic_number": 17, "atomic_mass": 35.45, "electronegativity_pauling": 3.16, "group": 17, "period": 3, "cpk-hex": "1FF01F", "category": "halogen", "covalent_radius": 0.99},
            {"name": "Helium", "symbol": "He", "atomic_number": 2, "atomic_mass": 4.0026, "electronegativity_pauling": None, "group": 18, "period": 1, "cpk-hex": "D9FFFF", "category": "noble gas", "covalent_radius": 0.28},
        ]
    }
    ensure_dir(DATA_DIR)
    safe_save_json(minimal, ELEMENTS_FILE)

raw_elements = safe_load_json(ELEMENTS_FILE)
if not raw_elements or "elements" not in raw_elements:
    raise RuntimeError(f"Failed to load elements data from {ELEMENTS_FILE}")

ELEMENT_DATA = {el['symbol'].upper(): el for el in raw_elements['elements']}

# Convenience lookup with defaults
def get_element(symbol: str) -> dict:
    s = symbol.upper()
    return ELEMENT_DATA.get(s, {"symbol": s, "atomic_mass": 0.0, "electronegativity_pauling": 0.0, "group": 0, "period": 0, "cpk-hex": "808080", "category": "", "covalent_radius": 0.7})

# -----------------------
# Chemistry Utilities
# -----------------------

def covalent_radius(symbol: str) -> float:
    el = get_element(symbol)
    return float(el.get("covalent_radius", 0.7))

def atomic_mass(symbol: str) -> float:
    el = get_element(symbol)
    return float(el.get("atomic_mass", 0.0))

def electronegativity(symbol: str) -> float:
    el = get_element(symbol)
    en = el.get("electronegativity_pauling")
    return float(en) if en not in (None, '') else 0.0

def is_noble(symbol: str) -> bool:
    return "noble" in (get_element(symbol).get("category","").lower())

def max_valence_guess(symbol: str) -> int:
    # crude valence estimation by group
    g = int(get_element(symbol).get("group", 0) or 0)
    if g == 0:
        return 4
    if g <= 2:
        return g
    if 13 <= g <= 18:
        return max(1, 18 - g)
    return 4

# -----------------------
# Data Structures
# -----------------------

class Atom:
    """A single atom/particle in the simulation."""
    __slots__ = ("uid","symbol","pos","vel","mass","charge","radius","color","state","bonds","history","properties")
    def __init__(self, uid: str, symbol: str, pos: Optional[np.ndarray] = None, charge: float = 0.0):
        self.uid = uid
        self.symbol = symbol.upper()
        self.pos = np.array(pos if pos is not None else np.random.rand(2), dtype=float)
        self.vel = np.zeros(2, dtype=float)
        # load element properties into the atom so rules/visualization can use them
        elem = get_element(self.symbol)
        self.mass = max(1e-6, float(elem.get('atomic_mass', 0.0)))
        self.charge = float(charge)
        self.radius = VDW_SCALE * max(0.2, covalent_radius(self.symbol))
        hexc = elem.get("cpk-hex", "808080")
        self.color = f"#{hexc}"
        self.state = "normal"  # normal, mutated, repaired, radical, ionized
        self.bonds: List["BondObj"] = []
        self.history: List[Dict[str,Any]] = []
        # copy element properties for downstream rules/training
        self.properties: Dict[str,Any] = dict(elem)

    def record(self, frame:int):
        self.history.append({"frame": frame, "pos": self.pos.tolist(), "state": self.state})

class BondObj:
    """Object representing a bond between two atoms."""
    __slots__ = ("atom1","atom2","order","rest_length","k_spring","energy","time_of_creation")
    def __init__(self, atom1: Atom, atom2: Atom, order:int = 1):
        self.atom1 = atom1
        self.atom2 = atom2
        self.order = order
        # rest length default to covalent radius sum
        self.rest_length = covalent_radius(atom1.symbol) + covalent_radius(atom2.symbol)
        self.rest_length *= VDW_SCALE * 0.6  # tuned scaling for visualization/physics
        self.k_spring = 100.0 * order  # bond stiffness (reduced units)
        # track creation time for short-lived highlights
        self.time_of_creation = time.time()
        self.energy = 0.0

    def compute_spring_force(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (force_on_atom1, force_on_atom2) from harmonic bond."""
        delta = self.atom2.pos - self.atom1.pos
        dist = np.linalg.norm(delta) + 1e-12
        direction = delta / dist
        # Hooke's law
        f_mag = self.k_spring * (dist - self.rest_length)
        f = f_mag * direction
        return f, -f

# -----------------------
# Spatial Hashing (for neighbor queries)
# -----------------------
class SpatialHash:
    """Simple grid-based spatial hashing for 2D positions in unit square."""
    def __init__(self, cell_size: float = 0.05):
        self.cell_size = cell_size
        self.cells = defaultdict(list)

    def _cell_coords(self, pos: np.ndarray) -> Tuple[int,int]:
        return (int(pos[0] // self.cell_size), int(pos[1] // self.cell_size))

    def clear(self):
        self.cells.clear()

    def add(self, atom: Atom):
        c = self._cell_coords(atom.pos)
        self.cells[c].append(atom)

    def neighbors(self, atom: Atom, radius: float) -> List[Atom]:
        cx, cy = self._cell_coords(atom.pos)
        rng = int(math.ceil(radius / self.cell_size))
        out = []
        for dx in range(-rng, rng+1):
            for dy in range(-rng, rng+1):
                for a in self.cells.get((cx+dx, cy+dy), []):
                    if a is atom:
                        continue
                    # cheap axis check then precise
                    if abs(a.pos[0] - atom.pos[0]) > radius or abs(a.pos[1] - atom.pos[1]) > radius:
                        continue
                    out.append(a)
        return out

# -----------------------
# Physics Engine
# -----------------------

class PhysicsEngine:
    """
    Core numerical integrator and force calculator:
      - Lennard-Jones for non-bonded
      - Coulomb (simple)
      - Bond springs
      - Langevin thermostat (temperature)
    """
    def __init__(self, atoms: List[Atom], bonds: List[BondObj], temperature: float = DEFAULT_TEMPERATURE):
        self.atoms = atoms
        self.bonds = bonds
        self.temperature = temperature
        self.dt = TIME_STEP
        self.spatial = SpatialHash(cell_size=0.06)
        self.frame = 0
        # reduced units scaling
        self.kT = 1.0 * (self.temperature / DEFAULT_TEMPERATURE)
        # thermostat damping
        self.gamma = 1.0

    def step(self):
        # 1) rebuild spatial hash
        self.spatial.clear()
        for a in self.atoms:
            # keep atoms inside unit box for visualization stability (periodic boundary could be used)
            a.pos = np.clip(a.pos, 0.0, 1.0)
            self.spatial.add(a)

        # 2) compute forces
        forces = {a: np.zeros(2, dtype=float) for a in self.atoms}

        # bond spring forces
        for b in list(self.bonds):
            f1, f2 = b.compute_spring_force()
            forces[b.atom1] += f1
            forces[b.atom2] += f2

        # non-bonded Lennard-Jones + Coulomb
        for a in self.atoms:
            # neighbor search
            neighs = self.spatial.neighbors(a, radius=0.2)
            for n in neighs:
                # skip if already bonded (we still compute weak LJ/short-range repulsion though)
                if any((b.atom1 is n or b.atom2 is n) for b in a.bonds):
                    continue
                delta = n.pos - a.pos
                r = np.linalg.norm(delta) + 1e-12
                # LJ
                sigma = (a.radius + n.radius) * LJ_SIGMA_FACTOR
                sr6 = (sigma / r) ** 6
                lj_force_mag = 24 * LJ_EPSILON * (2 * sr6 * sr6 - sr6) / r
                lj_force = lj_force_mag * (delta / r)
                forces[a] += lj_force
                # Coulomb (simple, reduced)
                if abs(a.charge) + abs(n.charge) > 1e-12:
                    coul = COULOMB_CONSTANT * (a.charge * n.charge) / (r*r + 1e-12)
                    forces[a] += coul * (delta / r)

        # Langevin thermostat & integrate (Velocity Verlet-like)
        for a in self.atoms:
            # thermostat random kick (scaled)
            rand_kick = np.sqrt(2.0 * self.gamma * self.kT / max(1.0, a.mass)) * np.random.normal(size=2)
            # friction
            friction = -self.gamma * a.vel
            total_force = forces[a] + rand_kick + friction
            # acceleration
            acc = total_force / max(1e-8, a.mass)
            # integrate velocities and positions
            a.vel += acc * self.dt
            a.pos += a.vel * self.dt
            # small damping to keep stable
            a.vel *= 0.999
            # clamp positions to [0,1]
            a.pos = np.clip(a.pos, 0.0, 1.0)

        self.frame += 1

    def set_temperature(self, T: float):
        self.temperature = T
        self.kT = 1.0 * (self.temperature / DEFAULT_TEMPERATURE)

# -----------------------
# Reaction Knowledge Base (upgraded)
# -----------------------

class ReactionKnowledgeBaseV2:
    """
    Upgraded KB that stores reaction event features and can persist datasets used for GNN training.
    - Stores per-event snapshots (atom types, positions, bonds, temperature, energies)
    - Exposes dataset exports for GNN training
    - Optionally persists a Torch model (if available)
    """
    def __init__(self, path_json: str = os.path.join(DATA_DIR, "reaction_patterns_v2.json"), model_prefix: str = os.path.join(MODEL_DIR, "reaction_gnn")):
        self.path_json = path_json
        self.model_prefix = model_prefix
        self.events: List[Dict[str,Any]] = []
        self.pair_counts = defaultdict(int)
        self.bond_counts = defaultdict(int)
        # Load existing
        if os.path.exists(self.path_json):
            try:
                obj = safe_load_json(self.path_json)
                if obj and isinstance(obj, dict):
                    self.events = obj.get("events", [])[:50000]
                    for k,v in obj.get("pair_counts", {}).items():
                        self.pair_counts[tuple(k.split("|"))] = v
                    for k,v in obj.get("bond_counts", {}).items():
                        self.bond_counts[tuple(k.split("|"))] = v
            except Exception:
                logger.exception("Failed to load existing KB JSON.")

    def add_event(self, event: Dict[str,Any]):
        """
        event should contain:
          - timestamp, frame
          - atoms: list of {uid, symbol, pos, charge}
          - bonds: list of (uid1, uid2, order)
          - temperature, energy, event_type (bond_form/break/other)
        """
        # minimal validation
        if "atoms" not in event or "bonds" not in event:
            raise ValueError("Invalid event format")
        self.events.append(event)
        # update counts
        syms = [a["symbol"].upper() for a in event["atoms"]]
        unique_syms = set(syms)
        for a,b in combinations(sorted(unique_syms), 2):
            self.pair_counts[(a,b)] += 1
        for b in event["bonds"]:
            k = tuple(sorted((b[0], b[1])))
            self.bond_counts[k] += 1
        # persistence (append-light): keep file reasonable size
        self._save_limited()

    def _save_limited(self, keep_last: int = 50000):
        out = {
            "events": self.events[-keep_last:],
            "pair_counts": {"|".join(k): v for k,v in self.pair_counts.items()},
            "bond_counts": {"|".join(k): v for k,v in self.bond_counts.items()},
        }
        safe_save_json(out, self.path_json)

    def export_ml_dataset(self, out_dir: str = os.path.join(DATA_DIR, "gnn_dataset")):
        """
        Export the internal events as an ML dataset consumable by PyTorch/PyG:
        - For each event: build node features, edge list, edge labels, global features.
        Will save a JSONL file with per-sample serialized fields, and a small index.
        """
        ensure_dir(out_dir)
        jsonl_path = os.path.join(out_dir, f"dataset_{now_str()}.jsonl")
        idx = []
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for i, ev in enumerate(self.events):
                # node features: Z, mass, en, degree, charge, pos
                nodes = []
                uid_to_idx = {}
                for j, a in enumerate(ev["atoms"]):
                    uid_to_idx[a["uid"]] = j
                    nodes.append({
                        "symbol": a["symbol"],
                        "mass": atomic_mass(a["symbol"]),
                        "en": electronegativity(a["symbol"]),
                        "charge": a.get("charge", 0.0),
                        "pos": a.get("pos", [0.0,0.0]),
                    })
                edges = []
                edge_labels = []
                for b in ev["bonds"]:
                    uid1, uid2, order = b
                    i1 = uid_to_idx.get(uid1)
                    i2 = uid_to_idx.get(uid2)
                    if i1 is None or i2 is None:
                        continue
                    edges.append([i1, i2])
                    edge_labels.append(order)
                sample = {
                    "nodes": nodes,
                    "edges": edges,
                    "edge_labels": edge_labels,
                    "global": {"temperature": ev.get("temperature", DEFAULT_TEMPERATURE), "energy": ev.get("energy", 0.0), "event_type": ev.get("event_type", "unknown")},
                }
                fh.write(json.dumps(sample) + "\n")
                idx.append({"sample_id": i, "file": os.path.basename(jsonl_path)})
        # write index
        safe_save_json({"index_file": os.path.basename(jsonl_path), "num": len(idx)}, os.path.join(out_dir, "index.json"))
        logger.info(f"Exported ML dataset to {jsonl_path} with {len(idx)} samples")
        return jsonl_path


class ReactionRuleEngine:
    """
    Simple deterministic reaction rule engine.
    - Matches common reactant patterns (e.g., H2 + O -> H2O, Na + Cl -> NaCl) and applies deterministic bond creation.
    - For arbitrary elements, uses valence and electronegativity heuristics to form bonds deterministically.
    """
    def __init__(self):
        # simple rule map: tuple of sorted symbols -> product formulas
        self.rules = {
            tuple(sorted(("H", "O"))): {"H": 2, "O": 1},  # H2 + O -> H2O
            tuple(sorted(("Na", "Cl"))): {"Na": 1, "Cl": 1},
            tuple(sorted(("H", "Cl"))): {"H": 1, "Cl": 1},
            tuple(sorted(("C", "O"))): {"C": 1, "O": 2},
            tuple(sorted(("C", "H"))): {"C": 1, "H": 4},  # CH4 heuristic: C-H bonds until valence filled
            tuple(sorted(("C", "O",))): {"C": 1, "O": 2},
        }

    def should_form_bond(self, a: Atom, b: Atom, physics: PhysicsEngine) -> Tuple[bool, float]:
        """Deterministic check whether a bond should form between a and b. Returns (should_form, score)"""
        # If atoms are already bonded, do nothing
        if any((bo.atom1 is b or bo.atom2 is b) for bo in a.bonds):
            return False, 0.0
        # quick distance heuristic: allow if within threshold unless a rule explicitly matches
        r = np.linalg.norm(a.pos - b.pos) + 1e-12
        cov_sum = covalent_radius(a.symbol) + covalent_radius(b.symbol)
        threshold = cov_sum * BOND_DISTANCE_FACTOR * VDW_SCALE
        # Check for defined rule combination first
        key = tuple(sorted((a.symbol, b.symbol)))
        if key in self.rules:
            # generate score based on bond energy; allow rule-driven bonds regardless of distance
            energy = estimate_bond_energy(a, b)
            score = 1.0 if energy > 0 else 0.0
            return True, float(score)
        # else use valence heuristics: if both have free valence, allow bonding
        # allow bond if both atoms have free valence and are close enough
        if len(a.bonds) < max_valence_guess(a.symbol) and len(b.bonds) < max_valence_guess(b.symbol):
            energy = estimate_bond_energy(a, b)
            # require that they are reasonably close and that energy is favorable
            if energy > -1.0 and r < threshold * 1.4:
                return True, float(min(1.0, max(0.05, (energy + 1.0) / 6.0)))
        # ionic check: large EN diff -> ionic pair
        en_diff = abs(electronegativity(a.symbol) - electronegativity(b.symbol))
        if en_diff > 1.2 and r < threshold * 2.0:
            # ionic attraction: score proportional to EN difference
            return True, float(min(1.0, en_diff / 4.0))
        return False, 0.0

# -----------------------
# GNN Modules (stubs + real if PyG available)
# -----------------------

if USE_TORCH and USE_PYG:
    # Real PyG-based GNNs (GraphConv / message passing)
    class BondPredictorGNN(torch.nn.Module):
        """
        GNN that predicts edge probabilities for bond formation between atoms.
        Uses node features and message passing.
        """
        def __init__(self, node_in_dim: int = 6, hidden_dim: int = GNN_HIDDEN_DIM, msg_passes: int = GNN_MESSAGE_PASSES):
            super().__init__()
            self.node_lin = nn.Linear(node_in_dim, hidden_dim)
            # stack of message-passing layers
            self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(msg_passes)])
            self.pool = global_mean_pool
            # edge scoring MLP
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim*2 + 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )

        def forward(self, data: PyGData):
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.node_lin(x))
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            # compute pairwise scores for candidate edges provided in data.candidate_edges (shape [2, M])
            ce = getattr(data, "candidate_edges", None)
            if ce is None:
                # fallback: score given edges
                src, dst = edge_index
                src_x = x[src]
                dst_x = x[dst]
                edge_feats = torch.cat([src_x, dst_x], dim=1)
                # if data has pairwise distances in data.edge_attr, use them
                if hasattr(data, "edge_attr"):
                    edge_feats = torch.cat([edge_feats, data.edge_attr.float()], dim=1)
                scores = self.edge_mlp(edge_feats)
                return scores.view(-1)
            else:
                src, dst = ce
                src_x = x[src]
                dst_x = x[dst]
                # optional distances
                d = data.candidate_edge_attr if hasattr(data, "candidate_edge_attr") else torch.zeros((src_x.shape[0],1))
                edge_feats = torch.cat([src_x, dst_x, d.float()], dim=1)
                scores = self.edge_mlp(edge_feats)
                return scores.view(-1)

    class ReactionPredictorGNN(torch.nn.Module):
        """
        Graph -> Graph or Graph -> Product prediction network.
        Here we implement a simple encoder -> pooled classifier that predicts reaction type.
        A more advanced graph-to-graph decoder can be plugged in later.
        """
        def __init__(self, node_in_dim: int = 6, hidden_dim: int = GNN_HIDDEN_DIM, msg_passes: int = GNN_MESSAGE_PASSES, out_classes: int = 8):
            super().__init__()
            self.node_lin = nn.Linear(node_in_dim, hidden_dim)
            self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(msg_passes)])
            self.pool = global_mean_pool
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, out_classes)
            )

        def forward(self, data: PyGData):
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.node_lin(x))
            for conv in self.convs:
                x = torch.relu(conv(x, edge_index))
            g = self.pool(x, data.batch)
            out = self.classifier(g)
            return out

else:
    # Fallback lightweight PyTorch-only or NumPy pseudo-GNN (dense)
    if USE_TORCH:
        class DenseGNN(torch.nn.Module):
            """
            Dense GNN approximation: compute node embeddings using MLP on node features,
            then create pairwise concatenations for edge scoring.
            Useful as a fallback when PyG isn't installed.
            """
            def __init__(self, node_in_dim: int = 6, hidden_dim: int = 128):
                super().__init__()
                self.node_mlp = nn.Sequential(
                    nn.Linear(node_in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                self.edge_mlp = nn.Sequential(
                    nn.Linear(hidden_dim*2 + 1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
                self.global_mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim//2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim//2, 8)
                )

            def forward_node_embeddings(self, x: torch.Tensor):
                return self.node_mlp(x)

            def score_edges(self, node_emb: torch.Tensor, pair_idx: torch.Tensor, pair_attr: Optional[torch.Tensor] = None):
                # pair_idx shape (M, 2)
                src = node_emb[pair_idx[:,0]]
                dst = node_emb[pair_idx[:,1]]
                pair_attr = pair_attr if pair_attr is not None else torch.zeros((src.shape[0],1), device=src.device)
                feats = torch.cat([src, dst, pair_attr], dim=1)
                return self.edge_mlp(feats).view(-1)

            def classify_graph(self, node_emb: torch.Tensor, batch_idx: torch.Tensor):
                # simple global mean pooling
                out = []
                device = node_emb.device
                for b in torch.unique(batch_idx):
                    mask = (batch_idx == b)
                    g = node_emb[mask].mean(dim=0)
                    out.append(g)
                g = torch.stack(out, dim=0)
                return self.global_mlp(g)

    else:
        # No torch at all: create tiny numpy-based predictors
        class NumpyPseudoGNN:
            def __init__(self):
                pass

            def predict_edge_scores(
                self,
                nodes: List[Dict],
                candidate_pairs: List[Tuple[int,int]],
                candidate_attrs: Optional[List[float]] = None
            ):
                # Return zero scores for each candidate pair by default (no torch model)
                scores = np.zeros(len(candidate_pairs), dtype=float)
                return scores

            # -------------------------------
            # Bond Energy Estimator
            # -------------------------------
            def estimate_bond_energy(self, atom1: "Atom", atom2: "Atom") -> float:
                """
                Estimate a heuristic bond energy (in reduced units) between two atoms based on
                electronegativity difference and element types.
                """
                # Get electronegativities
                en1 = electronegativity(atom1.symbol)
                en2 = electronegativity(atom2.symbol)
                en_diff = abs(en1 - en2)

                # Compute mass-based factor
                mass_factor = (atomic_mass(atom1.symbol) + atomic_mass(atom2.symbol)) / 50.0

                # Base energy by element identity heuristic
                base = 1.0 + 0.2 * mass_factor

                # Covalent bonding favored when EN difference is small
                covalent_term = max(0.1, 1.0 - 0.5 * en_diff)

                # Apply penalty for noble gases
                noble_penalty = 0.1 if (is_noble(atom1.symbol) or is_noble(atom2.symbol)) else 1.0

                # Final heuristic bond energy
                return float(BOND_ENERGY_SCALE * base * covalent_term * noble_penalty)


# Note: helper electronegativity/atomic_mass/is_noble are defined earlier using
# ELEMENT_DATA; we rely on those functions for element-specific values.


# -------------------------------
# Bond viability logic
# -------------------------------
# Note: basic can_form_bond logic is implemented below with a temperature-aware signature.


# -------------------------------
# Bond energy estimation
# -------------------------------
def estimate_bond_energy(atom1: "Atom", atom2: "Atom") -> float:
    """
    Heuristic bond-energy scoring using:
    - electronegativity difference
    - atomic mass factor
    - noble gas penalty
    """

    en1 = electronegativity(atom1.symbol)
    en2 = electronegativity(atom2.symbol)
    en_diff = abs(en1 - en2)

    # Mass-based influence (heavier atoms give slightly stronger bonds)
    mass_factor = (atomic_mass(atom1.symbol) + atomic_mass(atom2.symbol)) / 50.0

    # Baseline
    base = 1.0 + 0.2 * mass_factor

    # Covalent bonding behavior: smaller EN difference → stronger
    covalent_term = max(0.1, 1.0 - 0.5 * en_diff)

    # Heavy penalty for noble gases
    noble_penalty = 0.1 if (is_noble(atom1.symbol) or is_noble(atom2.symbol)) else 1.0

    # Final energy
    energy = BOND_ENERGY_SCALE * base * covalent_term * noble_penalty
    return float(energy)

def can_form_bond(
    atom1: "Atom",
    atom2: "Atom",
    temperature: float,
    prefer_strict_valence: bool = True
) -> tuple[bool, float]:
    """
    Determine whether a bond can form between atom1 and atom2 using geometric and
    energetic heuristics. Returns (can_form, score) where score is a
    probability-like confidence.
    """

    # Distance criterion
    r = np.linalg.norm(atom1.pos - atom2.pos) + 1e-12
    cov_sum = covalent_radius(atom1.symbol) + covalent_radius(atom2.symbol)
    threshold = cov_sum * BOND_DISTANCE_FACTOR * VDW_SCALE

    if r > max(0.0001, threshold):
        return False, 0.0
    # Valence check (optional)
    if prefer_strict_valence:
        current_bonds_a = len(atom1.bonds)
        current_bonds_b = len(atom2.bonds)
        if current_bonds_a >= max_valence_guess(atom1.symbol) or current_bonds_b >= max_valence_guess(atom2.symbol):
            return False, 0.0

    # Energy / temperature check: bond more likely at lower T if energy positive
    bond_energy = estimate_bond_energy(atom1, atom2)
    # Convert temperature effect into Boltzmann-like probability (reduced units)
    # Avoid actual physical units; treat kT scaled to [0,2]
    kT_red = max(0.001, temperature / DEFAULT_TEMPERATURE)
    prob = math.exp(-max(0.0, bond_energy) / (kT_red + 1e-12))
    # But restructure so stronger bonds (low bond_energy value) are more probable: invert
    prob = 1.0 - prob
    # clamp
    prob = float(max(0.0, min(1.0, prob)))
    # additional heuristic: electronegativity complement
    en_diff = abs(electronegativity(atom1.symbol) - electronegativity(atom2.symbol))
    en_bonus = 0.2 * (1.0 / (1.0 + en_diff))
    prob = min(1.0, prob + en_bonus * 0.6)
    return (prob > 0.15), prob

def should_break_bond(bond: BondObj, physics_engine: PhysicsEngine) -> Tuple[bool, float]:
    """
    Decide if a bond should break based on instantaneous forces, bond stretch, and temperature.
    Returns (break_flag, severity_score).
    """
    a = bond.atom1
    b = bond.atom2
    delta = b.pos - a.pos
    dist = np.linalg.norm(delta) + 1e-12
    # Stretch ratio
    stretch = dist / (bond.rest_length + 1e-12)
    # Approximate instantaneous force magnitude from spring: k * (dist - rest)
    f_mag = abs(bond.k_spring * (dist - bond.rest_length))
    # Temperature effect: high temperature increases break chance
    temp_factor = physics_engine.temperature / DEFAULT_TEMPERATURE
    severity = (stretch - 1.0) * 0.5 + f_mag * 0.02 + (temp_factor - 1.0) * 0.05
    severity = max(0.0, severity)
    # threshold
    if f_mag > BOND_BREAK_FORCE or stretch > BOND_BREAK_DISTANCE_FACTOR:
        return True, float(min(1.0, severity))
    # probabilistic break at elevated severity
    if severity > 0.5 and random.random() < min(0.5, severity):
        return True, float(severity)
    return False, float(severity)

# -----------------------
# Reaction Engine (manages bond updates)
# -----------------------

class ReactionEngine:
    """
    Responsible for:
      - scanning proximity graph for potential bond formation
      - applying can_form_bond heuristics
      - creating BondObj and updating Atom.bonds
      - evaluating existing bonds for breakage
      - logging bond events to ReactionKnowledgeBaseV2
    """
    def __init__(self, atoms: List[Atom], bonds: List[BondObj], physics: PhysicsEngine, kb: ReactionKnowledgeBaseV2):
        self.atoms = atoms
        self.bonds = bonds
        self.physics = physics
        self.kb = kb
        self.scan_radius = 0.18
        self.enabled = True
        self.deterministic = False
        self.rule_engine = ReactionRuleEngine() if 'ReactionRuleEngine' in globals() else None

    def step(self, frame: int):
        if not getattr(self, 'enabled', True):
            # still evaluate bond breakage when disabled to allow safe behavior
            for bond in list(self.bonds):
                brk, sev = should_break_bond(bond, self.physics)
                if brk:
                    try:
                        self.bonds.remove(bond)
                    except ValueError:
                        pass
                    if bond in bond.atom1.bonds:
                        bond.atom1.bonds.remove(bond)
                    if bond in bond.atom2.bonds:
                        bond.atom2.bonds.remove(bond)
                    ev = {
                        "timestamp": now_str(),
                        "frame": frame,
                        "event_type": "bond_broken",
                        "atoms": [{"uid": bond.atom1.uid, "symbol": bond.atom1.symbol, "pos": bond.atom1.pos.tolist(), "charge": bond.atom1.charge},
                                  {"uid": bond.atom2.uid, "symbol": bond.atom2.symbol, "pos": bond.atom2.pos.tolist(), "charge": bond.atom2.charge}],
                        "bonds": [(bond.atom1.uid, bond.atom2.uid, bond.order)],
                        "temperature": self.physics.temperature,
                        "severity": sev
                    }
                    try:
                        self.kb.add_event(ev)
                    except Exception:
                        logger.exception("KB add_event failed during bond breakage while disabled.")
            return
        # 1) Attempt bond formation among nearby atoms
        # Build spatial hash neighbor lists already exists in physics.spatial
        candidates = []
        if getattr(self, 'deterministic', False):
            # Evaluate all pairs deterministically to ensure rules apply regardless of proximity
            for i, a in enumerate(self.atoms):
                for j in range(i+1, len(self.atoms)):
                    n = self.atoms[j]
                    if any((b.atom1 is n or b.atom2 is n) for b in a.bonds):
                        continue
                    candidates.append((a, n))
        else:
            for a in self.atoms:
                neighs = self.physics.spatial.neighbors(a, radius=self.scan_radius)
                for n in neighs:
                    # avoid duplicates: evaluate only when uid order matches
                    if a.uid >= n.uid:
                        continue
                    # skip atoms already directly bonded
                    if any((b.atom1 is n or b.atom2 is n) for b in a.bonds):
                        continue
                    candidates.append((a, n))
        # Evaluate candidates
        logger.debug(f"Evaluating {len(candidates)} candidate pairs for bonding (deterministic={self.deterministic})")
        # collect preexisting bonds before any new formation to avoid immediate removal
        preexisting_bonds = list(self.bonds)
        for a, n in candidates:
            if self.deterministic and self.rule_engine is not None:
                # deterministic decision via rule engine
                should, score = self.rule_engine.should_form_bond(a, n, self.physics)
                if should:
                    logger.debug(f"Deterministic rule: forming bond between {a.uid} ({a.symbol}) and {n.uid} ({n.symbol}) score={score}")
                    before_len = len(self.bonds)
                    bond = BondObj(a, n, order=1)
                    self.bonds.append(bond)
                    logger.debug(f"Bond list length {before_len} -> {len(self.bonds)} after append")
                    a.bonds.append(bond)
                    n.bonds.append(bond)
                    # Log deterministic bond formation to the KB
                    ev = {
                        "timestamp": now_str(),
                        "frame": frame,
                        "event_type": "bond_formed",
                        "atoms": [{"uid": a.uid, "symbol": a.symbol, "pos": a.pos.tolist(), "charge": a.charge},
                                  {"uid": n.uid, "symbol": n.symbol, "pos": n.pos.tolist(), "charge": n.charge}],
                        "bonds": [(a.uid, n.uid, bond.order)],
                        "temperature": self.physics.temperature,
                        "energy": estimate_bond_energy(a, n)
                    }
                    try:
                        self.kb.add_event(ev)
                    except Exception:
                        logger.exception("KB add_event failed during deterministic bond formation.")
            else:
                can_form, score = can_form_bond(a, n, self.physics.temperature)
                if can_form:
                    # probabilistic formation
                    if random.random() < score:
                        # form bond
                        bond = BondObj(a, n, order=1)
                        self.bonds.append(bond)
                        a.bonds.append(bond)
                        n.bonds.append(bond)
                    ev = {
                        "timestamp": now_str(),
                        "frame": frame,
                        "event_type": "bond_formed",
                        "atoms": [{"uid": a.uid, "symbol": a.symbol, "pos": a.pos.tolist(), "charge": a.charge},
                                  {"uid": n.uid, "symbol": n.symbol, "pos": n.pos.tolist(), "charge": n.charge}],
                        "bonds": [(a.uid, n.uid, 1)],
                        "temperature": self.physics.temperature,
                        "energy": estimate_bond_energy(a, n)
                    }
                    try:
                        self.kb.add_event(ev)
                    except Exception:
                        logger.exception("KB add_event failed during bond formation.")

        # 2) Evaluate existing bonds for breakage. Only consider bonds that existed before this step
        for bond in list(preexisting_bonds):
            brk, sev = should_break_bond(bond, self.physics)
            if brk:
                # remove bond from bond list and atom bond lists
                try:
                    self.bonds.remove(bond)
                except ValueError:
                    pass
                logger.debug(f"Bond removed by break check: {bond.atom1.uid}({bond.atom1.symbol})-{bond.atom2.uid}({bond.atom2.symbol}) severity={sev}")
                if bond in bond.atom1.bonds:
                    bond.atom1.bonds.remove(bond)
                if bond in bond.atom2.bonds:
                    bond.atom2.bonds.remove(bond)
                ev = {
                    "timestamp": now_str(),
                    "frame": frame,
                    "event_type": "bond_broken",
                    "atoms": [{"uid": bond.atom1.uid, "symbol": bond.atom1.symbol, "pos": bond.atom1.pos.tolist(), "charge": bond.atom1.charge},
                              {"uid": bond.atom2.uid, "symbol": bond.atom2.symbol, "pos": bond.atom2.pos.tolist(), "charge": bond.atom2.charge}],
                    "bonds": [(bond.atom1.uid, bond.atom2.uid, bond.order)],
                    "temperature": self.physics.temperature,
                    "severity": sev
                }
                try:
                    self.kb.add_event(ev)
                except Exception:
                    logger.exception("KB add_event failed during bond breakage.")

# -----------------------
# Product Generation Scaffolding
# -----------------------

def graph_to_formula(atoms: List[Atom], bonds: List[BondObj]) -> Dict[str,int]:
    """
    Convert a list of atoms into a formula-like count dict.
    """
    counts = defaultdict(int)
    for a in atoms:
        counts[a.symbol] += 1
    return dict(counts)

def extract_connected_components(atoms: List[Atom], bonds: List[BondObj]) -> List[Tuple[List[Atom], List[BondObj]]]:
    """
    Return list of connected components (sub-molecules) given atoms and bonds.
    """
    uid_to_atom = {a.uid: a for a in atoms}
    # adjacency
    adj = defaultdict(list)
    for b in bonds:
        adj[b.atom1.uid].append(b.atom2.uid)
        adj[b.atom2.uid].append(b.atom1.uid)
    visited = set()
    components = []
    for a in atoms:
        if a.uid in visited:
            continue
        stack = [a.uid]
        comp_uids = []
        comp_bonds = []
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            comp_uids.append(u)
            for v in adj.get(u, []):
                if v not in visited:
                    stack.append(v)
        # collect atoms and bonds in component
        comp_atoms = [uid_to_atom[uid] for uid in comp_uids]
        for b in bonds:
            if b.atom1.uid in comp_uids and b.atom2.uid in comp_uids:
                comp_bonds.append(b)
        components.append((comp_atoms, comp_bonds))
    return components

def components_to_products(components: List[Tuple[List[Atom], List[BondObj]]]) -> List[Dict[str,int]]:
    """
    Convert components to product formulas.
    """
    prods = []
    for atoms, _ in components:
        prods.append(graph_to_formula(atoms, []))
    return prods

# -----------------------
# Simulation Manager
# -----------------------

class SimulationManager:
    """
    Top-level orchestrator: creates atoms from input formulas, runs physics + reaction engine,
    logs events, produces visualizations, supports training triggers.
    """
    def __init__(self, formula_list: List[Dict[str,int]], temperature: float = DEFAULT_TEMPERATURE, auto_train_kb: bool = True, deterministic_mode: bool = False):
        # flatten formula_list into atoms
        self.atoms: List[Atom] = []
        self.bonds: List[BondObj] = []
        self.kb = ReactionKnowledgeBaseV2()
        self.temperature = temperature
        uid_counter = 0
        for midx, fdict in enumerate(formula_list):
            # place molecules at different regions
            center = np.clip(np.random.rand(2) * 0.6 + 0.2, 0.0, 1.0)
            for sym, cnt in fdict.items():
                for i in range(max(1, int(cnt))):
                    uid = f"m{midx}_a{uid_counter}"
                    pos = center + np.random.normal(scale=0.02, size=2)
                    a = Atom(uid, sym, pos=pos)
                    self.atoms.append(a)
                    uid_counter += 1
        self.physics = PhysicsEngine(self.atoms, self.bonds, temperature=self.temperature)
        self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, self.kb)
        self.frame = 0
        self.auto_train_kb = auto_train_kb
        self.deterministic_mode = deterministic_mode
        self._initial_formulas = formula_list
        self.enable_bonds = True
        self._running = False
        self._thread = None
        # visualization handles
        self.fig = None
        self.ax = None
        self.artist_atoms = []
        self.artist_bonds = []
        self.text_overlay = None
        # event log for simulation end export
        self.event_log = []

    def step(self):
        # advance physics
        self.physics.step()
        self.reaction_engine.deterministic = self.deterministic_mode
        # reaction engine processes bonds based on new positions
        self.reaction_engine.step(self.frame)
        # record atom histories
        for a in self.atoms:
            a.record(self.frame)
        self.frame += 1

    def run_steps(self, n_steps: int = 1000, vis_interval: int = 10, update_callback: Optional[Any] = None):
        for _ in range(n_steps):
            self.step()
            if update_callback is not None and (self.frame % vis_interval == 0):
                update_callback(self)
            # periodic retrain/export
            if self.auto_train_kb and (self.frame % 400 == 0) and self.frame > 0:
                try:
                    # quick export dataset
                    self.kb.export_ml_dataset()
                except Exception:
                    logger.exception("Dataset export failed during run.")

    def start(self, interval: float = 0.02):
        """Start the simulation loop in a background thread."""
        if self._running:
            return
        self._running = True
        def loop():
            while self._running:
                self.step()
                time.sleep(interval)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background simulation thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def set_enable_bonds(self, enable: bool):
        """Enable or disable bond formation during reaction engine steps."""
        self.enable_bonds = bool(enable)
        # reaction engine relies on checks in ReactionEngine.step. We add a lightweight toggle.
        if hasattr(self.reaction_engine, 'enabled'):
            self.reaction_engine.enabled = self.enable_bonds

    def set_temperature(self, T: float):
        """Set the simulation temperature and update physics engine."""
        self.temperature = float(T)
        try:
            self.physics.set_temperature(self.temperature)
        except Exception:
            pass

    def reset_simulation(self):
        """Reset the simulation to initial formulas and clear KB/events."""
        # stop running thread if active
        self.stop()
        # rebuild atoms/bonds from original formulas
        self.atoms = []
        self.bonds = []
        self.kb = ReactionKnowledgeBaseV2()
        uid_counter = 0
        for midx, fdict in enumerate(self._initial_formulas):
            center = np.clip(np.random.rand(2) * 0.6 + 0.2, 0.0, 1.0)
            for sym, cnt in fdict.items():
                for i in range(max(1, int(cnt))):
                    uid = f"m{midx}_a{uid_counter}"
                    pos = center + np.random.normal(scale=0.02, size=2)
                    a = Atom(uid, sym, pos=pos)
                    self.atoms.append(a)
                    uid_counter += 1
        self.physics = PhysicsEngine(self.atoms, self.bonds, temperature=self.temperature)
        self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, self.kb)
        self.frame = 0

    def export_results(self, out_prefix: Optional[str] = None):
        if out_prefix is None:
            out_prefix = now_str()
        out_dir = os.path.join(OUTPUT_DIR, f"run_{out_prefix}")
        ensure_dir(out_dir)
        # export atom traces
        atoms_out = []
        for a in self.atoms:
            atoms_out.append({
                "uid": a.uid, "symbol": a.symbol, "history": a.history, "final_pos": a.pos.tolist(), "state": a.state
            })
        safe_save_json({"atoms": atoms_out, "frame_count": self.frame}, os.path.join(out_dir, "atoms.json"))
        # export bond list
        bond_list = []
        for b in self.bonds:
            bond_list.append((b.atom1.uid, b.atom2.uid, b.order))
        safe_save_json({"bonds": bond_list}, os.path.join(out_dir, "bonds.json"))
        # export KB snapshot
        try:
            self.kb._save_limited()
            safe_save_json({"kb_path": self.kb.path_json}, os.path.join(out_dir, "kb_snapshot.json"))
        except Exception:
            logger.exception("Failed to snapshot KB.")
        logger.info(f"Exported simulation results to {out_dir}")

    def detect_products(self) -> Dict[str, Tuple[List[Atom], List[BondObj]]]:
        """
        Detect product components in the current simulation and return a mapping
        label -> (atoms, bonds) for each connected component.
        """
        comps = extract_connected_components(self.atoms, self.bonds)
        out = {}
        for i, (atoms, bonds) in enumerate(comps):
            out[f"comp_{i}"] = (atoms, bonds)
        return out

# -----------------------
# Visualization Helpers (improved aesthetics)
# -----------------------

def draw_bezier(ax, p0, p1, control_offset=0.02, linewidth=2, color="#FFFFFF", alpha=0.8):
    """
    Draw a quadratic Bezier between p0 and p1 with a control point offset perpendicular to the segment
    to create a smooth curved bond visualization.
    """
    p0 = np.array(p0)
    p1 = np.array(p1)
    mid = (p0 + p1) / 2.0
    diff = p1 - p0
    perp = np.array([-diff[1], diff[0]])
    perp = perp / (np.linalg.norm(perp) + 1e-12)
    ctrl = mid + perp * control_offset
    path = np.array([p0, ctrl, p1])
    line = Line2D(path[:,0], path[:,1], linewidth=linewidth, color=color, alpha=alpha, solid_capstyle='round')
    ax.add_line(line)
    return line

def atom_radius_visual(atom: Atom) -> float:
    # map atom.radius (which is VDW scaled) into visual radius within [NODE_MIN_RADIUS, NODE_MAX_RADIUS]
    r = np.clip(atom.radius, NODE_MIN_RADIUS, NODE_MAX_RADIUS)
    return r

def render_simulation_frame(sim: SimulationManager):
    # set up if not exists
    if sim.fig is None or sim.ax is None:
        sim.fig, sim.ax = plt.subplots(figsize=CANVAS_SIZE)
        sim.fig.patch.set_facecolor(VISUAL_BG)
        sim.ax.set_facecolor(VISUAL_BG)
        sim.ax.set_xlim(0,1)
        sim.ax.set_ylim(0,1)
        sim.ax.set_xticks([])
        sim.ax.set_yticks([])
        # overlay text
        sim.text_overlay = sim.ax.text(0.5, 1.02, f"Frame: {sim.frame} KB events: {len(sim.kb.events)}", color="white", ha="center", va="bottom", transform=sim.ax.transAxes)

    # clear previous artists
    for art in (sim.artist_atoms + sim.artist_bonds):
        try:
            art.remove()
        except Exception:
            pass
    sim.artist_atoms = []
    sim.artist_bonds = []

    # draw bonds as bezier curves
    for b in sim.bonds:
        p0 = b.atom1.pos
        p1 = b.atom2.pos
        # color by average atom color with slight brightness
        col = b.atom1.color
        line = draw_bezier(sim.ax, p0, p1, control_offset=0.02, linewidth=2.5, color=col, alpha=0.7)
        sim.artist_bonds.append(line)

    # draw atoms on top
    for a in sim.atoms:
        r = atom_radius_visual(a)
        circ = Circle(a.pos, r, facecolor=a.color, edgecolor=VISUAL_ATOM_OUTLINE, linewidth=0.8, alpha=0.95, zorder=10)
        sim.ax.add_patch(circ)
        sim.artist_atoms.append(circ)
        # label
        sim.ax.text(a.pos[0], a.pos[1]-r-0.01, a.symbol, fontsize=8, color="white", ha="center", va="top", zorder=11)

    # update overlay
    sim.text_overlay.set_text(f"Frame: {sim.frame} KB events: {len(sim.kb.events)}")
    plt.pause(0.001) 
    # -----------------------
# CHUNK 3/10
# GNN training, dataset loading, training loop, evaluation, visualization niceties
# -----------------------

# -----------------------
# Dataset Loading Utilities
# -----------------------

def load_jsonl_samples(jsonl_path: str) -> list[dict]:
    samples = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
    except Exception:
        logger.exception(f"Failed to load jsonl {jsonl_path}")
    return samples

def build_pyg_dataset_from_jsonl(
    jsonl_path: str,
    elements_json_path: str = r"C:\Users\andre\Downloads\ElementsProject\data\elements.json"
) -> "Optional[list[PyGData]]":
    """
    Converts a JSONL exported by ReactionKnowledgeBaseV2.export_ml_dataset
    into a list of PyG Data objects suitable for GNN training.

    Node features now include a comprehensive set of chemical properties from elements.json:
        - atomic_mass, electronegativity, charge, electron_affinity
        - density, molar_heat, period, group
        - block (s/p/d/f encoded), phase (gas/liquid/solid encoded)
        - number of shells, first N ionization energies
        - x/y position, bias term
    Missing values are replaced with sensible defaults (0.0).

    Args:
        jsonl_path (str): Path to the JSONL file containing the dataset.
        elements_json_path (str): Path to elements.json containing atomic data.

    Returns:
        Optional[list[PyGData]]: List of PyG Data objects, or None if PyG/Torch unavailable.
    """
    if not (USE_PYG and USE_TORCH):
        logger.warning("PyG or Torch not available; cannot build PyG dataset.")
        return None

    # Load elements.json
    try:
        with open(elements_json_path, "r") as f:
            raw_elements = json.load(f)["elements"]
        elements_data = {e["symbol"].capitalize(): e for e in raw_elements}
    except Exception as e:
        logger.exception(f"Failed to load elements.json from {elements_json_path}: {e}")
        return None

    # Encode categorical features
    block_map = {"s": 0, "p": 1, "d": 2, "f": 3}
    phase_map = {"Gas": 0, "Liquid": 1, "Solid": 2}

    # Load JSONL samples
    try:
        samples = load_jsonl_samples(jsonl_path)
    except Exception as e:
        logger.exception(f"Failed to load JSONL samples from {jsonl_path}: {e}")
        return None

    if not samples:
        logger.warning(f"No samples found in {jsonl_path}")
        return []

    data_list = []

    for idx, s in enumerate(samples):
        nodes = s.get("nodes", [])
        edges = s.get("edges", [])
        edge_labels = s.get("edge_labels", [])

        if not nodes:
            logger.warning(f"Sample {idx} has no nodes, skipping.")
            continue

        # Build node feature tensor
        x = []
        for n_idx, n in enumerate(nodes):
            elem_symbol = n.get("element", "").capitalize()
            elem_info = elements_data.get(elem_symbol, {})

            # Node position
            pos = n.get("pos", [0.0, 0.0])
            if not isinstance(pos, (list, tuple)) or len(pos) < 2:
                logger.warning(f"Node {n_idx} in sample {idx} has invalid position {pos}, using [0.0, 0.0]")
                pos = [0.0, 0.0]

            # Extract features (JSONL overrides elements.json)
            atomic_mass = float(n.get("mass", elem_info.get("atomic_mass", 0.0)))
            en = float(n.get("en", elem_info.get("electronegativity_pauling", 0.0) or 0.0))
            charge = float(n.get("charge", elem_info.get("charge", 0.0)))
            electron_affinity = float(elem_info.get("electron_affinity", 0.0) or 0.0)
            density = float(elem_info.get("density", 0.0) or 0.0)
            molar_heat = float(elem_info.get("molar_heat", 0.0) or 0.0)
            period = float(elem_info.get("period", 0))
            group = float(elem_info.get("group", 0))
            block = float(block_map.get(elem_info.get("block", "s").lower(), 0))
            phase = float(phase_map.get(elem_info.get("phase", "Solid"), 2))
            shells_count = float(len(elem_info.get("shells", [])))
            
            # Include up to first 3 ionization energies as separate features
            ionization_energies = elem_info.get("ionization_energies", [0.0])
            ie_features = [float(ionization_energies[i]) if i < len(ionization_energies) else 0.0 for i in range(3)]

            x.append([
                atomic_mass, en, charge, electron_affinity, density, molar_heat,
                period, group, block, phase, shells_count, *ie_features,
                float(pos[0]), float(pos[1]), 1.0  # position + bias
            ])

        x_tensor = torch.tensor(x, dtype=torch.float)

        # Build edge tensors
        if not edges:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
            edge_label_tensor = torch.empty((0,), dtype=torch.long)
        else:
            try:
                ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
                edge_index = ei
                edge_attr = torch.ones((ei.shape[1], 1), dtype=torch.float)
                if edge_labels and len(edge_labels) == ei.shape[1]:
                    edge_label_tensor = torch.tensor(edge_labels, dtype=torch.long)
                else:
                    edge_label_tensor = torch.zeros((ei.shape[1],), dtype=torch.long)
                    if edge_labels:
                        logger.warning(f"Edge label count mismatch in sample {idx}; defaulting to zeros.")
            except Exception as e:
                logger.exception(f"Failed to process edges for sample {idx}: {e}")
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.float)
                edge_label_tensor = torch.empty((0,), dtype=torch.long)

        # Create PyG Data object
        data = PyGData(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data.y = torch.tensor([0], dtype=torch.long)  # Placeholder for reaction class
        data.edge_label = edge_label_tensor
        data_list.append(data)

    logger.info(f"Built {len(data_list)} PyG Data samples from {jsonl_path} using full elements.json features")
    return data_list

# -----------------------
# Model Persistence Utilities
# -----------------------

def save_model(model: Any, path: str):
    try:
        if USE_TORCH and isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
        elif joblib is not None:
            joblib.dump(model, path)
        else:
            with open(path, "wb") as fh:
                fh.write(b"")
        logger.info(f"Saved model to {path}")
    except Exception:
        logger.exception("Failed to save model")

def load_model(model_cls: Any, path: str):
    try:
        if USE_TORCH and issubclass(model_cls, torch.nn.Module):
            model = model_cls()
            model.load_state_dict(torch.load(path))
            return model
        elif joblib is not None:
            return joblib.load(path)
    except Exception:
        logger.exception("Failed to load model")
    return None

# -----------------------
# Training Loops
# -----------------------

def train_bond_predictor_from_jsonl(jsonl_path: str, epochs: int = TRAIN_EPOCHS, batch_size: int = TRAIN_BATCH_SIZE, save_path: Optional[str] = None):
    """
    Train an edge/bond predictor GNN using PyG if available, otherwise fall back to DenseGNN.
    Expects dataset exported by ReactionKnowledgeBaseV2.export_ml_dataset.
    """
    if USE_TORCH and USE_PYG:
        data_list = build_pyg_dataset_from_jsonl(jsonl_path)
        if not data_list:
            logger.error("No data for training.")
            return None
        loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=True)
        model = BondPredictorGNN(node_in_dim=6, hidden_dim=GNN_HIDDEN_DIM, msg_passes=GNN_MESSAGE_PASSES)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCELoss()
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                # assume batch.edge_index and batch.edge_attr represent candidate edges with labels in batch.edge_label (if present)
                scores = model(batch)
                # if batch has labels
                if hasattr(batch, "edge_label") and batch.edge_label is not None:
                    labels = batch.edge_label.float().to(device)
                    loss = loss_fn(scores, labels)
                    loss.backward()
                    opt.step()
                    total_loss += loss.item()
            logger.info(f"[BondTrain] Epoch {ep+1}/{epochs} Loss: {total_loss:.4f}")
        if save_path:
            save_model(model, save_path)
        return model

    elif USE_TORCH:
        # Build a synthetic dataset from JSONL using node features and candidate pairs
        samples = load_jsonl_samples(jsonl_path)
        # Create training arrays
        pair_feats = []
        labels = []
        for s in samples:
            nodes = s.get("nodes", [])
            n = len(nodes)
            for i in range(n):
                for j in range(i+1, n):
                    ni = nodes[i]; nj = nodes[j]
                    feat = [ni.get("mass",0.0), nj.get("mass",0.0), ni.get("en",0.0), nj.get("en",0.0), np.linalg.norm(np.array(ni.get("pos",[0,0])) - np.array(nj.get("pos",[0,0])))]
                    pair_feats.append(feat)
                    # label heuristics: if there is an edge between i,j in s.edges -> label 1 else 0
                    exists = False
                    for e in s.get("edges", []):
                        if (e[0] == i and e[1] == j) or (e[0] == j and e[1] == i):
                            exists = True; break
                    labels.append(1 if exists else 0)
        X = torch.tensor(np.array(pair_feats), dtype=torch.float)
        y = torch.tensor(np.array(labels), dtype=torch.float).view(-1,1)
        model = DenseGNN(node_in_dim=5, hidden_dim=GNN_HIDDEN_DIM)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.BCELoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                # split into node-like chunks for DenseGNN: here we fake node embeddings by projecting pairs
                # NOTE: DenseGNN expects node features; we'll project pair features into node_emb dim
                node_emb = model.node_mlp(xb)  # misuse: treat pair as node embedding
                scores = model.edge_mlp(torch.cat([node_emb, torch.zeros_like(node_emb[:, :1])], dim=1)).view(-1,1)
                loss = loss_fn(scores, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            logger.info(f"[DenseBondTrain] Epoch {ep+1}/{epochs} Loss: {total_loss:.4f}")
        if save_path:
            save_model(model, save_path)
        return model
    else:
        logger.error("No torch available to train GNN. Skipping training.")
        return None

def train_reaction_predictor_from_jsonl(jsonl_path: str, epochs: int = TRAIN_EPOCHS, batch_size: int = TRAIN_BATCH_SIZE, save_path: Optional[str] = None):
    """
    Train a graph-level reaction classifier using PyG or fallback DenseGNN.
    """
    if USE_TORCH and USE_PYG:
        data_list = build_pyg_dataset_from_jsonl(jsonl_path)
        if not data_list:
            logger.error("No data for reaction training.")
            return None
        # assign synthetic labels if missing (here event_type mapping)
        # For now, just reuse BondPredictorGNN pipeline structure
        loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=True)
        model = ReactionPredictorGNN(node_in_dim=6, hidden_dim=GNN_HIDDEN_DIM, msg_passes=GNN_MESSAGE_PASSES, out_classes=8)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for batch in loader:
                batch = batch.to(device)
                opt.zero_grad()
                out = model(batch)
                # synthetic target zeros for structure
                target = torch.zeros((out.shape[0],), dtype=torch.long).to(device)
                loss = loss_fn(out, target)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            logger.info(f"[ReactTrain] Epoch {ep+1}/{epochs} Loss: {total_loss:.4f}")
        if save_path:
            save_model(model, save_path)
        return model

    elif USE_TORCH:
        # Fallback simple MLP on pooled node features
        samples = load_jsonl_samples(jsonl_path)
        X = []
        y = []
        for s in samples:
            nodes = s.get("nodes", [])
            node_feats = np.array([[n.get("mass",0.0), n.get("en",0.0), n.get("charge",0.0)] for n in nodes])
            if node_feats.size == 0:
                pooled = np.zeros(6)
            else:
                pooled = node_feats.mean(axis=0)
            X.append(pooled)
            y.append(0)
        X = torch.tensor(np.array(X), dtype=torch.float)
        y = torch.tensor(np.array(y), dtype=torch.long)
        model = DenseGNN(node_in_dim=X.shape[1], hidden_dim=GNN_HIDDEN_DIM)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        opt = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        loss_fn = nn.CrossEntropyLoss()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for ep in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(device); yb = yb.to(device)
                opt.zero_grad()
                node_emb = model.node_mlp(xb)
                out = model.global_mlp(node_emb)
                loss = loss_fn(out, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            logger.info(f"[DenseReactTrain] Epoch {ep+1}/{epochs} Loss: {total_loss:.4f}")
        if save_path:
            save_model(model, save_path)
        return model
    else:
        logger.error("No torch available to train reaction predictor.")
        return None

# -----------------------
# Evaluation Metrics
# -----------------------

def evaluate_bond_predictor(model: Any, jsonl_path: str, batch_size: int = 256):
    """
    Evaluate bond predictor over dataset and print simple metrics (precision/recall).
    Works with PyG model or DenseGNN / numpy pseudo.
    """
    samples = load_jsonl_samples(jsonl_path)
    y_true = []
    y_pred = []
    for s in samples:
        nodes = s.get("nodes", [])
        n = len(nodes)
        # build candidate pairs
        for i in range(n):
            for j in range(i+1, n):
                exists = any((e[0]==i and e[1]==j) or (e[0]==j and e[1]==i) for e in s.get("edges", []))
                y_true.append(1 if exists else 0)
                # predict
                if USE_TORCH and USE_PYG and isinstance(model, BondPredictorGNN):
                    # skipped for brevity: full evaluation requires building batch PyG Data
                    y_pred.append(0)
                elif USE_TORCH and isinstance(model, DenseGNN):
                    feat = [nodes[i].get("mass",0.0), nodes[j].get("mass",0.0), nodes[i].get("en",0.0), nodes[j].get("en",0.0), np.linalg.norm(np.array(nodes[i].get("pos",[0,0]))-np.array(nodes[j].get("pos",[0,0])))]
                    x = torch.tensor([feat], dtype=torch.float)
                    with torch.no_grad():
                        node_emb = model.node_mlp(x)
                        score = model.edge_mlp(torch.cat([node_emb, torch.zeros_like(node_emb[:, :1])], dim=1)).item()
                    y_pred.append(1 if score > 0.5 else 0)
                else:
                    # numpy heuristic
                    npg = NumpyPseudoGNN()
                    score = npg.predict_edge_scores(nodes, [(i,j)])[0]
                    y_pred.append(1 if score > 0.5 else 0)
    # compute metrics
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt==0 and yp==1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt==1 and yp==0)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    logger.info(f"[Evaluate] Precision: {precision:.3f} Recall: {recall:.3f} TP:{tp} FP:{fp} FN:{fn}")

# -----------------------
# Visualization niceties
# -----------------------

def save_frame_png(sim: SimulationManager, filename: Optional[str] = None):
    if filename is None:
        filename = os.path.join(OUTPUT_DIR, f"frame_{now_str()}.png")
    if sim.fig is None:
        render_simulation_frame(sim)
    sim.fig.savefig(filename, dpi=200, bbox_inches='tight')
    logger.info(f"Saved frame to {filename}")


def export_simulation_to_plotly_html(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """
    Export the simulation history as an interactive Plotly HTML animation.
    This builds atom positions from their recorded `.history` and overlays bond
    events logged in `sim.kb.events` to draw bonds as they are formed/broken.
    The output is a self-contained HTML file that can be shared or hosted.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("plotly is not installed. Install with `pip install plotly` to use HTML export.")

    # Detect atom-like units and histories (support 'atoms' or 'units' field)
    units = None
    if hasattr(sim, 'atoms') and sim.atoms:
        units = sim.atoms
    elif hasattr(sim, 'units') and sim.units:
        units = sim.units
    if units is None or not any(getattr(u, 'history', None) for u in units):
        raise RuntimeError("No recorded atom/unit histories. Run simulation with .run_steps() or record frames first.")
        # compute number of frames available
    min_hist = min(len(u.history) for u in units)
    if n_frames is None:
        n_frames = min_hist
    n_frames = min(n_frames, min_hist)

    # parse bond lifecycle from KB events: map (uid1,uid2) -> (formed_frame, broken_frame)
    bond_lifecycles = {}
    for ev in getattr(sim.kb, 'events', []):
        if ev.get('event_type') == 'bond_formed':
            b = ev.get('bonds', [])
            for pair in b:
                uid1, uid2, _ = pair
                key = tuple(sorted((uid1, uid2)))
                # record earliest formation
                if key not in bond_lifecycles:
                    bond_lifecycles[key] = [ev.get('frame', 0), None]
        elif ev.get('event_type') == 'bond_broken':
            b = ev.get('bonds', [])
            for pair in b:
                uid1, uid2, _ = pair
                key = tuple(sorted((uid1, uid2)))
                if key in bond_lifecycles:
                    bond_lifecycles[key][1] = ev.get('frame', sim.frame)

    # Build frames
    frames = []
    # Obtain atom metadata in a generic way
    atom_symbols = [getattr(u, 'symbol', getattr(u, 'element', '?')) for u in units]
    atom_colors = [getattr(u, 'color', '#888888') for u in units]
    atom_sizes = [max(6, int(atom_radius_visual(u) * 2000)) if hasattr(u, 'radius') else 10 for u in units]

    uid_to_index = {getattr(u,'uid', getattr(u,'id', f'u{idx}')): idx for idx, u in enumerate(units)}

    for f in range(n_frames):
        # atom positions
        # extract positions from history or position attribute
        xs = [u.history[f]['pos'][0] if getattr(u, 'history', None) and len(u.history) > f else (getattr(u, 'position', getattr(u, 'pos', [0,0]))[0]) for u in units]
        ys = [u.history[f]['pos'][1] if getattr(u, 'history', None) and len(u.history) > f else (getattr(u, 'position', getattr(u, 'pos', [0,0]))[1]) for u in units]
        atom_trace = go.Scatter(x=xs, y=ys, mode='markers+text', text=atom_symbols,
                                marker=dict(size=atom_sizes, color=atom_colors, line=dict(width=1, color='black')),
                                textposition='bottom center', hoverinfo='text')

        # bonds present in this frame
        bond_x = []
        bond_y = []
        for (uid1, uid2), (formed, broken) in bond_lifecycles.items():
            if formed is None:
                continue
            if f < formed:
                continue
            if broken is not None and f >= broken:
                continue
            # include bond
            i1 = uid_to_index.get(uid1); i2 = uid_to_index.get(uid2)
            if i1 is None or i2 is None:
                continue
            u1 = units[i1]; u2 = units[i2]
            x0 = u1.history[f]['pos'][0] if getattr(u1, 'history', None) and len(u1.history) > f else (getattr(u1, 'position', getattr(u1, 'pos', [0,0]))[0])
            y0 = u1.history[f]['pos'][1] if getattr(u1, 'history', None) and len(u1.history) > f else (getattr(u1, 'position', getattr(u1, 'pos', [0,0]))[1])
            x1 = u2.history[f]['pos'][0] if getattr(u2, 'history', None) and len(u2.history) > f else (getattr(u2, 'position', getattr(u2, 'pos', [0,0]))[0])
            y1 = u2.history[f]['pos'][1] if getattr(u2, 'history', None) and len(u2.history) > f else (getattr(u2, 'position', getattr(u2, 'pos', [0,0]))[1])
            bond_x.extend([x0, x1, None])
            bond_y.extend([y0, y1, None])

        bond_trace = go.Scatter(x=bond_x, y=bond_y, mode='lines', line=dict(color='#888888', width=2), hoverinfo='none')

        frames.append(go.Frame(data=[atom_trace, bond_trace], name=str(f), traces=[0,1]))

    # initial traces from frame 0
    fig = go.Figure(data=[frames[0].data[0], frames[0].data[1]], frames=frames)

    fig.update_layout(title=f"Simulation export ({n_frames} frames)", showlegend=False,
                      xaxis=dict(range=[0,1], showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(range=[0,1], showgrid=False, zeroline=False, visible=False),
                      width=800, height=800,
                      updatemenus=[dict(type='buttons', showactive=False,
                                         y=1, x=1.12, xanchor='right', yanchor='top',
                                         pad=dict(t=0, r=10),
                                         buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': int(1000/fps), 'redraw': True}, 'fromcurrent': True}]),
                                                  dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0}, 'mode': 'immediate', 'transition': {'duration': 0}}])])])

    # slider
    sliders = [dict(steps=[dict(method='animate', args=[[fr.name], {'mode': 'immediate', 'frame': {'duration': int(1000/fps), 'redraw': True}, 'transition': {'duration': 0}}], label=str(i)) for i, fr in enumerate(frames)], active=0, x=0, y=0, len=1.0)]
    fig.update_layout(sliders=sliders)

    # write to file
    fig.write_html(filename, include_plotlyjs='cdn')
    logger.info(f"Exported interactive HTML to {filename}")


def _render_frames_to_images(sim: SimulationManager, n_frames: int, folder: str):
    """Render n_frames from sim history and save PNG frames into folder."""
    ensure_dir(folder)
    # determine frames: use length of atoms' history
    if not sim.atoms or not getattr(sim.atoms[0], 'history', None):
        raise RuntimeError("No atom history available for rendering frames.")
    total = len(sim.atoms[0].history)
    n = min(n_frames, total)
    # store original positions to restore
    orig_positions = [a.pos.copy() for a in sim.atoms]
    for i in range(n):
        # set positions for frame i
        for a in sim.atoms:
            try:
                a.pos = np.array(a.history[i]['pos'])
            except Exception:
                a.pos = a.pos
        # render
        render_simulation_frame(sim)
        out_path = os.path.join(folder, f"frame_{i:04d}.png")
        sim.fig.savefig(out_path, dpi=150, bbox_inches='tight')
    # restore positions
    for a, p in zip(sim.atoms, orig_positions):
        a.pos = p
    return n


def export_simulation_to_gif(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """Export the simulation to animated GIF using available writer (imageio or pillow)."""
    if not getattr(sim, 'atoms', None) or not sim.atoms:
        raise RuntimeError("No atoms present in simulation for export.")
    if n_frames is None:
        n_frames = min(len(sim.atoms[0].history) if sim.atoms and sim.atoms[0].history else sim.frame, sim.frame)
    import tempfile
    import imageio
    tmpdir = tempfile.mkdtemp(prefix='sim_frames_')
    try:
        n = _render_frames_to_images(sim, n_frames, tmpdir)
        images = []
        for i in range(n):
            im = imageio.imread(os.path.join(tmpdir, f"frame_{i:04d}.png"))
            images.append(im)
        imageio.mimsave(filename, images, fps=fps)
        logger.info(f"Saved GIF to {filename}")
    except Exception:
        # fallback: try pillow
        try:
            from PIL import Image
            frames = []
            for i in range(n):
                frames.append(Image.open(os.path.join(tmpdir, f"frame_{i:04d}.png")))
            frames[0].save(filename, save_all=True, append_images=frames[1:], duration=int(1000/fps), loop=0)
            logger.info(f"Saved GIF to {filename} (Pillow)")
        except Exception:
            logger.exception("Failed to export GIF; ensure imageio or pillow is installed.")
            raise
    finally:
        # cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_simulation_to_mp4(sim: SimulationManager, filename: str, n_frames: Optional[int] = None, fps: int = 10):
    """Export the simulation to MP4 using ffmpeg (via imageio-ffmpeg) if available."""
    if not getattr(sim, 'atoms', None) or not sim.atoms:
        raise RuntimeError("No atoms present in simulation for export.")
    if n_frames is None:
        n_frames = min(len(sim.atoms[0].history) if sim.atoms and sim.atoms[0].history else sim.frame, sim.frame)
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix='sim_frames_')
    try:
        n = _render_frames_to_images(sim, n_frames, tmpdir)
        # try use imageio to write mp4
        try:
            import imageio
            writer = imageio.get_writer(filename, fps=fps, codec='libx264')
            for i in range(n):
                img = imageio.imread(os.path.join(tmpdir, f"frame_{i:04d}.png"))
                writer.append_data(img)
            writer.close()
        except Exception:
            # fallback to ffmpeg via subprocess if installed
            try:
                import subprocess
                cmd = [
                    'ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(tmpdir, 'frame_%04d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', filename
                ]
                subprocess.check_call(cmd)
            except Exception:
                logger.exception('MP4 export failed; ensure imageio or ffmpeg is installed.')
                raise
        logger.info(f"Saved MP4 to {filename}")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def publish_directory_to_github(dir_path: str, repo_url: str, branch: str = 'gh-pages', token: Optional[str] = None, commit_message: str = 'Publish simulation'):
    """Publish a directory to GitHub Pages by creating a temporary repo, committing dir content, and pushing to `branch` in `repo_url`.
    If token is provided, it will be embedded into the remote URL (insecure). Ensure the token has repo:public_repo scope or repo scope for private repos.
    """
    import tempfile, subprocess, shutil
    tmpdir = tempfile.mkdtemp(prefix='publish_repo_')
    try:
        # copy dir content into tmpdir
        shutil.copytree(dir_path, os.path.join(tmpdir, os.path.basename(dir_path)))
    except Exception:
        # maybe dir_path is already a folder; copy files
        for root, dirs, files in os.walk(dir_path):
            rel = os.path.relpath(root, dir_path)
            dest = os.path.join(tmpdir, rel)
            os.makedirs(dest, exist_ok=True)
            for f in files:
                shutil.copyfile(os.path.join(root, f), os.path.join(dest, f))
    try:
        # init git repo
        subprocess.check_call(['git', 'init'], cwd=tmpdir)
        subprocess.check_call(['git', 'add', '.'], cwd=tmpdir)
        subprocess.check_call(['git', 'commit', '-m', commit_message], cwd=tmpdir)
        # ensure a deterministic local branch name (master)
        try:
            subprocess.check_call(['git', 'branch', '-M', 'master'], cwd=tmpdir)
        except Exception:
            pass
        # build remote url with token if provided
        remote = repo_url
        if token:
            # Insert token into URL, support https://github.com/owner/repo.git
            if remote.startswith('https://'):
                remote = remote.replace('https://', f'https://{token}@')
        subprocess.check_call(['git', 'remote', 'add', 'origin', remote], cwd=tmpdir)
        subprocess.check_call(['git', 'push', '-f', 'origin', f'master:{branch}'], cwd=tmpdir)
    except subprocess.CalledProcessError as e:
        logger.exception('Failed to publish to GitHub Pages')
        raise
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def export_product_timeline(sim: SimulationManager, comp_key: str, out_path: str):
    """Export timeline events from KB for the given component as JSON.
    This collects all KB events where any atom in the component is involved.
    """
    comps = sim.detect_products()
    if comp_key not in comps:
        raise KeyError(f"Component {comp_key} not found")
    atoms, _ = comps[comp_key]
    uids = {a.uid for a in atoms}
    events = [ev for ev in getattr(sim.kb, 'events', []) if any(a.get('uid') in uids for a in ev.get('atoms', []))]
    safe_save_json(events, out_path)
    logger.info(f"Saved product timeline for {comp_key} to {out_path}")


def generate_reaction_story(sim: SimulationManager, comp_key: str, products: dict):
    """Generate a short textual 'story' describing bond events that built the given product.
    Returns a string summary.
    """
    comps = products
    if comp_key not in comps:
        raise KeyError(f"Component {comp_key} not found")
    atoms, bonds = comps[comp_key]
    # get events
    uids = {a.uid for a in atoms}
    events = [ev for ev in getattr(sim.kb, 'events', []) if any(a.get('uid') in uids for a in ev.get('atoms', []))]
    # build sentences
    lines = []
    for ev in sorted(events, key=lambda e: e.get('frame', 0)):
        frame = ev.get('frame', 0)
        typ = ev.get('event_type', '')
        sym = ",".join(sorted({a.get('symbol') for a in ev.get('atoms', [])}))
        if typ == 'bond_formed':
            lines.append(f"At frame {frame}, a bond formed between {sym}.")
        elif typ == 'bond_broken':
            lines.append(f"At frame {frame}, a bond broke between {sym}.")
    if not lines:
        lines = ["No events recorded for this component."]
    story = "\n".join(lines[:100])
    return story

def annotate_products(sim: SimulationManager):
    """
    Detect connected components and annotate the visualization with product boxes/labels.
    """
    comps = extract_connected_components(sim.atoms, sim.bonds)
    for idx, (atoms, bonds) in enumerate(comps):
        # compute centroid
        pts = np.array([a.pos for a in atoms])
        centroid = pts.mean(axis=0)
        formula = graph_to_formula(atoms, bonds)
        label = "".join([f"{k}{v if v>1 else ''}" for k,v in sorted(formula.items())])
        # draw a small rounded rectangle
        txt = sim.ax.text(centroid[0], centroid[1]+0.08, label, fontsize=9, color="white", ha="center", va="bottom", bbox=dict(facecolor="#222222", alpha=0.7, boxstyle="round,pad=0.2"))
        sim.artist_atoms.append(txt)
def compute_reactivity_heatmap(sim: SimulationManager, grid_size: int = 80, radius: float = 0.2):
    """
    Compute a reactivity heatmap over the 2D canvas. For each grid cell, we sum the
    bond formation scores from nearby pairs (or deterministic rule engine) so that hotspots
    indicate where bonds would be most likely to form.
    """
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    heat = np.zeros_like(xx)
    # Precompute pair midpoints and scores
    pairs = []
    for i, a in enumerate(sim.atoms):
        for j in range(i+1, len(sim.atoms)):
            b = sim.atoms[j]
            midpoint = (a.pos + b.pos) / 2.0
            # Determine score: use deterministic rule engine if available, else probabilistic
            score = 0.0
            if getattr(sim, 'deterministic_mode', False) and sim.reaction_engine and getattr(sim.reaction_engine, 'rule_engine', None):
                should, s = sim.reaction_engine.rule_engine.should_form_bond(a, b, sim.physics)
                score = float(s if should else 0.0)
            else:
                can, s = can_form_bond(a, b, sim.temperature)
                score = float(s if can else 0.0)
            pairs.append((midpoint, score))

    # For each pair, spread contribution to nearby grid cells using gaussian falloff
    for mid, score in pairs:
        if score <= 0.0:
            continue
        dx = xx - mid[0]
        dy = yy - mid[1]
        dist2 = dx*dx + dy*dy
        # gaussian with sigma related to radius
        sigma = radius * 0.5
        contrib = score * np.exp(-dist2 / (2 * sigma * sigma + 1e-12))
        heat += contrib
    # normalize
    maxv = heat.max()
    if maxv > 0:
        heat /= maxv
    return xx, yy, heat

def plot_reactivity_heatmap(sim: SimulationManager, grid_size: int = 80, radius: float = 0.2, cmap: str = 'inferno'):
    try:
        import matplotlib.pyplot as plt
        xx, yy, heat = compute_reactivity_heatmap(sim, grid_size=grid_size, radius=radius)
        fig, ax = plt.subplots(figsize=(8,8))
        ax.imshow(heat, extent=[0,1,0,1], origin='lower', cmap=cmap, alpha=0.8)
        ax.set_title('Reaction potential heatmap')
        ax.set_xticks([])
        ax.set_yticks([])
        # overlay atom positions
        xs = [a.pos[0] for a in sim.atoms]
        ys = [a.pos[1] for a in sim.atoms]
        ax.scatter(xs, ys, c='white', s=40, edgecolors='black')
        plt.show()
    except Exception:
        logger.exception("Failed to plot heatmap")
# -----------------------
# CHUNK 4/10
# GUI Panel (Tkinter), Advanced Visual Effects, Simulation ↔ Model Integration, 
# Product Merging Utilities, and Structural Safety Tools
# -----------------------

# -----------------------
# Tkinter GUI Panel
# -----------------------

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    tk = None
    ttk = None
    logger.warning("Tkinter not available; GUI panel disabled.")

class ControlPanel:
    """
    A small floating GUI panel allowing runtime changes:
    - Temperature slider
    - Bond enable/disable
    - Reset simulation
    - Save frame
    """
    def __init__(self, sim: SimulationManager):
        self.sim = sim
        self.root = None
        if tk is not None:
            self.build_panel()
        else:
            logger.info("ControlPanel not built because Tkinter is missing.")

    def build_panel(self):
        self.root = tk.Tk()
        self.root.title("Reaction Simulation Controls")
        self.root.geometry("320x220")

        # Temperature slider
        ttk.Label(self.root, text="Temperature").pack(pady=4)
        self.temp_var = tk.DoubleVar(value=self.sim.temperature)
        temp_slider = ttk.Scale(self.root, from_=0.0, to=2000.0,
                                orient=tk.HORIZONTAL,
                                variable=self.temp_var,
                                command=self.on_temp_change)
        temp_slider.pack(fill="x", padx=10)

        # Bonding toggle
        self.bond_var = tk.BooleanVar(value=self.sim.enable_bonds)
        bond_chk = ttk.Checkbutton(self.root, text="Enable Bonding",
                                   variable=self.bond_var,
                                   command=self.on_bond_toggle)
        bond_chk.pack(pady=5)

        # Reset button
        reset_btn = ttk.Button(self.root, text="Reset Simulation",
                               command=self.on_reset)
        reset_btn.pack(pady=5)

        # Save frame button
        save_btn = ttk.Button(self.root, text="Save Frame",
                              command=self.on_save_frame)
        save_btn.pack(pady=5)

        # Deterministic toggle
        self.det_var = tk.BooleanVar(value=getattr(self.sim, 'deterministic_mode', False))
        det_chk = ttk.Checkbutton(self.root, text="Deterministic Reaction",
                       variable=self.det_var,
                       command=self.on_det_toggle)
        det_chk.pack(pady=5)

        # Predict reaction button
        pred_btn = ttk.Button(self.root, text="Predict Reaction", command=self.on_predict)
        pred_btn.pack(pady=5)

        # Start GUI in a separate thread in start_gui()

    def on_temp_change(self, event=None):
        try:
            val = float(self.temp_var.get())
            self.sim.set_temperature(val)
        except:
            pass

    def on_bond_toggle(self):
        self.sim.set_enable_bonds(self.bond_var.get())

    def on_reset(self):
        self.sim.reset_simulation()

    def on_save_frame(self):
        save_frame_png(self.sim)

    def on_det_toggle(self):
        try:
            val = bool(self.det_var.get())
            self.sim.deterministic_mode = val
            if hasattr(self.sim.reaction_engine, 'deterministic'):
                self.sim.reaction_engine.deterministic = val
        except Exception:
            pass

    def on_predict(self):
        try:
            # Run a deterministic prediction pass: use the rule engine to form bonds
            if hasattr(self.sim, 'reaction_engine') and hasattr(self.sim.reaction_engine, 'rule_engine'):
                self.sim.reaction_engine.deterministic = True
                self.sim.reaction_engine.rule_engine = self.sim.reaction_engine.rule_engine or ReactionRuleEngine()
                # call a single reaction pass
                self.sim.reaction_engine.step(self.sim.frame)
                render_simulation_frame(self.sim)
                # also show heatmap
                try:
                    plot_reactivity_heatmap(self.sim, grid_size=80)
                except Exception:
                    pass
                try:
                    comps = extract_connected_components(self.sim.atoms, self.sim.bonds)
                    label = "\n".join([" + ".join(["".join([f"{k}{v if v>1 else ''}" for k,v in sorted(graph_to_formula(atoms, bonds).items())])]) for atoms,bonds in comps])
                    # Use messagebox from tkinter
                    try:
                        from tkinter import messagebox
                        messagebox.showinfo('Predicted Products', label if label else 'No products predicted')
                    except Exception:
                        print('Predicted Products:\n', label)
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Prediction error in GUI")

    def start(self):
        """
        Start the GUI main loop on another thread.
        """
        if self.root is None:
            return
        t = threading.Thread(target=self.root.mainloop, daemon=True)
        t.start()

# -----------------------
# Advanced Visual Effects
# -----------------------

def draw_energy_field(sim: SimulationManager):
    """
    Overlay a "fog" field showing potential energy or random noise
    for better aesthetic appeal.
    """
    if sim.ax is None:
        return
    # Generate a soft background gradient field
    gx = np.linspace(-1, 1, 200)
    gy = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X*X + Y*Y)
    field = np.exp(-4 * R)  # radial fade
    sim.ax.imshow(field, extent=[-1,1,-1,1], origin="lower",
                  cmap="inferno", alpha=0.12, interpolation="bilinear")

def highlight_bonds(sim: SimulationManager):
    """
    Draw a short-lived glow on newly formed bonds.
    """
    now = time.time()
    for b in list(sim.bonds):
        dt = now - getattr(b, 'time_of_creation', 0.0)
        if dt < 0.4:
            # A glow radius that shrinks over time
            glow = max(0.0, 0.4 - dt) * 10
            sim.ax.plot([b.atom1.pos[0], b.atom2.pos[0]],
                        [b.atom1.pos[1], b.atom2.pos[1]],
                        linewidth=glow, color="#55aaff", alpha=0.1)

def activate_ghost_trails(sim: SimulationManager, length: int = 20):
    """
    Track and visualize motion trails for each atom.
    """
    if not hasattr(sim, "_trail_buffer"):
        sim._trail_buffer = defaultdict(lambda: deque(maxlen=length))
    for a in sim.atoms:
        sim._trail_buffer[a.uid].append(tuple(a.pos))

def draw_ghost_trails(sim: SimulationManager):
    """
    Draw faint trails following atomic motion.
    """
    if not hasattr(sim, "_trail_buffer"):
        return
    for aid, trail in sim._trail_buffer.items():
        if len(trail) > 1:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            sim.ax.plot(xs, ys, color="white", alpha=0.15, linewidth=1)

# Minimal stub replacements for missing functions
def update_visual(sim: SimulationManager):
    """Update the main simulation visualization (render frame and annotate products)."""
    try:
        render_simulation_frame(sim)
        try:
            annotate_products(sim)
        except Exception:
            pass
    except Exception:
        logger.exception("update_visual failed")

# Integrate these effects into update_visual_with_effects
def update_visual_with_effects(sim: SimulationManager):
    """
    Wrapper around update_visual() that adds fancy visual touches.
    """
    update_visual(sim)
    draw_energy_field(sim)
    highlight_bonds(sim)
    activate_ghost_trails(sim)
    draw_ghost_trails(sim)

def merge_predicted_products(sim: SimulationManager, reactants_products: Dict, model_pred: Optional[Any] = None):
    """
    Takes the result of SimulationManager's detect_products() and optionally
    merges with a ML model prediction.
    """
    merged = {}
    # Step 1: Simulation detection
    for label, items in reactants_products.items():
        merged[label] = items

    # Step 2: ML Prediction
    if model_pred is not None:
        try:
            ml = model_pred.get("ml_products", {})
            for lbl, items in ml.items():
                merged[lbl] = items
        except Exception:
            logger.exception("Failed merging ML predictions.")

    return merged

def products_to_string(product_dict: Dict[str, Any]) -> str:
    """
    Convert merged product information to a readable string.
    """
    lines = []
    for name, comp in product_dict.items():
        atoms, bonds = comp
        formula = graph_to_formula(atoms, bonds)
        fstr = "".join([f"{k}{v if v>1 else ''}" for k,v in sorted(formula.items())])
        lines.append(f"{name}: {fstr}")
    return "\n".join(lines)

# -----------------------
# Reaction Diagram Generation
# -----------------------

def reaction_to_text_diagram(reactants, products):
    """
    Converts reactants and products into a simple ASCII diagram.
    """
    r_strs = []
    p_strs = []
    for atoms, bonds in reactants:
        f = graph_to_formula(atoms, bonds)
        s = "".join([f"{k}{v if v>1 else ''}" for k, v in sorted(f.items())])
        r_strs.append(s)
    for atoms, bonds in products:
        f = graph_to_formula(atoms, bonds)
        s = "".join([f"{k}{v if v>1 else ''}" for k, v in sorted(f.items())])
        p_strs.append(s)
    return " + ".join(r_strs) + "  →  " + " + ".join(p_strs)


def sanitize_graph(atoms, bonds):
    """
    Clean impossible or unstable structures (e.g., too many bonds on one atom).
    """
    max_neigh = {
        "H": 1, "C": 4, "N": 3, "O": 2, "Cl": 1, "F": 1, "Br": 1
    }
    neigh_count = {}
    for b in list(bonds):
        neigh_count[b.atom1.uid] = neigh_count.get(b.atom1.uid, 0) + 1
        neigh_count[b.atom2.uid] = neigh_count.get(b.atom2.uid, 0) + 1
    for b in list(bonds):
        if neigh_count.get(b.atom1.uid, 0) > max_neigh.get(b.atom1.symbol, 8) or \
           neigh_count.get(b.atom2.uid, 0) > max_neigh.get(b.atom2.symbol, 8):
            # remove the problematic bond
            try:
                bonds.remove(b)
            except ValueError:
                pass
            neigh_count[b.atom1.uid] = max(0, neigh_count.get(b.atom1.uid, 0) - 1)
            neigh_count[b.atom2.uid] = max(0, neigh_count.get(b.atom2.uid, 0) - 1)


def sanitize_simulation(sim):
    """
    Applies structural safety constraints to the live simulation.
    """
    sanitize_graph(sim.atoms, sim.bonds)


class ReactionMLRouter:
    """
    Connects a trained GNN reaction predictor to the SimulationManager.
    """
    def __init__(self, model=None):
        self.model = model

    def predict_products(self, atoms, bonds):
        """
        Produce ML predictions from the graph.
        """
        if self.model is None:
            return {"ml_products": {}}

        if USE_TORCH:
            node_feats = [[a.mass, electronegativity(a.symbol), a.charge, a.pos[0], a.pos[1], 1.0] for a in atoms]
            x = torch.tensor(node_feats, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                try:
                    logits = self.model.global_mlp(self.model.node_mlp(x))
                    pred = torch.argmax(logits, dim=1).item()
                except:
                    pred = None
        else:
            pred = None

        if pred == 0:
            return {"ml_products": {"predicted_product": (atoms, bonds)}}
        else:
            return {"ml_products": {}}

# -----------------------
# End of CHUNK 4/10
# -----------------------
# -----------------------
# CHUNK 5/10
# High-Level Orchestrator, Reaction Pipeline Controller,
# Predictive Synthesis Engine, Runtime Analysis,
# Logging hooks, and Simulation ↔ Prediction alignment.
# -----------------------

# -----------------------
# High-Level Reaction Orchestrator
# -----------------------

class ReactionOrchestrator:
    """
    Coordinates:
    - SimulationManager (physics engine)
    - ReactionMLRouter (model predictions)
    - ReactionKnowledgeBaseV2 (history & examples)
    - Visualization (auto-capture, auto-diagram)
    - ControlPanel (if GUI enabled)
    """
    def __init__(self,
                 sim: SimulationManager,
                 ml_router: Optional[ReactionMLRouter] = None,
                 kb: Optional[ReactionKnowledgeBaseV2] = None,
                 gui: bool = True):
        self.sim = sim
        self.ml_router = ml_router
        self.kb = kb
        self.gui_enabled = gui
        self.gui_panel = None
        if gui and tk is not None:
            self.gui_panel = ControlPanel(sim)

        self.auto_capture_interval = 1.5  # minutes
        self.last_capture = time.time()
        self.auto_diagrams = True

        self._stop_flag = False
        self._thread = None

    def start(self):
        self._stop_flag = False
        self.sim.start()
        if self.gui_panel:
            self.gui_panel.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        self.sim.stop()

    def _loop(self):
        """
        Orchestrator main loop: monitors simulation, predicts products,
        auto-saves, stores reaction events in the knowledge base.
        """
        while not self._stop_flag:
            try:
                # 1. Sanitize
                sanitize_simulation(self.sim)

                # 2. Detect simulation products
                sim_products_raw = self.sim.detect_products()
                comps = extract_connected_components(self.sim.atoms, self.sim.bonds)
                reactants = comps if len(comps) > 1 else comps[:1]
                products = [comps] if len(comps) == 1 else comps

                # 3. ML prediction
                ml_pred = None
                if self.ml_router:
                    ml_pred = self.ml_router.predict_products(self.sim.atoms, self.sim.bonds)

                # 4. Merge
                merged = merge_predicted_products(self.sim, sim_products_raw, ml_pred)

                # 5. Save to KB
                if self.kb is not None:
                    event = {
                        "reactants": [graph_to_formula(a,b) for a,b in reactants],
                        "products": [graph_to_formula(a,b) for a,b in products],
                        "merged_products": {k:graph_to_formula(v[0],v[1]) for k,v in merged.items()},
                        "temperature": self.sim.temperature,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    try:
                        self.kb.add_event(event)
                    except Exception:
                        logger.exception("Failed to add orchestrator event to KB")

                # 6. Auto-capture
                if time.time() - self.last_capture > self.auto_capture_interval * 60:
                    fname = os.path.join(OUTPUT_DIR, f"auto_capture_{now_str()}.png")
                    save_frame_png(self.sim, fname)
                    self.last_capture = time.time()

                # 7. Auto reaction diagrams
                if self.auto_diagrams:
                    diag = reaction_to_text_diagram(reactants, products)
                    diag_path = os.path.join(OUTPUT_DIR, f"diagram_{now_str()}.txt")
                    with open(diag_path, "w") as fh:
                        fh.write(diag)

                time.sleep(0.15)
            except Exception:
                logger.exception("Error in orchestrator loop.")
                time.sleep(0.5)

# -----------------------
# Predictive Synthesis Engine
# -----------------------

class PredictiveSynthesisEngine:
    """
    Generates new reaction setups based on the knowledge base.
    Suggests new atom placement, starting temperature, expected behavior.
    """
    def __init__(self, kb: Optional[ReactionKnowledgeBaseV2]):
        self.kb = kb

    def propose_reaction(self) -> Dict:
        if not self.kb or not self.kb.events:
            return {
                "atoms": ["H", "H", "O"],
                "layout": "triangle",
                "temperature": 350
            }

        # simple heuristic: pick a past event with interesting products
        ev = random.choice(self.kb.events)
        prods = ev.get("products", [])
        guess = {"atoms": [], "layout": "spread", "temperature": ev.get("temperature", 300)}

        # suggest atoms
        for p in prods:
            for elem, count in p.items():
                for _ in range(count):
                    guess["atoms"].append(elem)

        # randomize slight modifications
        if random.random() < 0.4:
            guess["atoms"].append("H")

        return guess

    def generate_initial_atoms(self, suggestion: Dict) -> List[Atom]:
        atoms = []
        pos_mode = suggestion.get("layout", "spread")
        elems = suggestion.get("atoms", [])
        if pos_mode == "triangle":
            # place in triangle
            n = len(elems)
            for i, el in enumerate(elems):
                angle = 2 * np.pi * i / max(1,n)
                pos = [0.5*np.cos(angle), 0.5*np.sin(angle)]
                atoms.append(Atom(el, pos, np.zeros(2)))
        else:
            # scatter randomly
            for el in elems:
                pos = [random.uniform(-0.6,0.6), random.uniform(-0.6,0.6)]
                vel = [0.0,0.0]
                atoms.append(Atom(el, pos, vel))

        return atoms

# -----------------------
# Runtime Reaction Analyzer
# -----------------------

class ReactionRuntimeAnalyzer:
    """
    Monitors live simulation reaction progress:
    - number of atoms
    - bond activity
    - potential formation events
    - energy trends
    """
    def __init__(self, sim: SimulationManager):
        self.sim = sim
        self.history = deque(maxlen=300)

    def update(self):
        # track number of bonds and kinetic energies
        total_E = 0.0
        for a in self.sim.atoms:
            m = a.mass
            total_E += 0.5 * m * np.dot(a.vel,a.vel)
        self.history.append({
            "time": time.time(),
            "bonds": len(self.sim.bonds),
            "kinE": total_E
        })

    def summarize(self) -> str:
        if not self.history:
            return "No data yet."
        avg_bonds = np.mean([h["bonds"] for h in self.history])
        avg_energy = np.mean([h["kinE"] for h in self.history])
        return f"Avg bonds={avg_bonds:.2f}, Avg kinE={avg_energy:.3f}"

# -----------------------
# Logging Hooks
# -----------------------

def attach_logging_to_sim(sim: SimulationManager):
    """
    Adds a simple hook that logs bond formation and breaking by wrapping sim.step
    and comparing bond sets before and after the step.
    """
    original_step = getattr(sim, 'step', None)
    if original_step is None:
        logger.warning("SimulationManager.step not found; cannot attach logging wrapper.")
        return

    def wrapped_step(*args, **kwargs):
        before = set(tuple(sorted((b.atom1.uid, b.atom2.uid))) for b in list(sim.bonds))
        original_step(*args, **kwargs)
        after = set(tuple(sorted((b.atom1.uid, b.atom2.uid))) for b in list(sim.bonds))
        added = after - before
        removed = before - after
        for uid1, uid2 in added:
            logger.info(f"Bond formed: {uid1}-{uid2}")
        for uid1, uid2 in removed:
            logger.info(f"Bond broken: {uid1}-{uid2}")

    sim.step = wrapped_step

# -----------------------
# Predictive Reaction Alignment
# -----------------------

class ReactionAlignmentEngine:
    """
    Aligns simulation results with ML predictions:
    - computes matching scores
    - flags surprising reactions
    - recommends retraining
    """
    def __init__(self):
        self.log = []

    def score_alignment(self, sim_products: Dict, ml_products: Dict) -> float:
        """
        Score: 1.0 = perfect match, 0.0 = mismatch
        """
        sim_formulas = {k:graph_to_formula(v[0],v[1]) for k,v in sim_products.items()}
        ml_formulas = {k:graph_to_formula(v[0],v[1]) for k,v in ml_products.items()} if ml_products else {}

        if not ml_formulas:
            return 0.5  # unknown

        # Compare sets
        score = 0.0
        for k in sim_formulas:
            if k in ml_formulas and sim_formulas[k] == ml_formulas[k]:
                score += 1.0
        total = max(1, len(sim_formulas))
        return score / total

    def log_event(self, score: float, details: Dict):
        entry = {
            "timestamp": now_str(),
            "score": score,
            "details": details
        }
        self.log.append(entry)

# -----------------------
# # End of CHUNK 5/10
# -----------------------
# -----------------------
# CHUNK 6/10
# Batch Manager, Config Loader, CLI Hooks, Multi-Run Simulation Controller, Data Export
# -----------------------

# -----------------------
# Configuration Loader
# -----------------------

class SimulationConfig:
    """
    Loads and validates simulation configuration from JSON/YAML.
    """
    def __init__(self, path="sim_config.json"):
        self.path = path
        self.config = {}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.config = json.load(f)
            except Exception:
                logger.warning(f"Failed to load config from {path}. Using defaults.")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def validate(self):
        # Basic validation
        if "molecules" not in self.config:
            self.config["molecules"] = []
        if "temperature" not in self.config:
            self.config["temperature"] = 300
        if "output_dir" not in self.config:
            self.config["output_dir"] = "outputs"
        return self.config

# -----------------------
# Multi-Run Simulation Manager
# -----------------------

class MultiRunSimulationManager:
    """
    Handles multiple independent simulation runs:
    - auto-generates seeds
    - auto-logs outputs
    - can run in parallel threads
    """
    def __init__(self, sim_cls, formula_lists: List[List[Dict]], kb: Optional[ReactionKnowledgeBaseV2] = None, max_threads=2):
        self.sim_cls = sim_cls
        self.formula_lists = formula_lists
        self.kb = kb
        self.max_threads = max_threads
        self.results = []

    def run_all(self):
        threads = []
        for idx, f_list in enumerate(self.formula_lists):
            while threading.active_count() > self.max_threads:
                time.sleep(0.1)
            t = threading.Thread(target=self._run_single, args=(idx, f_list))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    def _run_single(self, run_id, f_list):
        sim = self.sim_cls(f_list, reaction_kb=self.kb)
        sim.run(frames=800, interval=50)
        # Collect results
        run_output = {
            "run_id": run_id,
            "event_count": len(sim.event_log),
            "kb_pairs": len(self.kb.cooccurrence) if self.kb else 0,
            "log_file": f"simulation_run_{run_id}.json"
        }
        # Save log
        with open(run_output["log_file"], "w") as f:
            json.dump(sim.event_log, f, indent=2)
        self.results.append(run_output)

# -----------------------
# Data Export Utilities
# -----------------------


def export_simulation_summary(sim_results: List[Dict], csv_path="multi_run_summary.csv"):
    """
    Export a summary CSV for multiple simulation runs.
    """
    lines = ["run_id,event_count,kb_pairs,log_file"]
    for r in sim_results:
        lines.append(f"{r['run_id']},{r['event_count']},{r['kb_pairs']},{r['log_file']}")
    try:
        with open(csv_path, "w") as f:
            f.write("\n".join(lines))
        logger.info(f"Simulation summary saved to {csv_path}")
    except Exception:
        logger.exception(f"Failed to save CSV summary to {csv_path}")


def parse_formula(formula: str, element_data) -> Dict[str,int]:
    """
    Convert formula string (H2O) into element count dictionary.
    """
    import re
    matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    fdict = {}
    for elem, count in matches:
        # Keep original capitalization for lookup, but normalize key to uppercase in dicts
        elem_norm = elem.upper()
        if elem_norm not in element_data:
            # If element not defined, insert a fallback with minimal properties
            logger.warning(f"Element '{elem_norm}' not found in elements.json — creating a fallback default entry.")
            element_data[elem_norm] = {
                "name": elem,
                "symbol": elem_norm,
                "atomic_number": 0,
                "atomic_mass": 0.0,
                "electronegativity_pauling": 0.0,
                "group": 0,
                "period": 0,
                "cpk-hex": "808080",
                "category": "unknown",
                "covalent_radius": 0.7
            }
        fdict[elem_norm] = fdict.get(elem_norm, 0) + (int(count) if count else 1)
    return fdict


def run_simulation_cli(element_data):
    """
    Run a single simulation from CLI using command line arguments.
    Example: python main.py --formulas H2,O2,H2O --frames 1000 --interval 50
    """
    parser = argparse.ArgumentParser(description="Run molecular reaction simulation.")
    parser.add_argument("--formulas", type=str, required=True, help="Comma-separated molecule formulas")
    parser.add_argument("--frames", type=int, default=1200, help="Number of frames")
    parser.add_argument("--interval", type=int, default=50, help="Animation interval in ms")
    parser.add_argument("--output", type=str, default="simulation_events.json", help="Output JSON file")
    parser.add_argument("--export-html", type=str, default=None, help="Path to export interactive HTML animation")
    parser.add_argument("--export-frames", type=int, default=500, help="Number of frames to export to HTML")
    parser.add_argument("--export-fps", type=int, default=10, help="Frames per second for exported HTML animation")
    parser.add_argument("--elements", type=str, default=None, help="Path to custom elements JSON file (override)")
    parser.add_argument("--save-elements", type=str, default=None, help="Path to save merged elements.json (useful to persist custom additions)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic rule-based reaction engine (no random) for run")
    parser.add_argument("--temperature", type=float, default=None, help="Simulation temperature (overrides default)")
    parser.add_argument("--show-products", action="store_true", help="Print detected product formulas after simulation")
    parser.add_argument("--explain-products", action="store_true", help="Explain bond events that led to each detected product (prints timeline)")
    parser.add_argument("--add-element", type=str, action='append', default=[], help="Add a custom element via command-line in the format 'SYMBOL,atomic_mass,enadius,covalent_radius,cpk_hex' (repeatable)")
    parser.add_argument("--publish", action="store_true", help="Copy exported HTML into docs/ for publishing (GitHub Pages)")
    parser.add_argument("--show-events", action='store_true', help='Print KB event log summary (bond formed/broken)')
    parser.add_argument("--docs-dir", type=str, default="docs", help="Directory to export published HTML to")
    parser.add_argument("--export-product-json", type=str, default=None, help="Directory to export product timeline JSON files (one per product)")
    parser.add_argument("--export-product-story", type=str, default=None, help="Directory to export product story text files (one per product)")
    args = parser.parse_args()

    # Export product timelines as JSON if requested
    if args.export_product_json:
        try:
            os.makedirs(args.export_product_json, exist_ok=True)
            comps = sim.detect_products()
            for k in comps.keys():
                out_path = os.path.join(args.export_product_json, f"product_{k}.json")
                export_product_timeline(sim, k, out_path)
            print(f"Exported product timelines to {args.export_product_json}")
        except Exception as e:
            print(f"Failed to export product timelines: {e}")

    # Export product stories as TXT if requested
    if args.export_product_story:
        try:
            os.makedirs(args.export_product_story, exist_ok=True)
            comps = sim.detect_products()
            for k in comps.keys():
                out_path = os.path.join(args.export_product_story, f"product_{k}.txt")
                story = generate_reaction_story(sim, k, comps)
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(story)
            print(f"Exported product stories to {args.export_product_story}")
        except Exception as e:
            print(f"Failed to export product stories: {e}")
    args = parser.parse_args()

    # If user supplied a custom elements file, load and merge
    # element_data is a mapping of SYMBOL -> properties
    if args.elements:
        try:
            raw = safe_load_json(args.elements)
            if isinstance(raw, dict) and "elements" in raw:
                edata = {el['symbol'].upper(): el for el in raw['elements']}
            else:
                edata = raw
            element_data = {**element_data, **edata}
            logger.info(f"Loaded custom elements data from {args.elements}")
        except Exception:
            logger.exception(f"Failed to load custom elements file {args.elements}. Using default elements.")

    # Process add_element commands if present
    for s in args.add_element:
        try:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            symbol = parts[0].upper()
            atomic_mass_val = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
            en_val = float(parts[2]) if len(parts) > 2 and parts[2] else 0.0
            cov_rad = float(parts[3]) if len(parts) > 3 and parts[3] else 0.7
            cpk = parts[4] if len(parts) > 4 and parts[4] else "808080"
            element_data[symbol] = {
                "name": symbol,
                "symbol": symbol,
                "atomic_number": 0,
                "atomic_mass": atomic_mass_val,
                "electronegativity_pauling": en_val,
                "group": 0,
                "period": 0,
                "cpk-hex": cpk,
                "category": "user-defined",
                "covalent_radius": cov_rad
            }
            logger.info(f"Added custom element {symbol} from CLI.")
        except Exception:
            logger.exception(f"Failed to parse CLI element definition: {s}")
    # Apply CLI-provided elements globally so Atom/get_element sees them
    try:
        ELEMENT_DATA.update({k.upper(): v for k, v in element_data.items()})
    except Exception:
        pass
    if args.save_elements:
        outpath = args.save_elements
        # Convert to serializable list
        try:
            out = {"elements": list(ELEMENT_DATA.values())}
            safe_save_json(out, outpath)
            logger.info(f"Saved merged elements to {outpath}")
        except Exception:
            logger.exception(f"Failed to save merged elements to {outpath}")

    formula_strings = [f.strip() for f in args.formulas.split(",") if f.strip()]
    formula_dicts = []
    for f in formula_strings:
        try:
            formula_dicts.append(parse_formula(f, element_data))
        except Exception as e:
            print(f"Invalid formula '{f}': {e}")
            return

    # Use SimulationManager to run a deterministic or probabilistic simulation for CLI
    # Temperature: use provided value, else default to 50 for deterministic (for bond stability) or 300 otherwise
    sim_temp = float(args.temperature) if args.temperature is not None else (50.0 if args.deterministic else DEFAULT_TEMPERATURE)
    sim = SimulationManager(formula_dicts, deterministic_mode=bool(args.deterministic), temperature=sim_temp)
    # Run the simulation for requested frames; use vis_interval 1 to record each frame
    sim.run_steps(n_steps=args.frames, vis_interval=1)

    # Export events
    # Export KB events recorded during the simulation
    try:
        events_to_export = getattr(sim.kb, 'events', [])
        with open(args.output, "w", encoding='utf-8') as f:
            json.dump(events_to_export, f, indent=2)
    except Exception:
        logger.exception("Failed to export simulation events")
    print(f"Simulation finished. Events saved to {args.output}")
    if args.show_events:
        try:
            print("Event log summary:")
            for ev in getattr(sim.kb, 'events', [])[:200]:
                t = ev.get('frame', 'NA')
                et = ev.get('event_type', 'unknown')
                atoms = ",".join([f"{a['symbol']}({a.get('uid')})" for a in ev.get('atoms', [])])
                if et in ('bond_formed', 'bond_broken'):
                    print(f"[frame {t}] {et}: {atoms}")
        except Exception:
            logger.exception('Failed to print events')
    # Optional: show detected products
    if args.show_products:
        try:
            comps = sim.detect_products()
            print("Detected products:")
            from collections import Counter
            for k, (atoms, bonds) in comps.items():
                formula = graph_to_formula(atoms, bonds)
                # print human readable formula
                pretty = "".join([f"{el}{c if c>1 else ''}" for el, c in sorted(formula.items())])
                print(f"- {k}: {pretty}")
        except Exception:
            logger.exception("Product detection failed")
    if args.explain_products:
        try:
            comps = sim.detect_products()
            for k in comps.keys():
                explain_component(sim, k, comps)
        except Exception:
            logger.exception("Product explain failed")
    # optional HTML export (best-effort)
    if args.export_html:
        # If sim has atom histories, try to export directly; else build a SimulationManager from formulas
        try:
            if hasattr(sim, 'atoms') and any(getattr(a, 'history', None) for a in sim.atoms):
                export_simulation_to_plotly_html(sim, args.export_html, n_frames=args.export_frames, fps=args.export_fps)
            else:
                # fallback: create a SimulationManager for the same formulas and run short simulation
                try:
                    sim_demo = SimulationManager(formula_dicts)
                    # run enough steps to record histories for export
                    sim_demo.run_steps(n_steps=args.export_frames, vis_interval=1)
                    export_simulation_to_plotly_html(sim_demo, args.export_html, n_frames=args.export_frames, fps=args.export_fps)
                except Exception:
                    # if fallback fails, still attempt export directly (may throw)
                    export_simulation_to_plotly_html(sim, args.export_html, n_frames=args.export_frames, fps=args.export_fps)
            print(f"Interactive HTML exported to {args.export_html}")
        except Exception as e:
            print(f"Export to HTML failed: {e}")
        # if publish requested, copy to docs
        if args.publish:
            try:
                docs_dir = args.docs_dir
                ensure_dir(docs_dir)
                # copy exported html into docs as index
                dest = os.path.join(docs_dir, os.path.basename(args.export_html))
                import shutil
                # Guard: avoid copying file onto itself
                if os.path.abspath(dest) != os.path.abspath(args.export_html):
                    shutil.copyfile(args.export_html, dest)
                # create a simple index.html that embeds the export
                idx = os.path.join(docs_dir, "index.html")
                with open(idx, "w", encoding="utf-8") as fh:
                    fh.write(f"<html><body><h1>Simulation Demo</h1><iframe src=\"{os.path.basename(args.export_html)}\" width=100% height=900></iframe></body></html>")
                print(f"Published HTML to {docs_dir}")
            except Exception:
                logger.exception("Failed to publish exported HTML to docs/")


def simulate_and_report(formula_str: str, frames:int = 200, interval: int = 1, deterministic: bool = True, export_html: Optional[str] = None, temperature: Optional[float] = None):
    """Utility function to run a simulation from a single formula string and report products.
    formula_str: comma-separated formulae or single formula like 'H2,O'
    returns (sim, products) where products is a dict mapping comp keys to (atoms,bonds)"""
    formula_strings = [f.strip() for f in formula_str.split(",") if f.strip()]
    formula_dicts = []
    for f in formula_strings:
        formula_dicts.append(parse_formula(f, ELEMENT_DATA))
    sim_temp = float(temperature) if temperature is not None else (50.0 if deterministic else DEFAULT_TEMPERATURE)
    sim = SimulationManager(formula_dicts, deterministic_mode=deterministic, temperature=sim_temp)
    sim.run_steps(n_steps=frames, vis_interval=interval)
    if export_html and PLOTLY_AVAILABLE:
        try:
            export_simulation_to_plotly_html(sim, export_html, n_frames=min(frames, sim.frame), fps=10)
        except Exception:
            logger.exception("Export failed in simulate_and_report")
    products = sim.detect_products()
    # convert to pretty formulas
    pretty = {k: "".join([f"{el}{c if c>1 else ''}" for el,c in sorted(graph_to_formula(atoms,bonds).items())]) for k,(atoms,bonds) in products.items()}
    return sim, pretty, products


def get_events_for_component(sim: SimulationManager, comp_atoms: list) -> list:
    """Return a list of KB events related to the atoms in the given component (match by uid)."""
    uids = {a.uid for a in comp_atoms}
    events = []
    for ev in getattr(sim.kb, 'events', []):
        # event atoms may contain dicts with uid keys
        ev_uids = {a.get('uid') for a in ev.get('atoms', [])}
        if uids & ev_uids:
            events.append(ev)
    return events


def explain_component(sim: SimulationManager, comp_key: str, products: dict) -> None:
    """Print a timeline of events that led to the formation (or breakage) of the component `comp_key`.
    `products` should be the mapping returned by sim.detect_products()."""
    if comp_key not in products:
        print(f"Component key {comp_key} not found")
        return
    atoms, bonds = products[comp_key]
    events = get_events_for_component(sim, atoms)
    print(f"Explaining component {comp_key}: {len(atoms)} atoms, {len(bonds)} bonds")
    for ev in sorted(events, key=lambda e: e.get('frame', 0)):
        frame = ev.get('frame', 'NA')
        typ = ev.get('event_type', 'unknown')
        atoms_str = ",".join([f"{a.get('symbol')}({a.get('uid')})" for a in ev.get('atoms', [])])
        print(f"[frame {frame}] {typ} -> {atoms_str}")


# Removed CRISPR placeholder classes and replaced them with the project-focused SimulationManager
# -----------------------
# End of CHUNK 9/10
# -----------------------
# -----------------------
# CHUNK 10/10
# Main Program Integration and Launch
# -----------------------

if __name__ == "__main__":
    # If CLI args passed (e.g., --formulas), run the CLI mode instead of GUI/demo
    if len(sys.argv) > 1 and any(arg.startswith("--formulas") or arg.startswith("--export-html") or arg.startswith("--elements") for arg in sys.argv[1:]):
        try:
            run_simulation_cli(ELEMENT_DATA)
            sys.exit(0)
        except SystemExit:
            raise
        except Exception as e:
            print(f"CLI run failed: {e}")
            # fallback to GUI flow below
    # -----------------------
    # Launch GUI
    # -----------------------
    try:
        # Use the project-local models.simulation_gui module
        from models.simulation_gui import SimulationGUIAdvanced

        # Use a general SimulationManager for GUI by default
        sim = SimulationManager([])
        gui = SimulationGUIAdvanced(sim=sim, title="Elements Simulation Viewer")
        gui.main_menu()  # setup GUI main menu
        gui.start()      # start Tkinter event loop
    except Exception as e:
        print(f"GUI launch failed: {e}")

    # -----------------------
    # Example: Run a simple SimulationManager demo and export results
    # -----------------------
    example_formulas = [
        {"H":2, "O":1},     # H2O
        {"C":6, "H":6},     # Benzene C6H6
        {"Na":1, "Cl":1},   # NaCl
        {"C":1, "O":2}      # CO2
    ]

    try:
        sim = SimulationManager(example_formulas)
        sim.run_steps(n_steps=600, vis_interval=1)
        print("Simulation complete.")
        # export a small KB snapshot next to docs
        try:
            ensure_dir('docs')
            sim.kb._save_limited()
            # export interactive HTML if plotly is available
            if PLOTLY_AVAILABLE:
                export_simulation_to_plotly_html(sim, os.path.join('docs', 'example_sim.html'), n_frames=600, fps=10)
                print("Exported docs/example_sim.html")
        except Exception:
            logger.exception("Failed to export example simulation data")
    except Exception as e:
        print(f"Simulation run failed: {e}")