from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any, Callable
import threading
import time
import json
import os
import logging
from collections import defaultdict

import numpy as np

# Import engine submodules
from .atoms import Atom
from .bonds import BondObj, can_form_bond, sanitize_bonds
from .physics import PhysicsEngine
from . import products
from .formula_parser import parse_formula

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# -----------------------
# Minimal Reaction Knowledge Base (persistable)
# -----------------------
class ReactionKnowledgeBase:
    """
    Lightweight, safe KB for storing reaction events and exporting ML datasets.
    Stores events in memory and persisting a JSONL file on demand.
    Event format (dict) is intentionally simple and serializable.
    """
    def __init__(self, path_json: Optional[str] = None):
        self.path_json = path_json or os.path.join(os.getcwd(), "data", "reaction_kb.jsonl")
        ensure_parent_dir(self.path_json)
        self.events: List[Dict[str, Any]] = []
        # If existing file found, load it into memory (line-by-line)
        if os.path.exists(self.path_json):
            try:
                with open(self.path_json, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            try:
                                self.events.append(json.loads(line))
                            except Exception:
                                # skip malformed lines
                                logger.debug("Skipping malformed KB line during load.")
                logger.info(f"Loaded {len(self.events)} events from KB at {self.path_json}")
            except Exception:
                logger.exception("Failed to load existing KB JSONL; starting fresh.")
        # lightweight counters for quick stats
        self.pair_counts = defaultdict(int)
        self.bond_counts = defaultdict(int)

    def add_event(self, ev: Dict[str, Any]) -> None:
        """
        Append an event to in-memory list and append to disk (JSONL).
        Event should be JSON-serializable.
        """
        try:
            # enrich with timestamp if not present
            if "timestamp" not in ev:
                ev["timestamp"] = time.strftime("%Y%m%dT%H%M%S")
            self.events.append(ev)
            with open(self.path_json, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            # update counters if event contains atoms/bonds
            for a in ev.get("atoms", []):
                sym = a.get("symbol") if isinstance(a, dict) else None
                if sym:
                    # no-op here; kept for parity with earlier structure
                    pass
            for b in ev.get("bonds", []):
                try:
                    u1, u2 = b[0], b[1]
                    key = tuple(sorted((u1, u2)))
                    self.bond_counts[key] += 1
                except Exception:
                    pass
        except Exception:
            logger.exception("Failed to persist KB event.")

    def export_jsonl(self, out_path: Optional[str] = None) -> str:
        """
        Write current in-memory events to out_path (or default KB path) as JSONL.
        Returns the path written.
        """
        target = out_path or self.path_json
        ensure_parent_dir(target)
        try:
            with open(target, "w", encoding="utf-8") as fh:
                for ev in self.events:
                    fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            logger.info(f"Exported KB events to {target} ({len(self.events)} events)")
        except Exception:
            logger.exception("Failed to export KB JSONL.")
        return target


# -----------------------
# Reaction Engine (bond formation / breakage)
# -----------------------
class ReactionEngine:
    """
    Responsible for:
      - scanning neighbor lists for candidate bond formation
      - using can_form_bond heuristics (from bonds.py) to decide on formation
      - evaluating existing bonds for breakage using BondObj.should_break()
      - logging bond events to the KB (if provided)

    This implementation keeps deterministic_mode flag that forces rule-driven bond formation
    to be evaluated for all pairs (useful for prediction mode).
    """
    def __init__(self,
                 atoms: List[Atom],
                 bonds: List[BondObj],
                 physics: PhysicsEngine,
                 kb: Optional[ReactionKnowledgeBase] = None,
                 scan_radius: float = 0.18):
        self.atoms = atoms
        self.bonds = bonds
        self.physics = physics
        self.kb = kb
        self.scan_radius = float(scan_radius)
        self.enabled = True
        self.deterministic_mode = False

    def step(self, frame: int) -> None:
        """
        Evaluate bond formation and breakage for the current simulation state.
        This function mutates self.bonds and Atom.bonds lists when bonds are formed/broken.
        """
        # If disabled, only process breakage to allow cleanup
        if not self.enabled:
            self._check_breakage(frame)
            return

        # Build candidate pairs
        candidates: List[Tuple[Atom, Atom]] = []
        if self.deterministic_mode:
            # evaluate all unique pairs
            n = len(self.atoms)
            for i in range(n):
                for j in range(i + 1, n):
                    candidates.append((self.atoms[i], self.atoms[j]))
        else:
            # local neighbor-based candidates via physics.spatial
            for a in self.atoms:
                neighs = self.physics.spatial.neighbors(a, radius=self.scan_radius)
                for n in neighs:
                    # avoid duplicates — only evaluate if uid ordering matches
                    if getattr(a, "uid") >= getattr(n, "uid"):
                        continue
                    # skip if already bonded directly
                    if a.is_bonded_to(n):
                        continue
                    candidates.append((a, n))

        # Evaluate candidates
        for a, b in candidates:
            try:
                can, score = can_form_bond(a, b, self.physics.temperature)
                if can:
                    # probabilistic acceptance unless deterministic_mode
                    if self.deterministic_mode or (score >= 1.0) or (np.random.rand() < float(score)):
                        # create bond
                        bond = BondObj(a, b, order=1)
                        # track bond creation time
                        bond.time_of_creation = time.time()
                        self.bonds.append(bond)
                        # log event
                        ev = {
                            "timestamp": time.strftime("%Y%m%dT%H%M%S"),
                            "frame": frame,
                            "event_type": "bond_formed",
                            "atoms": [
                                {"uid": a.uid, "symbol": a.symbol, "pos": list(map(float, a.pos)), "charge": float(a.charge)},
                                {"uid": b.uid, "symbol": b.symbol, "pos": list(map(float, b.pos)), "charge": float(b.charge)}
                            ],
                            "bonds": [(a.uid, b.uid, bond.order)],
                            "temperature": float(self.physics.temperature),
                            "energy": float(bond.estimate_bond_energy())
                        }
                        try:
                            if self.kb:
                                self.kb.add_event(ev)
                        except Exception:
                            logger.exception("KB add_event failed for bond formation.")
            except Exception:
                logger.exception("Error evaluating candidate pair for bonding.")

        # Evaluate existing bonds for breakage
        self._check_breakage(frame)

    def _check_breakage(self, frame: int) -> None:
        # we iterate over a snapshot of bonds that existed before this step
        existing = list(self.bonds)
        for b in existing:
            try:
                brk, severity = b.should_break(self.physics.temperature)
                if brk:
                    # safely remove bond
                    try:
                        self.bonds.remove(b)
                    except ValueError:
                        pass
                    try:
                        b.atom1.remove_bond(b)
                        b.atom2.remove_bond(b)
                    except Exception:
                        logger.debug("Failed to remove bond from atom lists during breakage.")
                    ev = {
                        "timestamp": time.strftime("%Y%m%dT%H%M%S"),
                        "frame": frame,
                        "event_type": "bond_broken",
                        "atoms": [
                            {"uid": b.atom1.uid, "symbol": b.atom1.symbol, "pos": list(map(float, b.atom1.pos)), "charge": float(b.atom1.charge)},
                            {"uid": b.atom2.uid, "symbol": b.atom2.symbol, "pos": list(map(float, b.atom2.pos)), "charge": float(b.atom2.charge)}
                        ],
                        "bonds": [(b.atom1.uid, b.atom2.uid, b.order)],
                        "temperature": float(self.physics.temperature),
                        "severity": float(severity)
                    }
                    try:
                        if self.kb:
                            self.kb.add_event(ev)
                    except Exception:
                        logger.exception("KB add_event failed for bond breakage.")
            except Exception:
                logger.exception("Error evaluating bond breakage.")


# -----------------------
# SimulationManager
# -----------------------
class SimulationManager:
    """
    High-level simulation manager.
    Usage:
        sim = SimulationManager([{"H":2, "O":1}], temperature=300.0, auto_train_kb=False)
        sim.run_steps(n_steps=1000, vis_interval=10)
    """

    def __init__(self,
                 formula_list: Optional[List[Dict[str, int]]] = None,
                 temperature: float = 300.0,
                 auto_train_kb: bool = True,
                 deterministic_mode: bool = False,
                 atom_history_maxlen: int = 2000,
                 seed: Optional[int] = None):
        """
        formula_list: list of molecule dicts, e.g. [{"H":2,"O":1}, {"Na":1,"Cl":1}]
                      if None or empty, the simulation will start with no atoms.
        """
        self.formula_list = formula_list or []
        self.temperature = float(temperature)
        self.auto_train_kb = bool(auto_train_kb)
        self.deterministic_mode = bool(deterministic_mode)
        # deterministic seed (optional): used to seed Python random and NumPy legacy RNGs
        self.seed = int(seed) if seed is not None else None

        # runtime state
        self.atoms: List[Atom] = []
        self.bonds: List[BondObj] = []
        self.frame: int = 0
        self._initial_formulas = list(self.formula_list)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # setup KB
        self.kb = ReactionKnowledgeBase(path_json=os.path.join("data", f"reaction_kb_{int(time.time())}.jsonl"))

        # deterministic seeding for reproducibility (affects random, numpy legacy, and PhysicsEngine)
        try:
            import random as _random
            if self.seed is not None:
                _random.seed(self.seed)
                # legacy numpy RNG seeding for code using np.random.*
                import numpy as _np
                _np.random.seed(self.seed)
        except Exception:
            logger.exception("Failed to set deterministic seeds.")

        # build atoms from formulas
        self._build_atoms_from_formulas(self._initial_formulas)

        # physics and reaction engine (pass seed to engine RNG)
        self.physics = PhysicsEngine(self.atoms, self.bonds, dt=1e-3, temperature=self.temperature, seed=self.seed)
        self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, kb=self.kb)
        self.reaction_engine.deterministic_mode = self.deterministic_mode

        # optional per-frame exporter path (set via start_frame_export)
        self._frame_export_path: Optional[str] = None

        # visualization handles (kept minimal here)
        self.fig = None
        self.ax = None

        # event log for end-of-run export (keeps references to KB events)
        self.event_log: List[Dict[str, Any]] = []
        # per-frame metrics for visualization / ML
        self.energy_history: List[float] = []
        self.training_loss_history: List[float] = []

        logger.info(f"SimulationManager initialized: atoms={len(self.atoms)} bonds={len(self.bonds)}")

    # -----------------------
    # Construction helpers
    # -----------------------
    def _build_atoms_from_formulas(self, formula_list: List[Dict[str, int]]) -> None:
        """
        Create Atom objects given a list of formula dictionaries.
        Atoms are placed randomly within a central region to avoid initial overlap.
        """
        self.atoms = []
        uid_counter = 0
        for midx, fdict in enumerate(formula_list):
            # choose a cluster center away from edges for nicer visuals
            center = np.clip(np.random.rand(2) * 0.5 + 0.25, 0.15, 0.85)
            for sym, cnt in fdict.items():
                for i in range(max(1, int(cnt))):
                    uid = f"m{midx}_a{uid_counter}"
                    # small scatter around the center proportional to covalent radius
                    # use element covalent radius to avoid overlaps
                    try:
                        from engine.elements_data import get_element
                        r = float(get_element(sym).get('covalent_radius', 0.7))
                    except Exception:
                        r = 0.7
                    dist_scale = max(0.005, r * 0.08)
                    pos = center + np.random.normal(scale=dist_scale, size=2)
                    pos = np.clip(pos, 0.05, 0.95)
                    # small thermal velocity based on mass / temperature
                    try:
                        import numpy as _np
                        vel = _np.random.normal(scale=0.005, size=2)
                    except Exception:
                        vel = np.zeros(2)
                    atom = Atom(symbol=sym, pos=pos, vel=vel, uid=uid)
                    self.atoms.append(atom)
                    uid_counter += 1
        # ensure physics engine (if present) references new atoms
        try:
            if hasattr(self, "physics") and self.physics is not None:
                self.physics.atoms = self.atoms
        except Exception:
            pass

    # -----------------------
    # Core stepping
    # -----------------------
    def step(self) -> None:
        """
        Perform a single simulation tick:
         - advance physics
         - perform reaction engine step (bond formation & breakage)
         - record atom history
         - optionally export / snapshot KB at intervals
        """
        try:
            # advance physics
            self.physics.step()

            # Reaction engine uses physics.spatial for neighbors
            self.reaction_engine.step(self.frame)

            # history recording for visualization / export
            for a in self.atoms:
                a.record(self.frame)

            # energy diagnostics for visualization/ML
            try:
                e = float(self.physics.total_energy().get('total', 0.0))
            except Exception:
                e = 0.0
            self.energy_history.append(e)

            # training loss placeholder: if ML trainer present, it can push values here
            if not self.training_loss_history:
                # seed with None or 0-length. downstream code can detect empty list.
                pass

            self.frame += 1

            # optional export/training trigger
            if self.auto_train_kb and (self.frame % 400 == 0) and self.frame > 0:
                try:
                    # quick KB snapshot
                    self.kb.export_jsonl()  # writes current KB out
                except Exception:
                    logger.exception("Dataset export failed during run.")

            # per-frame exporter (if enabled)
            try:
                if getattr(self, "_frame_export_path", None):
                    # lazy import to avoid circular issues
                    from .exporter import append_frame_jsonl
                    append_frame_jsonl(self, out_path=self._frame_export_path)
            except Exception:
                logger.exception("Per-frame export failed.")

        except Exception:
            logger.exception("SimulationManager.step failed.")

    def run_steps(self, n_steps: int = 1000, vis_interval: int = 10, update_callback: Optional[Callable[["SimulationManager"], None]] = None):
        """
        Run a synchronous loop of steps. Useful for batch runs or CLI mode.
        update_callback is called every vis_interval frames with self as arg.
        """
        for _ in range(n_steps):
            self.step()
            if update_callback is not None and (self.frame % vis_interval == 0):
                try:
                    update_callback(self)
                except Exception:
                    logger.exception("update_callback failed during run_steps.")

    # -----------------------
    # Background threaded run support
    # -----------------------
    def start(self, interval: float = 0.02):
        """
        Start a background thread running the simulation. interval is the sleep
        time between steps (seconds). This method is safe to call multiple times.
        """
        if self._running:
            logger.debug("Simulation already running; start() ignored.")
            return
        self._running = True

        def loop():
            while self._running:
                try:
                    self.step()
                except Exception:
                    logger.exception("Exception in simulation loop.")
                time.sleep(interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        logger.info("Simulation background thread started.")

    def stop(self):
        """Stop a background run and join the thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        logger.info("Simulation background thread stopped.")

    # -----------------------
    # Utility & exports
    # -----------------------
    def detect_products(self) -> Dict[str, Tuple[List[Atom], List[BondObj]]]:
        """Detect connected components (products) and return mapping comp_key -> (atoms,bonds)."""
        return products.detect_products_from_scene(self.atoms, self.bonds)

    def export_results(self, out_prefix: Optional[str] = None) -> str:
        """
        Export run results (atoms history, bonds, KB snapshot) into an output directory.
        Returns the path to the exported directory.
        """
        out_prefix = out_prefix or time.strftime("%Y%m%dT%H%M%S")
        out_dir = os.path.join("outputs", f"run_{out_prefix}")
        os.makedirs(out_dir, exist_ok=True)
        # atoms.json
        atoms_out = []
        for a in self.atoms:
            atoms_out.append({
                "uid": a.uid,
                "symbol": a.symbol,
                "final_pos": list(map(float, a.pos)),
                "history": list(a.history),
                "state": getattr(a, "state", None)
            })
        with open(os.path.join(out_dir, "atoms.json"), "w", encoding="utf-8") as fh:
            json.dump({"atoms": atoms_out, "frame_count": self.frame}, fh, indent=2, ensure_ascii=False)

        # bonds.json
        bond_list = []
        for b in self.bonds:
            bond_list.append((b.atom1.uid, b.atom2.uid, b.order))
        with open(os.path.join(out_dir, "bonds.json"), "w", encoding="utf-8") as fh:
            json.dump({"bonds": bond_list}, fh, indent=2, ensure_ascii=False)

        # KB snapshot
        try:
            kb_path = os.path.join(out_dir, os.path.basename(self.kb.path_json))
            self.kb.export_jsonl(kb_path)
        except Exception:
            logger.exception("KB snapshot failed during export_results.")

        logger.info(f"Exported simulation results to {out_dir}")
        return out_dir

    def reset_simulation(self) -> None:
        """
        Reset the simulation to the initial formulas: rebuild atoms/bonds and reset time/frame/KB.
        """
        try:
            self.stop()
        except Exception:
            pass
        # clear state
        self.atoms = []
        self.bonds = []
        self.frame = 0
        self.kb = ReactionKnowledgeBase(path_json=os.path.join("data", f"reaction_kb_{int(time.time())}.jsonl"))
        # rebuild
        self._build_atoms_from_formulas(self._initial_formulas)
        self.physics = PhysicsEngine(self.atoms, self.bonds, dt=self.physics.dt, temperature=self.temperature)
        self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, kb=self.kb)
        self.reaction_engine.deterministic_mode = self.deterministic_mode
        logger.info("Simulation reset to initial formulas.")

    # -----------------------
    # Convenience / analysis helpers
    # -----------------------
    def detect_and_describe(self) -> Dict[str, Any]:
        """
        Convenience: detect products and return structured descriptions (uses products.detect_and_describe).
        """
        return products.detect_and_describe(self.atoms, self.bonds, kb_events=self.kb.events)

    def attach_logging(self):
        """
        Attach simple logging hook to record bond changes between steps.
        """
        original_step = getattr(self, "step", None)
        if original_step is None:
            logger.warning("No step method to attach logging to.")
            return

        def wrapped_step(*args, **kwargs):
            before = set(tuple(sorted((b.atom1.uid, b.atom2.uid))) for b in list(self.bonds))
            original_step(*args, **kwargs)
            after = set(tuple(sorted((b.atom1.uid, b.atom2.uid))) for b in list(self.bonds))
            added = after - before
            removed = before - after
            for uid1, uid2 in added:
                logger.info(f"Bond formed: {uid1}-{uid2}")
            for uid1, uid2 in removed:
                logger.info(f"Bond broken: {uid1}-{uid2}")

        self.step = wrapped_step

    # -----------------------
    # Frame export control
    # -----------------------
    def start_frame_export(self, out_path: Optional[str] = None) -> str:
        """Enable per-frame JSONL export. If out_path is a directory, a timestamped file
        will be created inside it. Returns the path that will be written to."""
        if out_path is None:
            out_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(out_dir, exist_ok=True)
            out_path = out_dir
        # store path and return
        self._frame_export_path = out_path
        logger.info(f"Per-frame export enabled -> {self._frame_export_path}")
        return self._frame_export_path

    def stop_frame_export(self) -> None:
        """Disable per-frame exporting."""
        self._frame_export_path = None
        logger.info("Per-frame export disabled.")

# -----------------------
# Helper utilities
# -----------------------
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)