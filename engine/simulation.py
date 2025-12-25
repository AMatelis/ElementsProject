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
                 kb: Optional[Any] = None,  # Will be defined later
                 scan_radius: float = 0.18,
                 rng: Optional[np.random.Generator] = None):
        self.atoms = atoms
        self.bonds = bonds
        self.physics = physics
        self.kb = kb
        self.scan_radius = float(scan_radius)
        self.enabled = True
        self.deterministic_mode = False
        self.rng = rng if rng is not None else np.random.default_rng()

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
                    if self.deterministic_mode or (score >= 1.0) or (self.rng.random() < float(score)):
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
# Simulation
# -----------------------
class Simulation:
    """
    Core simulation engine: handles atoms, bonds, physics, reaction engine, and stepping.
    Decoupled from GUI, KB, and exports for clean headless runs and testing.
    """

    def __init__(self,
                 formula_list: Optional[List[Dict[str, int]]] = None,
                 temperature: float = 300.0,
                 deterministic_mode: bool = False,
                 atom_history_maxlen: int = 2000,
                 seed: Optional[int] = None,
                 kb: Optional[Any] = None):
        """
        formula_list: list of molecule dicts, e.g. [{"H":2,"O":1}, {"Na":1,"Cl":1}]
                      if None or empty, the simulation will start with no atoms.
        """
        self.formula_list = formula_list or []
        self.temperature = float(temperature)
        self.deterministic_mode = bool(deterministic_mode)
        # deterministic seed (optional): used to seed Python random and NumPy legacy RNGs
        self.seed = int(seed) if seed is not None else None
        self.kb = kb

        # centralized RNG for reproducibility
        self.rng = np.random.default_rng(seed=self.seed)

        # runtime state
        self.atoms: List[Atom] = []
        self.bonds: List[BondObj] = []
        self.frame: int = 0
        self._initial_formulas = list(self.formula_list)

        # build atoms from formulas
        self._build_atoms_from_formulas(self._initial_formulas, self.rng)

        # physics and reaction engine
        self.physics = PhysicsEngine(self.atoms, self.bonds, dt=1e-5, temperature=self.temperature, rng=self.rng)
        self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, kb=self.kb, rng=self.rng)
        self.reaction_engine.deterministic_mode = self.deterministic_mode

        # visualization handles
        self.fig = None
        self.main_ax = None

        # energy conservation tracking (approximate, due to thermostat)
        self.initial_energy = self.physics.total_energy()['total']
        self.energy_drift_history: List[float] = []
        self.max_energy_drift = 0.0

        logger.info(f"Simulation initialized: atoms={len(self.atoms)} bonds={len(self.bonds)} initial_energy={self.initial_energy:.6f}")

    # -----------------------
    # Construction helpers
    # -----------------------
    def _build_atoms_from_formulas(self, formula_list: List[Dict[str, int]], rng: np.random.Generator) -> None:
        """
        Create Atom objects given a list of formula dictionaries.
        Atoms are placed randomly within a central region to avoid initial overlap.
        """
        self.atoms = []
        uid_counter = 0
        for midx, fdict in enumerate(formula_list):
            # choose a cluster center away from edges for nicer visuals
            center = np.clip(rng.random(2) * 1.5 + 0.25, 0.25, 1.75)
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
                    pos = center + rng.normal(scale=dist_scale, size=2)
                    pos = np.clip(pos, 0.05, 1.95)
                    # small thermal velocity based on mass / temperature
                    try:
                        vel = rng.normal(scale=0.005, size=2)
                    except Exception:
                        vel = np.zeros(2)
                    atom = Atom(symbol=sym, pos=pos, vel=vel, uid=uid)
                    self.atoms.append(atom)
                    uid_counter += 1
        # ensure physics engine references new atoms
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
        """
        try:
            # advance physics
            self.physics.step()

            # Reaction engine uses physics.spatial for neighbors
            self.reaction_engine.step(self.frame)

            # history recording for visualization / export
            for a in self.atoms:
                a.record(self.frame)

            self.frame += 1

        except Exception:
            logger.exception("Simulation.step failed.")

    def run(self, n_steps: int, update_callback: Optional[Callable[["Simulation"], None]] = None) -> None:
        """
        Run n_steps synchronously. Useful for batch runs or CLI mode.
        update_callback is called every step with self as arg.
        """
        for _ in range(n_steps):
            self.step()
            if update_callback is not None:
                try:
                    update_callback(self)
                except Exception:
                    logger.exception("update_callback failed during run.")

    # -----------------------
    # Utility & analysis helpers
    # -----------------------
    def detect_products(self) -> Dict[str, Tuple[List[Atom], List[BondObj]]]:
        """Detect connected components (products) and return mapping comp_key -> (atoms,bonds)."""
        return products.detect_products_from_scene(self.atoms, self.bonds)

    def reset(self) -> None:
        """
        Reset the simulation to the initial formulas: rebuild atoms/bonds and reset time/frame.
        """
        try:
            # clear state
            self.atoms = []
            self.bonds = []
            self.frame = 0
            # rebuild
            self._build_atoms_from_formulas(self._initial_formulas)
            self.physics = PhysicsEngine(self.atoms, self.bonds, dt=self.physics.dt, temperature=self.temperature)
            self.reaction_engine = ReactionEngine(self.atoms, self.bonds, self.physics, kb=None)
            self.reaction_engine.deterministic_mode = self.deterministic_mode
            logger.info("Simulation reset to initial formulas.")
        except Exception:
            logger.exception("Failed to reset simulation.")