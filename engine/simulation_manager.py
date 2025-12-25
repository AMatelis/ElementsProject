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
from .metrics import EnergyMetrics
from .atoms import Atom
from .bonds import BondObj, can_form_bond, sanitize_bonds
from .physics import PhysicsEngine
from . import products
from .formula_parser import parse_formula
from .simulation import Simulation

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
        self.frame: int = 0
        self._initial_formulas = list(self.formula_list)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # setup KB
        self.kb = ReactionKnowledgeBase(path_json=os.path.join("data", f"reaction_kb_{int(time.time())}.jsonl"))

        # create core simulation engine
        self.sim = Simulation(
            formula_list=self.formula_list,
            temperature=self.temperature,
            deterministic_mode=self.deterministic_mode,
            seed=self.seed,
            kb=self.kb
        )

        # optional per-frame exporter path (set via start_frame_export)
        self._frame_export_path: Optional[str] = None

        # visualization handles (kept minimal here)
        self.fig = None
        self.main_ax = None

        # event log for end-of-run export (keeps references to KB events)
        self.event_log: List[Dict[str, Any]] = []

    @property
    def fig(self):
        """Delegate fig to internal simulation"""
        return self.sim.fig if hasattr(self.sim, 'fig') else None

    @fig.setter
    def fig(self, value):
        """Set fig on internal simulation"""
        self.sim.fig = value

    @property
    def main_ax(self):
        """Delegate main_ax to internal simulation"""
        return self.sim.main_ax if hasattr(self.sim, 'main_ax') else None

    @main_ax.setter
    def main_ax(self, value):
        """Set main_ax on internal simulation"""
        self.sim.main_ax = value
        # energy metrics for stable visualization
        self.energy_metrics = EnergyMetrics(max_history=1000, smoothing_window=5)
        # legacy energy history for backward compatibility
        self.energy_history: List[float] = []
        self.training_loss_history: List[float] = []
        # optional trainer callback: function(sim) -> Optional[float]
        self.loss_callback: Optional[Callable[["SimulationManager"], Optional[float]]] = None

        logger.info(f"SimulationManager initialized: atoms={len(self.sim.atoms)} bonds={len(self.sim.bonds)}")

    # -----------------------
    # Properties delegating to simulation
    # -----------------------
    @property
    def atoms(self) -> List[Atom]:
        return self.sim.atoms

    @property
    def bonds(self) -> List[BondObj]:
        return self.sim.bonds

    @property
    def physics(self):
        return self.sim.physics

    # -----------------------
    # Core stepping
    # -----------------------
    def step(self) -> None:
        """
        Perform a single simulation tick:
         - delegate to core simulation step
         - record atom history
         - optionally export / snapshot KB at intervals
        """
        try:
            # delegate core stepping to simulation engine
            self.sim.step()

            # energy diagnostics for visualization/ML (approximate estimates)
            try:
                # Update comprehensive energy metrics
                self.energy_metrics.update(
                    self.sim.physics,
                    len(self.sim.atoms),
                    len(self.sim.bonds)
                )
                # Keep legacy total energy for backward compatibility
                e = float(self.sim.physics.total_energy().get('total', 0.0))
                self.energy_history.append(e)
            except Exception as e:
                logger.exception(f"Error updating energy metrics: {e}")
                e = 0.0
                self.energy_history.append(e)

            # energy conservation validation (approximate due to thermostat)
            try:
                drift = (e - self.sim.initial_energy) / abs(self.sim.initial_energy) if self.sim.initial_energy != 0 else 0.0
                self.sim.energy_drift_history.append(drift)
                self.sim.max_energy_drift = max(self.sim.max_energy_drift, abs(drift))
                if abs(drift) > 0.1:
                    logger.warning(f"Timestep stability warning: energy drift {abs(drift):.6f} exceeds threshold 0.1")
            except Exception:
                logger.exception("Error computing energy drift")

            # training loss callback: allow external trainers to push per-step loss values
            try:
                if self.loss_callback is not None:
                    val = self.loss_callback(self)
                    if val is not None:
                        try:
                            self.training_loss_history.append(float(val))
                        except Exception:
                            # ignore non-numeric returns
                            pass
            except Exception:
                logger.exception("Error running loss_callback")

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

        # log energy conservation summary
        logger.info(f"Run completed: max energy drift = {self.sim.max_energy_drift:.6f}")

    # -----------------------
    # Trainer callback API
    # -----------------------
    def register_loss_callback(self, cb: Optional[Callable[["SimulationManager"], Optional[float]]]) -> None:
        """Register a loss callback. The callback receives the SimulationManager and
        can return a numeric loss value (or None). Returned values are appended to
        `self.training_loss_history` when provided."""
        if cb is None:
            self.loss_callback = None
        else:
            self.loss_callback = cb

    def push_training_loss(self, value: float) -> None:
        """Append a training loss value into the history (thread-safe enough for simple use)."""
        try:
            self.training_loss_history.append(float(value))
        except Exception:
            logger.exception("Failed to push training loss value")

    def clear_training_loss_history(self) -> None:
        """Clear stored training loss history."""
        self.training_loss_history = []

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
    # System properties
    # -----------------------
    def get_net_charge(self) -> float:
        """
        Calculate the net charge of the system.

        Returns:
            float: Sum of all atom charges
        """
        return sum(getattr(atom, 'charge', 0.0) for atom in self.atoms)

    def get_charge_distribution(self) -> Dict[str, float]:
        """
        Get charge distribution statistics.

        Returns:
            Dict[str, float]: Dictionary with charge statistics
        """
        charges = [getattr(atom, 'charge', 0.0) for atom in self.atoms]
        return {
            'net_charge': sum(charges),
            'positive_atoms': sum(1 for c in charges if c > 0.1),
            'negative_atoms': sum(1 for c in charges if c < -0.1),
            'neutral_atoms': sum(1 for c in charges if abs(c) <= 0.1),
            'max_positive': max(charges) if charges else 0.0,
            'max_negative': min(charges) if charges else 0.0,
        }

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