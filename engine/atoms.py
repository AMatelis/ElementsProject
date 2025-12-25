from __future__ import annotations
from typing import List, Optional, Dict
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)


class Atom:
    """
    Represents a single atom in the simulation.
    """

    def __init__(
        self,
        symbol: str,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        uid: Optional[str] = None,
        oxidation_state: Optional[int] = None
    ):
        """
        Initialize an Atom.

        Args:
            symbol (str): Element symbol, e.g., "H", "C".
            pos (np.ndarray, optional): 2D or 3D position vector. Defaults to origin.
            vel (np.ndarray, optional): Velocity vector. Defaults to zero.
            uid (str, optional): Unique identifier. Auto-generated if None.
            oxidation_state (int, optional): Oxidation state. If None, defaults to 0 (neutral).
        """
        # Delay import to avoid circular dependency
        from engine.elements_data import get_element
        from engine.visuals import get_element_color, get_element_radius

        self.symbol: str = symbol.capitalize()
        self.uid: str = uid or f"{self.symbol}_{np.random.randint(1e6)}"

        # Load properties from elements.json
        elem_props = get_element(self.symbol)
        self.mass: float = float(elem_props.get("atomic_mass", 12.0))
        self.radius: float = get_element_radius(self.symbol)
        self.en: float = float(elem_props.get("electronegativity_pauling", 0.0))
        self.group: int = int(elem_props.get("group", 0))
        self.period: int = int(elem_props.get("period", 0))
        self.color: str = get_element_color(self.symbol)
        self.category: str = elem_props.get("category", "unknown")

        # Oxidation state and charge
        self.oxidation_state: int = oxidation_state if oxidation_state is not None else 0
        # Calculate charge from oxidation state (simplified: charge = oxidation_state for main group elements)
        self.charge: float = float(self.oxidation_state)

        # Physics properties
        self.pos: np.ndarray = np.array(pos if pos is not None else np.zeros(2), dtype=float)
        self.vel: np.ndarray = np.array(vel if vel is not None else np.zeros_like(self.pos), dtype=float)
        self.force: np.ndarray = np.zeros_like(self.pos)

        # Bonds (import delayed inside methods that use BondObj)
        self.bonds: List = []

        # History buffer for visualization and analysis
        self.history: deque = deque(maxlen=1000)

        # Internal state (for ML, reaction tracking, etc.)
        self.state: Dict = {}

        logger.debug(f"Created Atom {self.uid}: {self.symbol} at {self.pos}, oxidation_state={self.oxidation_state}, charge={self.charge}")

    def set_oxidation_state(self, oxidation_state: int):
        """
        Set the oxidation state and update charge accordingly.

        Args:
            oxidation_state (int): New oxidation state.
        """
        self.oxidation_state = oxidation_state
        # For simplicity, charge equals oxidation state for main group elements
        # This could be made more sophisticated based on element type
        self.charge = float(oxidation_state)
        logger.debug(f"Atom {self.uid} oxidation state set to {oxidation_state}, charge={self.charge}")

    def add_bond(self, bond):
        """
        Add a bond reference to this atom.

        Args:
            bond (BondObj): Bond object connecting this atom to another.
        """
        if bond not in self.bonds:
            self.bonds.append(bond)

    def remove_bond(self, bond):
        """
        Remove a bond reference from this atom.

        Args:
            bond (BondObj): Bond object to remove.
        """
        try:
            self.bonds.remove(bond)
        except ValueError:
            pass

    def is_bonded_to(self, other: Atom) -> bool:
        """
        Check if this atom is bonded to another.

        Args:
            other (Atom): Another atom to check.

        Returns:
            bool: True if bonded.
        """
        # Delay import to avoid circular dependency
        from engine.bonds import BondObj
        return any(isinstance(b, BondObj) and (b.atom1 is other or b.atom2 is other) for b in self.bonds)

    def record(self, frame: int):
        """
        Record current position and state for visualization or logging.

        Args:
            frame (int): Current simulation frame number.
        """
        try:
            entry = {
                "frame": frame,
                "pos": self.pos.copy().tolist(),
                "vel": self.vel.copy().tolist(),
                "state": dict(self.state)
            }
            self.history.append(entry)
        except Exception:
            logger.exception(f"Failed to record history for Atom {self.uid}")

    def kinetic_energy(self) -> float:
        """
        Compute kinetic energy of the atom.

        Returns:
            float: 0.5 * mass * |velocity|^2
        """
        return 0.5 * self.mass * np.dot(self.vel, self.vel)

    def __repr__(self) -> str:
        return (
            f"<Atom {self.uid} symbol={self.symbol} pos={self.pos} "
            f"vel={self.vel} mass={self.mass:.3f}>"
        )