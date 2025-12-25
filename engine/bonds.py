from __future__ import annotations 
from typing import Optional, Tuple
import numpy as np
import math
import logging

logger = logging.getLogger(__name__)

# -----------------------
# Bond constants
# -----------------------
DEFAULT_BOND_SPRING = 5.0  # reduced for stability
BOND_BREAK_FORCE = 150.0     # arbitrary units
BOND_BREAK_DISTANCE_FACTOR = 1.5  # fraction of rest length

# -----------------------
# Bond object
# -----------------------

class BondObj:
    """
    Represents a bond between two atoms.
    """
    def __init__(self,
                 atom1,
                 atom2,
                 order: int = 1):
        """
        Initialize a bond.

        Args:
            atom1 (Atom): First atom in the bond.
            atom2 (Atom): Second atom in the bond.
            order (int): Bond order (1=single, 2=double, 3=triple)
        """
        # Delay imports to avoid circular dependency
        from engine import atoms
        from engine.elements_data import get_element

        if atom1 is atom2:
            raise ValueError("Cannot bond an atom to itself")

        self.atom1: atoms.Atom = atom1
        self.atom2: atoms.Atom = atom2
        self.order: int = max(1, min(order, 3))
        self.rest_length: float = self.estimate_rest_length()
        self.k_spring: float = self.estimate_spring_constant()
        self.time_of_creation: float = 0.0  # timestamp for visualization effects

        # Register bond with atoms
        atom1.add_bond(self)
        atom2.add_bond(self)

        logger.debug(f"Created Bond: {atom1.uid}-{atom2.uid} order={self.order} rest_length={self.rest_length:.3f}")

    def get_bond_type(self) -> str:
        """
        Determine if bond is ionic or covalent based on electronegativity difference and charges.

        Returns:
            str: "ionic", "covalent", or "polar_covalent"
        """
        en1 = getattr(self.atom1, "en", 0.0)
        en2 = getattr(self.atom2, "en", 0.0)
        charge1 = getattr(self.atom1, "charge", 0.0)
        charge2 = getattr(self.atom2, "charge", 0.0)

        en_diff = abs(en1 - en2)

        # Ionic bonds: large electronegativity difference (>1.7) or opposite charges
        if en_diff > 1.7 or (charge1 * charge2 < -0.5):
            return "ionic"
        # Polar covalent: moderate difference (0.5-1.7)
        elif en_diff > 0.5:
            return "polar_covalent"
        # Covalent: small difference
        else:
            return "covalent"

    def estimate_rest_length(self) -> float:
        """
        Estimate rest length of bond based on covalent radii and bond order.

        Returns:
            float: Rest length in simulation units
        """
        r1 = getattr(self.atom1, "radius", 0.7)
        r2 = getattr(self.atom2, "radius", 0.7)
        base_length = r1 + r2
        # simple adjustment for bond order (shorter for multiple bonds)
        adjustment = 0.9 ** (self.order - 1)
        return base_length * adjustment

    def estimate_spring_constant(self) -> float:
        """
        Estimate bond spring constant (stiffness) based on bond order and atom masses.

        Returns:
            float: Spring constant
        """
        mass_factor = (self.atom1.mass * self.atom2.mass) / (self.atom1.mass + self.atom2.mass)
        k = DEFAULT_BOND_SPRING * math.sqrt(self.order) * (mass_factor / 12.0)
        return k

    def current_length(self) -> float:
        """
        Current distance between bonded atoms.

        Returns:
            float: Distance
        """
        delta = self.atom2.pos - self.atom1.pos
        return np.linalg.norm(delta)

    def current_force_magnitude(self) -> float:
        """
        Approximate instantaneous bond force magnitude using Hooke's law.

        Returns:
            float: Force magnitude
        """
        delta = self.atom2.pos - self.atom1.pos
        dist = np.linalg.norm(delta) + 1e-12
        force = self.k_spring * (dist - self.rest_length)
        return abs(force)

    def estimate_bond_energy(self) -> float:
        """
        Estimate bond energy in arbitrary units based on atom electronegativities and bond order.

        Returns:
            float: Energy estimate
        """
        en1 = getattr(self.atom1, "en", 0.0)
        en2 = getattr(self.atom2, "en", 0.0)
        en_diff = abs(en1 - en2)
        # simplistic heuristic: higher difference = polar = stronger, multiple bonds stronger
        energy = (1.0 / (1.0 + en_diff)) * (1.0 + 0.5 * (self.order - 1))
        return energy

    def should_break(self, temperature: float = 300.0) -> Tuple[bool, float]:
        """
        Decide if a bond should break based on stretch, force, and temperature.

        Args:
            temperature (float): simulation temperature (affects probability)

        Returns:
            Tuple[bool, float]: (break_flag, severity_score)
        """
        stretch = self.current_length() / (self.rest_length + 1e-12)
        force_mag = self.current_force_magnitude()
        temp_factor = temperature / 300.0

        severity = max(0.0, (stretch - 1.0) * 0.5 + 0.02 * force_mag + 0.05 * (temp_factor - 1.0))
        # deterministic thresholds
        if force_mag > BOND_BREAK_FORCE or stretch > BOND_BREAK_DISTANCE_FACTOR:
            return True, min(severity, 1.0)
        # probabilistic break
        import random
        if severity > 0.5 and random.random() < min(0.5, severity):
            return True, min(severity, 1.0)
        return False, min(severity, 1.0)

    def __repr__(self):
        return f"<Bond {self.atom1.uid}-{self.atom2.uid} order={self.order} rest_len={self.rest_length:.3f}>"

# -----------------------
# Utility functions
# -----------------------

def can_form_bond(atom1, atom2, temperature: float = 300.0) -> Tuple[bool, float]:
    """
    Heuristic: decide if two atoms can form a bond, returns (can_form, score)
    Considers both covalent (electronegativity) and ionic (charge) bonding.
    """
    from engine import atoms

    delta = atom2.pos - atom1.pos
    dist = np.linalg.norm(delta)
    max_dist = atom1.radius + atom2.radius + 0.15  # arbitrary threshold
    if dist > max_dist:
        return False, 0.0

    # Covalent bonding score (electronegativity difference)
    en_diff = abs(atom1.en - atom2.en)
    covalent_score = 1.0 / (1.0 + en_diff)

    # Ionic bonding score (opposite charges attract)
    charge1 = getattr(atom1, 'charge', 0.0)
    charge2 = getattr(atom2, 'charge', 0.0)
    ionic_score = max(0.0, -charge1 * charge2)  # positive when opposite charges

    # Combined bonding score
    total_score = covalent_score + ionic_score

    kT_red = max(0.001, temperature / 300.0)
    prob = 1.0 - math.exp(-total_score / (kT_red + 1e-12))
    prob = float(max(0.0, min(1.0, prob)))
    return (prob > 0.15), prob

def sanitize_bonds(bonds: list, atoms_list: list):
    """
    Ensure no atom exceeds its allowed max bonds.
    """
    from engine import atoms
    from engine.elements_data import get_element

    neigh_count = {}
    for b in bonds:
        neigh_count[b.atom1.uid] = neigh_count.get(b.atom1.uid, 0) + 1
        neigh_count[b.atom2.uid] = neigh_count.get(b.atom2.uid, 0) + 1
    for b in list(bonds):
        element1 = get_element(b.atom1.symbol)
        element2 = get_element(b.atom2.symbol)
        max1 = element1.get('max_bonds', 8)
        max2 = element2.get('max_bonds', 8)
        if neigh_count.get(b.atom1.uid, 0) > max1 or neigh_count.get(b.atom2.uid, 0) > max2:
            try:
                bonds.remove(b)
                b.atom1.remove_bond(b)
                b.atom2.remove_bond(b)
            except ValueError:
                pass
            neigh_count[b.atom1.uid] = max(0, neigh_count.get(b.atom1.uid, 0) - 1)
            neigh_count[b.atom2.uid] = max(0, neigh_count.get(b.atom2.uid, 0) - 1)