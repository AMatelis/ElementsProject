from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import numpy as np
import math
import logging
import time

from .atoms import Atom
from .bonds import BondObj
from .spatial_hash import SpatialHash

logger = logging.getLogger(__name__)

# -----------------------
# Default physical constants and simulation config
# -----------------------
# These defaults are "reduced" units suitable for interactive simulation.
DEFAULT_DT = 1e-3
DEFAULT_TEMPERATURE = 300.0
DEFAULT_GAMMA = 0.6  # friction coefficient for Langevin thermostat (softer damping)
DEFAULT_CELL_SIZE = 0.06

# Non-bonded interactions
LJ_EPSILON = 0.05         # depth of LJ well (reduced units) — softened for smoother motion
LJ_SIGMA_FACTOR = 0.95    # factor applied to sum of visual radii to get sigma
NONBONDED_CUTOFF = 0.35   # neighbor cutoff radius (reduced units)

# Coulomb (reduced; user can scale using element_data or external factors)
COULOMB_K = 1.0  # reduced Coulomb prefactor (tune to get reasonable behavior)

# Bond force safety limits
DEFAULT_MAX_FORCE = 1e5

# small eps to avoid div-by-zero
_EPS = 1e-12

# -----------------------
# PhysicsEngine
# -----------------------

class PhysicsEngine:
    """
    PhysicsEngine manages positions, velocities, forces and integration.

    Typical usage:
        engine = PhysicsEngine(atoms, bonds, dt=1e-3)
        engine.step()   # advances simulation by dt
        engine.set_temperature(400.0)
        engine.enable_periodic(True)
    """

    def __init__(self,
                 atoms: Optional[List[Atom]] = None,
                 bonds: Optional[List[BondObj]] = None,
                 dt: float = DEFAULT_DT,
                 temperature: float = DEFAULT_TEMPERATURE,
                 gamma: float = DEFAULT_GAMMA,
                 seed: Optional[int] = None,
                 periodic: bool = False,
                 box_size: float = 1.0,
                 cell_size: float = DEFAULT_CELL_SIZE):
        # Simulation state
        self.atoms: List[Atom] = atoms if atoms is not None else []
        self.bonds: List[BondObj] = bonds if bonds is not None else []

        # Integration and thermostat parameters
        self.dt = float(dt)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.periodic = bool(periodic)
        self.box_size = float(box_size)

        # neighbor acceleration
        self.spatial = SpatialHash(cell_size=cell_size)

        # bookkeeping
        self.frame = 0
        self.time = 0.0

        # random number generator for thermostat
        # Create a dedicated Generator so Langevin noise is reproducible when seed provided
        self.rng = np.random.default_rng(seed=seed)

        # energy diagnostics cache
        self._last_energy: Dict[str, float] = {}

        logger.info(f"PhysicsEngine initialized dt={self.dt} T={self.temperature} periodic={self.periodic}")

    # -----------------------
    # Scene management
    # -----------------------
    def add_atom(self, atom: Atom) -> None:
        """Append an Atom to the simulation."""
        self.atoms.append(atom)

    def remove_atom(self, atom: Atom) -> None:
        """Remove an atom (and any bonds referencing it must be handled externally)."""
        try:
            self.atoms.remove(atom)
        except ValueError:
            logger.warning("Attempted to remove atom not present in engine.")

    def rebuild_spatial(self) -> None:
        """Rebuild the spatial hash from current atom positions."""
        self.spatial.clear()
        for a in self.atoms:
            # If periodic, wrap positions to canonical box before hashing
            if self.periodic:
                a.pos = self._wrap_position(a.pos)
            self.spatial.add(a)

    # -----------------------
    # Integration step
    # -----------------------
    def step(self) -> None:
        """
        Advance the simulation by one time-step (self.dt).

        Algorithm:
          - Rebuild spatial hash
          - Compute forces (bond springs, non-bonded LJ, Coulomb)
          - Integrate velocities & positions with Velocity-Verlet-like scheme
          - Apply Langevin thermostat as stochastic force + friction (optional)
          - Update time/frame
        """
        if not self.atoms:
            return

        # 1) spatial index (for neighbor queries)
        self.rebuild_spatial()

        # 2) clear forces
        forces = {a: np.zeros_like(a.pos) for a in self.atoms}

        # 3) bond spring forces
        for b in list(self.bonds):
            try:
                f_on_1, f_on_2 = self._bond_force(b)
                forces[b.atom1] += f_on_1
                forces[b.atom2] += f_on_2
            except Exception:
                logger.exception("Error computing bond force for bond %s", repr(b))

        # 4) non-bonded forces (LJ + Coulomb) using neighbor lists
        cutoff = NONBONDED_CUTOFF
        for a in self.atoms:
            neighbors = self.spatial.neighbors(a, radius=cutoff)
            for n in neighbors:
                # skip if same atom
                if n is a:
                    continue
                # avoid double accounting: compute only for ordered pair by id
                if id(n) <= id(a):
                    continue
                # compute pairwise forces
                try:
                    f_a_on_b, f_b_on_a = self._pairwise_nonbonded_forces(a, n)
                except Exception:
                    logger.exception("Error in pairwise force computation between %s and %s", a.uid, n.uid)
                    continue
                # apply forces (note signs)
                forces[a] += f_a_on_b
                forces[n] += f_b_on_a

        # 5) Langevin thermostat (stochastic force + friction) and integrate (Velocity-Verlet-ish)
        dt = self.dt
        for a in self.atoms:
            # compute random noise (Langevin): sqrt(2 * gamma * kT / m) * Normal(0,1)
            if self.gamma > 0.0:
                sigma = math.sqrt(max(0.0, 2.0 * self.gamma * self._kT() / max(a.mass, 1e-12)))
                rand_kick = self.rng.normal(0.0, sigma, size=a.pos.shape)
            else:
                rand_kick = np.zeros_like(a.pos)

            friction = -self.gamma * a.vel
            total_force = forces[a] + rand_kick + friction

            # acceleration
            acc = total_force / max(a.mass, 1e-12)

            # integrate velocities and positions
            # velocity half-step
            a.vel += acc * dt
            # position update
            a.pos += a.vel * dt

            # optional position wrapping / clamping
            if self.periodic:
                a.pos = self._wrap_position(a.pos)
            else:
                # clamp to [0, box_size]
                a.pos = np.clip(a.pos, 0.0, self.box_size)

            # tiny damping to improve stability
            a.vel *= 0.999

        # 6) post-step bookkeeping
        self.frame += 1
        self.time += dt

    # -----------------------
    # Force computations
    # -----------------------
    def _bond_force(self, bond: BondObj) -> Tuple[np.ndarray, np.ndarray]:
        """
        Harmonic spring force for bond: F = -k * (r - r0) * direction
        Returns: (force_on_atom1, force_on_atom2)
        """
        a = bond.atom1
        b = bond.atom2
        delta = b.pos - a.pos
        if self.periodic:
            delta = self._minimum_image(delta)
        dist = np.linalg.norm(delta) + _EPS
        dirn = delta / dist
        fmag = bond.k_spring * (dist - bond.rest_length)
        f = fmag * dirn
        # clamp extremely large forces for numerical safety
        f = np.clip(f, -DEFAULT_MAX_FORCE, DEFAULT_MAX_FORCE)
        return f, -f

    def _pairwise_nonbonded_forces(self, a: Atom, b: Atom) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute LJ + Coulomb forces between two atoms (non-bonded).
        Returns force on a (f_ab) and force on b (f_ba = -f_ab).
        """
        delta = b.pos - a.pos
        if self.periodic:
            delta = self._minimum_image(delta)
        r = np.linalg.norm(delta) + _EPS
        # simple exclusion: skip if atoms are bonded (bonded forces handled by bonds)
        if any((bo.atom1 is b or bo.atom2 is b) for bo in a.bonds):
            # still include a softened short-range repulsion to avoid overlap
            # return small repulsive force if extremely close
            if r < 1e-3:
                rep = (delta / (r + _EPS)) * 1e-2
                return rep, -rep
            return np.zeros_like(delta), np.zeros_like(delta)

        # Lennard-Jones (12-6) using sigma derived from radii
        sigma = (a.radius + b.radius) * LJ_SIGMA_FACTOR
        if r > 0 and r < NONBONDED_CUTOFF:
            sr6 = (sigma / r) ** 6
            lj_mag = 24.0 * LJ_EPSILON * (2.0 * sr6 * sr6 - sr6) / (r + _EPS)
            lj_force = lj_mag * (delta / r)
        else:
            lj_force = np.zeros_like(delta)

        # Coulomb (reduced)
        if abs(a.charge) + abs(b.charge) > 1e-12 and r < NONBONDED_CUTOFF:
            coul_mag = COULOMB_K * (a.charge * b.charge) / (r * r + _EPS)
            coul_force = coul_mag * (delta / r)
        else:
            coul_force = np.zeros_like(delta)

        f_ab = lj_force + coul_force
        # numerical safety clamp
        f_ab = np.clip(f_ab, -DEFAULT_MAX_FORCE, DEFAULT_MAX_FORCE)
        return f_ab, -f_ab

    # -----------------------
    # Energy diagnostics
    # -----------------------
    def kinetic_energy(self) -> float:
        """Total kinetic energy of all atoms."""
        ke = 0.0
        for a in self.atoms:
            ke += 0.5 * a.mass * np.dot(a.vel, a.vel)
        return float(ke)

    def potential_energy(self) -> float:
        """
        Compute approximate potential energy: bond springs + pairwise LJ + Coulomb.
        WARNING: O(n^2) if spatial hash not used for pairs; here we use neighbor queries.
        """
        pe_bond = 0.0
        for b in self.bonds:
            # harmonic energy 0.5 * k * (r - r0)^2
            r = np.linalg.norm(b.atom2.pos - b.atom1.pos) + _EPS
            pe_bond += 0.5 * b.k_spring * (r - b.rest_length) ** 2

        pe_nb = 0.0
        # accumulate pairwise energies once per pair
        seen = set()
        for a in self.atoms:
            neighs = self.spatial.neighbors(a, radius=NONBONDED_CUTOFF)
            for n in neighs:
                if a is n:
                    continue
                key = tuple(sorted((a.uid, n.uid)))
                if key in seen:
                    continue
                seen.add(key)
                r = np.linalg.norm(n.pos - a.pos) + _EPS
                # LJ potential: 4*eps * ((sigma/r)^12 - (sigma/r)^6)
                sigma = (a.radius + n.radius) * LJ_SIGMA_FACTOR
                if r < NONBONDED_CUTOFF:
                    sr6 = (sigma / r) ** 6
                    pe_nb += 4.0 * LJ_EPSILON * (sr6 * sr6 - sr6)
                # Coulomb
                if abs(a.charge) + abs(n.charge) > 1e-12:
                    pe_nb += COULOMB_K * (a.charge * n.charge) / (r + _EPS)
        total_pe = float(pe_bond + pe_nb)
        return total_pe

    def total_energy(self) -> Dict[str, float]:
        """Return a dict with kinetic, potential, and total energies."""
        ke = self.kinetic_energy()
        pe = self.potential_energy()
        tot = ke + pe
        self._last_energy = {"kinetic": ke, "potential": pe, "total": tot}
        return self._last_energy

    # -----------------------
    # Thermostat helpers
    # -----------------------
    def _kT(self) -> float:
        """
        Reduced kT value for thermostat scaling. This is user-tunable.
        For interactive / reduced-unit sims we simply map temperature -> multiplier.
        """
        # keep it simple and stable: map 300 K -> 1.0
        return max(1e-6, self.temperature / DEFAULT_TEMPERATURE)

    def set_temperature(self, T: float) -> None:
        """Set simulation temperature (affects Langevin noise scale)."""
        self.temperature = float(T)
        logger.info(f"PhysicsEngine temperature set to {self.temperature}")

    def set_gamma(self, gamma: float) -> None:
        """Set Langevin friction coefficient."""
        self.gamma = float(gamma)
        logger.info(f"PhysicsEngine gamma set to {self.gamma}")

    # -----------------------
    # Boundary / periodic helpers
    # -----------------------
    def enable_periodic(self, enable: bool = True, box_size: Optional[float] = None) -> None:
        self.periodic = bool(enable)
        if box_size is not None:
            self.box_size = float(box_size)
        logger.info(f"PhysicsEngine periodic={self.periodic} box_size={self.box_size}")

    def _wrap_position(self, pos: np.ndarray) -> np.ndarray:
        """Wrap position into [0, box_size) for periodic boundary conditions."""
        # works for both >2D shapes as we only use 2D
        wrapped = np.mod(pos, self.box_size)
        return wrapped

    def _minimum_image(self, delta: np.ndarray) -> np.ndarray:
        """
        Apply minimum image convention to a displacement vector for periodic boxes.
        """
        half = 0.5 * self.box_size
        # shift components into [-half, half]
        delta = (delta + half) % self.box_size - half
        return delta

    # -----------------------
    # Utilities & debugging
    # -----------------------
    def summary(self) -> str:
        """Return a short textual summary of engine state."""
        return (f"PhysicsEngine frame={self.frame} atoms={len(self.atoms)} bonds={len(self.bonds)} "
                f"T={self.temperature} dt={self.dt} periodic={self.periodic}")

    def debug_dump_positions(self) -> Dict[str, Tuple[float, float]]:
        """Return a uid->position mapping for debugging / visualization."""
        return {a.uid: (float(a.pos[0]), float(a.pos[1])) for a in self.atoms}