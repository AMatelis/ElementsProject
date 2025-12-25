from __future__ import annotations
from typing import List, Optional
import numpy as np
import logging

from .atoms import Atom

logger = logging.getLogger(__name__)


class Integrator:
    """
    Base class for numerical integrators.
    """

    def __init__(self, dt: float, atoms: List[Atom], box_size: float, periodic: bool = False):
        self.dt = dt
        self.atoms = atoms
        self.box_size = box_size
        self.periodic = periodic

    def update_positions(self, accelerations: List[np.ndarray]) -> None:
        """
        Update positions using current velocities and accelerations.
        """
        raise NotImplementedError

    def update_velocities(self, accelerations: List[np.ndarray]) -> None:
        """
        Update velocities using accelerations.
        """
        raise NotImplementedError

    def _wrap_position(self, pos: np.ndarray) -> np.ndarray:
        """Wrap position to box for periodic boundary conditions."""
        if self.periodic:
            return (pos + self.box_size / 2) % self.box_size - self.box_size / 2
        return pos


class VelocityVerletIntegrator(Integrator):
    """
    Velocity Verlet integrator for molecular dynamics.
    """

    def __init__(self, dt: float, atoms: List[Atom], box_size: float, periodic: bool = False):
        super().__init__(dt, atoms, box_size, periodic)
        self.old_accelerations: Optional[List[np.ndarray]] = None

    def update_positions(self, accelerations: List[np.ndarray]) -> None:
        """
        Update positions: r(t+dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        Store accelerations for velocity update.
        """
        self.old_accelerations = accelerations.copy()
        dt = self.dt
        dt2 = dt * dt
        for a, acc in zip(self.atoms, accelerations):
            a.pos += a.vel * dt + 0.5 * acc * dt2
            a.pos = self._wrap_position(a.pos)

    def update_velocities(self, accelerations: List[np.ndarray]) -> None:
        """
        Update velocities: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
        """
        if self.old_accelerations is None:
            # Fallback if no old accelerations
            self.old_accelerations = accelerations
        dt = self.dt
        for a, old_acc, new_acc in zip(self.atoms, self.old_accelerations, accelerations):
            a.vel += 0.5 * (old_acc + new_acc) * dt


def create_integrator(integrator_type: str, dt: float, atoms: List[Atom], box_size: float, periodic: bool = False) -> Integrator:
    """
    Factory function to create an integrator instance.
    """
    if integrator_type.lower() == "velocity_verlet":
        return VelocityVerletIntegrator(dt, atoms, box_size, periodic)
    else:
        raise ValueError(f"Unknown integrator type: {integrator_type}")