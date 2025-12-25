"""
Publication-Ready Metrics Module for Molecular Dynamics Simulation
Provides stable, normalized energy tracking and plotting for scientific visualization.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)


class EnergyMetrics:
    """
    Tracks and normalizes energy metrics for stable scientific visualization.

    Separates raw energies from display energies, applies normalization and smoothing.
    """

    def __init__(self, max_history: int = 1000, smoothing_window: int = 10):
        """
        Initialize energy metrics tracker.

        Args:
            max_history: Maximum number of data points to keep
            smoothing_window: Window size for moving average smoothing
        """
        self.max_history = max_history
        self.smoothing_window = smoothing_window

        # Raw energy data (stored for analysis)
        self.raw_kinetic: deque[float] = deque(maxlen=max_history)
        self.raw_potential: deque[float] = deque(maxlen=max_history)
        self.raw_total: deque[float] = deque(maxlen=max_history)

        # Normalized and smoothed data (for plotting)
        self.norm_kinetic: deque[float] = deque(maxlen=max_history)
        self.norm_potential: deque[float] = deque(maxlen=max_history)
        self.norm_total: deque[float] = deque(maxlen=max_history)

        # System size tracking for normalization
        self.atom_counts: deque[int] = deque(maxlen=max_history)
        self.bond_counts: deque[int] = deque(maxlen=max_history)

        # Statistics for stable plotting
        self._energy_stats = {
            'kinetic_min': float('inf'),
            'kinetic_max': float('-inf'),
            'potential_min': float('inf'),
            'potential_max': float('-inf'),
            'total_min': float('inf'),
            'total_max': float('-inf'),
        }

        # Initial reference values
        self.initial_total_energy = 0.0
        self.normalization_mode = 'per_atom'  # 'per_atom', 'per_bond', 'absolute'

    def update(self, physics_engine, atom_count: int, bond_count: int) -> None:
        """
        Update energy metrics from physics engine.

        Args:
            physics_engine: PhysicsEngine instance with energy calculation methods
            atom_count: Current number of atoms in system
            bond_count: Current number of bonds in system
        """
        try:
            # Get raw energies
            energies = physics_engine.total_energy()
            ke_raw = energies.get('kinetic', 0.0)
            pe_raw = energies.get('potential', 0.0)
            total_raw = energies.get('total', 0.0)

            # Store raw data
            self.raw_kinetic.append(ke_raw)
            self.raw_potential.append(pe_raw)
            self.raw_total.append(total_raw)

            # Store system size
            self.atom_counts.append(atom_count)
            self.bond_counts.append(bond_count)

            # Normalize energies
            ke_norm, pe_norm, total_norm = self._normalize_energies(
                ke_raw, pe_raw, total_raw, atom_count, bond_count
            )

            # Apply smoothing
            ke_smooth = self._moving_average(self.norm_kinetic, ke_norm)
            pe_smooth = self._moving_average(self.norm_potential, pe_norm)
            total_smooth = self._moving_average(self.norm_total, total_norm)

            # Store normalized/smoothed data
            self.norm_kinetic.append(ke_smooth)
            self.norm_potential.append(pe_smooth)
            self.norm_total.append(total_smooth)

            # Update statistics for stable plotting
            self._update_statistics(ke_smooth, pe_smooth, total_smooth)

            # Set initial energy reference
            if len(self.raw_total) == 1:
                self.initial_total_energy = total_raw

        except Exception as e:
            logger.exception(f"Error updating energy metrics: {e}")

    def _normalize_energies(self, ke: float, pe: float, total: float,
                          atom_count: int, bond_count: int) -> Tuple[float, float, float]:
        """
        Normalize energies based on system size.

        Returns:
            Tuple of (normalized_kinetic, normalized_potential, normalized_total)
        """
        if self.normalization_mode == 'per_atom' and atom_count > 0:
            # Normalize per atom
            norm_factor = atom_count
        elif self.normalization_mode == 'per_bond' and bond_count > 0:
            # Normalize per bond
            norm_factor = bond_count
        elif self.normalization_mode == 'per_degree' and atom_count > 0:
            # Normalize per degree of freedom (approximate)
            # 2D system: each atom has 2 position DOF + 2 velocity DOF = 4 DOF
            # But subtract constraints from bonds (each bond removes ~1 DOF)
            norm_factor = max(1, 4 * atom_count - bond_count)
        else:
            # Absolute energies
            norm_factor = 1.0

        return ke / norm_factor, pe / norm_factor, total / norm_factor

    def _moving_average(self, history: deque, new_value: float) -> float:
        """
        Apply moving average smoothing to new value.

        Args:
            history: Previous values in deque
            new_value: New raw value to smooth

        Returns:
            Smoothed value
        """
        if self.smoothing_window <= 1:
            return new_value

        # Add new value to history for smoothing calculation
        temp_history = list(history) + [new_value]

        # Use available data points (up to smoothing window)
        window_size = min(self.smoothing_window, len(temp_history))
        recent_values = temp_history[-window_size:]

        return float(np.mean(recent_values))

    def _update_statistics(self, ke: float, pe: float, total: float) -> None:
        """Update min/max statistics for stable axis limits."""
        self._energy_stats['kinetic_min'] = min(self._energy_stats['kinetic_min'], ke)
        self._energy_stats['kinetic_max'] = max(self._energy_stats['kinetic_max'], ke)
        self._energy_stats['potential_min'] = min(self._energy_stats['potential_min'], pe)
        self._energy_stats['potential_max'] = max(self._energy_stats['potential_max'], pe)
        self._energy_stats['total_min'] = min(self._energy_stats['total_min'], total)
        self._energy_stats['total_max'] = max(self._energy_stats['total_max'], total)

    def get_plot_data(self, max_points: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get smoothed, normalized data suitable for plotting.

        Args:
            max_points: Maximum number of points to return (for performance)

        Returns:
            Dict with 'kinetic', 'potential', 'total' arrays and 'x' (time steps)
        """
        n_points = len(self.norm_total)
        if n_points == 0:
            return {'x': np.array([]), 'kinetic': np.array([]),
                   'potential': np.array([]), 'total': np.array([])}

        if max_points and n_points > max_points:
            # Return last max_points for recent history
            start_idx = n_points - max_points
            x = np.arange(start_idx, n_points)
            kinetic = np.array(list(self.norm_kinetic)[start_idx:])
            potential = np.array(list(self.norm_potential)[start_idx:])
            total = np.array(list(self.norm_total)[start_idx:])
        else:
            x = np.arange(n_points)
            kinetic = np.array(list(self.norm_kinetic))
            potential = np.array(list(self.norm_potential))
            total = np.array(list(self.norm_total))

        return {
            'x': x,
            'kinetic': kinetic,
            'potential': potential,
            'total': total
        }

    def get_axis_limits(self, energy_type: str, padding: float = 0.1) -> Tuple[float, float]:
        """
        Get stable axis limits for plotting to prevent autoscaling spikes.

        Args:
            energy_type: 'kinetic', 'potential', or 'total'
            padding: Fraction of range to add as padding

        Returns:
            Tuple of (ymin, ymax)
        """
        if energy_type not in self._energy_stats:
            return (-1, 1)  # Default range

        min_val = self._energy_stats[f'{energy_type}_min']
        max_val = self._energy_stats[f'{energy_type}_max']

        if min_val == float('inf') or max_val == float('-inf'):
            return (-1, 1)  # No data yet

        if min_val == max_val:
            # Flat line, add some padding
            center = min_val
            return (center - 0.5, center + 0.5)

        # Add padding
        range_val = max_val - min_val
        padding_val = padding * range_val
        return (min_val - padding_val, max_val + padding_val)

    def get_current_summary(self) -> Dict[str, Any]:
        """Get current energy summary for display."""
        if not self.norm_total:
            return {
                'kinetic': 0.0, 'potential': 0.0, 'total': 0.0,
                'drift': 0.0, 'normalization': self.normalization_mode
            }

        current_ke = self.norm_kinetic[-1] if self.norm_kinetic else 0.0
        current_pe = self.norm_potential[-1] if self.norm_potential else 0.0
        current_total = self.norm_total[-1] if self.norm_total else 0.0

        # Calculate energy drift from initial value
        initial_norm = self._normalize_energies(
            self.initial_total_energy,
            0.0,  # We don't track initial PE separately
            self.initial_total_energy,
            self.atom_counts[0] if self.atom_counts else 1,
            self.bond_counts[0] if self.bond_counts else 0
        )[2] if self.initial_total_energy != 0 else 0.0

        drift = (current_total - initial_norm) / abs(initial_norm) if initial_norm != 0 else 0.0

        return {
            'kinetic': current_ke,
            'potential': current_pe,
            'total': current_total,
            'drift': drift,
            'normalization': self.normalization_mode,
            'smoothing_window': self.smoothing_window
        }

    def reset(self):
        """Reset all metrics data."""
        self.raw_kinetic.clear()
        self.raw_potential.clear()
        self.raw_total.clear()
        self.norm_kinetic.clear()
        self.norm_potential.clear()
        self.norm_total.clear()
        self.atom_counts.clear()
        self.bond_counts.clear()

        # Reset statistics
        for key in self._energy_stats:
            if 'min' in key:
                self._energy_stats[key] = float('inf')
            else:
                self._energy_stats[key] = float('-inf')

        self.initial_total_energy = 0.0

    def set_normalization_mode(self, mode: str):
        """
        Set energy normalization mode.

        Args:
            mode: 'per_atom', 'per_bond', 'per_degree', or 'absolute'
        """
        valid_modes = ['per_atom', 'per_bond', 'per_degree', 'absolute']
        if mode in valid_modes:
            self.normalization_mode = mode
            logger.info(f"Energy normalization mode set to: {mode}")
        else:
            logger.warning(f"Invalid normalization mode: {mode}. Using 'per_atom'")

    def get_normalization_description(self) -> str:
        """Get human-readable description of current normalization."""
        descriptions = {
            'per_atom': 'per atom',
            'per_bond': 'per bond',
            'per_degree': 'per degree of freedom',
            'absolute': 'absolute'
        }
        return descriptions.get(self.normalization_mode, self.normalization_mode)