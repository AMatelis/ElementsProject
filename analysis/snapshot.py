"""
Static snapshot rendering for simulation states.
Provides GUI-independent visualization of atomic configurations.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Union, Optional
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.visuals import get_bond_visual_properties

# Set rendering defaults
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0


def render_snapshot(state: object, filename: str, format: str = 'png') -> None:
    """
    Render a static snapshot of the simulation state.

    Parameters:
    -----------
    state : object
        Simulation state object with attributes:
        - atoms: list of Atom objects with pos, radius, color
        - bonds: list of BondObj with atom1, atom2
        - physics.box_size: float for domain size
    filename : str
        Output filename (extension will be added if not present)
    format : str
        Output format ('png' or 'svg', default: 'png')
    """
    # Extract data from state
    try:
        atoms = state.atoms
        bonds = state.bonds
        box_size = state.physics.box_size
        current_time = state.time
    except AttributeError as e:
        raise ValueError(f"State object missing required attributes: {e}")

    # Create figure with square aspect
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    # Set square domain
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)

    # Remove all axes and ticks
    ax.axis('off')

    # Plot bonds first (behind atoms)
    for bond in bonds:
        x1, y1 = bond.atom1.pos[:2]
        x2, y2 = bond.atom2.pos[:2]
        width, color = get_bond_visual_properties(bond, current_time)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, zorder=1)

    # Plot atoms
    for atom in atoms:
        x, y = atom.pos[:2]
        # Scale radius for visibility (adjust multiplier as needed)
        size = atom.radius * 200  # pixels
        color = atom.color if atom.color.startswith('#') else f'#{atom.color}'
        ax.scatter(x, y, s=size, c=color, edgecolors='black',
                  linewidth=0.5, alpha=0.9, zorder=2)

    # Ensure filename has extension
    if not filename.lower().endswith(f'.{format}'):
        filename = f'{filename}.{format}'

    # Save with high quality
    if format == 'png':
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
    elif format == 'svg':
        plt.savefig(filename, format='svg', bbox_inches='tight', pad_inches=0)
    else:
        raise ValueError(f"Unsupported format: {format}")

    plt.close(fig)


def render_snapshot_from_data(atoms: list, bonds: list, box_size: float,
                             filename: str, format: str = 'png', time: float = 0.0) -> None:
    """
    Render snapshot from raw data (alternative interface).

    Parameters:
    -----------
    atoms : list
        List of atom dicts/objects with 'pos', 'radius', 'color' attributes
    bonds : list
        List of bond dicts/objects with 'atom1', 'atom2' attributes
    box_size : float
        Size of the simulation box
    filename : str
        Output filename
    format : str
        Output format ('png' or 'svg')
    time : float
        Current simulation time for bond lifetime calculation
    """
    # Create a simple state-like object
    class SimpleState:
        def __init__(self, atoms, bonds, box_size, time):
            self.atoms = atoms
            self.bonds = bonds
            self.physics = type('obj', (object,), {'box_size': box_size})()
            self.time = time

    state = SimpleState(atoms, bonds, box_size, time)
    render_snapshot(state, filename, format)


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    class MockAtom:
        def __init__(self, pos, radius, color):
            self.pos = np.array(pos)
            self.radius = radius
            self.color = color

    class MockBond:
        def __init__(self, atom1, atom2, order=1, time_of_creation=0.0):
            self.atom1 = atom1
            self.atom2 = atom2
            self.order = order
            self.time_of_creation = time_of_creation

    # Sample atoms (water molecule)
    atoms = [
        MockAtom([1.0, 1.0], 0.7, '#FF0000'),  # O
        MockAtom([0.5, 0.5], 0.3, '#FFFFFF'),  # H
        MockAtom([1.5, 0.5], 0.3, '#FFFFFF'),  # H
    ]

    # Sample bonds
    bonds = [
        MockBond(atoms[0], atoms[1]),
        MockBond(atoms[0], atoms[2]),
    ]

    # Render snapshot
    render_snapshot_from_data(atoms, bonds, 2.0, 'sample_snapshot', 'png', time=0.0)
    render_snapshot_from_data(atoms, bonds, 2.0, 'sample_snapshot', 'svg', time=0.0)

    print("Sample snapshots saved as PNG and SVG.")