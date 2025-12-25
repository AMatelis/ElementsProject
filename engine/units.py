"""
Physical units for the molecular simulation.

This module defines the nondimensional reduced units used throughout the simulation.
Reduced units make quantities dimensionless by scaling with reference values, improving
numerical stability and allowing for easier parameter tuning.

Reference units:
- Length: σ (LJ sigma parameter, typically ~0.1 nm)
- Mass: m (atomic mass unit, ~1.66e-27 kg)
- Time: τ = sqrt(m σ² / ε), where ε is LJ epsilon (~4.2e-21 J)
- Energy: ε (LJ epsilon)
- Temperature: ε/k_B, where k_B is Boltzmann constant

All simulation quantities are expressed in these reduced units:
- Positions: in units of σ
- Masses: in units of m
- Time: in units of τ
- Energies: in units of ε
- Temperatures: in units of ε/k_B

For plotting and output, labels should reference these units.
"""

# Reduced unit definitions (dimensionless reference values)
LENGTH_UNIT = 1.0  # reduced length unit (σ)
MASS_UNIT = 1.0    # reduced mass unit (m)
TIME_UNIT = 1.0    # reduced time unit (τ)
ENERGY_UNIT = 1.0  # reduced energy unit (ε)
TEMPERATURE_UNIT = 1.0  # reduced temperature unit (ε/k_B)

# Conversion factors to SI units (for reference)
# These are approximate values used in the simulation
SI_LENGTH = 1e-10  # meters (Angstrom)
SI_MASS = 1.66053906660e-27  # kg (unified atomic mass)
SI_TIME = 1e-12   # seconds (picosecond)
SI_ENERGY = 4.2e-21  # joules (typical LJ epsilon)
SI_TEMPERATURE = 120.0  # Kelvin (ε/k_B for typical LJ)

def format_unit(quantity: str, reduced: bool = True) -> str:
    """
    Format a unit string for plotting labels.

    Args:
        quantity: The physical quantity (e.g., 'length', 'time', 'energy')
        reduced: If True, use reduced units; if False, use SI units

    Returns:
        Formatted unit string for plot labels
    """
    if reduced:
        units = {
            'length': 'σ',
            'time': 'τ',
            'mass': 'm',
            'energy': 'ε',
            'temperature': 'ε/k_B',
            'velocity': 'σ/τ',
            'force': 'ε/σ',
            'pressure': 'ε/σ³'
        }
    else:
        units = {
            'length': 'Å',
            'time': 'ps',
            'mass': 'u',
            'energy': 'eV',
            'temperature': 'K',
            'velocity': 'Å/ps',
            'force': 'eV/Å',
            'pressure': 'eV/Å³'
        }

    return units.get(quantity.lower(), '')