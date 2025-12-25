"""
Standard chemical visualization conventions.
Provides consistent colors and radii for atomic elements.
"""

from typing import Dict, Optional

# Standard CPK (Corey-Pauling-Koltun) colors for elements
# Based on common chemistry visualization standards
ELEMENT_COLORS: Dict[str, str] = {
    'H': '#FFFFFF',   # White
    'He': '#D9FFFF',  # Light cyan
    'Li': '#CC80FF',  # Purple
    'Be': '#C2FF00',  # Yellow-green
    'B': '#FFB5B5',   # Pink
    'C': '#909090',   # Gray
    'N': '#3050F8',   # Blue
    'O': '#FF0D0D',   # Red
    'F': '#90E050',   # Light green
    'Ne': '#B3E3F5',  # Light blue
    'Na': '#AB5CF2',  # Purple
    'Mg': '#8AFF00',  # Green
    'Al': '#BFA6A6',  # Light gray
    'Si': '#F0C8A0',  # Tan
    'P': '#FF8000',   # Orange
    'S': '#FFFF30',   # Yellow
    'Cl': '#1FF01F',  # Green
    'Ar': '#80D1E3',  # Light blue
    'K': '#8F40D4',   # Purple
    'Ca': '#3DFF00',  # Green
    'Sc': '#E6E6E6',  # Silver
    'Ti': '#BFC2C7',  # Gray
    'V': '#A6A6AB',   # Gray
    'Cr': '#8A99C7',  # Blue-gray
    'Mn': '#9C7AC7',  # Purple-gray
    'Fe': '#E06633',  # Orange-brown
    'Co': '#F090A0',  # Pink
    'Ni': '#50D050',  # Green
    'Cu': '#C88033',  # Copper
    'Zn': '#7D80B0',  # Blue-gray
    'Ga': '#C28F8F',  # Light red
    'Ge': '#668F8F',  # Teal
    'As': '#BD80E3',  # Purple
    'Se': '#FFA100',  # Orange
    'Br': '#A62929',  # Brown
    'Kr': '#5CB8D1',  # Light blue
    'Rb': '#702EB0',  # Purple
    'Sr': '#00FF00',  # Green
    'Y': '#94FFFF',   # Cyan
    'Zr': '#94E0E0',  # Cyan
    'Nb': '#73C2C9',  # Blue-cyan
    'Mo': '#54B5B5',  # Teal
    'Tc': '#3B9E9E',  # Teal
    'Ru': '#248F8F',  # Dark teal
    'Rh': '#0A7D8C',  # Dark blue
    'Pd': '#006985',  # Dark blue
    'Ag': '#C0C0C0',  # Silver
    'Cd': '#FFD98F',  # Yellow
    'In': '#A67573',  # Brown
    'Sn': '#668080',  # Gray
    'Sb': '#9E63B5',  # Purple
    'Te': '#D47A00',  # Orange
    'I': '#940094',   # Purple
    'Xe': '#429EB0',  # Blue
    'Cs': '#57178F',  # Purple
    'Ba': '#00C900',  # Green
    'La': '#70D4FF',  # Light blue
    'Ce': '#FFFFC7',  # Yellow
    'Pr': '#D9FFC7',  # Light yellow
    'Nd': '#C7FFC7',  # Light green
    'Pm': '#A3FFC7',  # Light green
    'Sm': '#8FFFC7',  # Light green
    'Eu': '#61FFC7',  # Light green
    'Gd': '#45FFC7',  # Light green
    'Tb': '#30FFC7',  # Light green
    'Dy': '#1FFFC7',  # Light green
    'Ho': '#00FF9C',  # Green-cyan
    'Er': '#00E675',  # Green
    'Tm': '#00D452',  # Green
    'Yb': '#00BF38',  # Green
    'Lu': '#00AB24',  # Green
    'Hf': '#4DC2FF',  # Light blue
    'Ta': '#4DA6FF',  # Light blue
    'W': '#2194D6',   # Blue
    'Re': '#267DAB',  # Blue
    'Os': '#266696',  # Blue
    'Ir': '#175487',  # Dark blue
    'Pt': '#D0D0E0',  # Light gray
    'Au': '#FFD123',  # Gold
    'Hg': '#B8B8D0',  # Light blue-gray
    'Tl': '#A6544D',  # Brown
    'Pb': '#575961',  # Gray
    'Bi': '#9E4FB5',  # Purple
    'Po': '#AB5C00',  # Brown
    'At': '#754F45',  # Brown
    'Rn': '#428296',  # Blue
    'Fr': '#420066',  # Purple
    'Ra': '#007D00',  # Green
    'Ac': '#70ABFA',  # Light blue
    'Th': '#00BAFF',  # Blue
    'Pa': '#00A1FF',  # Blue
    'U': '#008FFF',   # Blue
    'Np': '#0080FF',  # Blue
    'Pu': '#006BFF',  # Blue
    'Am': '#545CF2',  # Purple-blue
    'Cm': '#785CE3',  # Purple
    'Bk': '#8A4FE3',  # Purple
    'Cf': '#A136D4',  # Purple
    'Es': '#B31FD4',  # Purple
    'Fm': '#B31FBA',  # Purple
    'Md': '#B30DA6',  # Purple
    'No': '#BD0D87',  # Purple
    'Lr': '#C70066',  # Purple-red
    'Rf': '#CC0059',  # Red
    'Db': '#D1004F',  # Red
    'Sg': '#D90045',  # Red
    'Bh': '#E00038',  # Red
    'Hs': '#E6002E',  # Red
    'Mt': '#EB0026',  # Red
}

# Covalent radii in Angstroms (Å)
# Based on standard values from chemistry literature
ELEMENT_RADII: Dict[str, float] = {
    'H': 0.31,
    'He': 0.28,
    'Li': 1.28,
    'Be': 0.96,
    'B': 0.84,
    'C': 0.76,
    'N': 0.71,
    'O': 0.66,
    'F': 0.57,
    'Ne': 0.58,
    'Na': 1.66,
    'Mg': 1.41,
    'Al': 1.21,
    'Si': 1.11,
    'P': 1.07,
    'S': 1.05,
    'Cl': 1.02,
    'Ar': 1.06,
    'K': 2.03,
    'Ca': 1.76,
    'Sc': 1.70,
    'Ti': 1.60,
    'V': 1.53,
    'Cr': 1.39,
    'Mn': 1.39,
    'Fe': 1.32,
    'Co': 1.26,
    'Ni': 1.24,
    'Cu': 1.32,
    'Zn': 1.22,
    'Ga': 1.22,
    'Ge': 1.20,
    'As': 1.19,
    'Se': 1.20,
    'Br': 1.20,
    'Kr': 1.16,
    'Rb': 2.20,
    'Sr': 1.95,
    'Y': 1.90,
    'Zr': 1.75,
    'Nb': 1.64,
    'Mo': 1.54,
    'Tc': 1.47,
    'Ru': 1.46,
    'Rh': 1.42,
    'Pd': 1.39,
    'Ag': 1.45,
    'Cd': 1.44,
    'In': 1.42,
    'Sn': 1.39,
    'Sb': 1.39,
    'Te': 1.38,
    'I': 1.39,
    'Xe': 1.40,
    'Cs': 2.44,
    'Ba': 2.15,
    'La': 2.07,
    'Ce': 2.04,
    'Pr': 2.03,
    'Nd': 2.01,
    'Pm': 1.99,
    'Sm': 1.98,
    'Eu': 1.98,
    'Gd': 1.96,
    'Tb': 1.94,
    'Dy': 1.92,
    'Ho': 1.92,
    'Er': 1.89,
    'Tm': 1.90,
    'Yb': 1.87,
    'Lu': 1.87,
    'Hf': 1.75,
    'Ta': 1.70,
    'W': 1.62,
    'Re': 1.51,
    'Os': 1.44,
    'Ir': 1.41,
    'Pt': 1.36,
    'Au': 1.36,
    'Hg': 1.32,
    'Tl': 1.45,
    'Pb': 1.46,
    'Bi': 1.48,
    'Po': 1.40,
    'At': 1.50,
    'Rn': 1.50,
    'Fr': 2.60,
    'Ra': 2.21,
    'Ac': 2.15,
    'Th': 2.06,
    'Pa': 2.00,
    'U': 1.96,
    'Np': 1.90,
    'Pu': 1.87,
    'Am': 1.80,
    'Cm': 1.69,
}


def get_element_color(symbol: str) -> str:
    """
    Get the standard CPK color for an element.

    Parameters:
    -----------
    symbol : str
        Element symbol (case-insensitive)

    Returns:
    --------
    str
        Hex color code (e.g., '#FF0000')
    """
    return ELEMENT_COLORS.get(symbol.upper(), '#808080')  # Default to gray


def get_element_radius(symbol: str) -> float:
    """
    Get the covalent radius for an element.

    Parameters:
    -----------
    symbol : str
        Element symbol (case-insensitive)

    Returns:
    --------
    float
        Covalent radius in Angstroms
    """
    return ELEMENT_RADII.get(symbol.upper(), 0.7)  # Default radius


def get_visual_properties(symbol: str) -> Dict[str, any]:
    """
    Get both color and radius for an element.

    Parameters:
    -----------
    symbol : str
        Element symbol (case-insensitive)

    Returns:
    --------
    dict
        Dictionary with 'color' and 'radius' keys
    """
    return {
        'color': get_element_color(symbol),
        'radius': get_element_radius(symbol)
    }


def get_bond_width(order: int) -> float:
    """
    Get line width for bond visualization based on bond order.

    Parameters:
    -----------
    order : int
        Bond order (1=single, 2=double, 3=triple)

    Returns:
    --------
    float
        Line width in points
    """
    return 1.0 + 0.5 * (order - 1)  # 1.0 for single, 1.5 for double, 2.0 for triple


def get_bond_color(age: float, max_age: float = 100.0, breaking: bool = False) -> str:
    """
    Get color for bond visualization based on age and breaking state.

    Parameters:
    -----------
    age : float
        Bond age in seconds
    max_age : float
        Maximum age for full fade (default: 100.0)
    breaking : bool
        Whether the bond is breaking (default: False)

    Returns:
    --------
    str
        Hex color code
    """
    if breaking:
        return '#FF4444'  # Red for breaking bonds

    # Fade from black to gray as age increases
    fade = min(age / max_age, 1.0)
    gray = int(255 * (1 - fade * 0.5))
    return f'#{gray:02x}{gray:02x}{gray:02x}'


def get_bond_visual_properties(order: int, age: float, max_age: float = 100.0, breaking: bool = False) -> Dict[str, any]:
    """
    Get complete visual properties for a bond.

    Parameters:
    -----------
    order : int
        Bond order
    age : float
        Bond age in seconds
    max_age : float
        Maximum age for fading
    breaking : bool
        Whether bond is breaking

    Returns:
    --------
    dict
        Dictionary with 'color' and 'width' keys
    """
    return {
        'color': get_bond_color(age, max_age, breaking),
        'width': get_bond_width(order)
    }