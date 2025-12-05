
from typing import Dict, Tuple, Optional
from engine.elements_data import ELEMENT_DATA, load_elements

# -----------------------------
# Utility functions
# -----------------------------

def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    """
    Convert a hex color string (#RRGGBB or RRGGBB) to RGB tuple scaled 0-1.
    """
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) != 6:
        raise ValueError(f"Invalid hex color: {hex_color}")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return (r, g, b)


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    """
    Convert an RGB tuple scaled 0-1 to hex string #RRGGBB
    """
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0, min(1, rgb[0])) * 255),
        int(max(0, min(1, rgb[1])) * 255),
        int(max(0, min(1, rgb[2])) * 255)
    )


# -----------------------------
# Element color management
# -----------------------------

def get_element_color(symbol: str, fallback: str = '#888888') -> str:
    """
    Return the hex color string for an element symbol.
    - Looks up ELEMENT_DATA first
    - Fallback if not found
    """
    symbol = symbol.upper()
    elem = ELEMENT_DATA.get(symbol)
    if elem:
        color = elem.get('cpk-hex', fallback)
        if not color.startswith('#'):
            color = f"#{color}"
        return color
    return fallback


def get_element_rgb(symbol: str, fallback: Tuple[float,float,float]=(0.5,0.5,0.5)) -> Tuple[float,float,float]:
    """
    Return RGB tuple (0-1) for an element symbol.
    """
    try:
        hex_col = get_element_color(symbol)
        return hex_to_rgb(hex_col)
    except Exception:
        return fallback


def set_element_color(symbol: str, hex_color: str):
    """
    Update or add a color for a specific element in ELEMENT_DATA
    """
    if not symbol or not hex_color:
        raise ValueError("Both symbol and hex_color must be provided")
    if not hex_color.startswith('#'):
        hex_color = f"#{hex_color.strip()}"
    if symbol.upper() not in ELEMENT_DATA:
        # create minimal element entry
        ELEMENT_DATA[symbol.upper()] = {
            "name": symbol.upper(),
            "symbol": symbol.upper(),
            "atomic_number": 0,
            "atomic_mass": 0.0,
            "electronegativity_pauling": 0.0,
            "group": 0,
            "period": 0,
            "cpk-hex": hex_color,
            "category": "user-defined",
            "covalent_radius": 0.7
        }
    else:
        ELEMENT_DATA[symbol.upper()]['cpk-hex'] = hex_color


def get_palette(symbols: Optional[list] = None, fallback: str = '#888888') -> Dict[str, str]:
    """
    Return a dict mapping element symbols to hex colors.
    If symbols is None, return all elements in ELEMENT_DATA.
    """
    palette = {}
    targets = symbols if symbols else ELEMENT_DATA.keys()
    for s in targets:
        palette[s.upper()] = get_element_color(s, fallback=fallback)
    return palette


# -----------------------------
# Matplotlib / GUI helpers
# -----------------------------

def map_elements_to_colors(atoms: list, default: str = '#888888') -> list:
    """
    Given a list of Atom objects (must have .symbol), return list of hex colors
    """
    colors = []
    for a in atoms:
        try:
            colors.append(get_element_color(a.symbol, fallback=default))
        except Exception:
            colors.append(default)
    return colors


def map_elements_to_rgb(atoms: list, default: Tuple[float,float,float]=(0.5,0.5,0.5)) -> list:
    """
    Given a list of Atom objects (must have .symbol), return list of RGB tuples (0-1)
    """
    rgb_list = []
    for a in atoms:
        try:
            rgb_list.append(get_element_rgb(a.symbol, fallback=default))
        except Exception:
            rgb_list.append(default)
    return rgb_list


# -----------------------------
# Defaults / palette examples
# -----------------------------

DEFAULT_PALETTE = get_palette()

# Visual constants
VISUAL_BG = "#FFFFFF"
VISUAL_ATOM_OUTLINE = "#000000"
NODE_MIN_RADIUS = 3
NODE_MAX_RADIUS = 15

# Optional: simple test when run as script
if __name__ == "__main__":
    print("Example colors for H, O, C, N, Cl:")
    for el in ['H','O','C','N','Cl']:
        print(f"{el}: {get_element_color(el)}  ->  {get_element_rgb(el)}")