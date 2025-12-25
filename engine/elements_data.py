from pathlib import Path
import json
import logging
from typing import Dict, Any, Union

logger = logging.getLogger(__name__)

# Default path to elements.json relative to the engine folder
ELEMENTS_JSON: Path = Path(__file__).parent.parent / "data" / "elements.json"

# In-memory cache for element properties
ELEMENT_DATA: Dict[str, Dict[str, Any]] = {}

# Valence and oxidation state data for common elements
ELEMENT_VALENCE_DATA: Dict[str, Dict[str, Any]] = {
    "H": {"valence_electrons": 1, "common_oxidation_states": [1, -1], "max_bonds": 1},
    "C": {"valence_electrons": 4, "common_oxidation_states": [4, 3, 2, 1, -1, -2, -3, -4], "max_bonds": 4},
    "N": {"valence_electrons": 5, "common_oxidation_states": [5, 4, 3, 2, 1, -1, -2, -3], "max_bonds": 3},
    "O": {"valence_electrons": 6, "common_oxidation_states": [2, 1, -1, -2], "max_bonds": 2},
    "F": {"valence_electrons": 7, "common_oxidation_states": [1, -1], "max_bonds": 1},
    "P": {"valence_electrons": 5, "common_oxidation_states": [5, 4, 3, 1, -1, -2, -3], "max_bonds": 5},
    "S": {"valence_electrons": 6, "common_oxidation_states": [6, 4, 2, -2], "max_bonds": 6},
    "Cl": {"valence_electrons": 7, "common_oxidation_states": [7, 5, 3, 1, -1], "max_bonds": 1},
    "Br": {"valence_electrons": 7, "common_oxidation_states": [7, 5, 3, 1, -1], "max_bonds": 1},
    "I": {"valence_electrons": 7, "common_oxidation_states": [7, 5, 3, 1, -1], "max_bonds": 1},
    "Na": {"valence_electrons": 1, "common_oxidation_states": [1], "max_bonds": 1},
    "Mg": {"valence_electrons": 2, "common_oxidation_states": [2], "max_bonds": 2},
    "Al": {"valence_electrons": 3, "common_oxidation_states": [3], "max_bonds": 3},
    "Si": {"valence_electrons": 4, "common_oxidation_states": [4], "max_bonds": 4},
    "K": {"valence_electrons": 1, "common_oxidation_states": [1], "max_bonds": 1},
    "Ca": {"valence_electrons": 2, "common_oxidation_states": [2], "max_bonds": 2},
    "Fe": {"valence_electrons": 8, "common_oxidation_states": [3, 2], "max_bonds": 6},
    "Cu": {"valence_electrons": 11, "common_oxidation_states": [2, 1], "max_bonds": 4},
    "Zn": {"valence_electrons": 12, "common_oxidation_states": [2], "max_bonds": 2},
    "Ag": {"valence_electrons": 11, "common_oxidation_states": [1], "max_bonds": 4},
    "Au": {"valence_electrons": 11, "common_oxidation_states": [3, 1], "max_bonds": 4},
}


def load_elements(path: Union[Path, str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load elements.json into the ELEMENT_DATA dictionary.
    If path is not provided, uses the default ELEMENTS_JSON.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        A mapping from element symbol (uppercase) to its properties.
    """
    global ELEMENT_DATA
    if path is None:
        path = ELEMENTS_JSON

    try:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        out: Dict[str, Dict[str, Any]] = {}

        # Support different JSON formats
        if isinstance(raw, dict) and "elements" in raw:
            for el in raw["elements"]:
                if "symbol" in el:
                    out[el["symbol"].upper()] = el
        elif isinstance(raw, dict):
            for k, v in raw.items():
                out[str(k).upper()] = v if isinstance(v, dict) else {"symbol": str(k).upper()}

        ELEMENT_DATA = out
        logger.info(f"Loaded {len(ELEMENT_DATA)} elements from {path}")

    except Exception as e:
        logger.exception(f"Failed to load elements.json from {path}: {e}")
        ELEMENT_DATA = {}

    return ELEMENT_DATA


def get_element(symbol: str) -> Dict[str, Any]:
    """
    Return element properties by symbol. If missing, return a safe default.
    Automatically loads ELEMENT_DATA if it is empty.

    Parameters
    ----------
    symbol : str
        The chemical symbol of the element (case-insensitive).

    Returns
    -------
    Dict[str, Any]
        Element properties dictionary.
    """
    symbol = symbol.upper()
    if not ELEMENT_DATA:
        load_elements()

    element = ELEMENT_DATA.get(symbol, {
        "symbol": symbol,
        "atomic_mass": 12.0,
        "covalent_radius": 0.7,
        "electronegativity_pauling": 0.0,
        "charge": 0.0,
        "group": 0,
        "period": 0,
        "cpk-hex": "#808080",
        "category": "unknown"
    })

    # Merge valence data if available
    valence_data = ELEMENT_VALENCE_DATA.get(symbol, {})
    element.update(valence_data)

    return element