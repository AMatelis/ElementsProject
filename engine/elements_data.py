from pathlib import Path
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Default path to elements.json relative to the engine folder
ELEMENTS_JSON = Path(__file__).parent.parent / "data" / "elements.json"

# In-memory cache for element properties
ELEMENT_DATA: Dict[str, Dict[str, Any]] = {}


def load_elements(path: Path | str = None) -> Dict[str, Dict[str, Any]]:
    """
    Load elements.json into the ELEMENT_DATA dictionary.
    If path is not provided, uses the default ELEMENTS_JSON.
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
    """
    symbol = symbol.upper()
    if not ELEMENT_DATA:
        load_elements()
    return ELEMENT_DATA.get(symbol, {
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