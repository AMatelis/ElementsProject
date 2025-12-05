from __future__ import annotations
import re
from typing import Dict, List
import logging
from engine.elements_data import ELEMENT_DATA, load_elements

logger = logging.getLogger(__name__)

# ============================================================
#   Core Parsing Logic
# ============================================================

ELEMENT_REGEX = r"([A-Z][a-z]?)(\d*)"
PAREN_REGEX = r"\(([^()]+)\)(\d*)"


def _parse_simple(formula: str) -> Dict[str, int]:
    """
    Parse simple formulas without parentheses.
    Ex: 'H2O' → {'H': 2, 'O': 1}
    """
    parts = re.findall(ELEMENT_REGEX, formula)
    result = {}
    for symbol, count in parts:
        result[symbol] = result.get(symbol, 0) + (int(count) if count else 1)
    return result


def _validate_elements(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Ensure all element symbols exist in elements.json.
    If missing → create safe fallback entry and log.
    """
    if not ELEMENT_DATA:
        load_elements()

    cleaned = {}
    for sym, c in counts.items():
        s = sym.upper()

        if s not in ELEMENT_DATA:
            # fallback creation
            logger.warning(f"[formula_parser] Element '{s}' not found in elements.json — creating fallback entry.")
            ELEMENT_DATA[s] = {
                "symbol": s,
                "atomic_mass": 12.0,
                "covalent_radius": 0.7,
                "electronegativity_pauling": 0.0,
                "charge": 0.0,
                "group": 0,
                "period": 0,
                "category": "unknown",
                "cpk-hex": "#808080",
            }

        cleaned[s] = cleaned.get(s, 0) + c
    return cleaned


def parse_formula(formula: str) -> Dict[str, int]:
    """
    Parse a fully general chemical formula into an element → count dictionary.

    Supports:
        - Nested parentheses
        - Arbitrary multipliers
        - Multi-element symbols (Mg, Cl, Na, Si, etc.)
    """
    working = formula.replace(" ", "")
    if not working:
        raise ValueError("Cannot parse empty formula string.")

    # --------------------------
    # STEP 1: resolve parentheses iteratively
    # --------------------------
    while True:
        match = re.search(PAREN_REGEX, working)
        if not match:
            break

        group, multiplier = match.groups()
        multiplier = int(multiplier) if multiplier else 1

        parsed_inner = _parse_simple(group)

        # expand group like "H2O" etc
        expanded = ""
        for sym, cnt in parsed_inner.items():
            expanded += f"{sym}{cnt * multiplier}"

        # Replace the parenthetical block
        start, end = match.span()
        working = working[:start] + expanded + working[end:]

    # --------------------------
    # STEP 2: parse final simple string
    # --------------------------
    counts = _parse_simple(working)

    # --------------------------
    # STEP 3: validate elements against ELEMENT_DATA
    # --------------------------
    validated = _validate_elements(counts)

    return validated


# ============================================================
#   Utility helpers
# ============================================================

def formula_to_pretty_string(counts: Dict[str, int]) -> str:
    """
    Convert processed element counts back to a human-friendly formula string.
    """
    out = []
    for sym in sorted(counts.keys()):
        c = counts[sym]
        out.append(f"{sym}{c if c > 1 else ''}")
    return "".join(out)


def merge_formula_dicts(dict_list: List[Dict[str, int]]) -> Dict[str, int]:
    """
    Merge many formula dicts (useful for multi-molecule systems).
    """
    merged = {}
    for d in dict_list:
        for sym, c in d.items():
            merged[sym] = merged.get(sym, 0) + c
    return merged


# ============================================================
#   Self-test
# ============================================================

if __name__ == "__main__":
    test_cases = [
        "H2O",
        "Mg(OH)2",
        "(NH4)2SO4",
        "C6H5(CH3)3",
        "Al2(SO4)3",
    ]
    load_elements()

    for t in test_cases:
        parsed = parse_formula(t)
        pretty = formula_to_pretty_string(parsed)
        print(f"{t:<20} → {parsed} → {pretty}") 