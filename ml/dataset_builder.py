from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Optional imports
USE_TORCH = False
USE_PYG = False
try:
    import torch
    USE_TORCH = True
    try:
        from torch_geometric.data import Data as PyGData  # type: ignore
        from torch_geometric.data import DataLoader as PyGDataLoader  # type: ignore
        USE_PYG = True
    except Exception:
        USE_PYG = False
except Exception:
    USE_TORCH = False
    USE_PYG = False


# -----------------------
# Paths and element loader
# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_ELEMENTS_JSON = DATA_DIR / "elements.json"


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        logger.warning(f"JSON file not found: {path}")
        return None
    except Exception:
        logger.exception(f"Failed to load JSON: {path}")
        return None


def load_elements(elements_json_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load elements.json and return a mapping SYMBOL -> element_info.
    This function normalizes symbol keys to uppercase.
    """
    path = Path(elements_json_path) if elements_json_path else DEFAULT_ELEMENTS_JSON
    raw = safe_load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    if raw is None:
        logger.warning("No elements.json found; returning empty element map.")
        return out

    # If file has top-level 'elements' list, use that
    if isinstance(raw, dict) and "elements" in raw and isinstance(raw["elements"], list):
        for el in raw["elements"]:
            sym = el.get("symbol")
            if not sym:
                continue
            out[str(sym).upper()] = dict(el)
    elif isinstance(raw, dict):
        # Maybe mapping SYMBOL -> props
        for k, v in raw.items():
            out[str(k).upper()] = dict(v) if isinstance(v, dict) else {"symbol": str(k).upper()}
    else:
        logger.warning("elements.json format not understood; expecting dict or {'elements': [...]} structure.")
    return out


# -----------------------
# JSONL loader / exporter
# -----------------------
def load_jsonl_samples(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL file where each line is a JSON object representing one sample/event.
    Returns a list of parsed dicts (possibly empty).
    """
    samples: List[Dict[str, Any]] = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except Exception:
                    logger.exception("Skipping malformed JSONL line.")
    except Exception:
        logger.exception(f"Failed to load jsonl {jsonl_path}")
    return samples


def export_events_to_jsonl(events: List[Dict[str, Any]], out_path: str) -> str:
    """
    Save a list of event dictionaries as a JSONL file. Returns the file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for ev in events:
            try:
                fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Skipping non-serializable event")
    logger.info(f"Exported {len(events)} events to {out_path}")
    return str(out_path)


# -----------------------
# Node feature construction
# -----------------------
def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def build_node_features_for_event(
    event: Dict[str, Any],
    elements_map: Dict[str, Dict[str, Any]]
) -> Tuple[List[List[float]], List[str], List[List[float]]]:
    """
    Build node features for each atom in the event.

    Returns:
      - x: List of per-node numeric features (list of floats)
      - symbols: List of element symbols (string)
      - positions: List of [x,y] positions for optional positional features
    """
    nodes = event.get("nodes") or event.get("atoms") or []
    x: List[List[float]] = []
    symbols: List[str] = []
    positions: List[List[float]] = []

    for n in nodes:
        # support both 'atoms' (with uid,symbol,pos) and 'nodes' style exported by KB
        symbol = str(n.get("symbol") or n.get("element") or n.get("symbol") or "?").upper()
        symbols.append(symbol)
        # element properties from elements_map
        elem_info = elements_map.get(symbol, {})

        mass = _safe_float(n.get("mass") or elem_info.get("atomic_mass"))
        en = _safe_float(n.get("en") or elem_info.get("electronegativity_pauling"))
        charge = _safe_float(n.get("charge") or elem_info.get("charge") or 0.0)
        covrad = _safe_float(elem_info.get("covalent_radius") or elem_info.get("covalentRadius") or 0.7)
        # optional extras from elements.json
        period = _safe_float(elem_info.get("period", 0))
        group = _safe_float(elem_info.get("group", 0))
        block = elem_info.get("block", "")
        block_idx = {"s": 0.0, "p": 1.0, "d": 2.0, "f": 3.0}.get(str(block).lower(), 0.0)
        phase = elem_info.get("phase", "")
        phase_idx = {"Gas": 0.0, "Liquid": 1.0, "Solid": 2.0}.get(str(phase), 2.0)

        pos = n.get("pos", n.get("position", [0.0, 0.0]))
        try:
            px = float(pos[0])
            py = float(pos[1])
        except Exception:
            px, py = 0.0, 0.0
        positions.append([px, py])

        # Ionization energies (take up to 3)
        ies = elem_info.get("ionization_energies", []) if isinstance(elem_info.get("ionization_energies", []), list) else []
        ie_feats = [_safe_float(ies[i]) if i < len(ies) else 0.0 for i in range(3)]

        # Compose feature vector. Order matters for downstream models.
        feat = [
            mass,
            en,
            charge,
            covrad,
            period,
            group,
            block_idx,
            phase_idx,
            _safe_float(elem_info.get("electron_affinity", 0.0)),
            *ie_feats[:3],
            px, py, 1.0  # position + bias
        ]
        x.append(feat)
    return x, symbols, positions


# -----------------------
# PyG dataset builder
# -----------------------
def build_pyg_dataset_from_jsonl(
    jsonl_path: str,
    elements_json_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Optional[List["PyGData"]]:
    """
    Build a list of torch_geometric.data.Data objects from a JSONL dataset file.
    Returns None if PyG / torch is not available.

    Each Data will have:
      - x: [N, F] node features (torch.float)
      - edge_index: [2, E] long tensor (existing bonds)
      - edge_attr: [E, 1] float (placeholder)
      - edge_label: [E] long (bond order if present or zeros)
      - pos: [N, 2] (positions) optional
      - y: [1] (placeholder)
    """
    if not (USE_TORCH and USE_PYG):
        logger.warning("PyTorch + PyG not available; cannot build PyG dataset.")
        return None

    elements_map = load_elements(elements_json_path)
    samples = load_jsonl_samples(jsonl_path)
    if not samples:
        logger.warning("No samples found in JSONL.")
        return []

    data_list: List["PyGData"] = []
    for idx, s in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break
        # Prefer the 'nodes' field (export_ml_dataset format) else fall back
        x_list, symbols, positions = build_node_features_for_event(s, elements_map)
        if len(x_list) == 0:
            logger.debug(f"Skipping empty sample {idx}")
            continue
        x_tensor = torch.tensor(x_list, dtype=torch.float)

        # build edge_index from s['edges'] where edges are [[i,j],...]
        edges = s.get("edges", []) or []
        if edges:
            try:
                ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            except Exception:
                logger.exception(f"Failed to build edge_index for sample {idx}; using empty edge_index.")
                ei = torch.empty((2, 0), dtype=torch.long)
        else:
            ei = torch.empty((2, 0), dtype=torch.long)

        edge_attr = torch.ones((ei.shape[1], 1), dtype=torch.float) if ei.shape[1] > 0 else torch.empty((0, 1), dtype=torch.float)

        # labels
        edge_labels = s.get("edge_labels", None)
        if edge_labels and len(edge_labels) == ei.shape[1]:
            edge_label_tensor = torch.tensor(edge_labels, dtype=torch.long)
        else:
            edge_label_tensor = torch.zeros((ei.shape[1],), dtype=torch.long)

        data = PyGData(x=x_tensor, edge_index=ei, edge_attr=edge_attr)
        # optional metadata
        data.pos = torch.tensor(positions, dtype=torch.float)
        data.edge_label = edge_label_tensor
        data.y = torch.tensor([0], dtype=torch.long)
        data_list.append(data)

    logger.info(f"Built {len(data_list)} PyG Data samples from {jsonl_path}")
    return data_list


# -----------------------
# Dense / Torch fallback builder
# -----------------------
def build_dense_pairs_dataset_from_jsonl(
    jsonl_path: str,
    elements_json_path: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a dense pairwise dataset -> (X_pairs, y_labels) where each row is a pair feature vector.

    Pair feature vector includes:
      [mass_i, mass_j, en_i, en_j, dist_ij, pos_ix, pos_iy, pos_jx, pos_jy]

    Returns:
      - X_pairs: numpy array shape [M, D]
      - y_labels: numpy array shape [M] (0/1)
    """
    elements_map = load_elements(elements_json_path)
    samples = load_jsonl_samples(jsonl_path)
    feats: List[List[float]] = []
    labels: List[int] = []

    for idx, s in enumerate(samples):
        if max_samples is not None and idx >= max_samples:
            break
        nodes = s.get("nodes") or s.get("atoms") or []
        n = len(nodes)
        if n == 0:
            continue
        # build node-level features (mass,en,pos)
        node_feats = []
        for nidx, n in enumerate(nodes):
            sym = str(n.get("symbol") or n.get("element") or "?").upper()
            elem = elements_map.get(sym, {})
            mass = _safe_float(n.get("mass") or elem.get("atomic_mass", 0.0))
            en = _safe_float(n.get("en") or elem.get("electronegativity_pauling", 0.0))
            pos = n.get("pos", n.get("position", [0.0, 0.0]))
            try:
                px, py = float(pos[0]), float(pos[1])
            except Exception:
                px, py = 0.0, 0.0
            node_feats.append((mass, en, px, py))

        # build a set of existing edges for label lookups
        edges = s.get("edges", []) or []
        edge_set = {tuple(sorted((int(e[0]), int(e[1])))) for e in edges if isinstance(e, (list, tuple)) and len(e) >= 2}

        for i in range(n):
            for j in range(i + 1, n):
                mi, eni, pix, piy = node_feats[i]
                mj, enj, pjx, pjy = node_feats[j]
                dx = pix - pjx
                dy = piy - pjy
                dist = math.sqrt(dx * dx + dy * dy)
                feat = [mi, mj, eni, enj, dist, pix, piy, pjx, pjy]
                feats.append(feat)
                labels.append(1 if (i, j) in edge_set else 0)

    if not feats:
        return np.zeros((0, 9), dtype=float), np.zeros((0,), dtype=int)
    X = np.array(feats, dtype=float)
    y = np.array(labels, dtype=int)
    logger.info(f"Built pairwise dataset with {X.shape[0]} pairs from {jsonl_path}")
    return X, y


# -----------------------
# Utilities
# -----------------------
def save_index(out_dir: str, jsonl_filename: str, num_samples: int) -> None:
    """
    Save an index file describing the created dataset file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = {"index_file": Path(jsonl_filename).name, "num": int(num_samples)}
    with open(out_dir / "index.json", "w", encoding="utf-8") as fh:
        json.dump(idx, fh, indent=2)
    logger.info(f"Wrote index.json to {out_dir}")


# -----------------------
# CLI / Demo
# -----------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(prog="dataset_builder", description="Build ML datasets from JSONL event exports.")
    p.add_argument("--jsonl", type=str, required=True, help="Path to JSONL exported by ReactionKnowledgeBaseV2.export_ml_dataset")
    p.add_argument("--elements", type=str, default=None, help="Path to elements.json (optional override)")
    p.add_argument("--mode", type=str, default="info", choices=["info", "pyg", "dense"], help="What to build / show")
    p.add_argument("--max", type=int, default=None, help="Max samples to process")
    args = p.parse_args()

    elems = load_elements(args.elements)
    print(f"Loaded {len(elems)} elements from {args.elements or DEFAULT_ELEMENTS_JSON}")

    if args.mode == "info":
        samples = load_jsonl_samples(args.jsonl)
        print(f"Loaded {len(samples)} samples; example keys: {list(samples[0].keys())[:10] if samples else 'N/A'}")
    elif args.mode == "pyg":
        if not (USE_TORCH and USE_PYG):
            print("PyG/Torch not available in this environment.")
        else:
            data_list = build_pyg_dataset_from_jsonl(args.jsonl, elements_json_path=args.elements, max_samples=args.max)
            print(f"Built {len(data_list)} PyG samples.")
    elif args.mode == "dense":
        X, y = build_dense_pairs_dataset_from_jsonl(args.jsonl, elements_json_path=args.elements, max_samples=args.max)
        print(f"Built dense dataset with X.shape={X.shape}, y.shape={y.shape}")