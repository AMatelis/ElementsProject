from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Iterable
import json
import os
import time
import threading
import logging
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Default locations (module located in ml/, elements.json expected in ../data/)
MODULE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(MODULE_DIR)
DEFAULT_ELEMENTS_PATH = os.path.join(PROJECT_ROOT, "data", "elements.json")
DEFAULT_KB_PATH = os.path.join(PROJECT_ROOT, "data", "reaction_kb.jsonl")


# -----------------------
# Utilities
# -----------------------
def safe_load_json(path: str) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        logger.exception("safe_load_json failed for %s", path)
        return None


def safe_save_json(obj: Any, path: str) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
    except Exception:
        logger.exception("safe_save_json failed for %s", path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# -----------------------
# Element data loader
# -----------------------
def load_elements(elements_path: Optional[str] = None) -> Dict[str, Dict]:
    """
    Load elements JSON and return a mapping SYMBOL -> properties dict.
    The JSON file is expected either as {"elements":[{...},...] } or as a mapping.
    """
    p = elements_path or DEFAULT_ELEMENTS_PATH
    raw = safe_load_json(p)
    out: Dict[str, Dict] = {}
    if raw is None:
        logger.warning("No elements.json found at %s. KB will work with minimal defaults.", p)
        return out

    # Support both list-of-elements format and dict mapping format
    if isinstance(raw, dict) and "elements" in raw and isinstance(raw["elements"], list):
        for el in raw["elements"]:
            sym = el.get("symbol")
            if not sym:
                continue
            out[sym.upper()] = el.copy()
    elif isinstance(raw, dict):
        # mapping of symbol -> props
        for k, v in raw.items():
            out[k.upper()] = v.copy() if isinstance(v, dict) else {"symbol": k.upper()}
    else:
        logger.warning("Unexpected elements.json structure at %s. Using minimal fallback.", p)
    return out


# -----------------------
# Knowledge Base
# -----------------------
class ReactionKnowledgeBase:
    """
    Thread-safe reaction knowledge base.

    Events are stored as dicts. Minimal event format expected:
      {
        "timestamp": "2025-11-29T20:01:37Z",
        "frame": 123,
        "event_type": "bond_formed" | "bond_broken" | "other",
        "atoms": [
            {"uid": "m0_a0", "symbol": "H", "pos": [x,y], "charge": 0.0, ...},
            ...
        ],
        "bonds": [ ("m0_a0","m0_a1", 1), ... ]  # tuples (uid1, uid2, order)
        "temperature": 300.0,
        "energy": 1.23
        ... extra fields ...
      }

    The KB persists as a JSONL file (one event per line) for append-friendly IO and
    supports exporting a consolidated JSON snapshot when requested.
    """

    def __init__(self,
                 path_jsonl: Optional[str] = None,
                 elements_path: Optional[str] = None,
                 retention: int = 100_000,
                 deduplicate: bool = True):
        """
        Args:
            path_jsonl: file path to store events in JSONL (append-friendly). Defaults to data/reaction_kb.jsonl
            elements_path: path to elements.json used to enrich node features.
            retention: maximum number of events to keep in memory (older events pruned on save).
            deduplicate: whether to attempt to deduplicate identical events on add.
        """
        self.path_jsonl = path_jsonl or DEFAULT_KB_PATH
        self.retention = int(retention)
        self.deduplicate = bool(deduplicate)

        self._lock = threading.RLock()
        self._events: List[Dict[str, Any]] = []
        self._index_by_uid: Dict[str, List[int]] = defaultdict(list)  # uid -> list of event indices
        self._pair_counts: Counter = Counter()
        self._bond_counts: Counter = Counter()
        self._loaded_from_disk = False

        # load element metadata
        self.elements = load_elements(elements_path)

        # load existing KB if file exists (incremental read)
        self._load_existing_jsonl()

    # -----------------------
    # Persistence - JSONL
    # -----------------------
    def _load_existing_jsonl(self) -> None:
        """Load existing JSONL into memory (safe, tolerant)."""
        if not os.path.exists(self.path_jsonl):
            logger.info("KB JSONL not found at %s — starting fresh.", self.path_jsonl)
            return
        try:
            with open(self.path_jsonl, "r", encoding="utf-8") as fh:
                idx = 0
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                        self._events.append(ev)
                        # build indices/counts
                        for a in ev.get("atoms", []):
                            uid = a.get("uid")
                            if uid:
                                self._index_by_uid[uid].append(idx)
                        for b in ev.get("bonds", []):
                            if isinstance(b, (list, tuple)) and len(b) >= 2:
                                key = tuple(sorted((b[0], b[1])))
                                self._bond_counts[key] += 1
                        syms = sorted({a.get("symbol", "").upper() for a in ev.get("atoms", []) if a.get("symbol")})
                        for i in range(len(syms)):
                            for j in range(i+1, len(syms)):
                                self._pair_counts[(syms[i], syms[j])] += 1
                        idx += 1
                    except Exception:
                        logger.exception("Skipping corrupt KB line while loading: %s", line[:200])
                        continue
            self._loaded_from_disk = True
            logger.info("Loaded %d events from KB %s", len(self._events), self.path_jsonl)
            # prune if over retention
            if len(self._events) > self.retention:
                self._prune_to_retention()
        except Exception:
            logger.exception("Failed to read KB JSONL at %s", self.path_jsonl)

    def _append_event_to_disk(self, event: Dict[str, Any]) -> None:
        """Append a single event to JSONL file (best-effort; does not raise)."""
        try:
            os.makedirs(os.path.dirname(self.path_jsonl), exist_ok=True) if os.path.dirname(self.path_jsonl) else None
            with open(self.path_jsonl, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False) + "\n")
        except Exception:
            logger.exception("Failed to append event to KB JSONL %s", self.path_jsonl)

    def save_snapshot(self, out_path: Optional[str] = None) -> str:
        """
        Save current KB as a consolidated JSON snapshot (not JSONL).
        Returns path saved to.
        """
        out = out_path or os.path.splitext(self.path_jsonl)[0] + "_snapshot.json"
        try:
            with self._lock:
                data = {
                    "generated_at": now_iso(),
                    "num_events": len(self._events),
                    "pair_counts": {f"{a}|{b}": int(v) for (a, b), v in self._pair_counts.items()},
                    "bond_counts": {f"{u}|{v}": int(c) for (u, v), c in self._bond_counts.items()},
                    "events": list(self._events[-self.retention:])  # limited
                }
            safe_save_json(data, out)
            logger.info("Saved KB snapshot to %s", out)
            return out
        except Exception:
            logger.exception("Failed to save KB snapshot to %s", out)
            raise

    # -----------------------
    # Event operations
    # -----------------------
    def _event_signature(self, event: Dict[str, Any]) -> str:
        """Generate a compact signature to detect duplicates (best-effort)."""
        try:
            etype = event.get("event_type", "")
            frame = int(event.get("frame", 0))
            atoms = sorted([f"{a.get('uid')}|{a.get('symbol','')}" for a in event.get("atoms", [])])
            bonds = sorted([f"{b[0]}-{b[1]}:{b[2] if len(b)>2 else 1}" for b in event.get("bonds", [])]) if event.get("bonds") else []
            # keep signature short
            sig = f"{etype}|{frame}|{'|'.join(atoms)}|{'|'.join(bonds)}"
            return sig
        except Exception:
            return json.dumps(event, sort_keys=True)[:512]

    def add_event(self, event: Dict[str, Any], persist: bool = True) -> bool:
        """
        Add a single event to the KB.

        Args:
            event: event dict (see class docstring)
            persist: whether to append to the JSONL on disk (default True)

        Returns:
            True if added, False if filtered/duplicate/not valid.
        """
        if not isinstance(event, dict):
            logger.warning("Attempted to add non-dict event: %r", type(event))
            return False

        # minimal validation
        if "atoms" not in event or "bonds" not in event:
            logger.warning("Event missing required fields 'atoms' or 'bonds'. Skipping.")
            return False

        # add timestamp if missing
        if "timestamp" not in event:
            event["timestamp"] = now_iso()

        # normalize simple types
        try:
            event["frame"] = int(event.get("frame", 0))
        except Exception:
            event["frame"] = 0

        with self._lock:
            # deduplicate
            if self.deduplicate:
                sig = self._event_signature(event)
                # quick linear duplicate check on last 200 entries (fast heuristic)
                tail = self._events[-200:]
                if any(self._event_signature(e) == sig for e in tail):
                    logger.debug("Duplicate event detected (tail) — skipping add.")
                    return False

            idx = len(self._events)
            self._events.append(event)

            # update indices and counters
            for a in event.get("atoms", []):
                uid = a.get("uid")
                if uid:
                    self._index_by_uid[uid].append(idx)
            # bond counts
            for b in event.get("bonds", []):
                if isinstance(b, (list, tuple)) and len(b) >= 2:
                    key = tuple(sorted((b[0], b[1])))
                    self._bond_counts[key] += 1

            # pair counts by element symbols
            syms = sorted({(a.get("symbol") or "").upper() for a in event.get("atoms", []) if a.get("symbol")})
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    self._pair_counts[(syms[i], syms[j])] += 1

            # append to disk (best-effort)
            if persist:
                try:
                    self._append_event_to_disk(event)
                except Exception:
                    logger.exception("Failed to persist KB event to disk")

            # prune if needed
            if len(self._events) > self.retention:
                self._prune_to_retention_locked()

            return True

    def add_events_bulk(self, events: Iterable[Dict[str, Any]], persist: bool = True) -> int:
        """
        Add multiple events. Returns number of events added.
        """
        added = 0
        for ev in events:
            if self.add_event(ev, persist=persist):
                added += 1
        return added

    # -----------------------
    # Pruning & maintenance
    # -----------------------
    def _prune_to_retention_locked(self) -> None:
        """
        Prune in-memory events to `self.retention` (assumes lock already held).
        Rebuild indices and counters from remaining events.
        """
        # drop oldest events
        keep = self._events[-self.retention:]
        self._events = keep
        # rebuild indices/counters
        self._index_by_uid = defaultdict(list)
        self._pair_counts = Counter()
        self._bond_counts = Counter()
        for idx, ev in enumerate(self._events):
            for a in ev.get("atoms", []):
                uid = a.get("uid")
                if uid:
                    self._index_by_uid[uid].append(idx)
            for b in ev.get("bonds", []):
                if isinstance(b, (list, tuple)) and len(b) >= 2:
                    key = tuple(sorted((b[0], b[1])))
                    self._bond_counts[key] += 1
            syms = sorted({(a.get("symbol") or "").upper() for a in ev.get("atoms", []) if a.get("symbol")})
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    self._pair_counts[(syms[i], syms[j])] += 1

    def _prune_to_retention(self) -> None:
        with self._lock:
            self._prune_to_retention_locked()

    # -----------------------
    # Query utilities
    # -----------------------
    def list_events(self, start: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Return a shallow copy of events in index range [start:end].
        """
        with self._lock:
            return list(self._events[start:end])

    def get_events_for_uid(self, uid: str) -> List[Dict[str, Any]]:
        """
        Return events that mention atom with `uid`.
        """
        with self._lock:
            idxs = list(self._index_by_uid.get(uid, []))
            return [self._events[i] for i in idxs if 0 <= i < len(self._events)]

    def query(self,
              symbol: Optional[str] = None,
              event_type: Optional[str] = None,
              frame_range: Optional[Tuple[int, int]] = None,
              limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generic query by element symbol, event_type and frame_range.
        """
        res = []
        s = symbol.upper() if symbol else None
        start, end = (frame_range if frame_range is not None else (None, None))

        with self._lock:
            for ev in self._events:
                if event_type and ev.get("event_type") != event_type:
                    continue
                if s:
                    syms = { (a.get("symbol") or "").upper() for a in ev.get("atoms", []) }
                    if s not in syms:
                        continue
                if start is not None and ev.get("frame", 0) < start:
                    continue
                if end is not None and ev.get("frame", 0) > end:
                    continue
                res.append(ev)
                if limit and len(res) >= limit:
                    break
        return res

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "num_events": len(self._events),
                "unique_uids": len(self._index_by_uid),
                "top_pairs": self._pair_counts.most_common(20),
                "top_bonds": self._bond_counts.most_common(20),
            }

    # -----------------------
    # Export to ML dataset (JSONL)
    # -----------------------
    def _resolve_element_features(self, symbol: str) -> Dict[str, Any]:
        """
        Return a sanitized small dict of element features for `symbol` from loaded elements.json.
        Ensures consistent keys for ML export.
        """
        s = (symbol or "").upper()
        el = self.elements.get(s, {}) if isinstance(self.elements, dict) else {}
        # pick common useful properties with safe defaults
        return {
            "symbol": s,
            "atomic_mass": float(el.get("atomic_mass", el.get("atomic_mass", 0.0) or 0.0)),
            "electronegativity_pauling": float(el.get("electronegativity_pauling", el.get("electronegativity_pauling", 0.0) or 0.0) or 0.0),
            "covalent_radius": float(el.get("covalent_radius", el.get("covalent_radius", 0.7) or 0.7)),
            "group": int(el.get("group", 0) or 0),
            "period": int(el.get("period", 0) or 0),
            "cpk-hex": str(el.get("cpk-hex", el.get("cpk-hex", "808080") or "808080")),
            "category": str(el.get("category", "")),
        }

    def export_ml_jsonl(self, out_dir: Optional[str] = None, max_samples: Optional[int] = None) -> str:
        """
        Export KB events as a JSONL dataset suitable for ML pipelines.
        Each line: {
            "nodes": [{"uid","symbol","mass","en","charge","pos":[x,y], ...}, ...],
            "edges": [[i,j], ...],
            "edge_labels": [order,...],
            "global": {"temperature":..., "energy":..., "event_type":...}
        }

        Returns the path to the JSONL file produced.
        """
        out_dir = out_dir or os.path.join(PROJECT_ROOT, "data", "gnn_dataset")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"reaction_kb_dataset_{int(time.time())}.jsonl"
        out_path = os.path.join(out_dir, fname)

        written = 0
        try:
            with open(out_path, "w", encoding="utf-8") as fh:
                with self._lock:
                    events = list(self._events[-(max_samples or len(self._events)):])
                for ev in events:
                    # build sample
                    nodes = []
                    uid_to_idx: Dict[str, int] = {}
                    for i, a in enumerate(ev.get("atoms", [])):
                        uid = a.get("uid", f"u{i}")
                        uid_to_idx[uid] = i
                        sym = (a.get("symbol") or "").upper()
                        elem_feats = self._resolve_element_features(sym)
                        nodes.append({
                            "uid": uid,
                            "symbol": sym,
                            "mass": elem_feats["atomic_mass"],
                            "en": elem_feats["electronegativity_pauling"],
                            "charge": float(a.get("charge", 0.0) or 0.0),
                            "pos": list(map(float, a.get("pos", [0.0, 0.0]))),
                            # keep extra metadata for potential feature expansion
                            "covalent_radius": elem_feats["covalent_radius"],
                            "group": elem_feats["group"],
                            "period": elem_feats["period"],
                            "cpk-hex": elem_feats["cpk-hex"],
                            "category": elem_feats["category"],
                        })
                    edges = []
                    edge_labels = []
                    for b in ev.get("bonds", []):
                        if not isinstance(b, (list, tuple)) or len(b) < 2:
                            continue
                        u1, u2 = b[0], b[1]
                        if u1 not in uid_to_idx or u2 not in uid_to_idx:
                            continue
                        i1 = uid_to_idx[u1]
                        i2 = uid_to_idx[u2]
                        edges.append([i1, i2])
                        edge_labels.append(int(b[2]) if len(b) > 2 else 1)
                    sample = {
                        "nodes": nodes,
                        "edges": edges,
                        "edge_labels": edge_labels,
                        "global": {
                            "temperature": float(ev.get("temperature", 300.0) or 300.0),
                            "energy": float(ev.get("energy", 0.0) or 0.0),
                            "event_type": ev.get("event_type", "unknown"),
                            "frame": int(ev.get("frame", 0) or 0),
                            "timestamp": ev.get("timestamp", now_iso())
                        }
                    }
                    fh.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    written += 1
            logger.info("Exported ML JSONL dataset to %s (%d samples)", out_path, written)
            return out_path
        except Exception:
            logger.exception("Failed to export ML JSONL to %s", out_path)
            raise

    # -----------------------
    # Merge / import other KBs
    # -----------------------
    def merge_from_jsonl(self, other_path: str, max_events: Optional[int] = None) -> int:
        """
        Merge events from another JSONL file into this KB. Returns number of events added.
        """
        if not os.path.exists(other_path):
            logger.warning("merge_from_jsonl: file not found %s", other_path)
            return 0
        added = 0
        try:
            with open(other_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        ev = json.loads(line)
                        if self.add_event(ev, persist=False):
                            added += 1
                            if max_events and added >= max_events:
                                break
                    except Exception:
                        logger.exception("Skipping bad event while merging from %s", other_path)
                        continue
            # optionally persist merged events
            # append new events to disk
            with self._lock:
                new_events = self._events[-added:]
            for ev in new_events:
                self._append_event_to_disk(ev)
            logger.info("Merged %d events from %s", added, other_path)
        except Exception:
            logger.exception("Failed to merge from %s", other_path)
        return added

    # -----------------------
    # Convenience & cleanup
    # -----------------------
    def clear(self, persist_remove_file: bool = False) -> None:
        """Clear in-memory events (optionally remove on-disk JSONL file)."""
        with self._lock:
            self._events.clear()
            self._index_by_uid.clear()
            self._pair_counts.clear()
            self._bond_counts.clear()
        if persist_remove_file:
            try:
                if os.path.exists(self.path_jsonl):
                    os.remove(self.path_jsonl)
            except Exception:
                logger.exception("Failed to remove KB file %s", self.path_jsonl)

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    # Nice repr
    def __repr__(self) -> str:
        s = f"<ReactionKnowledgeBase events={len(self)} elements={len(self.elements)} retention={self.retention}>"
        return s


# -----------------------
# Example usage (for debugging)
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    kb = ReactionKnowledgeBase()
    ev = {
        "frame": 1,
        "event_type": "bond_formed",
        "atoms": [
            {"uid": "m0_a0", "symbol": "H", "pos": [0.1, 0.1], "charge": 0.0},
            {"uid": "m0_a1", "symbol": "O", "pos": [0.11, 0.11], "charge": 0.0}
        ],
        "bonds": [("m0_a0", "m0_a1", 1)],
        "temperature": 300.0,
        "energy": 1.23
    }
    added = kb.add_event(ev)
    print("Added:", added)
    print("Stats:", kb.stats())
    out = kb.export_ml_jsonl(out_dir=os.path.join(PROJECT_ROOT, "data", "gnn_dataset_test"))
    print("Exported:", out)