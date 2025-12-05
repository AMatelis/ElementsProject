from __future__ import annotations
from typing import Dict, Tuple, List, Iterable, Optional, Set
from collections import defaultdict
import math
import threading
import numpy as np
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


Coord = Tuple[int, int]


def _to_pos_array(pos) -> np.ndarray:
    """Safely convert various position types to numpy array shape (2,)."""
    arr = np.asarray(pos, dtype=float)
    if arr.size == 1:
        arr = np.array([float(arr), 0.0])
    if arr.size >= 2:
        return arr[:2].astype(float)
    # fallback zero
    return np.zeros(2, dtype=float)


class SpatialHash:
    """
    Grid-based spatial hash for 2D positions.

    Parameters
    ----------
    cell_size : float
        Size of each grid cell (in same units as atom positions).
    bounds : Optional[float]
        If provided, the simulation box is assumed to be [0, bounds) in both axes.
        If `bounds` is provided you may enable `periodic=True` for wrapping.
    periodic : bool
        If True and bounds is set, neighbor queries and coordinate wrapping use
        minimum-image convention.
    """

    def __init__(self, cell_size: float = 0.06, bounds: Optional[float] = 1.0, periodic: bool = False):
        if cell_size <= 0:
            raise ValueError("cell_size must be > 0")
        self.cell_size: float = float(cell_size)
        self.bounds: Optional[float] = float(bounds) if bounds is not None else None
        self.periodic: bool = bool(periodic)
        # internal mapping: (i,j) -> list of atom references
        self._cells: Dict[Coord, List[object]] = defaultdict(list)
        # mapping uid -> cell coord for quick update/remove
        self._uid_to_cell: Dict[str, Coord] = {}
        self._lock = threading.RLock()

    # -----------------------
    # Internal helpers
    # -----------------------
    def _cell_coords(self, pos: np.ndarray) -> Coord:
        """Return integer cell coords for a 2D position (numpy array)."""
        # allow negative positions if present
        i = int(math.floor(float(pos[0]) / self.cell_size))
        j = int(math.floor(float(pos[1]) / self.cell_size))
        return (i, j)

    def _wrap_pos(self, pos: np.ndarray) -> np.ndarray:
        """Wrap a position into the periodic box [0,bounds)."""
        if self.bounds is None:
            return pos
        # modulo operation that keeps values in [0, bounds)
        return np.mod(pos, self.bounds)

    def _min_image_delta(self, delta: np.ndarray) -> np.ndarray:
        """Apply minimum-image convention to a displacement vector (2D)."""
        if (not self.periodic) or (self.bounds is None):
            return delta
        half = 0.5 * self.bounds
        # shift components into [-half, half]
        dx = (delta + half) % self.bounds - half
        return dx

    # -----------------------
    # Core operations
    # -----------------------
    def clear(self) -> None:
        """Remove all atoms from the spatial hash."""
        with self._lock:
            self._cells.clear()
            self._uid_to_cell.clear()

    def add(self, atom: object) -> None:
        """
        Add an atom-like object into the spatial hash.

        The object must expose:
          - `.pos` : array-like length 2
          - `.uid` : unique identifier (string)
        """
        with self._lock:
            try:
                uid = getattr(atom, "uid")
                pos = _to_pos_array(getattr(atom, "pos"))
            except Exception:
                logger.exception("Failed to read atom.pos/atom.uid in SpatialHash.add; skipping atom.")
                return

            if self.periodic and self.bounds is not None:
                pos = self._wrap_pos(pos)

            cell = self._cell_coords(pos)
            self._cells[cell].append(atom)
            self._uid_to_cell[str(uid)] = cell

    def remove(self, atom: object) -> None:
        """Remove an atom from the spatial hash (no-op if not present)."""
        with self._lock:
            uid = getattr(atom, "uid", None)
            if uid is None:
                return
            cell = self._uid_to_cell.get(str(uid))
            if cell is None:
                return
            try:
                self._cells[cell].remove(atom)
            except ValueError:
                # atom not found in expected cell; fallback search and remove
                for c, lst in list(self._cells.items()):
                    if atom in lst:
                        lst.remove(atom)
                        break
            self._uid_to_cell.pop(str(uid), None)

    def update(self, atom: object) -> None:
        """
        Update the cell of a single atom after its position changed.
        This is efficient when atoms move incrementally.
        """
        with self._lock:
            try:
                uid = getattr(atom, "uid")
                pos = _to_pos_array(getattr(atom, "pos"))
            except Exception:
                logger.exception("Failed to read atom.pos/atom.uid in SpatialHash.update; skipping atom.")
                return

            if self.periodic and self.bounds is not None:
                pos = self._wrap_pos(pos)

            new_cell = self._cell_coords(pos)
            old_cell = self._uid_to_cell.get(str(uid))
            if old_cell == new_cell:
                # nothing to do
                return
            # remove from old
            if old_cell is not None:
                try:
                    self._cells[old_cell].remove(atom)
                except Exception:
                    # fallback remove if inconsistent
                    for c, lst in list(self._cells.items()):
                        if atom in lst:
                            lst.remove(atom)
                            break
            # add to new
            self._cells[new_cell].append(atom)
            self._uid_to_cell[str(uid)] = new_cell

    def bulk_update(self, atoms: Iterable[object]) -> None:
        """
        Rebuild the spatial hash from a sequence of atoms. This is the recommended
        approach when many atoms moved significantly (cheaper than many .update()).
        """
        with self._lock:
            self._cells.clear()
            self._uid_to_cell.clear()
            for a in atoms:
                try:
                    pos = _to_pos_array(getattr(a, "pos"))
                    if self.periodic and self.bounds is not None:
                        pos = self._wrap_pos(pos)
                    cell = self._cell_coords(pos)
                    self._cells[cell].append(a)
                    self._uid_to_cell[str(getattr(a, "uid"))] = cell
                except Exception:
                    # skip malformed atom
                    logger.debug("Skipping malformed atom during bulk_update.")

    # -----------------------
    # Querying
    # -----------------------
    def neighbors(self, atom: object, radius: float) -> List[object]:
        """
        Return a list of atoms within `radius` of `atom` (excluding `atom` itself).
        This uses the cell grid to restrict search to nearby cells.

        Note: does not exclude bonded atoms; caller may filter them.
        """
        with self._lock:
            try:
                pos = _to_pos_array(getattr(atom, "pos"))
            except Exception:
                logger.exception("Failed to read atom.pos in SpatialHash.neighbors; returning [].")
                return []

            if self.periodic and self.bounds is not None:
                # For the minimum-image distance, comparisons are done using wrapped delta
                pos_wrapped = self._wrap_pos(pos)
            else:
                pos_wrapped = pos

            # determine cell index range to search
            cell_x, cell_y = self._cell_coords(pos_wrapped)
            rng = int(math.ceil(float(radius) / self.cell_size))
            out = []
            r2 = float(radius) * float(radius)

            for dx in range(-rng, rng + 1):
                for dy in range(-rng, rng + 1):
                    cell = (cell_x + dx, cell_y + dy)
                    cell_list = self._cells.get(cell)
                    if not cell_list:
                        continue
                    for cand in cell_list:
                        if cand is atom:
                            continue
                        try:
                            cand_pos = _to_pos_array(getattr(cand, "pos"))
                        except Exception:
                            continue
                        # compute displacement
                        if self.periodic and self.bounds is not None:
                            delta = self._min_image_delta(cand_pos - pos_wrapped)
                        else:
                            delta = cand_pos - pos_wrapped
                        if (delta * delta).sum() <= r2 + 1e-12:
                            out.append(cand)
            return out

    def query_region(self, center: Iterable[float], radius: float) -> List[object]:
        """
        Return atoms contained within radius from center point (center may be tuple/list/numpy).
        """
        center_pos = _to_pos_array(center)
        # create a temporary dummy atom with .pos to reuse neighbors logic
        class _Dummy:
            def __init__(self, pos):
                self.pos = pos
        return self.neighbors(_Dummy(center_pos), radius)

    def neighbor_pairs(self, radius: float) -> List[Tuple[object, object]]:
        """
        Enumerate unique unordered pairs of atoms within `radius` of each other.
        This is useful for O(n) pair generation for force computations.
        """
        with self._lock:
            pairs: List[Tuple[object, object]] = []
            seen: Set[Tuple[str, str]] = set()
            # iterate cells and only check neighbor cells with monotonic ordering to avoid duplicates
            for (ci, cj), cell_list in list(self._cells.items()):
                if not cell_list:
                    continue
                # compare within cell
                n = len(cell_list)
                for i in range(n):
                    a = cell_list[i]
                    for j in range(i + 1, n):
                        b = cell_list[j]
                        uid_pair = tuple(sorted((str(getattr(a, "uid", "")), str(getattr(b, "uid", "")))))
                        if uid_pair in seen:
                            continue
                        seen.add(uid_pair)
                        # quick axis-aligned bounding box check before precise distance
                        try:
                            apos = _to_pos_array(getattr(a, "pos"))
                            bpos = _to_pos_array(getattr(b, "pos"))
                        except Exception:
                            continue
                        if self.periodic and self.bounds is not None:
                            delta = self._min_image_delta(bpos - apos)
                        else:
                            delta = bpos - apos
                        if (delta * delta).sum() <= radius * radius + 1e-12:
                            pairs.append((a, b))
                # compare with neighboring cells in a half-plane to avoid duplicates
                for dx in (0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy <= 0:
                            # skip cells we've already processed (including own cell's lower dy)
                            continue
                        neighbor_cell = (ci + dx, cj + dy)
                        neighbor_list = self._cells.get(neighbor_cell)
                        if not neighbor_list:
                            continue
                        for a in cell_list:
                            for b in neighbor_list:
                                uid_pair = tuple(sorted((str(getattr(a, "uid", "")), str(getattr(b, "uid", "")))))
                                if uid_pair in seen:
                                    continue
                                seen.add(uid_pair)
                                try:
                                    apos = _to_pos_array(getattr(a, "pos"))
                                    bpos = _to_pos_array(getattr(b, "pos"))
                                except Exception:
                                    continue
                                if self.periodic and self.bounds is not None:
                                    delta = self._min_image_delta(bpos - apos)
                                else:
                                    delta = bpos - apos
                                if (delta * delta).sum() <= radius * radius + 1e-12:
                                    pairs.append((a, b))
            return pairs

    # -----------------------
    # Utility helpers
    # -----------------------
    def approximate_density(self) -> float:
        """
        Approximate number of atoms per unit area using currently stored atoms and bounds.
        Returns atoms / area (if bounds present), else returns 0.
        """
        with self._lock:
            n_atoms = len(self._uid_to_cell)
            if self.bounds:
                area = float(self.bounds) * float(self.bounds)
                return n_atoms / (area + 1e-12)
            return float(n_atoms)

    @staticmethod
    def suggest_cell_size_from_atoms(atoms: Iterable[object], factor: float = 1.2, fallback: float = 0.06) -> float:
        """
        Suggest a cell_size using element covalent radii (if available) from element_data.
        Typical heuristic: cell_size ~= factor * average_visual_radius.

        - atoms: iterable of atom-like objects with .symbol attr or .pos
        - factor: how much larger than average radius (controls neighbor coverage)
        - fallback: cell_size if no radius info available
        """
        from . import element_data
        radii = []
        for a in atoms:
            sym = getattr(a, "symbol", None)
            if sym:
                try:
                    el = element_data.get_element(sym)
                    cov = float(el.get("covalent_radius", 0.7))
                    # convert to visual VDW-scaled if element_data stores covalent radii in Å
                    # we do not assume scaling here — caller decides absolute units
                    radii.append(max(0.01, cov))
                except Exception:
                    continue
        if not radii:
            return float(fallback)
        avg = float(sum(radii) / len(radii))
        return max(1e-3, float(avg) * factor)

    # -----------------------
    # Debug / representation
    # -----------------------
    def __len__(self) -> int:
        """Total number of atoms tracked."""
        with self._lock:
            return len(self._uid_to_cell)

    def cell_count(self) -> int:
        """Number of non-empty cells."""
        with self._lock:
            return sum(1 for v in self._cells.values() if v)

    def __repr__(self) -> str:
        return f"<SpatialHash cell_size={self.cell_size} bounds={self.bounds} periodic={self.periodic} atoms={len(self)}>"

# -----------------------
# Self-test / demo
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # lightweight demo with simple atom-like objects
    class Dummy:
        def __init__(self, uid, x, y):
            self.uid = uid
            self.pos = np.array([x, y], dtype=float)
            self.symbol = "H"

        def __repr__(self):
            return f"<D {self.uid} {self.pos.tolist()}>"

    # make 50 random atoms
    rng = np.random.default_rng(42)
    atoms = [Dummy(f"a{i}", *rng.random(2)) for i in range(50)]

    sh = SpatialHash(cell_size=0.1, bounds=1.0, periodic=False)
    sh.bulk_update(atoms)
    print("cells:", sh.cell_count(), "atoms:", len(sh))
    # query neighbors for first atom within radius 0.15
    nb = sh.neighbors(atoms[0], 0.15)
    print("neighbors of a0:", nb)
    pairs = sh.neighbor_pairs(0.12)
    print("pairs (<=0.12):", len(pairs))
    # suggest cell size
    suggested = SpatialHash.suggest_cell_size_from_atoms(atoms, factor=1.5, fallback=0.06)
    print("suggested cell_size:", suggested)