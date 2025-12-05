import numpy as np
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

from engine.simulation_manager import SimulationManager
from engine.atoms import Atom
from engine.elements_data import ELEMENT_DATA
from engine.bonds import can_form_bond
from visual.colors import get_element_color

def compute_reactivity_heatmap(sim: SimulationManager, grid_size: int = 80, radius: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a 2D reaction potential heatmap over the simulation canvas.
    Each grid cell accumulates likelihood of bond formation from nearby atom pairs.

    Returns:
        xx, yy: Meshgrid arrays
        heat: Normalized 2D array of reaction likelihoods (0-1)
    """
    if not sim.atoms or len(sim.atoms) < 2:
        logger.warning("Not enough atoms to compute heatmap")
        return np.array([]), np.array([]), np.array([])

    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x, y)
    heat = np.zeros_like(xx)

    # Precompute pair midpoints and bond formation scores
    pairs = []
    for i, a1 in enumerate(sim.atoms):
        for j, a2 in enumerate(sim.atoms[i+1:], start=i+1):
            midpoint = (a1.pos + a2.pos) / 2.0
            score = 0.0
            try:
                # Use deterministic mode if available, else probabilistic
                if getattr(sim, 'deterministic_mode', False) and getattr(sim, 'reaction_engine', None):
                    should, s = sim.reaction_engine.rule_engine.should_form_bond(a1, a2, getattr(sim, 'physics', None))
                    score = float(s if should else 0.0)
                else:
                    can, s = can_form_bond(a1, a2, getattr(sim, 'temperature', 300))
                    score = float(s if can else 0.0)
            except Exception as e:
                logger.warning(f"Failed to compute bond score for {a1.symbol}-{a2.symbol}: {e}")
            if score > 0:
                pairs.append((midpoint, score))

    # Spread contribution to nearby grid cells using Gaussian falloff
    sigma = radius * 0.5
    for mid, score in pairs:
        dx = xx - mid[0]
        dy = yy - mid[1]
        dist2 = dx**2 + dy**2
        contrib = score * np.exp(-dist2 / (2 * sigma**2 + 1e-12))
        heat += contrib

    # Normalize heatmap to 0-1
    maxv = heat.max()
    if maxv > 0:
        heat /= maxv
    return xx, yy, heat


def plot_reactivity_heatmap(sim: SimulationManager, grid_size: int = 80, radius: float = 0.2, cmap: str = 'inferno', ax=None):
    """
    Plot a reaction potential heatmap over the current simulation state.
    Can optionally overlay on a provided Matplotlib axis.

    Args:
        sim: SimulationManager instance
        grid_size: Resolution of the heatmap
        radius: Gaussian spread radius
        cmap: Matplotlib colormap
        ax: Optional Matplotlib axis to overlay
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not installed, cannot plot heatmap")
        return

    xx, yy, heat = compute_reactivity_heatmap(sim, grid_size, radius)
    if heat.size == 0:
        logger.warning("Heatmap empty, nothing to plot")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.imshow(heat, extent=[0, 1, 0, 1], origin='lower', cmap=cmap, alpha=0.8, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Reaction Potential Heatmap')

    # Overlay atoms
    xs = [a.pos[0] for a in sim.atoms]
    ys = [a.pos[1] for a in sim.atoms]
    colors = [get_element_color(a.symbol) for a in sim.atoms]
    sizes = [max(6, int(getattr(a, 'radius', 0.02)*2000)) for a in sim.atoms]

    ax.scatter(xs, ys, c=colors, s=sizes, edgecolors='black', zorder=1)
    plt.show()