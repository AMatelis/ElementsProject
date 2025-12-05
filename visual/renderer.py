import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from typing import Optional
import time
import logging

logger = logging.getLogger(__name__)

from engine.atoms import Atom
from engine.elements_data import ELEMENT_DATA, load_elements
from engine.bonds import BondObj
from visual.colors import get_element_color


def render_simulation_frame(sim) -> None:
    """
    Render the current frame of the simulation into sim.fig and sim.ax.
    Creates a new figure if not present.
    """
    if getattr(sim, 'fig', None) is None or getattr(sim, 'ax', None) is None:
        sim.fig, sim.ax = plt.subplots(figsize=(6, 6))

    sim.ax.clear()
    sim.ax.set_xlim(0, 1)
    sim.ax.set_ylim(0, 1)
    sim.ax.set_aspect('equal')
    sim.ax.set_xticks([])
    sim.ax.set_yticks([])

    # Draw energy field overlay if available
    draw_energy_field(sim)

    # Draw bonds
    for b in getattr(sim, 'bonds', []):
        try:
            x = [b.atom1.pos[0], b.atom2.pos[0]]
            y = [b.atom1.pos[1], b.atom2.pos[1]]
            linewidth = 2
            color = "#888888"
            if hasattr(b, 'time_of_creation'):
                dt = time.time() - b.time_of_creation
                if dt < 0.4:
                    linewidth += (0.4 - dt) * 10
                    color = "#55aaff"
            sim.ax.plot(x, y, color=color, linewidth=linewidth, zorder=1)
        except Exception:
            continue

    # Draw atoms
    for a in getattr(sim, 'atoms', []):
        try:
            pos = a.pos
            size = max(6, int(getattr(a, 'radius', 0.02) * 2000))
            color = get_element_color(a.symbol)
            sim.ax.scatter(pos[0], pos[1], s=size, c=color, edgecolors='black', zorder=2)
            sim.ax.text(pos[0], pos[1] + 0.015, a.symbol, ha='center', va='bottom', fontsize=8, color='white', zorder=3)
        except Exception:
            continue

    # Draw ghost trails if available
    draw_ghost_trails(sim)

    sim.fig.tight_layout()


def draw_energy_field(sim) -> None:
    """
    Optional aesthetic: overlay a soft energy/fog field to show potential regions.
    """
    if getattr(sim, 'ax', None) is None:
        return
    gx = np.linspace(-1, 1, 200)
    gy = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)
    field = np.exp(-4 * R)
    sim.ax.imshow(field, extent=[-1, 1, -1, 1], origin='lower',
                  cmap="inferno", alpha=0.12, interpolation="bilinear", zorder=0)


def activate_ghost_trails(sim, length: int = 20) -> None:
    """
    Maintain motion trails for atoms, stored in sim._trail_buffer.
    """
    if not hasattr(sim, "_trail_buffer"):
        sim._trail_buffer = defaultdict(lambda: deque(maxlen=length))
    for a in getattr(sim, 'atoms', []):
        sim._trail_buffer[a.uid].append(tuple(a.pos))


def draw_ghost_trails(sim) -> None:
    """
    Draw faint trails following atomic motion.
    """
    if not hasattr(sim, "_trail_buffer"):
        return
    for trail in sim._trail_buffer.values():
        if len(trail) > 1:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            sim.ax.plot(xs, ys, color="white", alpha=0.15, linewidth=1, zorder=0.5)


def highlight_bonds(sim) -> None:
    """
    Draw a glow for newly formed bonds (short-lived visual effect).
    """
    if not hasattr(sim, 'bonds'):
        return
    now = time.time()
    for b in list(sim.bonds):
        dt = now - getattr(b, 'time_of_creation', 0.0)
        if dt < 0.4:
            glow = max(0.0, 0.4 - dt) * 10
            sim.ax.plot([b.atom1.pos[0], b.atom2.pos[0]],
                        [b.atom1.pos[1], b.atom2.pos[1]],
                        linewidth=glow, color="#55aaff", alpha=0.1, zorder=1)


def update_visual_with_effects(sim) -> None:
    """
    Convenience wrapper: renders simulation frame and overlays
    ghost trails, energy field, and highlights.
    """
    try:
        render_simulation_frame(sim)
        activate_ghost_trails(sim)
        draw_ghost_trails(sim)
        highlight_bonds(sim)
    except Exception as e:
        logger.exception(f"Failed to update visual effects: {e}")