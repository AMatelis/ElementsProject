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
    # Slight margin around [0,1] for nicer framing
    sim.ax.set_xlim(-0.05, 1.05)
    sim.ax.set_ylim(-0.05, 1.05)
    sim.ax.set_aspect('equal')
    sim.ax.set_xticks([])
    sim.ax.set_yticks([])

    # Draw energy / potential field overlay (subtle)
    draw_energy_field(sim)

    # Draw bonds
    for b in getattr(sim, 'bonds', []):
        try:
            x = [b.atom1.pos[0], b.atom2.pos[0]]
            y = [b.atom1.pos[1], b.atom2.pos[1]]
            # thinner, more professional bond lines
            base_width = 1.0
            linewidth = base_width
            color = "#A8A8A8"
            if hasattr(b, 'time_of_creation'):
                dt = time.time() - b.time_of_creation
                if dt < 0.4:
                    # small transient highlight for new bonds
                    linewidth += (0.4 - dt) * 2.5
                    color = "#66bfff"
            # Draw single/double/triple bond representation
            if getattr(b, 'order', 1) == 1:
                sim.ax.plot(x, y, color=color, linewidth=linewidth, zorder=1, antialiased=True)
            else:
                # offset small perpendicular vector to draw parallel lines for double/triple bonds
                dx = x[1] - x[0]
                dy = y[1] - y[0]
                length = (dx*dx + dy*dy) ** 0.5 + 1e-12
                ux, uy = dx/length, dy/length
                # perpendicular
                px, py = -uy, ux
                offset = 0.004
                # draw lines based on order
                if b.order >= 2:
                    sim.ax.plot([x[0]+px*offset, x[1]+px*offset], [y[0]+py*offset, y[1]+py*offset], color=color, linewidth=linewidth, zorder=1, antialiased=True)
                    sim.ax.plot([x[0]-px*offset, x[1]-px*offset], [y[0]-py*offset, y[1]-py*offset], color=color, linewidth=linewidth, zorder=1, antialiased=True)
                if b.order >= 3:
                    # center line for triple
                    sim.ax.plot(x, y, color=color, linewidth=linewidth, zorder=1, antialiased=True)
        except Exception:
            continue

    # Draw atoms
    for a in getattr(sim, 'atoms', []):
        try:
            pos = a.pos
            # More refined size scaling: use radius -> marker area
            radius = getattr(a, 'radius', 0.2)
            base_scale = 40.0
            area = max(20.0, (radius * base_scale) ** 2)
            color = get_element_color(a.symbol)
            # soft outline and slightly smaller markers for a cleaner look
            sim.ax.scatter(pos[0], pos[1], s=area, c=[color], edgecolors=[(0,0,0,0.45)], linewidths=0.6, zorder=3, alpha=0.95)
            # element label: smaller, slightly transparent
            sim.ax.text(pos[0], pos[1] + 0.02, a.symbol, ha='center', va='bottom', fontsize=7, color='white', alpha=0.95, zorder=4)
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
    # energy/potential visualization constrained to simulation domain [0,1]
    gx = np.linspace(0, 1, 200)
    gy = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)
    # subtle radial-like field for aesthetics (user can replace with real potential)
    field = np.exp(-4 * ((X-0.5)**2 + (Y-0.5)**2))
    sim.ax.imshow(field, extent=[0, 1, 0, 1], origin='lower',
                  cmap="inferno", alpha=0.08, interpolation="bilinear", zorder=0)


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
            # draw trail with fading alpha along its length
            n = len(xs)
            for i in range(1, n):
                a = max(0.03, 0.25 * (i / n))
                sim.ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], color=(1,1,1), alpha=a*0.25, linewidth=0.8, zorder=0.5)


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