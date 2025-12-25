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
from visual.colors import get_element_color, get_element_rgb, VISUAL_BG
from engine.units import format_unit
from engine.visuals import get_bond_visual_properties


def render_simulation_frame(sim) -> None:
    """
    Render the current frame of the simulation into sim.fig and sim.ax.
    Creates a clean single-panel figure for publication.
    """
    # Use existing figure if available (from GUI), otherwise create new one
    if getattr(sim, 'fig', None) is None or getattr(sim, 'main_ax', None) is None:
        from gui.ui_constants import PLOT_STYLE
        # Use smaller figsize to match GUI canvas
        sim.fig = plt.figure(figsize=(6, 4.5), facecolor=PLOT_STYLE['figure_bg'], tight_layout=True)
        sim.main_ax = sim.fig.add_subplot(1, 1, 1)

    # Prepare main simulation axes with clean styling
    sim.main_ax.clear()
    sim.main_ax.set_xlim(-0.05, 1.05)
    sim.main_ax.set_ylim(-0.05, 1.05)
    sim.main_ax.set_aspect('equal')
    sim.main_ax.set_xticks([])
    sim.main_ax.set_yticks([])
    sim.main_ax.set_facecolor(VISUAL_BG)
    sim.main_ax.set_title('Molecular Dynamics Trajectory', fontsize=12, pad=10)

    # Remove all spines for clean look
    for spine in sim.main_ax.spines.values():
        spine.set_visible(False)

    # Draw energy / potential field overlay (subtle)
    draw_energy_field(sim)

    # Draw bonds
    for b in getattr(sim, 'bonds', []):
        try:
            x = [b.atom1.pos[0], b.atom2.pos[0]]
            y = [b.atom1.pos[1], b.atom2.pos[1]]
            width, color = get_bond_visual_properties(b, getattr(sim, 'time', time.time()))
            sim.main_ax.plot(x, y, color=color, linewidth=width, zorder=1, antialiased=True)
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
            sim.main_ax.scatter(pos[0], pos[1], s=area, c=[color], edgecolors=[(0,0,0,0.45)], linewidths=0.6, zorder=3, alpha=0.95)
            # element label: choose dark label for white background
            symbol_text = a.symbol
            charge = getattr(a, 'charge', 0.0)
            if abs(charge) > 1e-6:
                charge_str = f"{charge:+.1f}" if charge % 1 != 0 else f"{int(charge):+d}"
                symbol_text += charge_str

            sim.main_ax.text(pos[0], pos[1] + 0.02, symbol_text, ha='center', va='bottom', fontsize=7, color='black', alpha=0.95, zorder=4)
        except Exception:
            continue

    # Draw ghost trails if available
    draw_ghost_trails(sim)

    sim.fig.tight_layout()


def draw_energy_field(sim) -> None:
    """
    Optional aesthetic: overlay a soft energy/fog field to show potential regions.
    """
    if getattr(sim, 'main_ax', None) is None:
        return
    # energy/potential visualization constrained to simulation domain [0,1]
    gx = np.linspace(0, 1, 200)
    gy = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(gx, gy)
    R = np.sqrt(X**2 + Y**2)
    # subtle radial-like field for aesthetics (user can replace with real potential)
    field = np.exp(-4 * ((X-0.5)**2 + (Y-0.5)**2))
    # draw onto main_ax so it appears behind atoms/bonds
    sim.main_ax.imshow(field, extent=[0, 1, 0, 1], origin='lower',
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
                sim.main_ax.plot([xs[i-1], xs[i]], [ys[i-1], ys[i]], color=(0.15, 0.15, 0.15), alpha=a*0.35, linewidth=0.8, zorder=0.5)


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
            sim.main_ax.plot([b.atom1.pos[0], b.atom2.pos[0]],
                             [b.atom1.pos[1], b.atom2.pos[1]],
                             linewidth=glow, color="#55aaff", alpha=0.12, zorder=1)


def draw_time_series_panel(sim) -> None:
    """Draw energy history and training loss on the side panel."""
    if getattr(sim, 'side_ax', None) is None:
        return
    ax = sim.side_ax
    ax.clear()
    ax.set_facecolor(VISUAL_BG)
    # fetch histories (may not exist yet)
    energy = list(getattr(sim, 'energy_history', []))
    loss = list(getattr(sim, 'training_loss_history', []))
    drift = list(getattr(sim.sim, 'energy_drift_history', []))
    frames = list(range(len(energy))) if energy else []
    ax.set_title('Energy Estimates / Drift / Loss', fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    # plot energy on primary y axis
    if energy:
        ax.plot(frames, energy, color='#ff8800', label='Estimated Total Energy')
    # plot training loss on secondary axis
    if loss:
        ax2 = ax.twinx()
        loss_frames = list(range(len(loss)))
        ax2.plot(loss_frames, loss, color='#0066cc', linestyle='--', label='Train Loss')
        ax2.tick_params(axis='y', labelsize=8)
    # plot energy drift on tertiary axis
    if drift:
        ax3 = ax.twinx()
        ax3.spines["right"].set_position(("axes", 1.1))  # offset the third axis
        drift_frames = list(range(len(drift)))
        ax3.plot(drift_frames, drift, color='#ff0000', linestyle='-', label='Energy Drift')
        ax3.set_ylabel('Relative Energy Drift', fontsize=8)
        ax3.tick_params(axis='y', labelsize=8)
    # small legend
    ax.legend(loc='upper left', fontsize=7)
    # minimal layout
    ax.set_xlabel('Time Step', fontsize=8)
    ax.set_ylabel(f'Energy ({format_unit("energy")})', fontsize=8)


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