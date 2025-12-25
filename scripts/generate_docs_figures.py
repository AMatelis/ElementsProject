#!/usr/bin/env python3
"""
Generate documentation figures for Chunk 7: Scientific Documentation
Creates UI screenshot, energy graph, and reaction walkthrough figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

# Create docs directory
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)

def create_ui_screenshot_mockup():
    """Create a labeled mockup of the GUI interface"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 800)
    ax.set_facecolor('#f5f5f5')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Figure 1: Simulation Interface Layout', fontsize=14, pad=20)

    # Main canvas area (simulation view)
    canvas = FancyBboxPatch((50, 200), 700, 550, boxstyle="round,pad=0.02",
                           facecolor='white', edgecolor='#333', linewidth=2)
    ax.add_patch(canvas)
    ax.text(400, 480, 'Molecular Dynamics\nTrajectory View', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Control panel (top)
    control_panel = FancyBboxPatch((50, 50), 1100, 120, boxstyle="round,pad=0.02",
                                  facecolor='#e8f4f8', edgecolor='#333', linewidth=2)
    ax.add_patch(control_panel)
    ax.text(150, 110, 'Formula Input', ha='left', va='center', fontsize=10)
    ax.text(350, 110, 'Steps Control', ha='left', va='center', fontsize=10)
    ax.text(550, 110, 'Random Seed', ha='left', va='center', fontsize=10)
    ax.text(750, 110, 'Deterministic Mode', ha='left', va='center', fontsize=10)
    ax.text(950, 110, 'Export Buttons', ha='left', va='center', fontsize=10)

    # Sidebar (right)
    sidebar = FancyBboxPatch((800, 200), 350, 550, boxstyle="round,pad=0.02",
                            facecolor='#f0f8e8', edgecolor='#333', linewidth=2)
    ax.add_patch(sidebar)
    ax.text(975, 480, 'Element Palette\n& Metrics Plots', ha='center', va='center',
            fontsize=12, fontweight='bold')

    # Labels
    ax.text(400, 780, 'A: Control Panel - Simulation parameters and export controls', fontsize=10)
    ax.text(400, 760, 'B: Main View - Real-time molecular dynamics visualization', fontsize=10)
    ax.text(975, 760, 'C: Sidebar - Element selection and energy metrics', fontsize=10)

    # Arrows
    arrow1 = ConnectionPatch((400, 170), (400, 200), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='#666')
    ax.add_artist(arrow1)

    arrow2 = ConnectionPatch((975, 170), (975, 200), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5, color='#666')
    ax.add_artist(arrow2)

    plt.tight_layout()
    plt.savefig(docs_dir / 'ui_layout_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_energy_graph_explanation():
    """Create an energy graph with detailed explanation"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Generate sample energy data
    frames = np.arange(0, 500, 10)
    np.random.seed(42)  # For reproducible example

    # Kinetic energy (fluctuating)
    kinetic = 100 + 20 * np.sin(frames * 0.02) + np.random.normal(0, 5, len(frames))

    # Potential energy (more stable with reaction events)
    potential = -50 - 10 * np.exp(-frames * 0.005) + np.random.normal(0, 3, len(frames))

    # Add reaction events (sudden energy changes)
    reaction_frames = [150, 300, 450]
    for rf in reaction_frames:
        idx = np.argmin(np.abs(frames - rf))
        potential[idx:] -= 15  # Exothermic reaction

    total_energy = kinetic + potential

    # Main energy plot
    ax1.plot(frames, kinetic, 'b-', label='Kinetic Energy', alpha=0.8)
    ax1.plot(frames, potential, 'r-', label='Potential Energy', alpha=0.8)
    ax1.plot(frames, total_energy, 'k-', label='Total Energy', linewidth=2)

    # Mark reaction events
    for rf in reaction_frames:
        ax1.axvline(x=rf, color='green', linestyle='--', alpha=0.7, label='Reaction Event' if rf == reaction_frames[0] else "")

    ax1.set_xlabel('Simulation Frame')
    ax1.set_ylabel('Energy (arbitrary units)')
    ax1.set_title('Figure 2: Energy Conservation and Reaction Dynamics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy drift analysis
    ax2.plot(frames, total_energy - total_energy[0], 'purple', label='Energy Drift')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Ideal Conservation')
    ax2.fill_between(frames, -2, 2, color='green', alpha=0.1, label='Acceptable Range')
    ax2.set_xlabel('Simulation Frame')
    ax2.set_ylabel('Energy Deviation')
    ax2.set_title('Energy Drift Analysis')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(docs_dir / 'energy_analysis_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()

def create_reaction_walkthrough():
    """Create a step-by-step reaction walkthrough diagram"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # Time points for reaction H2 + O → H-OH + H
    times = ['t=0\nInitial State', 't=100\nApproach', 't=200\nBond Formation', 't=300\nProducts']

    for i, (ax, time_label) in enumerate(zip(axes, times)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(time_label, fontsize=12)

        if i == 0:
            # Initial state: H2 and O separate
            ax.add_patch(plt.Circle((2, 5), 0.3, color='lightblue', label='H'))
            ax.add_patch(plt.Circle((3, 5), 0.3, color='lightblue', label='H'))
            ax.plot([2, 3], [5, 5], 'k-', linewidth=2)  # H-H bond

            ax.add_patch(plt.Circle((7, 5), 0.3, color='red', label='O'))

            ax.text(2.5, 6, 'H₂', ha='center')
            ax.text(7, 6, 'O', ha='center')

        elif i == 1:
            # Approach: molecules getting closer
            ax.add_patch(plt.Circle((3, 5), 0.3, color='lightblue'))
            ax.add_patch(plt.Circle((4, 5), 0.3, color='lightblue'))
            ax.plot([3, 4], [5, 5], 'k-', linewidth=2)

            ax.add_patch(plt.Circle((6, 5), 0.3, color='red'))

            # Collision trajectory arrows
            ax.arrow(4.3, 5, 1.4, 0, head_width=0.2, head_length=0.3, fc='gray', ec='gray')

        elif i == 2:
            # Bond formation: H from H2 bonds with O
            ax.add_patch(plt.Circle((4, 5), 0.3, color='lightblue'))
            ax.add_patch(plt.Circle((5, 5), 0.3, color='red'))
            ax.plot([4, 5], [5, 5], 'k-', linewidth=3)  # New H-O bond

            ax.add_patch(plt.Circle((6.5, 5), 0.3, color='lightblue'))  # Free H

            ax.text(4.5, 6, 'H-OH', ha='center')
            ax.text(6.5, 6, 'H', ha='center')

        elif i == 3:
            # Final products: H-OH and H separated
            ax.add_patch(plt.Circle((2, 5), 0.3, color='lightblue'))
            ax.add_patch(plt.Circle((3, 5), 0.3, color='red'))
            ax.plot([2, 3], [5, 5], 'k-', linewidth=3)

            ax.add_patch(plt.Circle((7, 5), 0.3, color='lightblue'))

            ax.text(2.5, 6, 'H-OH', ha='center')
            ax.text(7, 6, 'H', ha='center')

    # Add legend
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.suptitle('Figure 3: Water Formation Reaction Walkthrough', fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(docs_dir / 'reaction_walkthrough.png', dpi=150, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating documentation figures...")

    create_ui_screenshot_mockup()
    print("✓ Created UI layout diagram")

    create_energy_graph_explanation()
    print("✓ Created energy analysis diagram")

    create_reaction_walkthrough()
    print("✓ Created reaction walkthrough diagram")

    print(f"\nFigures saved to {docs_dir}/")
    print("Ready for README integration")