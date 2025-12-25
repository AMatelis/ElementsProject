"""
Publication-Ready Molecular Dynamics Simulator GUI
Clean three-panel layout optimized for scientific figures and readability.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from tkinter import BooleanVar, StringVar, IntVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import matplotlib
import threading
import time
from typing import Optional, Dict, List, Tuple
import os
import json
import logging
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Import UI constants
from .ui_constants import (
    FONT_FAMILY, FONT_SIZES, COLORS, PADDING, MARGINS, SIZES,
    BORDER, PLOT_STYLE, PALETTE_CONFIG, LAYOUT_RATIOS
)

# Configure matplotlib for clean publication style
matplotlib.style.use('default')
plt.rcParams.update({
    'figure.facecolor': PLOT_STYLE['figure_bg'],
    'axes.facecolor': PLOT_STYLE['axes_bg'],
    'font.family': FONT_FAMILY,
    'font.size': FONT_SIZES['small'],
    'axes.labelsize': FONT_SIZES['label'],
    'axes.titlesize': FONT_SIZES['subtitle'],
    'xtick.labelsize': FONT_SIZES['small'],
    'ytick.labelsize': FONT_SIZES['small'],
    'axes.grid': True,
    'grid.color': PLOT_STYLE['grid_color'],
    'grid.alpha': 0.3,
    'axes.edgecolor': COLORS['border'],
    'axes.linewidth': 0.8,
})

# Engine imports
from engine.simulation_manager import SimulationManager
from engine.elements_data import ELEMENT_DATA, load_elements, get_element
from engine.atoms import Atom
from engine.formula_parser import parse_formula
from engine.products import graph_to_formula
from engine.metrics import EnergyMetrics

# Visualization imports
from visual.renderer import render_simulation_frame
from visual.heatmap import compute_reactivity_heatmap
from visual.export_tools import (
    export_simulation_to_plotly_html,
    export_simulation_to_gif,
    export_simulation_to_mp4,
    ensure_dir,
    now_str,
    save_sidepanel_png,
)


class PeriodicTableWidget(tk.Frame):
    """Interactive periodic table widget for element selection"""

    def __init__(self, parent, on_element_select, **kwargs):
        super().__init__(parent, **kwargs)
        self.on_element_select = on_element_select
        self.element_buttons = {}
        self.oxidation_dialog = None

        # Load element data
        load_elements()

        self._create_periodic_table()

    def _create_periodic_table(self):
        """Create the periodic table grid layout"""

        # Periodic table layout (simplified - showing main elements)
        # Row 1: H
        # Row 2: Li Be B C N O F Ne
        # Row 3: Na Mg Al Si P S Cl Ar
        # Row 4: K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr
        # Row 5: Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
        # Row 6: Cs Ba La Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
        # Row 7: Fr Ra Ac Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og

        # Define element positions (row, col) for main periodic table
        element_positions = {
            'H': (0, 0), 'He': (0, 17),
            'Li': (1, 0), 'Be': (1, 1), 'B': (1, 12), 'C': (1, 13), 'N': (1, 14), 'O': (1, 15), 'F': (1, 16), 'Ne': (1, 17),
            'Na': (2, 0), 'Mg': (2, 1), 'Al': (2, 12), 'Si': (2, 13), 'P': (2, 14), 'S': (2, 15), 'Cl': (2, 16), 'Ar': (2, 17),
            'K': (3, 0), 'Ca': (3, 1), 'Sc': (3, 2), 'Ti': (3, 3), 'V': (3, 4), 'Cr': (3, 5), 'Mn': (3, 6), 'Fe': (3, 7),
            'Co': (3, 8), 'Ni': (3, 9), 'Cu': (3, 10), 'Zn': (3, 11), 'Ga': (3, 12), 'Ge': (3, 13), 'As': (3, 14), 'Se': (3, 15), 'Br': (3, 16), 'Kr': (3, 17),
            'Rb': (4, 0), 'Sr': (4, 1), 'Y': (4, 2), 'Zr': (4, 3), 'Nb': (4, 4), 'Mo': (4, 5), 'Tc': (4, 6), 'Ru': (4, 7),
            'Rh': (4, 8), 'Pd': (4, 9), 'Ag': (4, 10), 'Cd': (4, 11), 'In': (4, 12), 'Sn': (4, 13), 'Sb': (4, 14), 'Te': (4, 15), 'I': (4, 16), 'Xe': (4, 17),
            'Cs': (5, 0), 'Ba': (5, 1), 'La': (5, 2), 'Hf': (5, 3), 'Ta': (5, 4), 'W': (5, 5), 'Re': (5, 6), 'Os': (5, 7),
            'Ir': (5, 8), 'Pt': (5, 9), 'Au': (5, 10), 'Hg': (5, 11), 'Tl': (5, 12), 'Pb': (5, 13), 'Bi': (5, 14), 'Po': (5, 15), 'At': (5, 16), 'Rn': (5, 17),
        }

        # Create buttons for elements that exist in our data
        for symbol, (row, col) in element_positions.items():
            if symbol in ELEMENT_DATA:
                element_data = get_element(symbol)
                color = element_data.get('cpk-hex', '#808080')
                if color and not color.startswith('#'):
                    color = f'#{color}'

                # Create button
                btn = tk.Button(
                    self,
                    text=f"{symbol}\n{element_data.get('atomic_mass', '?'):.1f}",
                    font=(FONT_FAMILY, 8, "bold"),
                    bg=color,
                    fg='white' if self._is_dark_color(color) else 'black',
                    relief=tk.RAISED,
                    bd=1,
                    width=4,
                    height=2,
                    command=lambda s=symbol: self._on_element_click(s)
                )
                btn.grid(row=row, column=col, padx=1, pady=1, sticky='nsew')
                self.element_buttons[symbol] = btn

        # Configure grid weights for proper layout
        for i in range(6):  # rows
            self.grid_rowconfigure(i, weight=1)
        for i in range(18):  # columns
            self.grid_columnconfigure(i, weight=1)

    def _is_dark_color(self, hex_color: str) -> bool:
        """Check if a hex color is dark (for text color selection)"""
        if not hex_color or len(hex_color) < 7:
            return True
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            brightness = (r * 299 + g * 587 + b * 114) / 1000
            return brightness < 128
        except:
            return True

    def _on_element_click(self, symbol: str):
        """Handle element button click - show oxidation state dialog"""
        self._show_oxidation_dialog(symbol)

    def _show_oxidation_dialog(self, symbol: str):
        """Show dialog for selecting oxidation state"""
        if self.oxidation_dialog:
            self.oxidation_dialog.destroy()

        element_data = get_element(symbol)
        common_states = element_data.get('common_oxidation_states', [0])

        # Create dialog
        self.oxidation_dialog = tk.Toplevel(self)
        self.oxidation_dialog.title(f"Select Oxidation State for {symbol}")
        self.oxidation_dialog.geometry("300x200")
        self.oxidation_dialog.resizable(False, False)

        # Center the dialog
        self.oxidation_dialog.transient(self)
        self.oxidation_dialog.grab_set()

        # Element info
        info_frame = tk.Frame(self.oxidation_dialog, bg=COLORS['panel_bg'], padx=10, pady=10)
        info_frame.pack(fill=tk.X)

        name = element_data.get('name', symbol)
        mass = element_data.get('atomic_mass', '?')
        tk.Label(
            info_frame,
            text=f"{name} ({symbol}) - Atomic Mass: {mass:.1f}",
            font=(FONT_FAMILY, FONT_SIZES['label'], "bold"),
            bg=COLORS['panel_bg'],
            fg=COLORS['text_primary']
        ).pack(anchor=tk.W)

        # Oxidation state selection
        state_frame = tk.Frame(self.oxidation_dialog, padx=10, pady=10)
        state_frame.pack(fill=tk.X)

        tk.Label(
            state_frame,
            text="Oxidation State:",
            font=(FONT_FAMILY, FONT_SIZES['label']),
            fg=COLORS['text_primary']
        ).pack(anchor=tk.W, pady=(0, 5))

        # Variable for selected state
        self.selected_state = tk.IntVar(value=0)

        # Radio buttons for common oxidation states
        for state in common_states:
            charge_text = f"{state:+d}" if state != 0 else "0 (neutral)"
            tk.Radiobutton(
                state_frame,
                text=f"{state} (charge: {charge_text})",
                variable=self.selected_state,
                value=state,
                font=(FONT_FAMILY, FONT_SIZES['body'])
            ).pack(anchor=tk.W)

        # Custom state entry
        custom_frame = tk.Frame(state_frame)
        custom_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(
            custom_frame,
            text="Or enter custom:",
            font=(FONT_FAMILY, FONT_SIZES['body']),
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT)

        self.custom_state_var = tk.StringVar()
        custom_entry = tk.Entry(
            custom_frame,
            textvariable=self.custom_state_var,
            width=5,
            font=(FONT_FAMILY, FONT_SIZES['body'])
        )
        custom_entry.pack(side=tk.LEFT, padx=(5, 0))

        # Buttons
        button_frame = tk.Frame(self.oxidation_dialog, padx=10, pady=10)
        button_frame.pack(fill=tk.X)

        tk.Button(
            button_frame,
            text="Add Atom",
            command=lambda: self._confirm_element_selection(symbol),
            bg=COLORS['accent'],
            fg=COLORS['background'],
            font=(FONT_FAMILY, FONT_SIZES['label'], "bold")
        ).pack(side=tk.RIGHT, padx=(5, 0))

        tk.Button(
            button_frame,
            text="Cancel",
            command=self.oxidation_dialog.destroy,
            bg=COLORS['error'],
            fg=COLORS['background'],
            font=(FONT_FAMILY, FONT_SIZES['label'])
        ).pack(side=tk.RIGHT)

    def _confirm_element_selection(self, symbol: str):
        """Confirm element selection with chosen oxidation state"""
        try:
            # Check if custom state was entered
            custom_text = self.custom_state_var.get().strip()
            if custom_text:
                oxidation_state = int(custom_text)
            else:
                oxidation_state = self.selected_state.get()

            # Validate the oxidation state is plausible
            element_data = get_element(symbol)
            common_states = element_data.get('common_oxidation_states', [])
            if oxidation_state not in common_states and common_states:
                # Show warning but allow
                result = messagebox.askyesno(
                    "Uncommon Oxidation State",
                    f"Oxidation state {oxidation_state} is not in the common states for {symbol}.\n"
                    f"Common states: {common_states}\n\nProceed anyway?",
                    parent=self.oxidation_dialog
                )
                if not result:
                    return

            # Call the callback with symbol and oxidation state
            if self.on_element_select:
                self.on_element_select(symbol, oxidation_state)

            self.oxidation_dialog.destroy()

        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Please enter a valid integer for oxidation state.",
                parent=self.oxidation_dialog
            )


class PublicationGUI(tk.Tk):
    """Publication-ready molecular dynamics simulator with clean three-panel layout"""

    def __init__(self, title: str = "Molecular Dynamics Simulation"):
        super().__init__()
        self.title(title)
        self.geometry("1400x900")

        # Set clean theme
        self.configure(bg=COLORS['background'])

        # Simulation state
        self.sim: Optional[SimulationManager] = None
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop = threading.Event()
        self._sim_running = False
        self._periodic_update_id: Optional[str] = None

        # UI Components
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.canvas_figure: Optional[Figure] = None
        self.metrics_canvas: Optional[FigureCanvasTkAgg] = None
        self.metrics_figure: Optional[Figure] = None

        # Control variables
        self.formula_var = StringVar(value="H2,O")
        self.frames_var = IntVar(value=600)
        self.deterministic_var = BooleanVar(value=True)
        self.seed_var = StringVar(value="")  # Empty means random seed

        # Setup
        load_elements()
        self._setup_ui()

    def _setup_ui(self):
        """Build clean three-panel layout"""
        # Header with minimal controls
        self._create_header()

        # Main content area
        content = tk.Frame(self, bg=COLORS['background'])
        content.pack(fill=tk.BOTH, expand=True, padx=MARGINS['content'], pady=MARGINS['content'])

        # Left panel: Simulation canvas
        self._create_simulation_panel(content)

        # Right panel: Element palette (top) and metrics/plots (bottom)
        self._create_sidebar(content)

    def _create_header(self):
        """Minimal header with essential controls"""
        header = tk.Frame(self, bg=COLORS['background'], height=60)
        header.pack(side=tk.TOP, fill=tk.X, padx=MARGINS['content'], pady=(MARGINS['content'], 0))
        header.pack_propagate(False)

        # Title
        title_label = tk.Label(
            header,
            text="Molecular Dynamics Simulation",
            font=(FONT_FAMILY, FONT_SIZES['title'], "bold"),
            bg=COLORS['background'],
            fg=COLORS['text_primary']
        )
        title_label.pack(anchor=tk.W)

        # Control row
        controls = tk.Frame(header, bg=COLORS['background'])
        controls.pack(fill=tk.X, pady=(PADDING['medium'], 0))

        # Formula input
        tk.Label(
            controls,
            text="Initial Molecules:",
            font=(FONT_FAMILY, FONT_SIZES['label']),
            bg=COLORS['background'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=(0, PADDING['small']))

        formula_entry = tk.Entry(
            controls,
            textvariable=self.formula_var,
            width=20,
            font=(FONT_FAMILY, FONT_SIZES['label']),
            relief=tk.SOLID,
            bd=BORDER['width']
        )
        formula_entry.pack(side=tk.LEFT, padx=(0, PADDING['large']))

        # Frames input
        tk.Label(
            controls,
            text="Simulation Steps:",
            font=(FONT_FAMILY, FONT_SIZES['label']),
            bg=COLORS['background'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=(0, PADDING['small']))

        frames_spin = tk.Spinbox(
            controls,
            from_=100, to=5000, increment=100,
            textvariable=self.frames_var,
            width=8,
            font=(FONT_FAMILY, FONT_SIZES['label']),
            relief=tk.SOLID,
            bd=BORDER['width']
        )
        frames_spin.pack(side=tk.LEFT, padx=(0, PADDING['large']))

        # Seed input
        tk.Label(
            controls,
            text="Random Seed:",
            font=(FONT_FAMILY, FONT_SIZES['label']),
            bg=COLORS['background'],
            fg=COLORS['text_primary']
        ).pack(side=tk.LEFT, padx=(0, PADDING['small']))

        seed_entry = tk.Entry(
            controls,
            textvariable=self.seed_var,
            width=10,
            font=(FONT_FAMILY, FONT_SIZES['label']),
            relief=tk.SOLID,
            bd=BORDER['width']
        )
        seed_entry.pack(side=tk.LEFT, padx=(0, PADDING['large']))

        # Deterministic checkbox
        det_cb = tk.Checkbutton(
            controls,
            text="Deterministic Mode",
            variable=self.deterministic_var,
            font=(FONT_FAMILY, FONT_SIZES['label']),
            bg=COLORS['background'],
            fg=COLORS['text_primary'],
            selectcolor=COLORS['accent_light']
        )
        det_cb.pack(side=tk.LEFT, padx=(0, PADDING['large']))

        # Run/Stop buttons
        self.run_button = tk.Button(
            controls,
            text="▶ Run Simulation",
            command=self._on_run,
            font=(FONT_FAMILY, FONT_SIZES['label'], "bold"),
            bg=COLORS['success'],
            fg=COLORS['background'],
            relief=tk.SOLID,
            bd=0,
            padx=PADDING['medium'],
            pady=PADDING['small']
        )
        self.run_button.pack(side=tk.LEFT, padx=(0, PADDING['small']))

        self.stop_button = tk.Button(
            controls,
            text="⏹ Stop",
            command=self._on_stop,
            font=(FONT_FAMILY, FONT_SIZES['label'], "bold"),
            bg=COLORS['error'],
            fg=COLORS['background'],
            relief=tk.SOLID,
            bd=0,
            padx=PADDING['medium'],
            pady=PADDING['small'],
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT)

        # Export buttons
        export_frame = tk.Frame(controls, bg=COLORS['background'])
        export_frame.pack(side=tk.LEFT, padx=(PADDING['large'], 0))

        self.export_config_btn = tk.Button(
            export_frame,
            text="📄 Config",
            command=self._export_config,
            font=(FONT_FAMILY, FONT_SIZES['small'], "bold"),
            bg=COLORS['accent_light'],
            fg=COLORS['accent'],
            relief=tk.SOLID,
            bd=BORDER['width'],
            padx=PADDING['small'],
            pady=PADDING['small']
        )
        self.export_config_btn.pack(side=tk.LEFT, padx=(0, PADDING['small']))

        self.export_state_btn = tk.Button(
            export_frame,
            text="💾 State",
            command=self._export_state,
            font=(FONT_FAMILY, FONT_SIZES['small'], "bold"),
            bg=COLORS['accent_light'],
            fg=COLORS['accent'],
            relief=tk.SOLID,
            bd=BORDER['width'],
            padx=PADDING['small'],
            pady=PADDING['small']
        )
        self.export_state_btn.pack(side=tk.LEFT, padx=(0, PADDING['small']))

        self.export_csv_btn = tk.Button(
            export_frame,
            text="📊 CSV",
            command=self._export_csv,
            font=(FONT_FAMILY, FONT_SIZES['small'], "bold"),
            bg=COLORS['accent_light'],
            fg=COLORS['accent'],
            relief=tk.SOLID,
            bd=BORDER['width'],
            padx=PADDING['small'],
            pady=PADDING['small']
        )
        self.export_csv_btn.pack(side=tk.LEFT, padx=(0, PADDING['small']))

        self.screenshot_btn = tk.Button(
            export_frame,
            text="📸 Screenshot",
            command=self._take_screenshot,
            font=(FONT_FAMILY, FONT_SIZES['small'], "bold"),
            bg=COLORS['accent_light'],
            fg=COLORS['accent'],
            relief=tk.SOLID,
            bd=BORDER['width'],
            padx=PADDING['small'],
            pady=PADDING['small']
        )
        self.screenshot_btn.pack(side=tk.LEFT)

        # Status
        self.status_label = tk.Label(
            controls,
            text="Ready",
            font=(FONT_FAMILY, FONT_SIZES['caption']),
            bg=COLORS['background'],
            fg=COLORS['text_secondary']
        )
        self.status_label.pack(side=tk.RIGHT, padx=(PADDING['medium'], 0))

        # Net charge display
        self.charge_label = tk.Label(
            controls,
            text="Net Charge: 0",
            font=(FONT_FAMILY, FONT_SIZES['caption'], "bold"),
            bg=COLORS['background'],
            fg=COLORS['accent']
        )
        self.charge_label.pack(side=tk.RIGHT, padx=(PADDING['medium'], 0))

    def _create_simulation_panel(self, parent):
        """Left panel: Clean simulation canvas"""
        # Canvas frame with minimal border
        canvas_frame = tk.Frame(
            parent,
            bg=COLORS['panel_bg'],
            relief=tk.SOLID,
            bd=BORDER['width']
        )
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, MARGINS['panel']))

        # Caption label
        caption = tk.Label(
            canvas_frame,
            text="Figure 1: Molecular Dynamics Trajectory",
            font=(FONT_FAMILY, FONT_SIZES['caption'], "italic"),
            bg=COLORS['panel_bg'],
            fg=COLORS['text_secondary']
        )
        caption.pack(anchor=tk.W, padx=PADDING['medium'], pady=(PADDING['medium'], 0))

        # Canvas
        self.canvas_figure = Figure(
            figsize=(6, 4.5),  # Smaller figure to fit better
            dpi=100,
            facecolor=PLOT_STYLE['figure_bg'],
            tight_layout=True
        )
        self.canvas = FigureCanvasTkAgg(self.canvas_figure, master=canvas_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(
            fill=tk.BOTH,
            expand=True,
            padx=PADDING['medium'],
            pady=(PADDING['small'], PADDING['medium'])
        )
        # Ensure canvas resizes properly
        canvas_widget.configure(bg=COLORS['panel_bg'])

    def _create_sidebar(self, parent):
        """Right panel: Element palette (top) and metrics/plots (bottom)"""
        sidebar = tk.Frame(parent, bg=COLORS['background'], width=SIZES['sidebar_width'])
        sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Element palette (top section)
        self._create_element_palette(sidebar)

        # Metrics and plots (bottom section)
        self._create_metrics_panel(sidebar)

    def _create_element_palette(self, parent):
        """Element palette with interactive periodic table"""
        palette_frame = tk.Frame(
            parent,
            bg=COLORS['panel_bg'],
            relief=tk.SOLID,
            bd=BORDER['width'],
            height=int(SIZES['sidebar_width'] * LAYOUT_RATIOS['palette_height_ratio'])
        )
        palette_frame.pack(fill=tk.X, pady=(0, MARGINS['panel']))
        palette_frame.pack_propagate(False)

        # Caption
        caption = tk.Label(
            palette_frame,
            text="Periodic Table",
            font=(FONT_FAMILY, FONT_SIZES['subtitle'], "bold"),
            bg=COLORS['panel_bg'],
            fg=COLORS['text_primary']
        )
        caption.pack(anchor=tk.W, padx=PADDING['medium'], pady=(PADDING['medium'], PADDING['small']))

        # Periodic table widget
        table_container = tk.Frame(palette_frame, bg=COLORS['panel_bg'])
        table_container.pack(fill=tk.BOTH, expand=True, padx=PADDING['medium'], pady=(0, PADDING['medium']))

        # Create periodic table
        self.periodic_table = PeriodicTableWidget(
            table_container,
            on_element_select=self._insert_element,
            bg=COLORS['panel_bg']
        )
        self.periodic_table.pack(fill=tk.BOTH, expand=True)

    def _create_metrics_panel(self, parent):
        """Metrics and plots panel"""
        metrics_frame = tk.Frame(
            parent,
            bg=COLORS['panel_bg'],
            relief=tk.SOLID,
            bd=BORDER['width']
        )
        metrics_frame.pack(fill=tk.BOTH, expand=True)

        # Caption
        caption = tk.Label(
            metrics_frame,
            text="Simulation Metrics",
            font=(FONT_FAMILY, FONT_SIZES['subtitle'], "bold"),
            bg=COLORS['panel_bg'],
            fg=COLORS['text_primary']
        )
        caption.pack(anchor=tk.W, padx=PADDING['medium'], pady=(PADDING['medium'], PADDING['small']))

        # Metrics canvas
        self.metrics_figure = Figure(
            figsize=(3.5, 4),
            dpi=100,
            facecolor=PLOT_STYLE['figure_bg']
        )
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_figure, master=metrics_frame)
        self.metrics_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
            padx=PADDING['medium'],
            pady=(0, PADDING['medium'])
        )

        # Initialize empty plots
        self._setup_metrics_plots()

    def _setup_metrics_plots(self):
        """Initialize clean metrics plots with separate energy components"""
        if not self.metrics_figure:
            return

        self.metrics_figure.clear()

        # Create subplots: energy components on top, products on bottom left, current state on bottom right
        gs = self.metrics_figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        # Energy plot with kinetic, potential, and total
        ax1 = self.metrics_figure.add_subplot(gs[0, :])
        ax1.set_title('Energy Components (Normalized)', fontsize=FONT_SIZES['caption'], pad=5)
        ax1.set_xlabel('Time Step', fontsize=FONT_SIZES['small'])
        ax1.set_ylabel('Energy (a.u.)', fontsize=FONT_SIZES['small'])
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right', fontsize=FONT_SIZES['small'])

        # Products over time
        ax2 = self.metrics_figure.add_subplot(gs[1, 0])
        ax2.set_title('Products Formed', fontsize=FONT_SIZES['caption'], pad=5)
        ax2.set_xlabel('Time Step', fontsize=FONT_SIZES['small'])
        ax2.set_ylabel('Count', fontsize=FONT_SIZES['small'])
        ax2.grid(True, alpha=0.3)

        # Current metrics summary
        ax3 = self.metrics_figure.add_subplot(gs[1, 1])
        ax3.set_title('Current State', fontsize=FONT_SIZES['caption'], pad=5)
        ax3.axis('off')

        # Initialize energy lines
        self.kinetic_line, = ax1.plot([], [], 'b-', linewidth=PLOT_STYLE['line_width'],
                                     label='Kinetic', alpha=0.8)
        self.potential_line, = ax1.plot([], [], 'r-', linewidth=PLOT_STYLE['line_width'],
                                       label='Potential', alpha=0.8)
        self.total_energy_line, = ax1.plot([], [], 'k-', linewidth=PLOT_STYLE['line_width'],
                                          label='Total', alpha=0.9)

        self.products_line, = ax2.plot([], [], 'g-', linewidth=PLOT_STYLE['line_width'],
                                      label='Products')

        self.metrics_canvas.draw()

    def _insert_element(self, symbol: str, oxidation_state: int = 0):
        """Insert selected element into running simulation"""
        if not self.sim:
            messagebox.showwarning("No Simulation", "Start a simulation first")
            return

        try:
            element_data = get_element(symbol)
            if not element_data:
                return

            # Add atom at random position with specified oxidation state
            pos = np.random.rand(2) * 0.8 + 0.1  # Keep within bounds
            atom = Atom(symbol, pos, oxidation_state=oxidation_state)
            self.sim.add_atom(atom)

            # Visual feedback
            btn = self.element_buttons.get(symbol)
            if btn:
                original_bg = btn.cget('bg')
                btn.config(bg=COLORS['success'])
                self.after(200, lambda: btn.config(bg=original_bg))

        except Exception as e:
            logger.exception(f"Failed to insert {symbol}: {e}")

    def _on_run(self):
        """Start simulation"""
        formula_str = self.formula_var.get().strip()
        if not formula_str:
            messagebox.showerror("Input Error", "Enter initial molecules (e.g., H2,O)")
            return

        try:
            frames = self.frames_var.get()
            formula_dicts = [parse_formula(f) for f in formula_str.split(",") if f.strip()]
            if not formula_dicts:
                raise ValueError("No valid formulas")
        except Exception as e:
            messagebox.showerror("Parse Error", str(e))
            return

        # Setup simulation
        det = self.deterministic_var.get()
        
        # Parse seed
        seed_str = self.seed_var.get().strip()
        seed = int(seed_str) if seed_str else None
        
        self.sim = SimulationManager(formula_dicts, deterministic_mode=det, seed=seed)

        # Reset UI state
        self._sim_stop.clear()
        self._sim_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text=f"Running simulation...")

        # Reset products history
        self._products_history = []

        # Start simulation thread
        self._sim_thread = threading.Thread(target=self._sim_loop, args=(frames,), daemon=True)
        self._sim_thread.start()

        # Start UI updates
        self._periodic_update()

    def _on_stop(self):
        """Stop simulation"""
        self._sim_stop.set()
        self._sim_running = False
        self.stop_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.NORMAL)
        self.status_label.config(text="Stopped")

    def _sim_loop(self, frames: int):
        """Main simulation loop"""
        try:
            for i in range(frames):
                if self._sim_stop.is_set():
                    break
                self.sim.step()
                time.sleep(0.005)  # Slightly slower for better visualization
        except Exception as e:
            logger.exception(f"Simulation error: {e}")
        finally:
            self._sim_running = False
            self.stop_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.NORMAL)
            self.status_label.config(text="Complete")

    def _periodic_update(self):
        """Update GUI with simulation state"""
        if not self.sim or not self._sim_running:
            return

        try:
            # Update main canvas
            if self.canvas and self.sim:
                # Ensure sim uses the canvas's figure
                self.sim.fig = self.canvas.figure
                if not hasattr(self.sim, 'main_ax') or self.sim.main_ax is None:
                    self.sim.main_ax = self.sim.fig.add_subplot(1, 1, 1)

                render_simulation_frame(self.sim)
                self.canvas.draw_idle()

            # Update charge display
            net_charge = self.sim.get_net_charge()
            charge_text = f"Net Charge: {net_charge:+.1f}"
            self.charge_label.config(text=charge_text)

            # Update metrics plots
            self._update_metrics_plots()

        except Exception as e:
            logger.exception(f"Update error: {e}")

        # Schedule next update
        self._periodic_update_id = self.after(100, self._periodic_update)

    def _update_metrics_plots(self):
        """Update metrics plots with stable, normalized energy data"""
        if not self.sim or not self.metrics_figure:
            return

        frame = self.sim.frame

        # Get normalized, smoothed energy data
        energy_data = self.sim.energy_metrics.get_plot_data(max_points=200)  # Limit for performance

        # Update energy plot with separate components
        if energy_data['x'].size > 0:
            # Update line data
            self.kinetic_line.set_data(energy_data['x'], energy_data['kinetic'])
            self.potential_line.set_data(energy_data['x'], energy_data['potential'])
            self.total_energy_line.set_data(energy_data['x'], energy_data['total'])

            # Set stable axis limits to prevent spikes from dominating view
            ax1 = self.metrics_figure.axes[0]
            ylim = self.sim.energy_metrics.get_axis_limits('total', padding=0.2)
            ax1.set_ylim(ylim)
            ax1.relim()
            ax1.autoscale_view(scalex=True, scaley=False)  # Only autoscale x-axis

        # Update products plot
        products_count = len(self.sim.detect_products())
        if hasattr(self, '_products_history'):
            self._products_history.append(products_count)
        else:
            self._products_history = [products_count]

        if len(self._products_history) > 1:
            x_prod = np.arange(len(self._products_history))
            self.products_line.set_data(x_prod, self._products_history)

            ax2 = self.metrics_figure.axes[1]
            ax2.relim()
            ax2.autoscale_view()

        # Update current state with detailed energy summary
        ax3 = self.metrics_figure.axes[2]
        ax3.clear()
        ax3.set_title('Current State', fontsize=FONT_SIZES['caption'], pad=5)
        ax3.axis('off')

        # Get current energy summary
        energy_summary = self.sim.energy_metrics.get_current_summary()

        state_text = f"""Step: {frame}
Atoms: {len(self.sim.atoms)}
Bonds: {len(self.sim.bonds)}
Products: {products_count}

Energy ({energy_summary['normalization']}):
  Kinetic: {energy_summary['kinetic']:.3f}
  Potential: {energy_summary['potential']:.3f}
  Total: {energy_summary['total']:.3f}
  Drift: {energy_summary['drift']:.1%}"""

        ax3.text(0.05, 0.95, state_text,
                fontsize=FONT_SIZES['small'],
                verticalalignment='top',
                fontfamily=FONT_FAMILY,
                transform=ax3.transAxes)

        self.metrics_canvas.draw_idle()

    def _export_config(self):
        """Export simulation configuration to JSON file"""
        if not self.sim:
            messagebox.showwarning("No Simulation", "Run a simulation first")
            return

        try:
            config = {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "formulas": self.formula_var.get(),
                "frames": self.frames_var.get(),
                "deterministic_mode": self.deterministic_var.get(),
                "seed": self.seed_var.get() or None,
                "temperature": self.sim.temperature,
                "current_frame": self.sim.frame,
                "net_charge": self.sim.get_net_charge(),
                "atom_count": len(self.sim.atoms),
                "bond_count": len(self.sim.bonds)
            }

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Configuration"
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config, f, indent=2)
                messagebox.showinfo("Export Complete", f"Configuration saved to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_state(self):
        """Export current simulation state"""
        if not self.sim:
            messagebox.showwarning("No Simulation", "Run a simulation first")
            return

        try:
            state = {
                "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                "frame": self.sim.frame,
                "atoms": [
                    {
                        "uid": atom.uid,
                        "symbol": atom.symbol,
                        "position": atom.pos.tolist(),
                        "velocity": atom.vel.tolist(),
                        "charge": atom.charge,
                        "oxidation_state": getattr(atom, 'oxidation_state', 0)
                    }
                    for atom in self.sim.atoms
                ],
                "bonds": [
                    {
                        "atom1_uid": bond.atom1.uid,
                        "atom2_uid": bond.atom2.uid,
                        "order": bond.order,
                        "bond_type": bond.get_bond_type()
                    }
                    for bond in self.sim.bonds
                ],
                "energy": self.sim.energy_metrics.get_current_energy()
            }

            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
                title="Save Simulation State"
            )
            if filename:
                with open(filename, 'w') as f:
                    json.dump(state, f, indent=2)
                messagebox.showinfo("Export Complete", f"Simulation state saved to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _export_csv(self):
        """Export metrics data to CSV"""
        if not self.sim:
            messagebox.showwarning("No Simulation", "Run a simulation first")
            return

        try:
            # Get energy data
            energy_data = self.sim.energy_metrics.get_plot_data(max_points=None)
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Metrics to CSV"
            )
            if filename:
                import csv
                with open(filename, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frame', 'Kinetic_Energy', 'Potential_Energy', 'Total_Energy', 'Temperature'])
                    
                    for i in range(len(energy_data['x'])):
                        writer.writerow([
                            energy_data['x'][i],
                            energy_data['kinetic'][i],
                            energy_data['potential'][i], 
                            energy_data['total'][i],
                            self.sim.temperature  # Constant temperature
                        ])
                messagebox.showinfo("Export Complete", f"Metrics exported to {filename}")

        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def _take_screenshot(self):
        """Take screenshot of the main simulation canvas"""
        if not self.sim or not self.canvas_figure:
            messagebox.showwarning("No Simulation", "Run a simulation first")
            return

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Screenshot"
            )
            if filename:
                self.canvas_figure.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Screenshot Saved", f"Image saved to {filename}")

        except Exception as e:
            messagebox.showerror("Screenshot Error", str(e))


def main():
    """Launch the publication-ready GUI"""
    app = PublicationGUI()
    app.mainloop()


if __name__ == "__main__":
    main()