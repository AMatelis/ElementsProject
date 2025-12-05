"""
Professional-grade Advanced Simulation GUI
- Dark mode with modern styling
- Large zoomed-out visualization
- Multi-panel analytics dashboard
- Energy & thermodynamic analysis
- Publication-ready exports
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

# Configure matplotlib for dark mode
matplotlib.style.use('dark_background')
plt.rcParams['figure.facecolor'] = '#1e1e1e'
plt.rcParams['axes.facecolor'] = '#2d2d2d'
plt.rcParams['font.size'] = 9
plt.rcParams['lines.linewidth'] = 1.5

# Engine imports
from engine.simulation_manager import SimulationManager
from engine.elements_data import ELEMENT_DATA, load_elements
from engine.formula_parser import parse_formula
from engine.products import graph_to_formula

# Visualization imports
from visual.renderer import render_simulation_frame
from visual.heatmap import compute_reactivity_heatmap
from visual.export_tools import (
    export_simulation_to_plotly_html,
    export_simulation_to_gif,
    export_simulation_to_mp4,
    ensure_dir,
    now_str,
)


# ────────────────────────────
# Theme Configuration
# ────────────────────────────

THEME = {
    'bg': '#1a1a1a',
    'fg': '#e0e0e0',
    'accent': '#00d4ff',
    'accent_dark': '#0099cc',
    'success': '#00cc44',
    'warning': '#ffaa00',
    'danger': '#ff4444',
    'panel': '#2d2d2d',
    'border': '#404040',
}


# ────────────────────────────
# Advanced Chemistry Analytics
# ────────────────────────────

class ChemistryAnalyzer:
    """Advanced chemistry metrics and analysis"""
    
    def __init__(self):
        self.history = {
            'frames': deque(maxlen=1000),
            'n_atoms': deque(maxlen=1000),
            'n_bonds': deque(maxlen=1000),
            'n_products': deque(maxlen=1000),
            'avg_bond_energy': deque(maxlen=1000),
            'kinetic_energy': deque(maxlen=1000),
            'reaction_rate': deque(maxlen=1000),
        }
    
    def update(self, sim: SimulationManager, frame: int):
        """Record metrics from simulation"""
        self.history['frames'].append(frame)
        self.history['n_atoms'].append(len(sim.atoms))
        self.history['n_bonds'].append(len(sim.bonds))
        
        products = sim.detect_products()
        self.history['n_products'].append(len(products))
        
        # Calculate average bond energy (simplified)
        if sim.bonds:
            avg_energy = sum(b.k_spring * 0.1 for b in sim.bonds) / len(sim.bonds)
        else:
            avg_energy = 0
        self.history['avg_bond_energy'].append(avg_energy)
        
        # Kinetic energy
        ke = sum(0.5 * getattr(a, 'mass', 12.0) * np.linalg.norm(a.vel)**2 
                 for a in sim.atoms) if sim.atoms else 0
        self.history['kinetic_energy'].append(ke)
        
        # Reaction rate (bonds formed per frame)
        rate = len(sim.bonds) / max(1, frame) if frame > 0 else 0
        self.history['reaction_rate'].append(rate)
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        if not self.history['frames']:
            return {}
        
        return {
            'avg_bonds': np.mean(list(self.history['n_bonds'])) if self.history['n_bonds'] else 0,
            'max_bonds': max(self.history['n_bonds']) if self.history['n_bonds'] else 0,
            'avg_ke': np.mean(list(self.history['kinetic_energy'])) if self.history['kinetic_energy'] else 0,
            'max_ke': max(self.history['kinetic_energy']) if self.history['kinetic_energy'] else 0,
            'max_products': max(self.history['n_products']) if self.history['n_products'] else 0,
        }


# ────────────────────────────
# Professional GUI
# ────────────────────────────

class SimulationGUIAdvanced(tk.Tk):
    """Production-grade molecular simulation GUI"""

    def __init__(self, title: str = "Molecular Dynamics Simulator"):
        super().__init__()
        self.title(title)
        self.geometry("1600x1000")
        
        # Set overall theme
        self.configure(bg=THEME['bg'])
        
        # Simulation state
        self.sim: Optional[SimulationManager] = None
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop = threading.Event()
        self._sim_running = False
        self._periodic_update_id: Optional[str] = None
        
        # Analytics
        self.analyzer = ChemistryAnalyzer()
        
        # UI Components
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.canvas_figure: Optional[Figure] = None
        self.charts_figure: Optional[Figure] = None
        self.charts_canvas: Optional[FigureCanvasTkAgg] = None
        
        # Control variables
        self.det_var = BooleanVar(value=True)
        self.heatmap_var = BooleanVar(value=True)
        self.show_charts_var = BooleanVar(value=True)
        self.formula_entry: Optional[tk.Entry] = None
        self.frames_spin: Optional[tk.Spinbox] = None
        self.run_button: Optional[tk.Button] = None
        self.stop_button: Optional[tk.Button] = None
        self.status_label: Optional[tk.Label] = None
        
        # Output panels
        self.product_list: Optional[tk.Listbox] = None
        self.metrics_text: Optional[tk.Text] = None
        
        # Setup
        load_elements()
        self._setup_ui()

    def _setup_ui(self):
        """Build professional UI layout"""
        # ═══════════════════════════════════════
        # HEADER / Control Panel
        # ═══════════════════════════════════════
        header = tk.Frame(self, bg=THEME['panel'], height=80)
        header.pack(side=tk.TOP, fill=tk.X)
        header.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header, text="⚛ Molecular Dynamics Simulator",
            font=("Helvetica", 16, "bold"), bg=THEME['panel'], fg=THEME['accent']
        )
        title_label.pack(anchor=tk.W, padx=15, pady=(8, 0))
        
        # Control row
        ctrl_frame = tk.Frame(header, bg=THEME['panel'])
        ctrl_frame.pack(fill=tk.X, padx=15, pady=(8, 8))
        
        tk.Label(ctrl_frame, text="Formulas:", bg=THEME['panel'], fg=THEME['fg'], 
                font=("Courier", 10)).pack(side=tk.LEFT, padx=(0, 5))
        
        self.formula_entry = tk.Entry(
            ctrl_frame, width=25, bg=THEME['border'], fg=THEME['accent'],
            insertbackground=THEME['accent'], font=("Courier", 10), relief=tk.FLAT, bd=1
        )
        self.formula_entry.pack(side=tk.LEFT, padx=5)
        self.formula_entry.insert(0, "H2,O")
        
        tk.Label(ctrl_frame, text="Frames:", bg=THEME['panel'], fg=THEME['fg'],
                font=("Courier", 10)).pack(side=tk.LEFT, padx=(15, 5))
        
        self.frames_spin = tk.Spinbox(
            ctrl_frame, from_=50, to=10000, increment=50, width=6,
            bg=THEME['border'], fg=THEME['accent'], font=("Courier", 10),
            relief=tk.FLAT, bd=1
        )
        self.frames_spin.pack(side=tk.LEFT, padx=5)
        self.frames_spin.delete(0, tk.END)
        self.frames_spin.insert(0, "800")
        
        # Checkboxes
        for i, (var, label) in enumerate([
            (self.det_var, "Deterministic"),
            (self.heatmap_var, "Heatmap"),
            (self.show_charts_var, "Charts"),
        ]):
            cb = tk.Checkbutton(
                ctrl_frame, text=label, variable=var,
                bg=THEME['panel'], fg=THEME['fg'], selectcolor=THEME['border'],
                activebackground=THEME['panel'], activeforeground=THEME['accent'],
                font=("Courier", 9)
            )
            cb.pack(side=tk.LEFT, padx=10)
        
        # Buttons
        self.run_button = self._make_button(ctrl_frame, "▶ RUN", self._on_run, THEME['success'], side=tk.LEFT)
        self.stop_button = self._make_button(ctrl_frame, "⏹ STOP", self._on_stop, THEME['danger'], 
                                             side=tk.LEFT, state=tk.DISABLED)
        
        # Status
        self.status_label = tk.Label(
            header, text="Ready", bg=THEME['panel'], fg=THEME['fg'],
            font=("Courier", 9)
        )
        self.status_label.pack(anchor=tk.E, padx=15, pady=(0, 8))
        
        # ═══════════════════════════════════════
        # MAIN CONTENT
        # ═══════════════════════════════════════
        content = tk.Frame(self, bg=THEME['bg'])
        content.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Left panel: Main visualization (larger)
        left = tk.Frame(content, bg=THEME['panel'], relief=tk.SUNKEN, bd=1)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        
        tk.Label(left, text="VISUALIZATION (Zoom: 0-1)", bg=THEME['panel'], 
                fg=THEME['accent'], font=("Courier", 9, "bold")).pack(anchor=tk.W, padx=8, pady=4)
        
        self._init_main_canvas(left)
        
        # Right panel: Analytics & Controls
        right = tk.Frame(content, bg=THEME['bg'], width=320)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)
        right.pack_propagate(False)
        
        # Right-top: Products
        prod_frame = tk.Frame(right, bg=THEME['panel'], relief=tk.SUNKEN, bd=1)
        prod_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        tk.Label(prod_frame, text="PRODUCTS", bg=THEME['panel'], fg=THEME['accent'],
                font=("Courier", 9, "bold")).pack(anchor=tk.W, padx=8, pady=4)
        
        self.product_list = tk.Listbox(
            prod_frame, bg=THEME['border'], fg=THEME['fg'], font=("Courier", 8),
            relief=tk.FLAT, bd=0, selectmode=tk.SINGLE, height=6
        )
        self.product_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        
        # Right-bottom: Metrics
        metrics_frame = tk.Frame(right, bg=THEME['panel'], relief=tk.SUNKEN, bd=1)
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(metrics_frame, text="METRICS", bg=THEME['panel'], fg=THEME['accent'],
                font=("Courier", 9, "bold")).pack(anchor=tk.W, padx=8, pady=4)
        
        self.metrics_text = tk.Text(
            metrics_frame, bg=THEME['border'], fg=THEME['fg'], font=("Courier", 8),
            relief=tk.FLAT, bd=0, height=12, wrap=tk.WORD, state=tk.DISABLED
        )
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Export buttons
        export_frame = tk.Frame(right, bg=THEME['bg'])
        export_frame.pack(fill=tk.X, pady=(8, 0))
        
        for label, cmd in [("HTML", self._export_html), ("GIF", self._export_gif), ("MP4", self._export_mp4)]:
            self._make_button(export_frame, label, cmd, THEME['accent'], side=tk.LEFT, width=8)

    def _make_button(self, parent, text, cmd, color, side=tk.LEFT, width=None, state=tk.NORMAL):
        """Create themed button"""
        btn = tk.Button(
            parent, text=text, command=cmd, bg=color, fg=THEME['bg'],
            font=("Helvetica", 9, "bold"), relief=tk.FLAT, bd=0,
            activebackground=color, activeforeground=THEME['bg'],
            padx=12, pady=6, state=state
        )
        if width:
            btn.config(width=width)
        btn.pack(side=side, padx=4)
        return btn

    def _init_main_canvas(self, parent):
        """Large zoomed-out main visualization"""
        self.canvas_figure = Figure(figsize=(9.5, 6.5), dpi=100, 
                                    facecolor=THEME['panel'], edgecolor=THEME['border'])
        self.canvas = FigureCanvasTkAgg(self.canvas_figure, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

    def _on_run(self):
        """Start simulation"""
        formula_str = self.formula_entry.get().strip()
        if not formula_str:
            messagebox.showerror("Input", "Enter formulas (e.g., H2,O)")
            return
        
        try:
            frames = int(self.frames_spin.get())
            formula_dicts = [parse_formula(f) for f in formula_str.split(",") if f.strip()]
            if not formula_dicts:
                raise ValueError("No valid formulas")
        except Exception as e:
            messagebox.showerror("Parse Error", str(e))
            return
        
        # Setup
        det = bool(self.det_var.get())
        self.sim = SimulationManager(formula_dicts, deterministic_mode=det)
        self.analyzer = ChemistryAnalyzer()
        self._sim_stop.clear()
        self._sim_running = True
        
        # UI update
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self._set_status(f"Running {len(formula_dicts)} molecules for {frames} frames...")
        
        # Start
        self._sim_thread = threading.Thread(target=self._sim_loop, args=(frames,), daemon=True)
        self._sim_thread.start()
        self._periodic_update()

    def _on_stop(self):
        """Stop simulation gracefully"""
        self._sim_stop.set()
        self._sim_running = False
        self.stop_button.config(state=tk.DISABLED)
        self.run_button.config(state=tk.NORMAL)
        self._set_status("Stopped")

    def _sim_loop(self, frames: int):
        """Main simulation loop"""
        try:
            for i in range(frames):
                if self._sim_stop.is_set():
                    break
                try:
                    self.sim.step()
                    self.analyzer.update(self.sim, i)
                except Exception as e:
                    logger.exception(f"Step {i}: {e}")
                time.sleep(0.002)
        finally:
            self._sim_running = False
            self.stop_button.config(state=tk.DISABLED)
            self.run_button.config(state=tk.NORMAL)
            self._set_status("Complete")

    def _periodic_update(self):
        """GUI refresh loop"""
        if not self.sim or not self._sim_running:
            if self._sim_running:
                self.after(50, self._periodic_update)
            return
        
        try:
            # Render main frame with extended zoom-out
            render_simulation_frame(self.sim)
            if self.canvas and self.sim.fig:
                ax = self.sim.ax
                if ax:
                    ax.set_xlim(-0.1, 1.1)
                    ax.set_ylim(-0.1, 1.1)
                
                # Heatmap overlay
                if self.heatmap_var.get() and ax:
                    try:
                        xx, yy, heat = compute_reactivity_heatmap(self.sim)
                        ax.imshow(heat, extent=[0, 1, 0, 1], origin='lower', 
                                 cmap='hot', alpha=0.2, zorder=1)
                    except:
                        pass
                
                self.canvas.figure = self.sim.fig
                self.canvas.draw_idle()
            
            # Update info panels
            self._update_products()
            self._update_metrics()
        except Exception as e:
            logger.exception(f"Update error: {e}")
        
        self._periodic_update_id = self.after(100, self._periodic_update)

    def _update_products(self):
        """Update product list"""
        if not self.sim:
            return
        
        self.product_list.delete(0, tk.END)
        products = self.sim.detect_products()
        
        for key, (atoms, bonds) in products.items():
            formula = graph_to_formula(atoms, bonds)
            pretty = ''.join([f"{el}{c if c > 1 else ''}" for el, c in sorted(formula.items())])
            self.product_list.insert(tk.END, f"{pretty}")

    def _update_metrics(self):
        """Update metrics display"""
        if not self.sim:
            return
        
        summary = self.analyzer.get_summary()
        
        text = f"""Frame: {self.sim.frame}
Atoms: {len(self.sim.atoms)}
Bonds: {len(self.sim.bonds)}

Products: {len(self.sim.detect_products())}

─────────────────
MAX BONDS: {summary.get('max_bonds', 0):.0f}
AVG BONDS: {summary.get('avg_bonds', 0):.1f}

KINETIC ENERGY:
  Max: {summary.get('max_ke', 0):.2e}
  Avg: {summary.get('avg_ke', 0):.2e}

PRODUCTS FOUND: {summary.get('max_products', 0)}

Deterministic: {'✓' if self.sim.deterministic_mode else '✗'}
"""
        
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete('1.0', tk.END)
        self.metrics_text.insert('1.0', text)
        self.metrics_text.config(state=tk.DISABLED)

    def _set_status(self, msg: str):
        """Update status label"""
        self.status_label.config(text=msg)
        self.update_idletasks()

    # ────────────────────────────
    # Export Functions
    # ────────────────────────────

    def _export_html(self):
        """Export to interactive HTML"""
        if not self.sim:
            messagebox.showwarning("No Sim", "Run simulation first")
            return
        fname = filedialog.asksaveasfilename(defaultext=".html", 
                                            filetypes=[("HTML", "*.html"), ("All", "*.*")])
        if fname:
            try:
                export_simulation_to_plotly_html(self.sim, fname)
                messagebox.showinfo("Export", f"Saved: {os.path.basename(fname)}")
                import webbrowser
                webbrowser.open(f"file:///{fname}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _export_gif(self):
        """Export to GIF"""
        if not self.sim:
            messagebox.showwarning("No Sim", "Run simulation first")
            return
        fname = filedialog.asksaveasfilename(defaultext=".gif",
                                            filetypes=[("GIF", "*.gif"), ("All", "*.*")])
        if fname:
            try:
                export_simulation_to_gif(self.sim, fname)
                messagebox.showinfo("Export", f"Saved: {os.path.basename(fname)}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _export_mp4(self):
        """Export to MP4"""
        if not self.sim:
            messagebox.showwarning("No Sim", "Run simulation first")
            return
        fname = filedialog.asksaveasfilename(defaultext=".mp4",
                                            filetypes=[("MP4", "*.mp4"), ("All", "*.*")])
        if fname:
            try:
                export_simulation_to_mp4(self.sim, fname)
                messagebox.showinfo("Export", f"Saved: {os.path.basename(fname)}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def start(self):
        """Start the GUI"""
        self.mainloop()

    def main_menu(self):
        """Compatibility stub"""
        pass


# Aliases for compatibility
class SimulationGUI(SimulationGUIAdvanced):
    pass
