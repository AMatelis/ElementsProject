from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from tkinter import BooleanVar, StringVar
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
from typing import Optional, Dict, List
import os
import shutil
import webbrowser

# -----------------------
# Engine / Simulation imports
# -----------------------
from engine.simulation_manager import SimulationManager
from engine.elements_data import ELEMENT_DATA
from engine.formula_parser import parse_formula
from engine.products import graph_to_formula, explain_component, generate_reaction_story
# -----------------------
# Visualization imports
# -----------------------
from visual.renderer import render_simulation_frame
from visual.heatmap import compute_reactivity_heatmap
from visual.export_tools import (
    export_simulation_to_plotly_html,
    export_simulation_to_gif,
    export_simulation_to_mp4,
    export_product_timeline,
    ensure_dir,
    now_str,
    publish_directory_to_github,
)


class SimulationGUI:
    """
    Base GUI for molecular simulation.
    Fully modular and publication-ready.
    """

    def __init__(self, sim: Optional[SimulationManager] = None, title: str = "Reaction Simulation GUI"):
        self.sim: Optional[SimulationManager] = sim
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("900x700")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self._sim_thread: Optional[threading.Thread] = None
        self._sim_stop = threading.Event()
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.canvas_figure = None

        # GUI state
        self.formula_entry = None
        self.frames_spin = None
        self.det_var = BooleanVar(value=True)
        self.product_list = None
        self.explain_text = None
        self.heatmap_var = BooleanVar(value=False)
        self.preset_var = StringVar()
        self.mol_presets = {'Water':'H2,O', 'Sodium Chloride':'Na,Cl', 'Carbon Dioxide':'C,O2', 'Benzene':'C6,H6'}
        self.mol_preset_var = StringVar(value='Water')
        self.auto_open_var = BooleanVar(value=True)

    # -----------------------
    # Tkinter GUI helpers
    # -----------------------
    def start(self):
        self.main_menu()
        self.root.mainloop()

    def _init_canvas(self):
        if self.canvas is not None:
            return
        self.canvas_figure = plt.figure(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.canvas_figure, master=self.main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    # -----------------------
    # Screens
    # -----------------------
    def main_menu(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        tk.Label(self.main_frame, text="Reaction Simulation GUI", font=("Arial", 16)).pack(pady=15)
        tk.Button(self.main_frame, text="Formula Input", command=self.formula_screen).pack(pady=10)
        tk.Button(self.main_frame, text="Run Simulation", command=self.run_simulation).pack(pady=10)

    def run_simulation(self):
        """Entry point for the Run Simulation action in the menu."""
        # Show the formula input screen where the user can run the sim
        self.formula_screen()

    def formula_screen(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        frame_controls = tk.Frame(self.main_frame)
        frame_controls.pack(side='top', fill='x', padx=6, pady=6)

        tk.Label(frame_controls, text="Formulas (comma-separated)").grid(row=0, column=0, sticky='w')
        self.formula_entry = tk.Entry(frame_controls, width=50)
        self.formula_entry.grid(row=0, column=1, padx=4)
        self.formula_entry.insert(0, "H2,O")

        tk.Label(frame_controls, text="Frames").grid(row=1, column=0, sticky='w')
        self.frames_spin = tk.Spinbox(frame_controls, from_=10, to=2000, increment=10, width=6)
        self.frames_spin.grid(row=1, column=1, sticky='w')
        self.frames_spin.delete(0, 'end'); self.frames_spin.insert(0, '200')

        tk.Checkbutton(frame_controls, text='Deterministic', variable=self.det_var).grid(row=0, column=2, padx=6)
        tk.Button(frame_controls, text="Run & Show", command=self._run_with_entry).grid(row=1, column=2, padx=6)
        tk.Button(frame_controls, text="Stop", command=self._stop_run, state='disabled').grid(row=1, column=3, padx=6)

        # Canvas
        self._init_canvas()

        # Products panel
        side_frame = tk.Frame(self.main_frame)
        side_frame.pack(side='right', fill='y', padx=6, pady=6)
        tk.Label(side_frame, text='Products').pack()
        self.product_list = tk.Listbox(side_frame, height=8)
        self.product_list.pack(fill='y')
        self.explain_text = tk.Text(side_frame, width=40, height=12)
        self.explain_text.pack(pady=4)
        tk.Checkbutton(side_frame, text='Heatmap overlay', variable=self.heatmap_var, command=self._refresh_products).pack(pady=4)

    # -----------------------
    # Simulation handlers
    # -----------------------
    def _run_with_entry(self):
        formulas = self.formula_entry.get().strip()
        if not formulas:
            messagebox.showerror("Input Error", "Enter at least one formula")
            return
        try:
            frames = int(self.frames_spin.get())
        except Exception:
            frames = 200
        det = bool(self.det_var.get())

        formula_dicts = [parse_formula(f) for f in formulas.split(",") if f.strip()]
        self.sim = SimulationManager(formula_dicts, deterministic_mode=det)
        self._sim_stop.clear()

        self._sim_thread = threading.Thread(target=self._sim_loop, args=(frames,), daemon=True)
        self._sim_thread.start()
        self._periodic_update()

    def _sim_loop(self, frames: int):
        try:
            for _ in range(frames):
                if self._sim_stop.is_set():
                    break
                self.sim.step()
                time.sleep(0.01)
        finally:
            self._refresh_products()

    def _periodic_update(self):
        if not self.sim:
            self.root.after(200, self._periodic_update)
            return
        try:
            render_simulation_frame(self.sim)
            if self.canvas and self.sim.fig:
                self.canvas.figure = self.sim.fig
                self.canvas.draw_idle()
            self._refresh_products()
            if self.heatmap_var.get():
                xx, yy, heat = compute_reactivity_heatmap(self.sim)
                self.sim.ax.imshow(heat, extent=[0,1,0,1], origin='lower', cmap='inferno', alpha=0.45)
        except Exception:
            pass
        self.root.after(200, self._periodic_update)

    def _stop_run(self):
        self._sim_stop.set()

    def _refresh_products(self):
        if not self.sim:
            return
        self.product_list.delete(0, 'end')
        products = self.sim.detect_products()
        for k, (atoms, bonds) in products.items():
            formula = ''.join([f"{el}{c if c>1 else ''}" for el, c in sorted(graph_to_formula(atoms, bonds).items())])
            self.product_list.insert('end', f"{k}: {formula}")

    # -----------------------
    # Example: show explanation
    # -----------------------
    def _explain_selected(self):
        sel = self.product_list.curselection()
        if not sel:
            messagebox.showerror("Selection Error", "Select a product to explain")
            return
        key = self.product_list.get(sel[0]).split(":")[0]
        self.explain_text.delete('1.0', 'end')
        try:
            products = self.sim.detect_products()
            explanation = explain_component(self.sim, key, products)
            # Display the explanation in the text widget
            if isinstance(explanation, str):
                self.explain_text.insert('1.0', explanation)
            else:
                self.explain_text.insert('1.0', str(explanation))
        except Exception as e:
            self.explain_text.insert('1.0', f"Explain failed: {e}")


class SimulationGUIAdvanced(SimulationGUI):
    """
    Advanced GUI with presets, export buttons, and optional ML integrations.
    This subclass can be extended later; for now it uses the base GUI behavior.
    """
    def __init__(self, sim: Optional[SimulationManager] = None, title: str = "Reaction Simulation GUI - Advanced"):
        super().__init__(sim=sim, title=title)


class ControlPanel:
    """Simple ControlPanel placeholder used by `gui.__init__`.

    This lightweight object groups a few control callbacks for integration
    with other parts of the application. It can be extended later.
    """
    def __init__(self, gui: SimulationGUI):
        self.gui = gui

    def start(self):
        # No-op for now; real app may embed controls into a parent window
        return