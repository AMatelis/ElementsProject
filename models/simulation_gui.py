import tkinter as tk
from tkinter import messagebox, ttk, filedialog, simpledialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import threading
import time
from typing import Optional

# Import simulation helpers
# Avoid circular import; import simulation_viewer functions on demand inside methods


class SimulationGUI:
    """
    Base GUI class for the molecular reaction simulation.
    Provides a basic window and placeholder screens.
    """

    def __init__(self, sim=None, title="Reaction Simulation GUI"):
        self.sim = sim
        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("400x300")

        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True)

        self.logger = None  # Optional: attach a logger later

    def start(self):
        """Start the Tkinter event loop."""
        self.root.mainloop()

    def formula_screen(self):
        """
        Advanced formula input screen with embedded Matplotlib canvas and controls to run simulations.
        """
        # Clear old screen
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Top controls
        frame_controls = tk.Frame(self.main_frame)
        frame_controls.pack(side='top', fill='x', padx=6, pady=6)

        tk.Label(frame_controls, text="Formulas (comma-separated)").grid(row=0, column=0, sticky='w')
        self.formula_entry = tk.Entry(frame_controls, width=50)
        self.formula_entry.grid(row=0, column=1, padx=4)
        self.formula_entry.insert(0, "H2,O")

        tk.Label(frame_controls, text="Frames").grid(row=1, column=0, sticky='w')
        self.frames_spin = tk.Spinbox(frame_controls, from_=10, to=2000, increment=10, width=6)
        self.frames_spin.grid(row=1, column=1, sticky='w')
        self.frames_spin.delete(0, 'end')
        self.frames_spin.insert(0, '200')

        self.det_var = tk.BooleanVar(value=True)
        self.det_check = tk.Checkbutton(frame_controls, text='Deterministic', variable=self.det_var)
        self.det_check.grid(row=0, column=2, padx=6)

        self.run_btn = tk.Button(frame_controls, text="Run & Show", command=self._run_with_entry)
        self.run_btn.grid(row=1, column=2, padx=6)

        self.stop_btn = tk.Button(frame_controls, text="Stop", command=self._stop_run, state='disabled')
        self.stop_btn.grid(row=1, column=3, padx=6)

        # Allow loading a custom element file
        self.load_elements_btn = tk.Button(frame_controls, text='Load Elements JSON', command=self._load_elements_json)
        self.load_elements_btn.grid(row=0, column=3, padx=6)
        self.add_element_btn = tk.Button(frame_controls, text='Add Element', command=self._add_element_popup)
        self.add_element_btn.grid(row=1, column=3, padx=6)
        # preset element selector
        from simulation_viewer import ELEMENT_DATA
        vm = sorted(ELEMENT_DATA.keys())[:30]
        self.preset_var = tk.StringVar(value=vm[0] if vm else '')
        self.preset_menu = tk.OptionMenu(frame_controls, self.preset_var, *vm)
        self.preset_menu.grid(row=0, column=4, padx=6)
        self.insert_preset_btn = tk.Button(frame_controls, text='Insert', command=self._insert_preset_element)
        self.insert_preset_btn.grid(row=1, column=4, padx=6)
        # Molecule presets
        self.mol_presets = {'Water':'H2,O', 'Sodium Chloride':'Na,Cl', 'Carbon Dioxide':'C,O2', 'Benzene':'C6,H6'}
        self.mol_preset_var = tk.StringVar(value='Water')
        self.mol_preset_menu = tk.OptionMenu(frame_controls, self.mol_preset_var, *list(self.mol_presets.keys()))
        self.mol_preset_menu.grid(row=0, column=6, padx=6)
        self.insert_mol_preset_btn = tk.Button(frame_controls, text='Insert Molecule', command=self._insert_molecule_preset)
        self.insert_mol_preset_btn.grid(row=1, column=6, padx=6)
        self.run_publish_btn = tk.Button(frame_controls, text='Run & Publish', command=self._run_and_publish)
        self.run_publish_btn.grid(row=1, column=5, padx=6)
        # ensure presets are current
        self._refresh_presets()

        # Plot canvas
        self.plot_frame = tk.Frame(self.main_frame)
        self.plot_frame.pack(side='top', fill='both', expand=True, padx=6, pady=6)
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.canvas_figure = None

        # Right side panels: product list and explanation
        side_frame = tk.Frame(self.main_frame)
        side_frame.pack(side='right', fill='y', padx=6, pady=6)

        tk.Label(side_frame, text='Products').pack()
        self.product_list = tk.Listbox(side_frame, height=8)
        self.product_list.pack(fill='y')
        self.explain_btn = tk.Button(side_frame, text='Explain Product', command=self._explain_selected)
        self.explain_btn.pack(pady=4)
        self.export_timeline_btn = tk.Button(side_frame, text='Export Timeline', command=self._export_timeline_selected)
        self.export_timeline_btn.pack(pady=4)
        self.save_image_btn = tk.Button(side_frame, text='Save Product Image', command=self._save_product_image)
        self.save_image_btn.pack(pady=4)
        self.heatmap_var = tk.BooleanVar(value=False)
        self.heatmap_check = tk.Checkbutton(side_frame, text='Heatmap overlay', variable=self.heatmap_var, command=self._refresh_products)
        self.heatmap_check.pack(pady=4)
        self.story_btn = tk.Button(side_frame, text='Generate Story', command=self._generate_story_selected)
        self.story_btn.pack(pady=4)
        self.export_html_btn = tk.Button(side_frame, text='Export HTML', command=self._export_html)
        self.export_html_btn.pack(pady=4)
        self.export_gif_btn = tk.Button(side_frame, text='Export GIF', command=self._export_gif)
        self.export_gif_btn.pack(pady=4)
        self.export_mp4_btn = tk.Button(side_frame, text='Export MP4', command=self._export_mp4)
        self.export_mp4_btn.pack(pady=4)
        self.view_final_btn = tk.Button(side_frame, text='View Final', command=self._open_final_frame)
        self.view_final_btn.pack(pady=4)
        self.publish_btn = tk.Button(side_frame, text='Publish to GitHub Pages', command=self._publish_to_github_dialog)
        self.publish_btn.pack(pady=4)
        # Quick add element input
        tk.Label(side_frame, text='Quick add element (SYMBOL,mass,en,cov_radius,hex)').pack()
        add_el_frame = tk.Frame(side_frame)
        add_el_frame.pack(pady=4)
        self.quick_add_entry = tk.Entry(add_el_frame, width=28)
        self.quick_add_entry.grid(row=0, column=0)
        self.quick_add_btn = tk.Button(add_el_frame, text='Add', command=self._parse_quick_add)
        self.quick_add_btn.grid(row=0, column=1)
        # Quick add preview area
        preview_frame = tk.Frame(side_frame)
        preview_frame.pack(pady=2)
        self.preview_color = tk.Label(preview_frame, text=' ', width=4, bg='#808080', relief='sunken')
        self.preview_color.grid(row=0, column=0, padx=4)
        self.preview_label = tk.Label(preview_frame, text='No preview')
        self.preview_label.grid(row=0, column=1)

        # Option: auto open final frame when run ends
        self.auto_open_var = tk.BooleanVar(value=True)
        self.auto_open_check = tk.Checkbutton(side_frame, text='Auto-open final', variable=self.auto_open_var)
        self.auto_open_check.pack(pady=4)

        self.explain_text = tk.Text(side_frame, width=40, height=12)
        self.explain_text.pack(pady=4)

        back_btn = tk.Button(self.main_frame, text="Back", command=self.main_menu)
        back_btn.pack(side='bottom', pady=6)

        # internal state
        self._sim_thread = None
        self._sim_stop = threading.Event()
        self._sim: Optional['SimulationManager'] = None

    def _init_canvas(self):
        if self.canvas is not None:
            return
        self.canvas_figure = plt.figure(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.canvas_figure, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def _run_with_entry(self):
        # read parameters and start simulation in background
        formulas = self.formula_entry.get().strip()
        if not formulas:
            messagebox.showerror("Input Error", "Please enter at least one formula.")
            return
        try:
            frames = int(self.frames_spin.get())
        except Exception:
            frames = 200
        det = bool(self.det_var.get())
        # build formula dicts
        formula_strings = [f.strip() for f in formulas.split(",") if f.strip()]
        # dynamic import to avoid circular module imports
        from simulation_viewer import parse_formula, ELEMENT_DATA, SimulationManager
        formula_dicts = [parse_formula(f, ELEMENT_DATA) for f in formula_strings]

        # create SimulationManager
        self._sim = SimulationManager(formula_dicts, deterministic_mode=det)
        self._sim_stop.clear()
        self._init_canvas()
        # start background thread
        self.run_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        self._sim_thread = threading.Thread(target=self._run_sim_in_thread, args=(self._sim, frames, ), daemon=True)
        self._sim_thread.start()
        # start periodic UI updater
        self._periodic_update()

    def _load_elements_json(self):
        fname = filedialog.askopenfilename(title='Select elements JSON', filetypes=[('JSON','*.json')])
        if not fname:
            return
        try:
            import json
            with open(fname, 'r', encoding='utf-8') as fh:
                raw = json.load(fh)
            # merge into ELEMENT_DATA
            from simulation_viewer import ELEMENT_DATA
            if isinstance(raw, dict) and 'elements' in raw:
                for el in raw['elements']:
                    ELEMENT_DATA[el['symbol'].upper()] = el
            elif isinstance(raw, dict):
                # assume mapping of SYMBOL -> el props
                for k,v in raw.items():
                    ELEMENT_DATA[k.upper()] = v
            messagebox.showinfo('Loaded', f'Loaded elements into memory from {fname}')
        except Exception as e:
            messagebox.showerror('Load failed', str(e))

    def _run_sim_in_thread(self, sim: 'SimulationManager', frames: int):
        try:
            for _ in range(frames):
                if self._sim_stop.is_set():
                    break
                sim.step()
                time.sleep(0.01)  # small sleep to yield
        except Exception as e:
            messagebox.showerror("Simulation Error", str(e))
        finally:
            self.run_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')
            # show final state: render once more and update product list
            try:
                from simulation_viewer import render_simulation_frame
                render_simulation_frame(self._sim)
                if self.canvas is not None:
                    self.canvas.figure = self._sim.fig
                    self.canvas.draw_idle()
                self._refresh_products()
            except Exception:
                pass
            # Optionally auto-open final frame
            try:
                if self.auto_open_var.get():
                    self._open_final_frame()
            except Exception:
                pass

    def _periodic_update(self):
        # update the plot from sim
        if self._sim is None:
            # schedule next update and return
            self.root.after(200, self._periodic_update)
            return
        try:
            # dynamic import local to avoid circular import
            from simulation_viewer import render_simulation_frame, compute_reactivity_heatmap
            # if the sim has its own fig, replace canvas figure to allow render
            if self._sim.fig is None:
                self._sim.fig, self._sim.ax = plt.subplots(figsize=(6,6))
            # render current frame
            render_simulation_frame(self._sim)
            # draw figure in tkinter canvas
            if self.canvas_figure is not None and self.canvas is not None:
                # replace canvas figure with sim.fig
                try:
                    self.canvas.figure = self._sim.fig
                    self.canvas.draw_idle()
                except Exception:
                    pass
            # update product list
            self._refresh_products()
            # if heatmap overlay requested, draw it
            if self.heatmap_var.get():
                try:
                    xx, yy, heat = compute_reactivity_heatmap(self._sim, grid_size=80, radius=0.2)
                    # overlay heatmap as an image on sim.ax
                    self._sim.ax.imshow(heat, extent=[0,1,0,1], origin='lower', cmap='inferno', alpha=0.45, zorder=0)
                except Exception:
                    pass
        except Exception:
            # quietly ignore drawing errors to keep UI responsive
            pass

    def _stop_run(self):
        self._sim_stop.set()
        self.stop_btn.configure(state='disabled')

    def _refresh_products(self):
        if not self._sim:
            return
        products = self._sim.detect_products()
        self.product_list.delete(0, 'end')
        for k, (atoms, bonds) in products.items():
            from simulation_viewer import graph_to_formula
            formula = ''.join([f"{el}{c if c>1 else ''}" for el,c in sorted(graph_to_formula(atoms, bonds).items())])
            self.product_list.insert('end', f"{k}: {formula}")

    def _explain_selected(self):
        sel = self.product_list.curselection()
        if not sel:
            messagebox.showerror("Selection Error", "Select a product to explain")
            return
        idx = sel[0]
        key = self.product_list.get(idx).split(":")[0]
        self.explain_text.delete('1.0', 'end')
        try:
            products = self._sim.detect_products()
            # capture printed output
            import io, sys
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            from simulation_viewer import explain_component
            explain_component(self._sim, key, products)
            out = sys.stdout.getvalue()
            sys.stdout = old_stdout
            self.explain_text.insert('1.0', out)
        except Exception as e:
            self.explain_text.insert('1.0', f"Explain failed: {e}")

    def _export_timeline_selected(self):
        sel = self.product_list.curselection()
        if not sel:
            messagebox.showerror("Selection Error", "Select a product to export timeline")
            return
        idx = sel[0]
        key = self.product_list.get(idx).split(":")[0]
        try:
            products = self._sim.detect_products()
            fname = tk.filedialog.asksaveasfilename(title='Save timeline JSON', defaultextension='.json', filetypes=[('JSON','*.json')])
            if not fname:
                return
            from simulation_viewer import export_product_timeline
            export_product_timeline(self._sim, key, fname)
            messagebox.showinfo('Export', f'Timeline saved to {fname}')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))

    def _save_product_image(self):
        sel = self.product_list.curselection()
        if not sel:
            messagebox.showerror("Selection Error", "Select a product to save image")
            return
        idx = sel[0]
        key = self.product_list.get(idx).split(":")[0]
        fname = tk.filedialog.asksaveasfilename(title='Save image PNG', defaultextension='.png', filetypes=[('PNG','*.png')])
        if not fname:
            return
        try:
            # We render and then save the current sim figure
            self._sim.fig.savefig(fname, dpi=200, bbox_inches='tight')
            messagebox.showinfo('Saved', f'Image saved to {fname}')
        except Exception as e:
            messagebox.showerror('Save failed', str(e))

    def _generate_story_selected(self):
        sel = self.product_list.curselection()
        if not sel:
            messagebox.showerror("Selection Error", "Select a product to generate story")
            return
        idx = sel[0]
        key = self.product_list.get(idx).split(":")[0]
        try:
            products = self._sim.detect_products()
            from simulation_viewer import generate_reaction_story
            story = generate_reaction_story(self._sim, key, products)
            # show story in explanation text widget
            self.explain_text.delete('1.0','end')
            self.explain_text.insert('1.0', story)
        except Exception as e:
            messagebox.showerror('Generation failed', str(e))

    def main_menu(self):
        """
        Main landing screen with navigation.
        """

        # Clear old widgets
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        title_label = tk.Label(self.main_frame, text="Reaction Simulation GUI", font=("Arial", 16))
        title_label.pack(pady=15)

        formula_btn = tk.Button(self.main_frame, text="Formula Input", command=self.formula_screen)
        formula_btn.pack(pady=10)

        run_btn = tk.Button(self.main_frame, text="Run Simulation", command=self.run_simulation)
        run_btn.pack(pady=10)

    def run_simulation(self):
        """
        Placeholder for running a simulation.
        """

        if self.sim is None:
            messagebox.showerror("Error", "No simulation attached to GUI!")
            return

        try:
            # Try common run() or run_with_logging or run_steps
            if hasattr(self.sim, 'run'):
                # run synchronously
                result = self.sim.run()
                messagebox.showinfo("Simulation Result", "Simulation completed")
                return
            if hasattr(self.sim, 'run_with_logging'):
                # run in background to keep UI responsive
                def run_task():
                    try:
                        self.sim.run_with_logging(frames=600, interval=50)
                        messagebox.showinfo("Simulation Result", "Simulation completed")
                    except Exception as e:
                        messagebox.showerror("Simulation Failed", str(e))
                import threading
                threading.Thread(target=run_task, daemon=True).start()
                messagebox.showinfo("Simulation", "Simulation started in background.")
                return
            if hasattr(self.sim, 'run_steps'):
                def run_task_steps():
                    try:
                        self.sim.run_steps(n_steps=600, vis_interval=10)
                        messagebox.showinfo("Simulation Result", "Simulation completed")
                    except Exception as e:
                        messagebox.showerror("Simulation Failed", str(e))
                import threading
                threading.Thread(target=run_task_steps, daemon=True).start()
                messagebox.showinfo("Simulation", "Simulation started in background.")
                return
            raise RuntimeError("Attached simulation does not implement run/run_with_logging/run_steps")
        except Exception as e:
            messagebox.showerror("Simulation Failed", str(e))
        """Open a small quick-run dialog to input formulas and options and run the sim in the GUI."""
        d = tk.Toplevel(self.root)
        d.title('Quick Run')
        tk.Label(d, text='Formulas (comma-separated)').grid(row=0, column=0, padx=6, pady=6)
        e = tk.Entry(d, width=40)
        e.grid(row=0, column=1, padx=6, pady=6)
        e.insert(0, 'H2,O')
        tk.Label(d, text='Frames').grid(row=1, column=0, padx=6, pady=6)
        fr = tk.Spinbox(d, from_=10, to=2000, increment=10, width=8)
        fr.grid(row=1, column=1, sticky='w', padx=6, pady=6)
        fr.delete(0, 'end'); fr.insert(0, '200')
        det_var = tk.BooleanVar(value=True)
        tk.Checkbutton(d, text='Deterministic', variable=det_var).grid(row=2, column=0, columnspan=2, padx=6, pady=6)
        def run_and_close():
            # set formula entry fields so we can reuse existing run handler
            self.formula_entry.delete(0, 'end')
            self.formula_entry.insert(0, e.get())
            self.frames_spin.delete(0, 'end'); self.frames_spin.insert(0, fr.get())
            self.det_var.set(det_var.get())
            d.destroy()
            self._run_with_entry()
        run_btn = tk.Button(d, text='Run', command=run_and_close)
        run_btn.grid(row=3, column=0, padx=6, pady=6)
        cancel_btn = tk.Button(d, text='Cancel', command=d.destroy)
        cancel_btn.grid(row=3, column=1, padx=6, pady=6)

    def _add_element_popup(self):
        d = tk.Toplevel(self.root)
        d.title('Add Element')
        tk.Label(d, text='Symbol').grid(row=0, column=0, padx=6, pady=6)
        sym_e = tk.Entry(d, width=8)
        sym_e.grid(row=0, column=1, padx=6, pady=6)
        tk.Label(d, text='Atomic mass').grid(row=1, column=0, padx=6, pady=6)
        mass_e = tk.Entry(d, width=12)
        mass_e.grid(row=1, column=1, padx=6, pady=6)
        mass_e.insert(0, '0')
        tk.Label(d, text='Electronegativity').grid(row=2, column=0, padx=6, pady=6)
        en_e = tk.Entry(d, width=12)
        en_e.grid(row=2, column=1, padx=6, pady=6)
        en_e.insert(0, '0')
        tk.Label(d, text='Covalent radius').grid(row=3, column=0, padx=6, pady=6)
        cr_e = tk.Entry(d, width=12)
        cr_e.grid(row=3, column=1, padx=6, pady=6)
        cr_e.insert(0, '0.7')
        tk.Label(d, text='Color (hex)').grid(row=4, column=0, padx=6, pady=6)
        cpk_e = tk.Entry(d, width=12)
        cpk_e.grid(row=4, column=1, padx=6, pady=6)
        cpk_e.insert(0, '808080')

        def add_element_and_close():
            sym = sym_e.get().strip().upper()
            if not sym:
                messagebox.showerror('Add Element', 'Symbol is required')
                return
            try:
                emass = float(mass_e.get())
            except Exception:
                emass = 0.0
            try:
                en = float(en_e.get())
            except Exception:
                en = 0.0
            try:
                cr = float(cr_e.get())
            except Exception:
                cr = 0.7
            cpk = cpk_e.get().strip() or '808080'
            # Basic validation
            if emass < 0:
                messagebox.showerror('Invalid mass', 'Atomic mass must be >= 0')
                return
            if cr <= 0:
                messagebox.showerror('Invalid radius', 'Covalent radius must be positive')
                return
            if en < 0:
                messagebox.showerror('Invalid electronegativity', 'Electronegativity must be >= 0')
                return
            # Validate hex color
            if not isinstance(cpk, str) or not len(cpk.strip()) in (6, 7):
                # allow #ff00ff or ff00ff
                if len(cpk.strip()) == 6:
                    pass
                else:
                    messagebox.showerror('Invalid color', 'Color should be a 6-digit hex string, e.g., 00ff00')
                    return
            try:
                from simulation_viewer import ELEMENT_DATA
                ELEMENT_DATA[sym] = {
                    'name': sym,
                    'symbol': sym,
                    'atomic_number': 0,
                    'atomic_mass': emass,
                    'electronegativity_pauling': en,
                    'group': 0,
                    'period': 0,
                    'cpk-hex': cpk,
                    'category': 'user-defined',
                    'covalent_radius': cr
                }
                messagebox.showinfo('Added', f'Element {sym} added to ELEMENT_DATA')
                d.destroy()
            except Exception as e:
                messagebox.showerror('Add failed', str(e))

        add_btn = tk.Button(d, text='Add', command=add_element_and_close)
        add_btn.grid(row=5, column=0, padx=6, pady=6)
        cancel_btn = tk.Button(d, text='Cancel', command=d.destroy)
        cancel_btn.grid(row=5, column=1, padx=6, pady=6)

    def _export_html(self):
        if not self._sim:
            messagebox.showerror('Export', 'No simulation to export. Run a simulation first.')
            return
        try:
            fname = filedialog.asksaveasfilename(title='Export interactive HTML', defaultextension='.html', filetypes=[('HTML','*.html')])
            if not fname:
                return
            from simulation_viewer import export_simulation_to_plotly_html
            # choose n_frames to export based on recorded history
            n_frames = min(self._sim.frame, 600)
            export_simulation_to_plotly_html(self._sim, fname, n_frames=n_frames, fps=10)
            messagebox.showinfo('Exported', f'HTML exported to {fname}')
            # open in default browser
            import webbrowser
            webbrowser.open(f'file://{fname}')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))

    def _open_final_frame(self):
        if not self._sim:
            messagebox.showerror('Open', 'No simulation available')
            return
        try:
            # render final frame
            from simulation_viewer import render_simulation_frame
            render_simulation_frame(self._sim)
            # open in a Toplevel window with canvas
            top = tk.Toplevel(self.root)
            top.title('Final Simulation Frame')
            fig = self._sim.fig
            canvas = FigureCanvasTkAgg(fig, master=top)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
        except Exception as e:
            messagebox.showerror('Open failed', str(e))

    def _parse_quick_add(self):
        s = self.quick_add_entry.get().strip()
        if not s:
            messagebox.showerror('Add element', 'Enter a CSV string like SYMBOL,mass,en,cov_radius,hex')
            return
        parts = [p.strip() for p in s.split(',') if p.strip()]
        try:
            symbol = parts[0].upper()
            mass = float(parts[1]) if len(parts) > 1 else 0.0
            en = float(parts[2]) if len(parts) > 2 else 0.0
            cr = float(parts[3]) if len(parts) > 3 else 0.7
            hexcol = parts[4] if len(parts) > 4 else '808080'
            from simulation_viewer import ELEMENT_DATA
            ELEMENT_DATA[symbol] = {
                'name': symbol,
                'symbol': symbol,
                'atomic_number': 0,
                'atomic_mass': mass,
                'electronegativity_pauling': en,
                'group': 0,
                'period': 0,
                'cpk-hex': hexcol,
                'category': 'user-defined',
                'covalent_radius': cr
            }
            messagebox.showinfo('Added', f'Element {symbol} added')
            # update preview
            try:
                self.preview_color.configure(bg=f"#{hexcol}")
            except Exception:
                pass
            self.preview_label.configure(text=f"{symbol} m={mass} en={en} r={cr}")
            # refresh presets
            self._refresh_presets()
        except ValueError:
            messagebox.showerror('Add failed', 'Invalid numeric values for mass/en/radius')
        except Exception as e:
            messagebox.showerror('Add failed', str(e))

    def _refresh_presets(self):
        try:
            from simulation_viewer import ELEMENT_DATA
            keys = sorted(ELEMENT_DATA.keys())
            # reconfigure OptionMenu
            menu = self.preset_menu['menu']
            menu.delete(0, 'end')
            for k in keys:
                name = ELEMENT_DATA.get(k, {}).get('name', '')
                label = f"{k} ({name})" if name else k
                menu.add_command(label=label, command=lambda v=k: self.preset_var.set(v))
            # keep current var if exists
            if not self.preset_var.get() and keys:
                self.preset_var.set(keys[0])
        except Exception:
            pass

    def _insert_preset_element(self):
        try:
            sym = self.preset_var.get().strip()
            if not sym:
                return
            cur = self.formula_entry.get().strip()
            if not cur:
                cur = sym
            else:
                cur = cur + ',' + sym
            self.formula_entry.delete(0, 'end')
            self.formula_entry.insert(0, cur)
        except Exception:
            pass

    def _insert_molecule_preset(self):
        try:
            k = self.mol_preset_var.get()
            if not k:
                return
            formula = self.mol_presets.get(k)
            cur = self.formula_entry.get().strip()
            if not cur:
                cur = formula
            else:
                cur = cur + ',' + formula
            self.formula_entry.delete(0, 'end')
            self.formula_entry.insert(0, cur)
        except Exception:
            pass

    def _run_and_publish(self):
        # run simulation in background and then export an interactive HTML to docs/ and open in browser
        try:
            # ensure current formula is in entry
            self._run_with_entry()
            # After run completes, ask for publish directory
            # We'll spawn a worker to wait for thread to finish and then export
            def wait_and_publish():
                if self._sim_thread is not None:
                    # wait for thread to finish (join)
                    self._sim_thread.join()
                # export html
                import os
                try:
                    from simulation_viewer import export_simulation_to_plotly_html, ensure_dir, now_str
                    out_html = os.path.join('docs', f"sim_preview_{now_str()}.html")
                    export_simulation_to_plotly_html(self._sim, out_html, n_frames=min(self._sim.frame, 600), fps=10)
                    # copy to docs index
                    ensure_dir('docs')
                    import shutil
                    dest = os.path.join('docs', os.path.basename(out_html))
                    if os.path.abspath(dest) != os.path.abspath(out_html):
                        shutil.copyfile(out_html, dest)
                    # create simple index
                    with open(os.path.join('docs', 'index.html'), 'w', encoding='utf-8') as fh:
                        fh.write(f"<html><body><h1>Published Simulation</h1><iframe src=\"{os.path.basename(out_html)}\" width=100% height=900></iframe></body></html>")
                    import webbrowser
                    webbrowser.open('file://' + os.path.abspath(out_html))
                    messagebox.showinfo('Published', f'Published HTML to docs/{os.path.basename(out_html)}')
                except Exception as e:
                    messagebox.showerror('Publish failed', str(e))
            t = threading.Thread(target=wait_and_publish, daemon=True)
            t.start()
        except Exception as e:
            messagebox.showerror('Run & Publish failed', str(e))

    def _export_gif(self):
        if not self._sim:
            messagebox.showerror('Export GIF', 'No simulation to export. Run a simulation first.')
            return
        fname = filedialog.asksaveasfilename(title='Export GIF', defaultextension='.gif', filetypes=[('GIF','*.gif')])
        if not fname:
            return
        try:
            from simulation_viewer import export_simulation_to_gif
            n_frames = min(self._sim.frame, 600)
            export_simulation_to_gif(self._sim, fname, n_frames=n_frames, fps=10)
            messagebox.showinfo('Saved', f'GIF saved to {fname}')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))

    def _export_mp4(self):
        if not self._sim:
            messagebox.showerror('Export MP4', 'No simulation to export. Run a simulation first.')
            return
        fname = filedialog.asksaveasfilename(title='Export MP4', defaultextension='.mp4', filetypes=[('MP4','*.mp4')])
        if not fname:
            return
        try:
            from simulation_viewer import export_simulation_to_mp4
            n_frames = min(self._sim.frame, 600)
            export_simulation_to_mp4(self._sim, fname, n_frames=n_frames, fps=10)
            messagebox.showinfo('Saved', f'MP4 saved to {fname}')
        except Exception as e:
            messagebox.showerror('Export failed', str(e))

    def _publish_to_github_dialog(self):
        if not self._sim:
            messagebox.showerror('Publish', 'No simulation to publish. Run a simulation and export to docs/ first.')
            return
        repo = simpledialog.askstring('Publish', 'GitHub repository URL (e.g., https://github.com/owner/repo.git)')
        if not repo:
            return
        token = simpledialog.askstring('Token', 'Personal access token (optional)', show='*')
        branch = simpledialog.askstring('Branch', 'Branch to push to (default: gh-pages)', initialvalue='gh-pages')
        if not branch:
            branch = 'gh-pages'
        try:
            # Ensure docs exist - usually produced by run_and_publish
            import os
            docs_dir = os.path.join(os.getcwd(), 'docs')
            if not os.path.exists(docs_dir):
                messagebox.showerror('Publish', 'No docs/ directory found. Use Run & Publish first to create an export.')
                return
            from simulation_viewer import publish_directory_to_github
            publish_directory_to_github(docs_dir, repo, branch=branch, token=token)
            messagebox.showinfo('Published', 'Published docs to GitHub Pages')
        except Exception as e:
            messagebox.showerror('Publish failed', str(e))

class SimulationGUIAdvanced(SimulationGUI):
    """An advanced version of the base GUI."""

    def formula_screen(self):
        # Add more widgets, instructions, or extra features
        super().formula_screen()
        extra_label = tk.Label(self.main_frame, text="Advanced features here")
        extra_label.pack(pady=10)

if __name__ == "__main__":
    gui = SimulationGUI()
    gui.main_menu()
    gui.start()