import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import random
import tkinter as tk
from tkinter import messagebox
from collections import defaultdict, Counter
from itertools import combinations, product

# -------------------------------
# Load element data from JSON
# -------------------------------

ELEMENTS_FILE = "elements.json"
with open(ELEMENTS_FILE, "r", encoding="utf-8") as f:
    raw = json.load(f)

ELEMENT_DATA = {el['symbol'].upper(): el for el in raw['elements']}

# Default visualization radius factor
VDW_SCALE = 0.05

# -------------------------------
# Reaction Knowledge Base
# -------------------------------

class ReactionKnowledgeBase:
    """
    Stores observed co-occurrences and reaction hints.
    Saves to reaction_patterns.json and persists a RandomForest model reaction_model.pkl.
    """
    def __init__(self, path_json="reaction_patterns.json", model_path="reaction_model.pkl"):
        self.path_json = path_json
        self.model_path = model_path

        self.cooccurrence = defaultdict(int)  # key: tuple(sorted pair), value: count of co-occurrence in same molecule
        self.bond_examples = defaultdict(int)  # key: tuple(sorted pair), value: count when they are bonded
        self.examples = []  # raw examples for model (features, label)
        self.model = None

        self._init_priors()
        self._load_existing()

    def _init_priors(self):
        priors = {
            ("H","H"): 10,
            ("O","O"): 8,
            ("N","N"): 6,
            ("Cl","Cl"): 6,
            ("F","F"): 6,
            ("H","O"): 30,
            ("Na","Cl"): 12,
            ("C","O"): 20,
            ("C","H"): 50,
        }
        for pair, count in priors.items():
            pair_key = tuple(sorted(pair))
            self.cooccurrence[pair_key] += count
            self.bond_examples[pair_key] += count

    def _load_existing(self):
        if os.path.exists(self.path_json):
            try:
                with open(self.path_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("cooccurrence", {}).items():
                    key = tuple(k.split("|"))
                    self.cooccurrence[key] = v
                for k, v in data.get("bond_examples", {}).items():
                    key = tuple(k.split("|"))
                    self.bond_examples[key] = v
                self.examples = data.get("examples", [])[:1000]
            except Exception:
                pass

        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def _save_json(self):
        data = {
            "cooccurrence": {"|".join(k): v for k, v in self.cooccurrence.items()},
            "bond_examples": {"|".join(k): v for k, v in self.bond_examples.items()},
            "examples": self.examples[-1000:],
        }
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def add_observation_from_molecule(self, formula_dict):
        elems = []
        for el_symbol, cnt in formula_dict.items():
            el = el_symbol.upper()
            elems.extend([el] * max(1, int(cnt)))

        unique_elems = list(set(elems))

        for a, b in combinations(unique_elems, 2):
            key = tuple(sorted((a, b)))
            self.cooccurrence[key] += 1
            self.bond_examples[key] += 1
            feat = self._features_for_pair(a, b)
            self.examples.append({"feat": feat, "label": 1, "pair": key})

        for el in unique_elems:
            if elems.count(el) > 1:
                key = (el, el)
                self.cooccurrence[key] += 1
                self.bond_examples[key] += 1
                feat = self._features_for_pair(el, el)
                self.examples.append({"feat": feat, "label": 1, "pair": key})

        all_elements = list(ELEMENT_DATA.keys())
        absent = [e for e in all_elements if e not in unique_elems]

        num_neg = min(3, max(1, len(unique_elems)))
        for _ in range(num_neg):
            if not unique_elems or not absent:
                break
            a = random.choice(unique_elems)
            b = random.choice(absent)
            key = tuple(sorted((a, b)))
            feat = self._features_for_pair(a, b)
            self.examples.append({"feat": feat, "label": 0, "pair": key})

        self._save_json()

    def _features_for_pair(self, a, b):
        a = a.upper()
        b = b.upper()
        ea = ELEMENT_DATA.get(a, {})
        eb = ELEMENT_DATA.get(b, {})

        mass_a = float(ea.get("atomic_mass") or 0.0)
        mass_b = float(eb.get("atomic_mass") or 0.0)
        en_a = ea.get("electronegativity_pauling") or 0.0
        en_b = eb.get("electronegativity_pauling") or 0.0
        group_a = ea.get("group") or 0
        group_b = eb.get("group") or 0
        period_a = ea.get("period") or 0
        period_b = eb.get("period") or 0

        diatomic = {"H","N","O","F","Cl","Br","I"}

        is_diatomic_a = 1 if a in diatomic else 0
        is_diatomic_b = 1 if b in diatomic else 0

        prior_key = tuple(sorted((a, b)))
        prior_count = float(self.cooccurrence.get(prior_key, 0))
        bond_count = float(self.bond_examples.get(prior_key, 0))

        same = 1 if a == b else 0
        noble_flags = (
            1 if ea.get("category","").lower().find("noble") != -1 else 0,
            1 if eb.get("category","").lower().find("noble") != -1 else 0
        )

        feat = [
            mass_a, mass_b,
            en_a, en_b,
            abs(en_a - en_b),
            group_a, group_b, abs(group_a - group_b),
            period_a, period_b, abs(period_a - period_b),
            is_diatomic_a, is_diatomic_b,
            prior_count, bond_count,
            same,
            noble_flags[0], noble_flags[1],
        ]

        return [float(x) for x in feat]

    def train_model_if_ready(self):
        if len(self.examples) < 30:
            return False

        X = np.array([ex["feat"] for ex in self.examples])
        y = np.array([ex["label"] for ex in self.examples])

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        try:
            model.fit(X, y)
            self.model = model
            joblib.dump(self.model, self.model_path)
            self._save_json()
            return True
        except Exception:
            return False

    def predict_reaction_probability(self, a, b):
        a = a.upper()
        b = b.upper()
        feat = self._features_for_pair(a, b)

        if self.model:
            try:
                prob = self.model.predict_proba([feat])[0]
                if prob.shape[0] == 2:
                    return float(prob[1])
            except Exception:
                pass

        key = tuple(sorted((a, b)))
        prior = self.cooccurrence.get(key, 0) + 1.0
        bond = self.bond_examples.get(key, 0) + 0.5

        score = min(
            1.0,
            (bond / (prior + 0.1)) * 0.8 + min(1.0, prior / 50.0) * 0.2
        )
        return float(score)

    def export_json(self, path=None):
        if path is None:
            path = self.path_json
        self._save_json()


# -------------------------------
# Molecular Classes (unchanged core behavior)
# -------------------------------

class MolecularUnit:
    def __init__(self, uid, element='C', subscript=1, charge=0, position=None):
        if element not in ELEMENT_DATA:
            raise ValueError(f"Element '{element}' not found in elements.json")

        self.uid = uid
        self.element = element
        self.subscript = subscript
        self.charge = charge

        self.mass = ELEMENT_DATA[element]['atomic_mass'] * subscript
        self.radius = VDW_SCALE * subscript

        hex_color = ELEMENT_DATA[element].get('cpk-hex', '808080')
        self.color = f"#{hex_color}"

        self.position = np.random.rand(2) if position is None else np.array(position, dtype=float)
        self.velocity = np.zeros(2)

        self.neighbors = []
        self.state = 'normal'
        self.history = []

    def compute_forces(self):
        K_BOND = 50.0
        K_CHARGE = 0.01
        MAX_FORCE = 0.1
        DT = 0.01

        force = np.zeros(2)

        for n in self.neighbors:
            delta = n.position - self.position
            dist = np.linalg.norm(delta)
            if dist > 1e-6:
                force += K_BOND * (dist - self.radius - n.radius) * delta / (dist + 1e-12)
                if hasattr(self, "charge") and hasattr(n, "charge"):
                    force += K_CHARGE * self.charge * n.charge * delta / ((dist**3) + 1e-12)

        return np.clip(force, -MAX_FORCE, MAX_FORCE)

    def update_position(self):
        DT = 0.01
        noise = np.random.normal(scale=0.001, size=2)
        self.velocity += (self.compute_forces() / max(1.0, self.mass)) * DT + noise
        self.position += self.velocity * DT
        self.position = np.clip(self.position, 0, 1)

    def record_history(self, frame):
        self.history.append((frame, self.state))


class Bond:
    def __init__(self, u1, u2):
        self.u1 = u1
        self.u2 = u2
        self.color = 'white'
        self.alpha = 0.3


class Molecule:
    def __init__(self, mol_id, formula_dict):
        self.mol_id = mol_id
        self.units = []

        start_pos = np.random.rand(2)
        uid_count = 0

        for elem, count in formula_dict.items():
            for _ in range(max(1, int(count))):
                unit = MolecularUnit(uid=f"{mol_id}_{uid_count}", element=elem, subscript=1)
                unit.position = start_pos + np.random.normal(scale=0.02, size=2)
                self.units.append(unit)
                uid_count += 1

        for u in self.units:
            u.neighbors = [v for v in self.units if v is not u]

        if len(self.units) >= 2:
            self.bonds = [Bond(self.units[i], self.units[i+1]) for i in range(len(self.units)-1)]
        else:
            self.bonds = []


# -------------------------------
# Cas9 Agent
# -------------------------------

class Cas9Agent:
    def __init__(self, start_pos=(1.2,1.2)):
        self.position = np.array(start_pos, dtype=float)
        self.radius = 0.03
        self.color = "yellow"
        self.target = None
        self.active = False

    def assign_target(self, unit):
        self.target = unit
        self.active = True

    def move_towards(self, dt=0.02):
        if self.target is None:
            return
        direction = self.target.position - self.position
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            speed = 0.06
            self.position += (direction / dist) * speed * dt

    def check_reached(self):
        if self.target is None:
            return False
        return np.linalg.norm(self.position - self.target.position) < 0.02

    def reset(self):
        self.target = None
        self.active = False
        self.position = np.array([1.2,1.2])


# -------------------------------
# ML Mutation Detector
# -------------------------------

class MutationDetector:
    def __init__(self, path="crispr_model.pkl"):
        self.path = path

        if os.path.exists(path):
            try:
                self.model = joblib.load(path)
            except Exception:
                self.model = None
        else:
            self.model = None

        if self.model is None:
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            X_dummy = np.random.rand(20,7)
            y_dummy = np.random.randint(0,2,size=20)
            self.model.fit(X_dummy, y_dummy)

        self.features = []
        self.labels = []

    def predict(self, feat):
        try:
            return int(self.model.predict([feat])[0])
        except Exception:
            return int(random.random() < 0.01)

    def update_training(self, feat, label):
        self.features.append(feat)
        self.labels.append(label)

    def retrain(self):
        if len(self.features) >= 100:
            X = np.array(self.features)
            y = np.array(self.labels)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            joblib.dump(self.model, self.path)
            self.features, self.labels = [], []


# -------------------------------
# CRISPR Simulation
# -------------------------------

class CRISPRSimulation:
    def __init__(self, formula_list, reaction_kb=None, auto_train_kb=True):
        self.reaction_kb = reaction_kb if reaction_kb is not None else ReactionKnowledgeBase()
        self.auto_train_kb = auto_train_kb

        for fdict in formula_list:
            try:
                self.reaction_kb.add_observation_from_molecule(fdict)
            except Exception:
                pass

        if auto_train_kb:
            self.reaction_kb.train_model_if_ready()

        self.molecules = [Molecule(i, fdict) for i, fdict in enumerate(formula_list)]
        self.units = [u for m in self.molecules for u in m.units]

        self.editor = Cas9Agent()
        self.detector = MutationDetector()

        self.fig, self.ax = plt.subplots(figsize=(8,8))
        self.fig.patch.set_facecolor('black')
        self.ax.set_facecolor('black')
        self.ax.set_xlim(0,1)
        self.ax.set_ylim(0,1)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.unit_circles = [
            Circle(u.position, max(0.005, u.radius), color=u.color, alpha=0.7)
            for u in self.units
        ]
        for c in self.unit_circles:
            self.ax.add_patch(c)

        self.editor_circle = Circle(self.editor.position, self.editor.radius, color=self.editor.color, alpha=0.8)
        self.ax.add_patch(self.editor_circle)

        kb_summary = f"KB pairs: {len(self.reaction_kb.cooccurrence)}"
        self.text_overlay = self.ax.text(0.5,1.02,kb_summary,color='white',ha='center',va='bottom',transform=self.ax.transAxes)

        self.bond_lines = []
        for m in self.molecules:
            for b in m.bonds:
                line, = self.ax.plot(
                    [b.u1.position[0], b.u2.position[0]],
                    [b.u1.position[1], b.u2.position[1]],
                    color=b.color,
                    alpha=b.alpha
                )
                self.bond_lines.append(line)

        self.event_log = []

    def update(self, frame):
        for u in self.units:
            u.update_position()

        if frame % 120 == 0:
            candidates = [u for u in self.units if u.state == 'normal']
            if candidates:
                u = random.choice(candidates)
                u.state = 'mutated'
                self.text_overlay.set_text(f"Mutation at {u.element}{u.subscript} ({u.uid})")
                self.event_log.append({"frame": frame, "event": "mutation", "uid": u.uid, "element": u.element})
                if not self.editor.active:
                    self.editor.assign_target(u)

        for u in self.units:
            feat = [
                u.position[0], u.position[1], len(u.history),
                u.charge, u.subscript,
                hash(u.element) % 4, 0
            ]
            pred = self.detector.predict(feat)
            if pred == 1 and u.state == 'normal':
                u.state = 'mutated'
                if not self.editor.active:
                    self.editor.assign_target(u)
                self.event_log.append({"frame": frame, "event": "predicted_mutation", "uid": u.uid, "element": u.element})

        if self.editor.active:
            self.editor.move_towards()
            self.editor_circle.center = self.editor.position
            if self.editor.check_reached():
                if np.random.rand() < 0.9:
                    self.editor.target.state = 'repaired'
                    self.text_overlay.set_text(f"Repair successful at {self.editor.target.element}{self.editor.target.subscript}")
                    self.event_log.append({"frame": frame, "event": "repair_success", "uid": self.editor.target.uid, "element": self.editor.target.element})
                else:
                    self.editor.target.state = 'mutated'
                    self.text_overlay.set_text(f"Repair FAILED at {self.editor.target.element}{self.editor.target.subscript}")
                    self.event_log.append({"frame": frame, "event": "repair_failed", "uid": self.editor.target.uid, "element": self.editor.target.element})

                self.editor.reset()

        for i, u in enumerate(self.units):
            self.unit_circles[i].center = u.position
            color = u.color if u.state == 'normal' else ('red' if u.state == 'mutated' else 'lime')
            self.unit_circles[i].set_color(color)

            u.record_history(frame)
            if u.state == 'mutated':
                feat = [
                    u.position[0], u.position[1], len(u.history),
                    u.charge, u.subscript,
                    hash(u.element) % 4, 0
                ]
                self.detector.update_training(feat, 1)

        if frame % 400 == 0 and frame > 0:
            self.detector.retrain()

        if self.auto_train_kb:
            trained = self.reaction_kb.train_model_if_ready()
            if trained:
                self.text_overlay.set_text(f"ReactionKB trained at frame {frame}")

        idx = 0
        for m in self.molecules:
            for b in m.bonds:
                if idx < len(self.bond_lines):
                    self.bond_lines[idx].set_data(
                        [b.u1.position[0], b.u2.position[0]],
                        [b.u1.position[1], b.u2.position[1]]
                    )
                idx += 1

        self.editor_circle.set_alpha(0.5 + 0.5 * math.sin(frame / 8.0))

        self.text_overlay.set_text(f"KB pairs: {len(self.reaction_kb.cooccurrence)} Events: {len(self.event_log)}")

        return self.unit_circles + [self.editor_circle, self.text_overlay] + self.bond_lines

    def run(self, frames=1200, interval=50):
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=frames, interval=interval, blit=True)
        plt.show()

        output_log = {
            "event_log": self.event_log,
            "kb_summary": {
                "cooccurrence_pairs": len(self.reaction_kb.cooccurrence),
                "bond_examples": len(self.reaction_kb.bond_examples),
            }
        }
        with open("simulation_events.json", "w", encoding="utf-8") as f:
            json.dump(output_log, f, indent=2)

        self.reaction_kb.export_json()


# -------------------------------
# GUI
# -------------------------------

class SimulationGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Molecular Reaction Learner")
        self.reaction_kb = ReactionKnowledgeBase()
        self.home_screen()

    def home_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Molecular Reaction Learner", font=("Arial", 16)).pack(pady=10)
        tk.Button(self.root, text="Enter Molecules", command=self.formula_screen).pack(pady=6)
        tk.Button(self.root, text="Inspect Knowledge Base", command=self.inspect_kb_screen).pack(pady=6)
        tk.Button(self.root, text="Export KB JSON", command=self.export_kb).pack(pady=6)
        tk.Button(self.root, text="Quit", command=self.root.destroy).pack(pady=10)

    def inspect_kb_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Knowledge Base Summary", font=("Arial", 14)).pack(pady=8)

        text = tk.Text(self.root, width=80, height=24)
        text.pack(pady=6)

        lines = []
        lines.append(f"Number of observed cooccurrence pairs: {len(self.reaction_kb.cooccurrence)}")
        lines.append("Top cooccurring pairs (by count):")

        top_pairs = sorted(self.reaction_kb.cooccurrence.items(), key=lambda x: -x[1])[:30]
        for k, v in top_pairs:
            lines.append(f"  {k[0]} - {k[1]} : {v} (bond evidence: {self.reaction_kb.bond_examples.get(k,0)})")

        lines.append("\nPrediction examples using current model (probability of bonding):")
        sample_pairs = [("H","O"), ("C","H"), ("Na","Cl"), ("He","O"), ("C","O")]

        for a, b in sample_pairs:
            try:
                p = self.reaction_kb.predict_reaction_probability(a, b)
                lines.append(f"  {a}-{b} : {p:.3f}")
            except Exception:
                lines.append(f"  {a}-{b} : n/a")

        text.insert("1.0", "\n".join(lines))

        tk.Button(self.root, text="Back", command=self.home_screen).pack(pady=8)

    def export_kb(self):
        self.reaction_kb.export_json()
        messagebox.showinfo("Exported", f"Knowledge base exported to {self.reaction_kb.path_json}")

    def formula_screen(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        tk.Label(self.root, text="Enter molecule formulas (e.g., H2O, C6H12O6), comma-separated").pack(pady=5)
        self.formula_entry = tk.Entry(self.root, width=70)
        self.formula_entry.pack(pady=5)

        tk.Button(self.root, text="Back", command=self.home_screen).pack(pady=5)
        tk.Button(self.root, text="Start Simulation", command=self.start_simulation).pack(pady=5)
        tk.Label(self.root, text="Tip: provide many known molecules to grow the KB. Examples: H2,O2,H2O,NaCl,CO2,C6H6").pack(pady=6)

    def parse_formula(self, formula):
        import re
        matches = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        fdict = {}
        for elem, count in matches:
            elem = elem.upper()
            if elem not in ELEMENT_DATA:
                raise ValueError(f"Element '{elem}' not found in elements.json")
            fdict[elem] = fdict.get(elem, 0) + (int(count) if count else 1)
        return fdict

    def start_simulation(self):
        raw_input = self.formula_entry.get()
        formulas = [f.strip() for f in raw_input.split(',') if f.strip()]

        if not formulas:
            messagebox.showerror("No input", "Please enter at least one molecule formula.")
            return

        parsed_list = []
        try:
            for f in formulas:
                parsed = self.parse_formula(f)
                parsed_list.append(parsed)
        except ValueError as e:
            messagebox.showerror("Invalid Input", str(e))
            return

        for fdict in parsed_list:
            try:
                self.reaction_kb.add_observation_from_molecule(fdict)
            except Exception:
                pass

        trained = self.reaction_kb.train_model_if_ready()
        if trained:
            messagebox.showinfo("KB Model", "Reaction model trained and saved to disk.")

        csv_lines = ["molecule,element_counts"]
        for fdict in parsed_list:
            items = ";".join([f"{k}:{v}" for k, v in fdict.items()])
            csv_lines.append(f"\"{''.join(list(fdict.keys()))}\",\"{items}\"")

        csv_path = "molecule_inputs_summary.csv"
        try:
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write("\n".join(csv_lines))
        except Exception:
            pass

        self.root.destroy()
        sim = CRISPRSimulation(parsed_list, reaction_kb=self.reaction_kb, auto_train_kb=True)
        sim.run()

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    gui = SimulationGUI()
    gui.run()
