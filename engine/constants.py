DEFAULT_TEMPERATURE = 300.0  # Kelvin
DEFAULT_DT = 1e-3            # time step in seconds
DEFAULT_CELL_SIZE = 0.06     # default spatial hash cell size
DEFAULT_BOUNDS = 1.0         # default simulation box size
DEFAULT_PERIODIC = False     # periodic boundary conditions

# -----------------------
# Reaction Engine
# -----------------------
DEFAULT_SCAN_RADIUS = 0.18   # neighbor search radius
BOND_FORMATION_THRESHOLD = 1.0  # deterministic threshold for bond formation

# -----------------------
# Knowledge Base
# -----------------------
DEFAULT_KB_DIR = "data"
DEFAULT_KB_FILENAME = "reaction_kb.jsonl"

# -----------------------
# Simulation Manager
# -----------------------
DEFAULT_AUTO_TRAIN_KB = True
DEFAULT_DETERMINISTIC_MODE = False
DEFAULT_ATOM_HISTORY_MAXLEN = 2000

# -----------------------
# Logging
# -----------------------
LOGGING_LEVEL = "INFO"  # options: DEBUG, INFO, WARNING, ERROR

# -----------------------
# Misc
# -----------------------
EPSILON = 1e-12  # small value to prevent div by zero or numerical issues