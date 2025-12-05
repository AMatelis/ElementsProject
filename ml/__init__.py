from .knowledge_base import ReactionKnowledgeBase

# Optional imports - only available if dependencies are installed
try:
    from .gnn_model import BondPredictorGNN, ReactionPredictorGNN, DenseGNN
except ImportError:
    BondPredictorGNN = None
    ReactionPredictorGNN = None
    DenseGNN = None

try:
    from .dataset_builder import build_pyg_dataset_from_jsonl, load_jsonl_samples
except ImportError:
    build_pyg_dataset_from_jsonl = None
    load_jsonl_samples = None

try:
    from .trainer import train_bond_predictor_from_jsonl, train_reaction_predictor_from_jsonl
except ImportError:
    train_bond_predictor_from_jsonl = None
    train_reaction_predictor_from_jsonl = None

__all__ = [
    "ReactionKnowledgeBase",
    "build_pyg_dataset_from_jsonl", "load_jsonl_samples",
    "BondPredictorGNN", "ReactionPredictorGNN", "DenseGNN",
    "train_bond_predictor_from_jsonl", "train_reaction_predictor_from_jsonl"
]