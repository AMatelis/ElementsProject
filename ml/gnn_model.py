from __future__ import annotations
from typing import Optional, Tuple, Any, List, Dict
import logging
import math
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Optional imports
USE_TORCH = False
USE_PYG = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    USE_TORCH = True
    try:
        # PyG imports
        from torch_geometric.nn import GraphConv, global_mean_pool  # type: ignore
        from torch_geometric.data import Data as PyGData  # type: ignore
        USE_PYG = True
    except Exception:
        USE_PYG = False
except Exception:
    USE_TORCH = False
    USE_PYG = False

# joblib fallback for saving non-torch objects
try:
    import joblib
except Exception:
    joblib = None


# -----------------------
# Helper utilities
# -----------------------
def get_device() -> str:
    if USE_TORCH and torch.cuda.is_available():
        return "cuda"
    if USE_TORCH:
        return "cpu"
    return "cpu"


def to_device(obj: Any, device: Optional[str] = None) -> Any:
    """
    Move a torch model or tensor to device. If torch isn't available, returns object unchanged.
    """
    if not USE_TORCH:
        return obj
    dev = device or get_device()
    try:
        return obj.to(dev)
    except Exception:
        return obj


def save_model(model: Any, path: str) -> None:
    """
    Save a model to disk.
    - If model is a torch.nn.Module and torch available -> save state_dict
    - Else if joblib available -> joblib.dump
    - Else attempt Python pickle (may fail for some objects)
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
        if USE_TORCH and isinstance(model, torch.nn.Module):
            torch.save(model.state_dict(), path)
            logger.info(f"Saved torch model state_dict to {path}")
            return
        if joblib is not None:
            joblib.dump(model, path)
            logger.info(f"Saved model with joblib to {path}")
            return
        # Last resort: use torch.save if available (it can save arbitrary python objects)
        if USE_TORCH:
            torch.save(model, path)
            logger.info(f"Saved model object with torch.save to {path}")
            return
        raise RuntimeError("No supported persistence available in this environment.")
    except Exception:
        logger.exception(f"Failed to save model to {path}")


def load_model(cls: Any, path: str, map_location: Optional[str] = None) -> Optional[Any]:
    """
    Load a model. If cls is a torch.nn.Module subclass and torch available, instantiate and load state_dict.
    Else try joblib/pickle.
    """
    try:
        if USE_TORCH and isinstance(cls, type) and issubclass(cls, torch.nn.Module):
            device = map_location or get_device()
            model = cls()
            state = torch.load(path, map_location=device)
            # If the file contains the object, attempt to set state dict
            if isinstance(state, dict):
                model.load_state_dict(state)
            else:
                # file saved full object -> attempt to return it
                return state
            return model
        if joblib is not None:
            return joblib.load(path)
        if USE_TORCH:
            return torch.load(path, map_location=map_location or get_device())
    except Exception:
        logger.exception(f"Failed to load model from {path}")
    return None


# -----------------------
# PyG-based models
# -----------------------
if USE_TORCH and USE_PYG:

    class BondPredictorGNN(nn.Module):
        """
        GNN that produces a score [0,1] for each candidate edge.
        Expects PyG Data objects where candidate edges may be provided as:
           data.candidate_edges -> Tensor shape [2, M]
           data.candidate_edge_attr -> Tensor shape [M, K] (optional)
        Otherwise fallback: scores existing edges in data.edge_index
        """

        def __init__(self, node_in_dim: int = 14, hidden_dim: int = 128, msg_passes: int = 3, edge_attr_dim: int = 1):
            super().__init__()
            self.node_lin = nn.Linear(node_in_dim, hidden_dim)
            self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(msg_passes)])
            # edge MLP: concat(src, dst, optional edge_attr)
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + edge_attr_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )

        def forward(self, data: "PyGData"):
            x, edge_index = data.x, data.edge_index
            x = F.relu(self.node_lin(x))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            # candidate edges?
            ce = getattr(data, "candidate_edges", None)
            if ce is not None:
                # ce: [2, M]
                src = ce[0, :]
                dst = ce[1, :]
                src_x = x[src]
                dst_x = x[dst]
                # candidate attrs
                cea = getattr(data, "candidate_edge_attr", None)
                if cea is None:
                    cea = torch.zeros((src_x.shape[0], 1), device=src_x.device)
                feats = torch.cat([src_x, dst_x, cea.float()], dim=1)
                return self.edge_mlp(feats).view(-1)
            else:
                # score existing edges
                src, dst = edge_index
                src_x = x[src]
                dst_x = x[dst]
                edge_attr = getattr(data, "edge_attr", None)
                if edge_attr is None:
                    edge_attr = torch.zeros((src_x.shape[0], 1), device=src_x.device)
                feats = torch.cat([src_x, dst_x, edge_attr.float()], dim=1)
                return self.edge_mlp(feats).view(-1)


    class ReactionPredictorGNN(nn.Module):
        """
        Graph-level predictor: encode node features with message passing, pool, then classify.
        """

        def __init__(self, node_in_dim: int = 14, hidden_dim: int = 128, msg_passes: int = 3, out_classes: int = 8):
            super().__init__()
            self.node_lin = nn.Linear(node_in_dim, hidden_dim)
            self.convs = nn.ModuleList([GraphConv(hidden_dim, hidden_dim) for _ in range(msg_passes)])
            self.pool = global_mean_pool
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, out_classes)
            )

        def forward(self, data: "PyGData"):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.node_lin(x))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))
            g = self.pool(x, batch)
            return self.classifier(g)


# -----------------------
# Dense torch fallback
# -----------------------
elif USE_TORCH:

    class DenseGNN(nn.Module):
        """
        Dense/fallback model using only torch.nn (no PyG).
        - node_mlp: projects node features -> node embeddings
        - score_edges: given embedded nodes and pair indices, compute a score per pair
        - global_mlp: produce graph-level output from pooled node embeddings
        """

        def __init__(self, node_in_dim: int = 14, hidden_dim: int = 128):
            super().__init__()
            self.node_mlp = nn.Sequential(
                nn.Linear(node_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            # edge scoring MLP expects [embed_i, embed_j, optional dist]
            self.edge_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            self.global_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 8)
            )

        def forward_node_embeddings(self, x: torch.Tensor) -> torch.Tensor:
            return self.node_mlp(x)

        def score_edges(self, node_emb: torch.Tensor, pair_idx: torch.Tensor, pair_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            node_emb: [N, D]
            pair_idx: [M, 2] (long)
            pair_attr: [M, 1] optional
            returns: [M] scores in [0,1]
            """
            src = node_emb[pair_idx[:, 0]]
            dst = node_emb[pair_idx[:, 1]]
            if pair_attr is None:
                pair_attr = torch.zeros((src.shape[0], 1), device=src.device)
            feats = torch.cat([src, dst, pair_attr.float()], dim=1)
            return self.edge_mlp(feats).view(-1)

        def classify_graph(self, node_emb: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
            """
            batch_idx: [N] tensor indicating batch membership of nodes (0..B-1)
            Returns logits for each graph in batch
            """
            # mean pooling
            unique_batches = torch.unique(batch_idx)
            outs = []
            for b in unique_batches:
                mask = (batch_idx == b)
                g = node_emb[mask].mean(dim=0)
                outs.append(g)
            g = torch.stack(outs, dim=0)
            return self.global_mlp(g)


# -----------------------
# Numpy pseudo fallback
# -----------------------
else:

    class NumpyPseudoGNN:
        """
        Minimal, deterministic pseudo-model using heuristics.
        Useful for debugging / minimal environments where torch is unavailable.
        """

        def __init__(self):
            logger.info("Using NumpyPseudoGNN (no torch).")

        def predict_edge_scores(self, nodes: List[Dict], candidate_pairs: List[Tuple[int, int]], candidate_attrs: Optional[List[float]] = None) -> List[float]:
            """
            nodes: list of node dicts with at least 'mass','en','pos' keys
            candidate_pairs: list of tuples (i,j)
            returns: list of float scores [0..1]
            """
            scores = []
            for k, (i, j) in enumerate(candidate_pairs):
                try:
                    ni = nodes[i]; nj = nodes[j]
                    mass_factor = (float(ni.get("mass", 0.0)) + float(nj.get("mass", 0.0))) / 50.0
                    en_diff = abs(float(ni.get("en", 0.0)) - float(nj.get("en", 0.0)))
                    posi = ni.get("pos", [0.0, 0.0]); posj = nj.get("pos", [0.0, 0.0])
                    dx = posi[0] - posj[0]; dy = posi[1] - posj[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    # heuristic: closer and moderate EN diff -> higher score
                    dscore = max(0.0, 1.0 - dist)
                    en_score = max(0.0, 1.0 - en_diff * 0.4)
                    score = 0.6 * dscore + 0.4 * en_score
                    scores.append(float(max(0.0, min(1.0, score))))
                except Exception:
                    scores.append(0.0)
            return scores

        def classify_graph(self, nodes: List[Dict]) -> int:
            """
            Very small heuristic: choose 0 as default class.
            """
            return 0


# -----------------------
# Factory helpers
# -----------------------
def create_bond_predictor(model_type: str = "pyg", **kwargs) -> Any:
    """
    Factory to create bond predictor model.
    model_type: "pyg" -> use PyG BondPredictorGNN (requires PyG)
                "dense" -> DenseGNN (requires torch)
                "numpy" -> NumpyPseudoGNN
    kwargs forwarded to model constructor (hidden_dim, node_in_dim, etc.)
    """
    if model_type == "pyg":
        if USE_TORCH and USE_PYG:
            return BondPredictorGNN(**kwargs)
        logger.warning("PyG requested but not available. Falling back.")
        model_type = "dense"
    if model_type == "dense":
        if USE_TORCH:
            return DenseGNN(**kwargs)
        logger.warning("Torch not available; falling back to numpy pseudo-model.")
        model_type = "numpy"
    if model_type == "numpy":
        return NumpyPseudoGNN()
    raise ValueError(f"Unknown model_type: {model_type}")


def create_reaction_predictor(model_type: str = "pyg", **kwargs) -> Any:
    if model_type == "pyg":
        if USE_TORCH and USE_PYG:
            return ReactionPredictorGNN(**kwargs)
        logger.warning("PyG requested but not available. Falling back.")
        model_type = "dense"
    if model_type == "dense":
        if USE_TORCH:
            return DenseGNN(**kwargs)
        logger.warning("Torch not available; falling back to numpy pseudo-model.")
        model_type = "numpy"
    if model_type == "numpy":
        return NumpyPseudoGNN()
    raise ValueError(f"Unknown model_type: {model_type}")
