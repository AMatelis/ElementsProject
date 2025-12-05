from __future__ import annotations
import os
import time
import json
import logging
import random
from typing import Optional, Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Optional: torch_geometric if using PyG GNNs
try:
    import torch_geometric
    from torch_geometric.data import Data, Dataset
    from torch_geometric.loader import DataLoader as PyGDataLoader
    USE_PYG = True
except ImportError:
    USE_PYG = False

from gnn_model import ReactionGNN  # your custom model
from dataset_builder import ReactionDataset  # wraps JSONL -> ML-ready samples
from knowledge_base import ReactionKnowledgeBase, load_elements

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_str() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

# -----------------------
# Trainer Class
# -----------------------
class Trainer:
    """
    Trainer for molecular reaction GNN models.

    Features:
      - Dataset loading from JSONL or PyG-compatible Data objects
      - Training loop with optimizer, scheduler, and loss
      - Checkpoint saving and loading
      - Optional GPU/CPU device selection
      - Logging metrics
    """

    def __init__(self,
                 model: ReactionGNN,
                 dataset: Optional[Dataset] = None,
                 elements_path: Optional[str] = None,
                 batch_size: int = 32,
                 lr: float = 1e-3,
                 device: Optional[str] = None,
                 seed: Optional[int] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler: Optional[Any] = None  # can set externally
        self.criterion = nn.CrossEntropyLoss()  # or adjust for regression tasks
        self.dataset = dataset
        self.seed = seed
        if self.seed is not None:
            set_seed(self.seed)
        self.elements = load_elements(elements_path)
        self._best_val_loss = float("inf")
        self._epoch = 0

    def prepare_dataloader(self, shuffle: bool = True) -> DataLoader:
        if self.dataset is None:
            raise ValueError("Dataset not set for trainer.")
        if USE_PYG and isinstance(self.dataset, Dataset):
            loader = PyGDataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle)
        else:
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=shuffle, collate_fn=lambda x: x)
        return loader

    # -----------------------
    # Training Loop
    # -----------------------
    def train(self,
              epochs: int = 50,
              val_dataset: Optional[Dataset] = None,
              checkpoint_dir: Optional[str] = None,
              log_interval: int = 10):
        """
        Train the model.

        Args:
            epochs: number of epochs
            val_dataset: optional validation dataset
            checkpoint_dir: directory to save checkpoints
            log_interval: iterations between logging
        """
        train_loader = self.prepare_dataloader(shuffle=True)
        val_loader = None
        if val_dataset is not None:
            val_loader = PyGDataLoader(val_dataset, batch_size=self.batch_size, shuffle=False) if USE_PYG else DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        checkpoint_dir = checkpoint_dir or os.path.join("checkpoints", now_str())
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(1, epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                self.optimizer.zero_grad()
                x, edge_index, edge_attr, y = self._prepare_batch(batch)
                x, edge_index, edge_attr, y = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), y.to(self.device)
                pred = self.model(x, edge_index, edge_attr)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += float(loss.item())
                if batch_idx % log_interval == 0:
                    logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] loss={loss.item():.6f}")

            avg_loss = epoch_loss / max(1, len(train_loader))
            logger.info(f"Epoch {epoch} completed. Avg training loss: {avg_loss:.6f}")

            # optional validation
            if val_loader:
                val_loss = self.evaluate(val_loader)
                logger.info(f"Validation loss: {val_loss:.6f}")
                if val_loss < self._best_val_loss:
                    self._best_val_loss = val_loss
                    self._save_checkpoint(checkpoint_dir, epoch, best=True)

            self._epoch = epoch
            self._save_checkpoint(checkpoint_dir, epoch)

    # -----------------------
    # Evaluation
    # -----------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            x, edge_index, edge_attr, y = self._prepare_batch(batch)
            x, edge_index, edge_attr, y = x.to(self.device), edge_index.to(self.device), edge_attr.to(self.device), y.to(self.device)
            pred = self.model(x, edge_index, edge_attr)
            loss = self.criterion(pred, y)
            total_loss += float(loss.item()) * x.size(0)
            total_samples += x.size(0)
        return total_loss / max(1, total_samples)

    # -----------------------
    # Batch preparation
    # -----------------------
    def _prepare_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert batch into tensors suitable for GNN forward pass.
        Expected batch: list of samples from ReactionDataset
        Returns: x, edge_index, edge_attr, y
        """
        if USE_PYG:
            # PyG Data object
            if isinstance(batch, Data):
                data = batch
            else:
                data = batch[0]
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            y = data.y
        else:
            # Non-PyG fallback: custom collate (simplified)
            x_list, edge_idx_list, edge_attr_list, y_list = [], [], [], []
            for sample in batch:
                nodes = sample.get("nodes", [])
                edges = sample.get("edges", [])
                edge_labels = sample.get("edge_labels", [])
                global_y = sample.get("global", {}).get("event_type", 0)
                x_list.append(torch.tensor([[n["mass"], n["en"], n["covalent_radius"]] for n in nodes], dtype=torch.float))
                edge_idx_list.append(torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2,0), dtype=torch.long))
                edge_attr_list.append(torch.tensor(edge_labels, dtype=torch.float) if edge_labels else torch.empty((0,), dtype=torch.float))
                y_list.append(torch.tensor([int(global_y) if isinstance(global_y,int) else 0], dtype=torch.long))
            # simple batching: concat
            x = torch.cat(x_list, dim=0)
            if edge_idx_list:
                edge_index = torch.cat(edge_idx_list, dim=1)
            else:
                edge_index = torch.empty((2,0), dtype=torch.long)
            if edge_attr_list:
                edge_attr = torch.cat(edge_attr_list, dim=0)
            else:
                edge_attr = torch.empty((0,), dtype=torch.float)
            y = torch.cat(y_list, dim=0)
        return x, edge_index, edge_attr, y

    # -----------------------
    # Checkpoint
    # -----------------------
    def _save_checkpoint(self, dir_path: str, epoch: int, best: bool = False):
        try:
            fname = f"model_epoch_{epoch}.pt" if not best else "model_best.pt"
            path = os.path.join(dir_path, fname)
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None
            }, path)
            logger.info(f"Saved checkpoint: {path}")
        except Exception:
            logger.exception("Failed to save checkpoint at epoch %d", epoch)

    def load_checkpoint(self, path: str, strict: bool = True):
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            self._epoch = checkpoint.get("epoch", 0)
            logger.info(f"Loaded checkpoint {path} (epoch {self._epoch})")
        except Exception:
            logger.exception("Failed to load checkpoint %s", path)

# -----------------------
# Example usage
# -----------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example with a minimal dummy dataset
    dummy_dataset = ReactionDataset(jsonl_path=os.path.join(os.path.dirname(__file__), "../data/reaction_kb.jsonl"))
    model = ReactionGNN(in_features=3, hidden_features=32, num_classes=2)
    trainer = Trainer(model=model, dataset=dummy_dataset, batch_size=16, lr=1e-3, seed=123)
    trainer.train(epochs=2)