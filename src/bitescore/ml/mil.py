"""Multiple Instance Learning (MIL) model for food-level digestibility prediction.

This module implements an attention-based MIL architecture that learns to
predict food-level digestibility (DIAAS) from per-protein feature vectors.

Architecture
------------
1. **Protein Encoder**: MLP that transforms per-protein features into a
   learned representation space.
2. **Attention Pooling**: Gated attention mechanism that learns which proteins
   in a food mixture contribute most to digestibility, optionally incorporating
   protein abundance fractions.
3. **Regression Head**: MLP that maps the attention-weighted food
   representation to a scalar digestibility score.

The model handles the fundamental mismatch between protein-level features and
food-level experimental labels by learning an aggregation function end-to-end.

References
----------
Ilse M, Tomczak JM, Welling M (2018). Attention-based Deep Multiple Instance
    Learning. ICML 2018.

Requires the optional ``torch`` package::

    pip install torch
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _check_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FoodBag:
    """A 'bag' of protein feature vectors representing a single food.

    In MIL terminology, the food is the bag and individual proteins are
    instances.
    """
    food_id: str
    protein_ids: List[str]
    features: np.ndarray          # (n_proteins, n_features)
    abundance: np.ndarray         # (n_proteins,) — sums to 1.0
    label: float                  # experimental DIAAS (normalized 0-1)


@dataclass
class MILConfig:
    """Hyperparameters for the MIL model."""
    hidden_dim: int = 256
    attention_dim: int = 128
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 200
    patience: int = 30  # early stopping patience
    min_delta: float = 1e-4
    label_scale: float = 140.0  # max DIAAS for normalization


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------

def _build_model(input_dim: int, cfg: MILConfig):
    """Build the MIL PyTorch model. Returns (model_class_instance)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ProteinEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            )

        def forward(self, x):
            return self.net(x)

    class GatedAttentionPooling(nn.Module):
        """Gated attention mechanism (Ilse et al. 2018).

        Computes attention weights as:
            a = softmax(V * tanh(W * h) ⊙ sigmoid(U * h))

        where h is the protein encoding, and the gating term (sigmoid)
        allows the model to suppress irrelevant proteins.
        """
        def __init__(self):
            super().__init__()
            self.W = nn.Linear(cfg.hidden_dim, cfg.attention_dim)
            self.U = nn.Linear(cfg.hidden_dim, cfg.attention_dim)
            self.V = nn.Linear(cfg.attention_dim, 1)

        def forward(self, h, abundance=None):
            # h: (n_proteins, hidden_dim)
            tanh_out = torch.tanh(self.W(h))        # (n, attn_dim)
            sigm_out = torch.sigmoid(self.U(h))      # (n, attn_dim)
            a = self.V(tanh_out * sigm_out)           # (n, 1)

            # Incorporate abundance as a prior on attention
            if abundance is not None:
                a = a + torch.log(abundance.unsqueeze(-1) + 1e-8)

            a = F.softmax(a, dim=0)                   # (n, 1)
            pooled = (a * h).sum(dim=0)               # (hidden_dim,)
            return pooled, a.squeeze(-1)

    class DigestibilityRegressor(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Sequential(
                nn.Linear(cfg.hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.head(x).squeeze(-1)

    class MILDigestibilityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = ProteinEncoder()
            self.attention = GatedAttentionPooling()
            self.regressor = DigestibilityRegressor()

        def forward(self, protein_features, abundance=None):
            """Predict food-level digestibility from protein features.

            Parameters
            ----------
            protein_features : Tensor of shape (n_proteins, input_dim)
            abundance : optional Tensor of shape (n_proteins,)

            Returns
            -------
            score : scalar Tensor — predicted normalized DIAAS (0-1)
            attention_weights : Tensor of shape (n_proteins,)
            """
            h = self.encoder(protein_features)
            pooled, attn_weights = self.attention(h, abundance)
            score = self.regressor(pooled)
            return score, attn_weights

        def protein_scores(self, protein_features):
            """Compute per-protein digestibility scores.

            Passes each protein's encoding individually through the regressor
            to get an intrinsic per-protein score, independent of food context.
            """
            h = self.encoder(protein_features)
            scores = self.regressor(h)
            return scores

    return MILDigestibilityModel()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_mil_model(
    food_bags: List[FoodBag],
    cfg: MILConfig | None = None,
    val_fraction: float = 0.2,
) -> Tuple:
    """Train the MIL model on food-level digestibility data.

    Parameters
    ----------
    food_bags : list of FoodBag instances
    cfg : MIL hyperparameters (defaults used if None)
    val_fraction : fraction of foods held out for validation/early stopping

    Returns
    -------
    (model, training_history) — trained PyTorch model and loss history dict
    """
    if not _check_torch():
        raise ImportError("PyTorch is required for MIL training: pip install torch")

    import torch
    import torch.nn.functional as F

    if cfg is None:
        cfg = MILConfig()

    if len(food_bags) < 2:
        raise ValueError("Need at least 2 food bags for MIL training.")

    input_dim = food_bags[0].features.shape[1]
    model = _build_model(input_dim, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Train/val split
    n_val = max(1, int(len(food_bags) * val_fraction))
    indices = np.random.RandomState(42).permutation(len(food_bags))
    val_indices = set(indices[:n_val].tolist())
    train_bags = [b for i, b in enumerate(food_bags) if i not in val_indices]
    val_bags = [b for i, b in enumerate(food_bags) if i in val_indices]

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.n_epochs):
        # --- Training ---
        model.train()
        train_losses = []
        perm = np.random.permutation(len(train_bags))
        for idx in perm:
            bag = train_bags[idx]
            features_t = torch.tensor(bag.features, dtype=torch.float32, device=device)
            abundance_t = torch.tensor(bag.abundance, dtype=torch.float32, device=device)
            label_t = torch.tensor(bag.label, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred, _ = model(features_t, abundance_t)
            loss = F.mse_loss(pred, label_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # --- Validation ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for bag in val_bags:
                features_t = torch.tensor(bag.features, dtype=torch.float32, device=device)
                abundance_t = torch.tensor(bag.abundance, dtype=torch.float32, device=device)
                label_t = torch.tensor(bag.label, dtype=torch.float32, device=device)
                pred, _ = model(features_t, abundance_t)
                val_losses.append(F.mse_loss(pred, label_t).item())

        avg_val = np.mean(val_losses) if val_losses else avg_train
        history["val_loss"].append(avg_val)
        scheduler.step(avg_val)

        # Early stopping
        if avg_val < best_val_loss - cfg.min_delta:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            logger.info("Early stopping at epoch %d (val_loss=%.6f)", epoch, best_val_loss)
            break

        if (epoch + 1) % 50 == 0:
            logger.info(
                "Epoch %d/%d — train_loss=%.6f val_loss=%.6f",
                epoch + 1, cfg.n_epochs, avg_train, avg_val,
            )

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return model, history


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_food_digestibility(
    model,
    protein_features: np.ndarray,
    abundance: np.ndarray | None = None,
    label_scale: float = 140.0,
) -> Tuple[float, np.ndarray]:
    """Predict food-level digestibility from protein feature matrix.

    Parameters
    ----------
    model : trained MIL model
    protein_features : (n_proteins, n_features) array
    abundance : optional (n_proteins,) array of abundance fractions
    label_scale : scale factor to convert normalized score back to DIAAS

    Returns
    -------
    (diaas_score, attention_weights) — predicted DIAAS and per-protein attention
    """
    import torch

    device = next(model.parameters()).device
    features_t = torch.tensor(protein_features, dtype=torch.float32, device=device)
    abundance_t = None
    if abundance is not None:
        abundance_t = torch.tensor(abundance, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        score, attn = model(features_t, abundance_t)

    diaas = score.item() * label_scale
    return diaas, attn.cpu().numpy()


def predict_protein_scores(
    model,
    protein_features: np.ndarray,
    label_scale: float = 140.0,
) -> np.ndarray:
    """Predict per-protein digestibility scores.

    Uses the MIL encoder + regressor to score each protein individually,
    without attention pooling (suitable for ranking individual proteins).

    Returns array of shape (n_proteins,) with DIAAS-scale scores.
    """
    import torch

    device = next(model.parameters()).device
    features_t = torch.tensor(protein_features, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        scores = model.protein_scores(features_t)

    return (scores.cpu().numpy() * label_scale)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_mil_model(model, cfg: MILConfig, path: Path):
    """Save MIL model weights and config."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "hidden_dim": cfg.hidden_dim,
                "attention_dim": cfg.attention_dim,
                "dropout": cfg.dropout,
                "label_scale": cfg.label_scale,
            },
        },
        path,
    )
    logger.info("MIL model saved to %s", path)


def load_mil_model(path: Path, input_dim: int):
    """Load a saved MIL model.

    Parameters
    ----------
    path : path to saved model file
    input_dim : number of input features (must match training)

    Returns
    -------
    (model, cfg)
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    saved_cfg = checkpoint["config"]
    cfg = MILConfig(
        hidden_dim=saved_cfg["hidden_dim"],
        attention_dim=saved_cfg["attention_dim"],
        dropout=saved_cfg["dropout"],
        label_scale=saved_cfg.get("label_scale", 140.0),
    )
    model = _build_model(input_dim, cfg)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, cfg
