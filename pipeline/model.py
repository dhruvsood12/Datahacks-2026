"""
pipeline/model.py — Stage 4: Multi-label MLP Species Distribution Model.

Architecture
────────────
    Linear(6 → 128) → BatchNorm → ReLU → Dropout(0.3)
    Linear(128 → 64) → BatchNorm → ReLU → Dropout(0.3)
    Linear(64 → 32)  → BatchNorm → ReLU → Dropout(0.3)
    Linear(32 → S)   # raw logits — sigmoid applied at inference only

Loss
────
BCEWithLogitsLoss with per-species pos_weight.

    pos_weight[j] = n_negative_j / n_positive_j  (clamped to ≤ 200)

This handles severe class imbalance (typical ratio 50:1 to 500:1) without
resampling and without discarding the signal in rare species.  The weight is
computed from the training split ONLY; it must not see val/test label counts.

Optimiser
─────────
Adam, lr=1e-3, weight_decay=1e-4 (light L2 regularisation)

Early stopping
──────────────
Monitors val loss.  Stops when it fails to improve by MIN_DELTA for PATIENCE
consecutive epochs.  Restores the best-epoch weights before returning.

Design contract
───────────────
SDMModel.fit() and predict_proba() accept PRE-SCALED arrays.  The caller
(notebook or Predictor in Stage 6) is responsible for scaling with SDMScaler.
This keeps the model class pure and testable in isolation.

save() persists both the model weights and enough metadata to reconstruct the
architecture on load() — no separate config file needed.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE,
    DROPOUT,
    HIDDEN_DIMS,
    INPUT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    MODEL_PATH,
    PATIENCE,
    SPECIES_LABELS_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class SDMNet(nn.Module):
    """
    Multi-label MLP core.  Forward pass returns raw logits (shape N × S).
    Do NOT apply sigmoid inside forward — BCEWithLogitsLoss is numerically
    more stable operating on logits directly.

    Architecture: [input_dim → *hidden_dims → n_species]
    Each hidden block: Linear → BatchNorm1d → ReLU → Dropout
    Output layer: Linear only (no activation, no BN)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        n_species: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, n_species))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------


class _EarlyStopping:
    """
    Stops training when validation loss does not improve by at least
    min_delta for `patience` consecutive epochs.  Stores the best model
    state in-memory so it can be restored after stopping.
    """

    def __init__(self, patience: int = PATIENCE, min_delta: float = MIN_DELTA) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_state: Optional[Dict] = None
        self.best_epoch: int = 0

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Returns True if training should stop.
        Copies model state when a new best is found.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best-epoch weights back into model (in-place)."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ---------------------------------------------------------------------------
# High-level model wrapper
# ---------------------------------------------------------------------------


class SDMModel:
    """
    sklearn-style wrapper around SDMNet.

    Responsibilities
    ────────────────
    fit()           — train with early stopping, log loss curves
    predict_proba() — return (N, S) probability matrix [0, 1]
    save()          — persist weights + metadata to .pt checkpoint
    load()          — class method: reconstruct from checkpoint

    Data contract
    ─────────────
    fit() and predict_proba() receive PRE-SCALED feature arrays (float32 or
    float64 ndarray, shape (N, INPUT_DIM)).  The caller must apply SDMScaler
    before calling these methods.  This class never touches raw features.

    Label arrays must be float32 (N, S) with values in {0, 1}.
    """

    def __init__(
        self,
        n_species: int,
        species_list: Optional[List[str]] = None,
        input_dim: int = INPUT_DIM,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = DROPOUT,
        lr: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        max_epochs: int = MAX_EPOCHS,
        patience: int = PATIENCE,
        min_delta: float = MIN_DELTA,
        device: Optional[str] = None,
    ) -> None:
        self.n_species = n_species
        self.species_list = species_list or []
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else list(HIDDEN_DIMS)
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.net = SDMNet(input_dim, self.hidden_dims, n_species, dropout).to(self.device)
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
        self._fitted: bool = False

        logger.info(
            "SDMModel initialised: input=%d  hidden=%s  n_species=%d  device=%s",
            input_dim, self.hidden_dims, n_species, self.device,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pos_weight(
        Y_train: np.ndarray,
        max_weight: float = 200.0,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """
        Compute BCEWithLogitsLoss pos_weight from training labels.

        pos_weight[j] = (N - pos_j) / pos_j, clamped to [1, max_weight].
        Species with zero positives in training get weight max_weight as a
        fallback (they will be filtered by Stage 5 spatial CV anyway).

        Data split: TRAINING DATA ONLY.  Must not use val or test labels.
        """
        N = Y_train.shape[0]
        pos_counts = Y_train.sum(axis=0).astype(np.float64)
        neg_counts = N - pos_counts

        # Avoid division by zero — species with no positives get max_weight
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = np.where(pos_counts > 0, neg_counts / pos_counts, max_weight)
        weights = np.clip(weights, 1.0, max_weight)

        logger.debug(
            "pos_weight: min=%.1f  median=%.1f  max=%.1f",
            weights.min(), float(np.median(weights)), weights.max(),
        )
        return torch.tensor(weights, dtype=torch.float32, device=device)

    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        """
        Train the MLP with early stopping on validation loss.

        Parameters
        ──────────
        X_train, X_val : float ndarray, shape (N, INPUT_DIM)  — PRE-SCALED
        Y_train, Y_val : float32 ndarray, shape (N, S)

        Returns
        ───────
        history dict with keys "train_loss" and "val_loss" (one value per epoch).

        Data split: X_train / Y_train must be the training split ONLY.
        pos_weight is derived from Y_train — calling this with the full dataset
        constitutes data leakage.
        """
        # Convert to float32 tensors
        Xt = torch.tensor(X_train, dtype=torch.float32)
        Yt = torch.tensor(Y_train, dtype=torch.float32)
        Xv = torch.tensor(X_val, dtype=torch.float32)
        Yv = torch.tensor(Y_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(Xt, Yt),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        pos_weight = self._compute_pos_weight(Y_train, device=self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimiser = torch.optim.Adam(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )
        stopper = _EarlyStopping(patience=self.patience, min_delta=self.min_delta)

        self.history = {"train_loss": [], "val_loss": []}

        logger.info(
            "Training: %d train samples, %d val samples, %d species, device=%s",
            len(X_train), len(X_val), self.n_species, self.device,
        )

        for epoch in range(1, self.max_epochs + 1):
            # ── training ──────────────────────────────────────────────
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimiser.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(X_train)

            # ── validation ────────────────────────────────────────────
            self.net.eval()
            with torch.no_grad():
                xv = Xv.to(self.device)
                yv = Yv.to(self.device)
                val_loss = criterion(self.net(xv), yv).item()

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if epoch % 10 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f",
                    epoch, self.max_epochs, train_loss, val_loss,
                )

            if stopper.step(val_loss, self.net, epoch):
                logger.info(
                    "Early stopping at epoch %d.  Best epoch: %d  val_loss=%.4f",
                    epoch, stopper.best_epoch, stopper.best_loss,
                )
                break

        stopper.restore_best(self.net)
        self._fitted = True
        logger.info(
            "Training complete.  Best epoch: %d  best_val_loss=%.4f",
            stopper.best_epoch, stopper.best_loss,
        )
        return self.history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return (N, S) probability matrix with values in [0, 1].

        Data split: any — but the scaler must have been fitted on training data.
        Raises RuntimeError if called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "SDMModel.predict_proba called before fit().  "
                "Call model.fit(X_train, Y_train, X_val, Y_val) first."
            )
        self.net.eval()
        Xt = torch.tensor(X, dtype=torch.float32)
        loader = DataLoader(TensorDataset(Xt), batch_size=self.batch_size, shuffle=False)
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device)
                logits = self.net(xb)
                probs.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(probs, axis=0)

    def predict_binary(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Return (N, S) binary predictions using a per-model threshold."""
        return (self.predict_proba(X) >= threshold).astype(np.float32)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = MODEL_PATH) -> None:
        """
        Persist model weights + architecture metadata to a .pt checkpoint.

        Checkpoint keys:
            state_dict      — SDMNet weights
            input_dim       — int
            hidden_dims     — list[int]
            n_species       — int
            dropout         — float
            species_list    — list[str]
            history         — {train_loss, val_loss}

        Correctness: call ONLY after final training is complete.  Do not
        call inside cross-validation loops — each fold's model is temporary
        and must not overwrite the final checkpoint.
        Raises RuntimeError if model has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError(
                "SDMModel.save called before fit().  Train the model first."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.net.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "n_species": self.n_species,
            "dropout": self.dropout,
            "species_list": self.species_list,
            "history": self.history,
        }
        torch.save(checkpoint, path)
        logger.info("Saved SDMModel → %s", path)

    @classmethod
    def load(cls, path: Path = MODEL_PATH, device: Optional[str] = None) -> "SDMModel":
        """
        Reconstruct an SDMModel from a .pt checkpoint.

        The loaded model is ready for predict_proba(); fit() should not be
        called on a loaded model (it would overwrite the trained weights).
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            n_species=checkpoint["n_species"],
            species_list=checkpoint.get("species_list", []),
            input_dim=checkpoint["input_dim"],
            hidden_dims=checkpoint["hidden_dims"],
            dropout=checkpoint["dropout"],
            device=device,
        )
        model.net.load_state_dict(checkpoint["state_dict"])
        model.history = checkpoint.get("history", {})
        model._fitted = True
        logger.info(
            "Loaded SDMModel from %s  (n_species=%d, device=%s)",
            path, model.n_species, model.device,
        )
        return model

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """True after fit() has been called successfully."""
        return self._fitted

    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"SDMModel(n_species={self.n_species}, "
            f"hidden={self.hidden_dims}, "
            f"fitted={self._fitted}, "
            f"device={self.device})"
        )


# ---------------------------------------------------------------------------
# Species label utilities
# ---------------------------------------------------------------------------


def save_species_labels(species_list: List[str], path: Path = SPECIES_LABELS_PATH) -> None:
    """
    Save the ordered species list to JSON.

    The list order is the canonical mapping from model output column index to
    species name.  It must be identical at training and inference time.

    Correctness: this must be called with the SAME species_list used to
    build the label matrix and the SDMModel.  Any mismatch silently corrupts
    predictions.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info("Saved %d species labels → %s", len(species_list), path)


def load_species_labels(path: Path = SPECIES_LABELS_PATH) -> List[str]:
    """Load the ordered species list from JSON."""
    with open(path) as f:
        labels = json.load(f)
    if not isinstance(labels, list):
        raise TypeError(f"Expected list in {path}, got {type(labels)}")
    logger.info("Loaded %d species labels from %s", len(labels), path)
    return labels


# ---------------------------------------------------------------------------
# Per-species AUC helper
# ---------------------------------------------------------------------------


def compute_species_auc(
    Y_true: np.ndarray,
    Y_prob: np.ndarray,
    species_list: List[str],
    min_positives: int = 5,
) -> Dict[str, float]:
    """
    Compute per-species ROC-AUC on a held-out split.

    Species with fewer than min_positives presences in Y_true are skipped
    (AUC is unreliable with too few positives) and logged as NaN.

    Returns a dict {taxon_name: auc_score} for species with enough positives.
    NaN-valued species are included with value float('nan') for completeness.

    Data split: Y_true / Y_prob must be from the test or val split.
    """
    from sklearn.metrics import roc_auc_score

    results: Dict[str, float] = {}
    for j, sp in enumerate(species_list):
        n_pos = int(Y_true[:, j].sum())
        if n_pos < min_positives:
            results[sp] = float("nan")
            continue
        try:
            auc = roc_auc_score(Y_true[:, j], Y_prob[:, j])
            results[sp] = float(auc)
        except ValueError:
            results[sp] = float("nan")

    valid = {sp: v for sp, v in results.items() if not np.isnan(v)}
    logger.info(
        "AUC computed for %d/%d species  mean=%.3f  median=%.3f",
        len(valid), len(species_list),
        float(np.mean(list(valid.values()))) if valid else float("nan"),
        float(np.median(list(valid.values()))) if valid else float("nan"),
    )
    return results


# ---------------------------------------------------------------------------
# Smoke test / CLI
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Minimal sanity check: build a tiny model, train 2 epochs, predict."""
    import pandas as pd
    from config import SAMPLED_PATH
    from pipeline.features import (
        build_features,
        build_label_matrix_fast,
        build_scaler,
        get_species_list,
    )

    logger.info("Running model.py smoke test…")

    df = pd.read_parquet(SAMPLED_PATH)
    X = build_features(df)
    species = get_species_list(df)
    Y = build_label_matrix_fast(df, species)

    # Tiny split
    n = len(X)
    n_train = int(0.7 * n)
    n_val = int(0.85 * n)
    X_tr, X_v, X_te = X[:n_train], X[n_train:n_val], X[n_val:]
    Y_tr, Y_v, Y_te = Y[:n_train], Y[n_train:n_val], Y[n_val:]

    scaler = build_scaler()
    X_tr_s = scaler.fit(X_tr).transform(X_tr)
    X_v_s = scaler.transform(X_v)
    X_te_s = scaler.transform(X_te)

    model = SDMModel(
        n_species=len(species),
        species_list=species,
        max_epochs=2,
        patience=5,
    )
    logger.info("Parameters: %d", model.parameter_count())
    history = model.fit(X_tr_s, Y_tr, X_v_s, Y_v)
    assert len(history["train_loss"]) == 2, "Expected 2 epochs"

    probs = model.predict_proba(X_te_s)
    assert probs.shape == (len(X_te), len(species)), f"Bad proba shape: {probs.shape}"
    assert 0.0 <= probs.min() and probs.max() <= 1.0, "Probs outside [0,1]"

    auc = compute_species_auc(Y_te, probs, species)
    valid_auc = [v for v in auc.values() if not np.isnan(v)]
    logger.info("Valid AUC scores: %d  mean=%.3f", len(valid_auc), np.mean(valid_auc))

    logger.info("Smoke test PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    _smoke_test()
