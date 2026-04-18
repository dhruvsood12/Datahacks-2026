"""
train.py — Standalone Stage 4 training script.

Trains the SDM, saves model / scaler / species_labels / species_metadata.
Equivalent to running notebooks/03_model.ipynb end-to-end.

Usage:
    python train.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    BATCH_SIZE, DROPOUT, HIDDEN_DIMS, INPUT_DIM, LEARNING_RATE,
    MAX_EPOCHS, MIN_DELTA, MODEL_PATH, MODELS_DIR, PATIENCE,
    SAMPLED_PATH, SCALER_PATH, SPECIES_LABELS_PATH,
)
from pipeline.features import (
    build_features, build_label_matrix_fast, build_scaler,
    build_species_metadata, get_species_list,
)
from pipeline.model import SDMModel, compute_species_auc, save_species_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("=== Stage 4: Training ===")

    # 1 — Load data
    df = pd.read_parquet(SAMPLED_PATH)
    logger.info("Loaded %d rows from %s", len(df), SAMPLED_PATH)

    X = build_features(df)
    species_list = get_species_list(df)
    S = len(species_list)
    Y = build_label_matrix_fast(df, species_list)
    logger.info("Feature matrix: %s  Label matrix: %s  Species: %d", X.shape, Y.shape, S)

    # 2 — Split (70 / 15 / 15, stratified)
    idx_all = np.arange(len(X))
    idx_train, idx_temp = train_test_split(
        idx_all, test_size=0.30, random_state=42,
        stratify=df["presence"].values,
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=42,
        stratify=df["presence"].values[idx_temp],
    )
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val,   Y_val   = X[idx_val],   Y[idx_val]
    X_test,  Y_test  = X[idx_test],  Y[idx_test]
    logger.info("Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test))

    # 3 — Scaler (train only)
    scaler = build_scaler()
    X_train_s = scaler.fit(X_train).transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    # 4 — Train
    model = SDMModel(
        n_species=S,
        species_list=species_list,
        input_dim=INPUT_DIM,
        hidden_dims=list(HIDDEN_DIMS),
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        min_delta=MIN_DELTA,
    )
    logger.info("Model parameters: %d", model.parameter_count())
    history = model.fit(X_train_s, Y_train, X_val_s, Y_val)
    logger.info(
        "Training done: %d epochs, best_val_loss=%.4f",
        len(history["train_loss"]), min(history["val_loss"]),
    )

    # 5 — Test AUC
    Y_prob_test = model.predict_proba(X_test_s)
    auc_scores = compute_species_auc(Y_test, Y_prob_test, species_list, min_positives=5)
    valid_auc = [v for v in auc_scores.values() if not np.isnan(v)]
    logger.info(
        "Test AUC: n=%d  mean=%.3f  median=%.3f",
        len(valid_auc), np.mean(valid_auc), float(np.median(valid_auc)),
    )

    # 6 — Save artefacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)
    scaler.save(SCALER_PATH)
    save_species_labels(species_list, SPECIES_LABELS_PATH)

    meta = build_species_metadata(df, species_list)
    with open(MODELS_DIR / "species_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Saved species_metadata.json")

    auc_out = {sp: (None if np.isnan(v) else round(v, 4)) for sp, v in auc_scores.items()}
    with open(MODELS_DIR / "species_auc_random_split.json", "w") as f:
        json.dump(auc_out, f, indent=2)
    logger.info("Saved species_auc_random_split.json")

    logger.info("=== Stage 4 complete ===")
    logger.info("Artefacts in %s:", MODELS_DIR)
    for p in sorted(MODELS_DIR.iterdir()):
        if p.suffix not in {".gitkeep"}:
            logger.info("  %-40s  %.1f KB", p.name, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
