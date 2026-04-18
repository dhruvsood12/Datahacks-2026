"""
pipeline/evaluation.py — Stage 5: Blocked spatial cross-validation.

Why spatial CV?
───────────────
Standard random k-fold inflates AUC because training and test points sit next
to each other — the model memorises spatial autocorrelation rather than genuine
habitat preference.  Blocked spatial CV holds out entire geographic blocks, so
the test set is spatially independent of training.

Method
──────
The study area is divided into a CV_GRID_SIZE × CV_GRID_SIZE (default 5×5)
regular grid.  Each non-empty block is held out once as the test fold while
all remaining blocks form the training set.  This produces up to 25 folds,
though only blocks that contain ≥1 observation are used.

Key correctness rules
─────────────────────
1. The scaler is refitted on each fold's training blocks — never on the
   held-out block.  Using the final scaler across folds would leak test
   statistics.
2. pos_weight is recomputed from each fold's training labels.
3. Models trained inside the CV loop are throw-away — they must NOT be
   saved with scaler.save() or model.save().  Only the final model
   (trained outside this module) should be persisted.
4. Species with fewer than min_positives presences in the held-out block
   are skipped for that fold (AUC is unreliable with too few positives).

Output
──────
A dict mapping taxon_name → list of per-fold AUC values (NaN for folds
where the species had too few positives).  Callers aggregate to mean/std.

MIN_SPECIES_AUC (0.65) is the threshold below which a species is flagged for
exclusion from the final model.  Stage 4's final model is already trained on
all species; the Stage 5 AUC list tells the inference layer (Stage 6) which
species predictions are reliable.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BATCH_SIZE,
    CV_GRID_SIZE,
    DROPOUT,
    HIDDEN_DIMS,
    INPUT_DIM,
    LEARNING_RATE,
    MAX_EPOCHS,
    MIN_DELTA,
    MIN_SPECIES_AUC,
    MODELS_DIR,
    PATIENCE,
    SAMPLED_PATH,
    SD_LAT_MAX,
    SD_LAT_MIN,
    SD_LON_MAX,
    SD_LON_MIN,
)
from pipeline.features import (
    build_features,
    build_label_matrix_fast,
    build_scaler,
    get_species_list,
)
from pipeline.model import SDMModel, compute_species_auc

logger = logging.getLogger(__name__)

_CV_AUC_PATH = MODELS_DIR / "species_auc_spatial_cv.json"
_FILTERED_LABELS_PATH = MODELS_DIR / "species_labels_filtered.json"


# ---------------------------------------------------------------------------
# Block assignment
# ---------------------------------------------------------------------------


def assign_blocks(
    lat: np.ndarray,
    lon: np.ndarray,
    n_lat: int = CV_GRID_SIZE,
    n_lon: int = CV_GRID_SIZE,
    lat_min: float = SD_LAT_MIN,
    lat_max: float = SD_LAT_MAX,
    lon_min: float = SD_LON_MIN,
    lon_max: float = SD_LON_MAX,
) -> np.ndarray:
    """
    Assign each observation to a geographic block index (0-based).

    The study area is divided into n_lat × n_lon equal-area grid cells.
    Block index = lat_band * n_lon + lon_band.

    Returns: integer array of shape (N,) with values in [0, n_lat*n_lon).
    Observations outside the bounding box are clipped to the nearest edge cell.
    """
    lat_edges = np.linspace(lat_min, lat_max, n_lat + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon + 1)

    lat_band = np.digitize(lat, lat_edges, right=False) - 1
    lon_band = np.digitize(lon, lon_edges, right=False) - 1

    # Clip to valid range (handles points exactly on the upper boundary)
    lat_band = np.clip(lat_band, 0, n_lat - 1)
    lon_band = np.clip(lon_band, 0, n_lon - 1)

    return lat_band * n_lon + lon_band


# ---------------------------------------------------------------------------
# Single CV fold
# ---------------------------------------------------------------------------


def _run_fold(
    X: np.ndarray,
    Y: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    species_list: List[str],
    n_species: int,
    cv_max_epochs: int,
    cv_patience: int,
    min_positives: int,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    Train one CV fold and return per-species AUC on the held-out block.

    Correctness: scaler is fitted on X[train_mask] only.  The returned model
    is discarded — do not save it.
    """
    X_tr, Y_tr = X[train_mask], Y[train_mask]
    X_te, Y_te = X[test_mask],  Y[test_mask]

    # Fit scaler on this fold's training data ONLY
    scaler = build_scaler()
    X_tr_s = scaler.fit(X_tr).transform(X_tr)
    X_te_s = scaler.transform(X_te)

    # Reserve 10% of training rows as a mini-val split for early stopping
    n_tr = len(X_tr_s)
    n_val = max(1, int(0.10 * n_tr))
    X_cv_tr, X_cv_v = X_tr_s[:-n_val], X_tr_s[-n_val:]
    Y_cv_tr, Y_cv_v = Y_tr[:-n_val],   Y_tr[-n_val:]

    fold_model = SDMModel(
        n_species=n_species,
        species_list=species_list,
        input_dim=INPUT_DIM,
        hidden_dims=list(HIDDEN_DIMS),
        dropout=DROPOUT,
        lr=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_epochs=cv_max_epochs,
        patience=cv_patience,
        min_delta=MIN_DELTA,
        device=device,
    )

    fold_model.fit(X_cv_tr, Y_cv_tr, X_cv_v, Y_cv_v)
    Y_prob = fold_model.predict_proba(X_te_s)

    auc = compute_species_auc(Y_te, Y_prob, species_list, min_positives=min_positives)
    return auc


# ---------------------------------------------------------------------------
# Full spatial CV
# ---------------------------------------------------------------------------


def run_spatial_cv(
    df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    species_list: List[str],
    n_grid: int = CV_GRID_SIZE,
    min_block_samples: int = 50,
    min_positives: int = 3,
    cv_max_epochs: int = 100,
    cv_patience: int = 8,
    device: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Blocked spatial leave-one-block-out cross-validation.

    For each non-empty geographic block:
      1. Assign all rows to the n_grid × n_grid block grid.
      2. Hold out the block as test; all other blocks form training.
      3. Refit scaler on training blocks only.
      4. Train a throw-away model with early stopping.
      5. Record per-species AUC on the held-out block.

    Parameters
    ──────────
    df                : sampled.parquet DataFrame (must have lat, lon columns)
    X                 : (N, 6) feature matrix — raw (unscaled)
    Y                 : (N, S) label matrix
    species_list      : ordered species names
    n_grid            : grid size each side (default 5 → up to 25 folds)
    min_block_samples : skip folds where the held-out block has fewer rows
    min_positives     : minimum presences in test block to compute AUC
    cv_max_epochs     : epoch cap for CV fold models (lower than final model)
    cv_patience       : early-stopping patience for CV fold models

    Returns
    ───────
    Dict mapping taxon_name → list of per-fold AUC values.
    NaN entries mean the species had too few positives in that fold.
    Folds where the block had < min_block_samples rows are not included.

    Correctness: never call model.save() or scaler.save() inside this function.
    """
    n_species = len(species_list)

    # Use actual data extent (not the full SD bounding box) so that all data
    # isn't crammed into one cell.  Add a tiny buffer to avoid edge clipping.
    lat_vals = df["lat"].values
    lon_vals = df["lon"].values
    buf = 1e-4
    data_lat_min = float(lat_vals.min()) - buf
    data_lat_max = float(lat_vals.max()) + buf
    data_lon_min = float(lon_vals.min()) - buf
    data_lon_max = float(lon_vals.max()) + buf

    logger.info(
        "Spatial CV grid extent (data-driven): lat [%.4f, %.4f]  lon [%.4f, %.4f]",
        data_lat_min, data_lat_max, data_lon_min, data_lon_max,
    )

    blocks = assign_blocks(
        lat_vals, lon_vals,
        n_lat=n_grid, n_lon=n_grid,
        lat_min=data_lat_min, lat_max=data_lat_max,
        lon_min=data_lon_min, lon_max=data_lon_max,
    )
    unique_blocks = np.unique(blocks)
    n_folds_total = len(unique_blocks)

    logger.info(
        "Spatial CV: %d×%d grid, %d non-empty blocks, %d species",
        n_grid, n_grid, n_folds_total, n_species,
    )

    # Accumulator: species → list of fold AUC values
    fold_aucs: Dict[str, List[float]] = {sp: [] for sp in species_list}
    folds_run = 0

    for fold_i, block_id in enumerate(unique_blocks):
        test_mask  = blocks == block_id
        train_mask = ~test_mask
        n_test  = test_mask.sum()
        n_train = train_mask.sum()

        if n_test < min_block_samples:
            logger.info(
                "Fold %d/%d  block=%d  SKIPPED (only %d test rows < %d)",
                fold_i + 1, n_folds_total, block_id, n_test, min_block_samples,
            )
            continue

        n_test_pos = int(Y[test_mask].sum())
        logger.info(
            "Fold %d/%d  block=%d  train=%d  test=%d  test_pos=%d",
            fold_i + 1, n_folds_total, block_id, n_train, n_test, n_test_pos,
        )

        fold_result = _run_fold(
            X, Y, train_mask, test_mask,
            species_list, n_species,
            cv_max_epochs, cv_patience,
            min_positives, device,
        )

        for sp in species_list:
            fold_aucs[sp].append(fold_result.get(sp, float("nan")))

        folds_run += 1

    logger.info("Spatial CV complete: %d folds run", folds_run)
    return fold_aucs


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_cv_auc(
    fold_aucs: Dict[str, List[float]],
) -> pd.DataFrame:
    """
    Summarise per-species spatial CV AUC results.

    Returns a DataFrame indexed by taxon_name with columns:
        mean_auc, std_auc, n_folds, min_auc, max_auc, above_threshold
    Sorted by mean_auc descending.
    """
    rows = []
    for sp, aucs in fold_aucs.items():
        valid = [a for a in aucs if not np.isnan(a)]
        n = len(valid)
        if n == 0:
            rows.append({
                "taxon_name": sp,
                "mean_auc": float("nan"),
                "std_auc": float("nan"),
                "n_folds": 0,
                "min_auc": float("nan"),
                "max_auc": float("nan"),
                "above_threshold": False,
            })
        else:
            m = float(np.mean(valid))
            rows.append({
                "taxon_name": sp,
                "mean_auc": round(m, 4),
                "std_auc": round(float(np.std(valid)), 4),
                "n_folds": n,
                "min_auc": round(float(np.min(valid)), 4),
                "max_auc": round(float(np.max(valid)), 4),
                "above_threshold": m >= MIN_SPECIES_AUC,
            })

    df = pd.DataFrame(rows).set_index("taxon_name")
    df = df.sort_values("mean_auc", ascending=False)
    return df


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------


def save_cv_auc(
    fold_aucs: Dict[str, List[float]],
    path: Path = _CV_AUC_PATH,
) -> None:
    """Save per-fold AUC lists to JSON for reproducibility."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = {
        sp: [None if np.isnan(v) else round(v, 4) for v in aucs]
        for sp, aucs in fold_aucs.items()
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    logger.info("Saved spatial CV AUC → %s", path)


def load_cv_auc(path: Path = _CV_AUC_PATH) -> Dict[str, List[float]]:
    """Load per-fold AUC lists from JSON."""
    with open(path) as f:
        raw = json.load(f)
    return {
        sp: [float("nan") if v is None else float(v) for v in aucs]
        for sp, aucs in raw.items()
    }


def save_filtered_species(
    species_list: List[str],
    path: Path = _FILTERED_LABELS_PATH,
) -> None:
    """Persist the post-CV filtered species list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(species_list, f, indent=2)
    logger.info(
        "Saved %d filtered species labels → %s", len(species_list), path
    )


# ---------------------------------------------------------------------------
# Full Stage 5 orchestrator
# ---------------------------------------------------------------------------


def run_evaluation(
    sampled_path: Path = SAMPLED_PATH,
    output_dir: Path = MODELS_DIR,
    n_grid: int = CV_GRID_SIZE,
    min_block_samples: int = 50,
    cv_max_epochs: int = 100,
    cv_patience: int = 8,
    device: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Full Stage 5 pipeline: load data → spatial CV → aggregate → report.

    Returns
    ───────
    (summary_df, species_above, species_below)
        summary_df    : per-species CV AUC summary DataFrame
        species_above : species with mean spatial CV AUC ≥ MIN_SPECIES_AUC
        species_below : species below threshold (flag for exclusion)
    """
    logger.info("=== Stage 5: Spatial Cross-Validation ===")

    df = pd.read_parquet(sampled_path)
    X = build_features(df)
    species_list = get_species_list(df)
    Y = build_label_matrix_fast(df, species_list)

    logger.info(
        "Data: %d rows, %d species, %d total positives",
        len(df), len(species_list), int(Y.sum()),
    )

    fold_aucs = run_spatial_cv(
        df, X, Y, species_list,
        n_grid=n_grid,
        min_block_samples=min_block_samples,
        cv_max_epochs=cv_max_epochs,
        cv_patience=cv_patience,
        device=device,
    )

    save_cv_auc(fold_aucs, output_dir / "species_auc_spatial_cv.json")

    summary = aggregate_cv_auc(fold_aucs)

    species_above = summary.index[summary["above_threshold"]].tolist()
    species_below = summary.index[~summary["above_threshold"]].tolist()

    save_filtered_species(species_above, output_dir / "species_labels_filtered.json")

    _print_cv_report(summary, species_above, species_below)

    return summary, species_above, species_below


def _print_cv_report(
    summary: pd.DataFrame,
    species_above: List[str],
    species_below: List[str],
) -> None:
    valid = summary.dropna(subset=["mean_auc"])
    sep = "=" * 64
    lines = [
        "",
        sep,
        "  STAGE 5 — SPATIAL CV REPORT",
        sep,
        f"  Total species evaluated  : {len(valid)}",
        f"  Mean spatial CV AUC      : {valid['mean_auc'].mean():.3f}",
        f"  Median spatial CV AUC    : {valid['mean_auc'].median():.3f}",
        f"  ≥ {MIN_SPECIES_AUC} (included)      : {len(species_above)}",
        f"  < {MIN_SPECIES_AUC} (flagged)       : {len(species_below)}",
        "",
        "  Top 5 species:",
    ]
    for sp, row in summary.head(5).iterrows():
        lines.append(
            f"    {row['mean_auc']:.3f} ± {row['std_auc']:.3f}  {sp}"
        )
    if len(species_below) > 0:
        lines += ["", "  Species below threshold:"]
        for sp in sorted(species_below)[:10]:
            row = summary.loc[sp]
            lines.append(
                f"    {row['mean_auc']:.3f}  {sp}"
                if not np.isnan(row["mean_auc"]) else f"    NaN  {sp}"
            )
        if len(species_below) > 10:
            lines.append(f"    ... and {len(species_below)-10} more")
    lines += ["", f"  Saved: species_labels_filtered.json ({len(species_above)} species)", sep]
    print("\n".join(lines))
