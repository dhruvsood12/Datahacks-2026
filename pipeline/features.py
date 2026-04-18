"""
pipeline/features.py — Stage 3: Feature engineering and train-safe scaling.

Computes the 6 model input features from sampled.parquet and exposes an
SDMScaler class that structurally prevents data leakage by refusing to
transform data before it has been fitted on a training split.

──────────────────────────────────────────────────────────────────────
Feature vector (INPUT_DIM = 6)
──────────────────────────────────────────────────────────────────────
  Index  Name            Description
  ─────  ──────────────  ─────────────────────────────────────────────
  0      temperature_c   sensor climatology mean temperature (°C)
  1      humidity_pct    sensor climatology mean relative humidity (%)
  2      lat             decimal degrees latitude
  3      lon             decimal degrees longitude
  4      sin_doy         sin(day_of_year × 2π / 365)
  5      cos_doy         cos(day_of_year × 2π / 365)

sin/cos encoding ensures the seasonal signal is continuous at the year
boundary (day 365 wraps smoothly to day 1).  Raw day_of_year would
create a discontinuity that would need special handling.

──────────────────────────────────────────────────────────────────────
Scaler design — structural leakage prevention
──────────────────────────────────────────────────────────────────────
build_scaler() returns a deliberately UNFITTED SDMScaler.  Callers must
call scaler.fit(X_train) before scaler.transform(X).  Calling transform()
on an unfitted scaler raises RuntimeError immediately rather than silently
returning wrong values.

Rule: fit() is called exactly ONCE per model — on the training split.
      It must NEVER be called on val, test, or on the full dataset.
      save() is only called after final training (Stage 4), not in CV loops.

──────────────────────────────────────────────────────────────────────
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import INPUT_DIM, MODELS_DIR, SAMPLED_PATH, SCALER_PATH, SPECIES_LABELS_PATH

logger = logging.getLogger(__name__)

# Canonical feature column names — index matches feature vector position
FEATURE_NAMES: List[str] = [
    "temperature_c",
    "humidity_pct",
    "lat",
    "lon",
    "sin_doy",
    "cos_doy",
]

assert len(FEATURE_NAMES) == INPUT_DIM, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but INPUT_DIM={INPUT_DIM}"
)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------


def compute_doy_features(day_of_year: np.ndarray) -> np.ndarray:
    """
    Encode day-of-year as (sin, cos) pair for circular continuity.

    Uses 365-day period so the encoding is consistent across leap and
    non-leap years.  The pair together encode both the phase and period
    of the annual cycle without a discontinuity at Dec 31 → Jan 1.

    Data split: any — this is a deterministic transformation with no
                fitted parameters.
    Returns: array of shape (N, 2), columns [sin_doy, cos_doy].
    """
    angle = day_of_year * (2.0 * np.pi / 365.0)
    return np.column_stack([np.sin(angle), np.cos(angle)])


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute the 6-dimensional feature matrix from a sampled DataFrame.

    Input columns required: temperature_c, humidity_pct, lat, lon, day_of_year.
    All six features are returned in the canonical order defined by FEATURE_NAMES.

    Data split: any split of sampled.parquet — this function has no fitted
                parameters.  The caller is responsible for passing only the
                appropriate split and scaling afterwards with SDMScaler.
    Returns: float64 ndarray of shape (N, 6).
    """
    required = {"temperature_c", "humidity_pct", "lat", "lon", "day_of_year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"build_features: missing columns {missing}")

    doy_enc = compute_doy_features(df["day_of_year"].values.astype(np.float64))

    X = np.column_stack([
        df["temperature_c"].values,
        df["humidity_pct"].values,
        df["lat"].values,
        df["lon"].values,
        doy_enc,                     # sin_doy, cos_doy
    ]).astype(np.float64)

    if X.shape[1] != INPUT_DIM:
        raise RuntimeError(
            f"build_features produced {X.shape[1]} columns, expected {INPUT_DIM}"
        )

    logger.debug("build_features: produced shape %s", X.shape)
    return X


# ---------------------------------------------------------------------------
# Label matrix
# ---------------------------------------------------------------------------


def build_label_matrix(
    df: pd.DataFrame,
    species_list: List[str],
) -> np.ndarray:
    """
    Build the binary multi-label target matrix Y of shape (N, S).

    Rows correspond to rows in df (presences + background points).
    Columns correspond to species in species_list.
    Y[i, j] = 1 if row i is a presence for species_list[j], else 0.
    Background rows (taxon_name is NaN) are all zeros — they are not
    absences for any specific species, they represent sampling locations.

    Correctness: species_list must be the same ordered list used to build
    the model's output layer.  Passing a different order produces silently
    wrong label assignments.

    Data split: any split of sampled.parquet.
    Returns: float32 ndarray of shape (N, len(species_list)).
    """
    N = len(df)
    S = len(species_list)
    Y = np.zeros((N, S), dtype=np.float32)

    sp_to_idx = {sp: i for i, sp in enumerate(species_list)}
    presence_mask = df["presence"] == 1

    for row_idx, (_, row) in enumerate(df[presence_mask].iterrows()):
        sp = row["taxon_name"]
        if sp in sp_to_idx:
            Y[df.index.get_loc(row.name), sp_to_idx[sp]] = 1.0

    logger.debug(
        "build_label_matrix: shape=%s  positive_entries=%d",
        Y.shape, int(Y.sum()),
    )
    return Y


def build_label_matrix_fast(
    df: pd.DataFrame,
    species_list: List[str],
) -> np.ndarray:
    """
    Vectorised version of build_label_matrix — preferred for large datasets.

    Same contract as build_label_matrix but avoids the Python-level row loop.

    Data split: any split of sampled.parquet.
    Returns: float32 ndarray of shape (N, len(species_list)).
    """
    sp_to_idx = {sp: i for i, sp in enumerate(species_list)}
    Y = np.zeros((len(df), len(species_list)), dtype=np.float32)

    presence_rows = df[df["presence"] == 1].copy()
    # Map taxon_name to column index; unknown species map to -1 (ignored)
    col_idx = presence_rows["taxon_name"].map(sp_to_idx).fillna(-1).astype(int)
    valid = col_idx >= 0
    row_positions = np.where(df["presence"] == 1)[0]

    for pos, col in zip(row_positions[valid.values], col_idx[valid].values):
        Y[pos, col] = 1.0

    logger.debug(
        "build_label_matrix_fast: shape=%s  positive_entries=%d",
        Y.shape, int(Y.sum()),
    )
    return Y


# ---------------------------------------------------------------------------
# Scaler
# ---------------------------------------------------------------------------


class SDMScaler:
    """
    StandardScaler wrapper that structurally prevents data leakage.

    Raises RuntimeError on transform() if fit() has not been called, making
    it impossible to accidentally scale validation or test data with
    statistics derived from those splits.

    Usage pattern (enforced by design):
        scaler = build_scaler()          # unfitted
        scaler.fit(X_train)              # fit on training split ONLY
        X_train_s = scaler.transform(X_train)
        X_val_s   = scaler.transform(X_val)   # safe — uses train statistics
        X_test_s  = scaler.transform(X_test)  # safe

    NEVER call:
        scaler.fit(X_full)               # leaks val/test statistics
        scaler.fit_transform(X_val)      # fits on val — leakage

    Correctness: fit() must receive only the training split.  This class
    cannot verify that programmatically — it is the caller's responsibility.
    The docstring is the enforcement contract.
    """

    def __init__(self) -> None:
        self._scaler: StandardScaler = StandardScaler()
        self._fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "SDMScaler":
        """
        Fit the scaler on X_train.

        Data split: TRAINING DATA ONLY.  Calling this on validation or test
        data constitutes data leakage and will invalidate model evaluation.
        Returns self for chaining.
        """
        if X_train.shape[1] != INPUT_DIM:
            raise ValueError(
                f"SDMScaler.fit: expected {INPUT_DIM} features, got {X_train.shape[1]}"
            )
        self._scaler.fit(X_train)
        self._fitted = True
        logger.info(
            "SDMScaler fitted on %d samples.  "
            "Means: %s  Stds: %s",
            len(X_train),
            np.round(self._scaler.mean_, 3).tolist(),
            np.round(self._scaler.scale_, 3).tolist(),
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale X using statistics from the training split.

        Data split: any split — but fit() MUST have been called on training
        data before this is called on any split.
        Raises RuntimeError if called before fit().
        """
        if not self._fitted:
            raise RuntimeError(
                "SDMScaler.transform called before fit().  "
                "Call scaler.fit(X_train) first — on training data only."
            )
        return self._scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Convenience: fit then transform in one call.

        Data split: TRAINING DATA ONLY.  This is a thin wrapper around
        fit() + transform() and carries the same leakage risk — only call
        on training data.
        """
        return self.fit(X_train).transform(X_train)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Reverse the scaling.  Requires fit() to have been called.
        Data split: any (typically used on model outputs for interpretability).
        """
        if not self._fitted:
            raise RuntimeError("SDMScaler.inverse_transform called before fit().")
        return self._scaler.inverse_transform(X_scaled)

    @property
    def is_fitted(self) -> bool:
        """True if fit() has been successfully called."""
        return self._fitted

    @property
    def mean_(self) -> np.ndarray:
        """Per-feature means from training data.  Requires fit()."""
        if not self._fitted:
            raise RuntimeError("SDMScaler not yet fitted.")
        return self._scaler.mean_

    @property
    def scale_(self) -> np.ndarray:
        """Per-feature standard deviations from training data.  Requires fit()."""
        if not self._fitted:
            raise RuntimeError("SDMScaler not yet fitted.")
        return self._scaler.scale_

    def save(self, path: Path = SCALER_PATH) -> None:
        """
        Persist the fitted scaler to disk as a pickle file.

        Correctness: call this ONLY after final training is complete (Stage 4
        train() method).  Do NOT call inside cross-validation loops — each
        fold uses a throw-away scaler that should not overwrite the final one.
        Raises RuntimeError if called on an unfitted scaler.
        """
        if not self._fitted:
            raise RuntimeError(
                "SDMScaler.save called on an unfitted scaler.  "
                "Fit the scaler on training data before saving."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Saved SDMScaler → %s", path)

    @classmethod
    def load(cls, path: Path = SCALER_PATH) -> "SDMScaler":
        """
        Load a previously saved SDMScaler from disk.

        Data split: loaded scaler retains the statistics from whatever
        training split it was fitted on — use only with the corresponding
        model.
        """
        with open(path, "rb") as f:
            scaler = pickle.load(f)
        if not isinstance(scaler, cls):
            raise TypeError(f"Expected SDMScaler, got {type(scaler)}")
        if not scaler._fitted:
            raise RuntimeError("Loaded scaler is not fitted — file may be corrupt.")
        logger.info("Loaded SDMScaler from %s", path)
        return scaler


def build_scaler() -> SDMScaler:
    """
    Return a fresh, unfitted SDMScaler.

    This factory function is the intended entry point.  Returning an unfitted
    scaler forces downstream code to call fit(X_train) explicitly, making the
    training-data-only contract visible at every call site.

    Data split: N/A — returns an unfitted object.
    """
    return SDMScaler()


# ---------------------------------------------------------------------------
# Species label utilities
# ---------------------------------------------------------------------------


def get_species_list(df: pd.DataFrame) -> List[str]:
    """
    Return a sorted list of unique species names from the presence rows.

    Sorted alphabetically for determinism across runs.
    Data split: sampled.parquet (or any split thereof that contains presences).
    """
    return sorted(df.loc[df["presence"] == 1, "taxon_name"].dropna().unique().tolist())


def build_species_metadata(df: pd.DataFrame, species_list: List[str]) -> dict:
    """
    Build a {taxon_name: {common, iconic_taxon}} metadata dict for the API.

    Reads common_name and iconic_taxon_name from the first presence row of
    each species.  Used by Stage 4 to write models/species_metadata.json.

    Data split: sampled.parquet presence rows — common_name and
                iconic_taxon_name must be present (added in Stage 1).
    """
    meta = {}
    presence_rows = df[df["presence"] == 1].dropna(subset=["taxon_name"])
    for sp in species_list:
        rows = presence_rows[presence_rows["taxon_name"] == sp]
        if rows.empty:
            meta[sp] = {"common": sp, "iconic_taxon": "Unknown"}
            continue
        row = rows.iloc[0]
        meta[sp] = {
            "common": row.get("common_name", sp) or sp,
            "iconic_taxon": row.get("iconic_taxon_name", "Unknown") or "Unknown",
        }
    return meta


# ---------------------------------------------------------------------------
# Smoke-test / CLI
# ---------------------------------------------------------------------------


def _smoke_test(sampled_path: Path = SAMPLED_PATH) -> None:
    """Quick sanity check: build features, fit scaler, verify shapes."""
    logger.info("Running features.py smoke test…")
    df = pd.read_parquet(sampled_path)

    X = build_features(df)
    logger.info("build_features: X.shape=%s  dtype=%s", X.shape, X.dtype)
    assert X.shape == (len(df), INPUT_DIM), "Shape mismatch"
    assert not np.any(np.isnan(X)), "NaN in feature matrix"

    # Scaler leakage test — transform before fit must raise
    scaler = build_scaler()
    try:
        scaler.transform(X[:5])
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError:
        pass  # expected
    logger.info("Leakage guard: transform-before-fit correctly raises RuntimeError")

    # Fit on first 80 %, transform all
    n_train = int(0.8 * len(X))
    X_scaled = scaler.fit(X[:n_train]).transform(X)
    logger.info("Scaler fitted and applied: X_scaled.shape=%s", X_scaled.shape)
    assert np.allclose(X_scaled[:n_train].mean(axis=0), 0.0, atol=1e-6), \
        "Training mean not ~0 after scaling"

    # Label matrix
    species = get_species_list(df)
    Y = build_label_matrix_fast(df, species)
    logger.info(
        "Label matrix: shape=%s  positives=%d  species=%d",
        Y.shape, int(Y.sum()), len(species),
    )
    assert Y.shape == (len(df), len(species))
    assert Y.sum() == (df["presence"] == 1).sum(), \
        "Label matrix positives != presence count"

    logger.info("Smoke test PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    _smoke_test()
