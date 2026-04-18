"""
pipeline/inference.py — Stage 6: Counterfactual inference interface.

Exposes a `Predictor` class that your partner's FastAPI backend imports
directly.  The public surface matches the agreed API contract:

    POST /predict       →  predictor.predict(observations)
    POST /predict_grid  →  predictor.predict_grid(lat_min, lat_max,
                                                   lon_min, lon_max,
                                                   day_of_year,
                                                   temperature_offset,
                                                   n_lat, n_lon,
                                                   species_filter)
    GET  /species       →  predictor.list_species()
    GET  /health        →  predictor.health()

Counterfactual warming
──────────────────────
`temperature_offset` (float, default 0.0) shifts `temperature_c` by that
many degrees before scaling and inference.  Other features are unchanged.
This is the climate counterfactual: hold humidity / geography / season fixed,
ask what happens if temperature increases by +1, +2, or +3 °C.

Correctness notes
─────────────────
- The scaler is loaded from disk and applied to every input.  temperature_c
  is shifted BEFORE scaling (shift in original units → correct mean-centre).
- The model predicts raw logits; sigmoid is applied inside predict_proba().
- Species with spatial CV AUC < MIN_SPECIES_AUC are still predicted but
  flagged as low_confidence in the response.  Callers can filter them.
- Temperature, humidity, lat, lon, and day_of_year must be provided for
  every observation.  Missing values raise ValueError immediately rather
  than propagating NaN silently through the model.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    SCALER_PATH,
    MODEL_PATH,
    SPECIES_LABELS_PATH,
    MIN_SPECIES_AUC,
)
from pipeline.features import SDMScaler, build_scaler
from pipeline.model import SDMModel, load_species_labels

logger = logging.getLogger(__name__)

_METADATA_PATH        = MODELS_DIR / "species_metadata.json"
_FILTERED_LABELS_PATH = MODELS_DIR / "species_labels_filtered.json"
_CV_AUC_PATH          = MODELS_DIR / "species_auc_spatial_cv.json"


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------


class Predictor:
    """
    End-to-end inference wrapper.  Loads all artefacts on construction and
    exposes three public methods matching the API contract.

    Parameters
    ──────────
    model_path    : path to SDMModel .pt checkpoint
    scaler_path   : path to SDMScaler .pkl
    labels_path   : path to species_labels.json (full ordered list)
    metadata_path : path to species_metadata.json {taxon:{common,iconic_taxon}}
    filtered_labels_path : path to species_labels_filtered.json (post-CV)
    cv_auc_path   : path to species_auc_spatial_cv.json
    device        : 'cuda' | 'cpu' | None (auto-detect)

    Usage
    ─────
        predictor = Predictor.load()   # load all artefacts from default paths
        result = predictor.predict(observations)
        grid   = predictor.predict_grid(lat_min=32.7, lat_max=33.0, ...)
        sp     = predictor.list_species()
    """

    def __init__(
        self,
        model: SDMModel,
        scaler: SDMScaler,
        species_list: List[str],
        metadata: Dict[str, Dict],
        filtered_species: Optional[List[str]] = None,
        cv_auc: Optional[Dict[str, float]] = None,
    ) -> None:
        self._model = model
        self._scaler = scaler
        self._species_list = species_list
        self._metadata = metadata
        self._filtered_set = set(filtered_species) if filtered_species else set(species_list)
        self._cv_auc = cv_auc or {}
        self._loaded_at = datetime.now(timezone.utc).isoformat()

        logger.info(
            "Predictor ready: %d species (%d with high spatial CV AUC)",
            len(species_list), len(self._filtered_set),
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        model_path: Path = MODEL_PATH,
        scaler_path: Path = SCALER_PATH,
        labels_path: Path = SPECIES_LABELS_PATH,
        metadata_path: Path = _METADATA_PATH,
        filtered_labels_path: Path = _FILTERED_LABELS_PATH,
        cv_auc_path: Path = _CV_AUC_PATH,
        device: Optional[str] = None,
    ) -> "Predictor":
        """Load all model artefacts from disk and return a ready Predictor."""
        logger.info("Loading SDM artefacts…")

        model   = SDMModel.load(model_path, device=device)
        scaler  = SDMScaler.load(scaler_path)
        species = load_species_labels(labels_path)

        with open(metadata_path) as f:
            metadata = json.load(f)

        filtered: Optional[List[str]] = None
        if filtered_labels_path.exists():
            with open(filtered_labels_path) as f:
                filtered = json.load(f)
        else:
            logger.warning(
                "species_labels_filtered.json not found — all species treated as high-confidence.  "
                "Run Stage 5 (04_evaluation.ipynb) to generate it."
            )

        cv_auc_map: Dict[str, float] = {}
        if cv_auc_path.exists():
            with open(cv_auc_path) as f:
                raw = json.load(f)
            # raw is {species: [fold_auc, ...]} — compute mean per species
            for sp, folds in raw.items():
                valid = [v for v in folds if v is not None and not (isinstance(v, float) and np.isnan(v))]
                cv_auc_map[sp] = float(np.mean(valid)) if valid else float("nan")

        return cls(model, scaler, species, metadata, filtered, cv_auc_map)

    # ------------------------------------------------------------------
    # Core feature builder
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_obs(obs: Dict[str, Any]) -> None:
        required = {"temperature_c", "humidity_pct", "lat", "lon", "day_of_year"}
        missing = required - set(obs.keys())
        if missing:
            raise ValueError(f"Observation missing required fields: {missing}")
        for field in required:
            v = obs[field]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                raise ValueError(f"Field '{field}' is null/NaN in observation")

    def _build_feature_row(
        self,
        obs: Dict[str, Any],
        temperature_offset: float = 0.0,
    ) -> np.ndarray:
        """
        Build one scaled feature row from an observation dict.

        temperature_offset is applied in original °C units BEFORE scaling.
        """
        doy = float(obs["day_of_year"])
        angle = doy * (2.0 * np.pi / 365.0)

        row = np.array([[
            float(obs["temperature_c"]) + temperature_offset,
            float(obs["humidity_pct"]),
            float(obs["lat"]),
            float(obs["lon"]),
            np.sin(angle),
            np.cos(angle),
        ]], dtype=np.float64)
        return self._scaler.transform(row)[0]

    def _build_feature_batch(
        self,
        observations: List[Dict[str, Any]],
        temperature_offset: float = 0.0,
    ) -> np.ndarray:
        """Build (N, 6) scaled feature matrix from a list of observation dicts."""
        for obs in observations:
            self._validate_obs(obs)

        doy = np.array([float(o["day_of_year"]) for o in observations])
        angle = doy * (2.0 * np.pi / 365.0)

        X = np.column_stack([
            np.array([float(o["temperature_c"]) for o in observations]) + temperature_offset,
            np.array([float(o["humidity_pct"])   for o in observations]),
            np.array([float(o["lat"])            for o in observations]),
            np.array([float(o["lon"])            for o in observations]),
            np.sin(angle),
            np.cos(angle),
        ]).astype(np.float64)

        return self._scaler.transform(X)

    # ------------------------------------------------------------------
    # POST /predict
    # ------------------------------------------------------------------

    def predict(
        self,
        observations: List[Dict[str, Any]],
        temperature_offset: float = 0.0,
        threshold: float = 0.5,
        include_low_confidence: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Predict species presence probabilities for a list of observations.

        Parameters
        ──────────
        observations : list of dicts, each with keys:
            temperature_c (float °C), humidity_pct (float %), lat (float °),
            lon (float °), day_of_year (int 1-365).
            Any extra keys are passed through unchanged.
        temperature_offset : float — added to temperature_c before inference.
        threshold          : binary presence threshold (default 0.5).
        include_low_confidence : if False, exclude species below MIN_SPECIES_AUC.

        Returns
        ───────
        List of result dicts, one per observation:
            {
                "lat": float,
                "lon": float,
                "day_of_year": int,
                "temperature_c": float,        # original (before offset)
                "effective_temperature_c": float,  # after offset
                "predictions": [
                    {
                        "taxon_name": str,
                        "common_name": str,
                        "iconic_taxon": str,
                        "probability": float,
                        "present": bool,
                        "low_confidence": bool,  # spatial CV AUC < threshold
                        "spatial_cv_auc": float | null,
                    },
                    ...
                ]
            }
        """
        if not observations:
            return []

        X_scaled = self._build_feature_batch(observations, temperature_offset)
        probs = self._model.predict_proba(X_scaled)  # (N, S)

        results = []
        for i, obs in enumerate(observations):
            preds = []
            for j, sp in enumerate(self._species_list):
                prob = float(probs[i, j])
                is_low_conf = sp not in self._filtered_set
                if not include_low_confidence and is_low_conf:
                    continue
                cv_auc = self._cv_auc.get(sp)
                preds.append({
                    "taxon_name": sp,
                    "common_name": self._metadata.get(sp, {}).get("common", sp),
                    "iconic_taxon": self._metadata.get(sp, {}).get("iconic_taxon", "Unknown"),
                    "probability": round(prob, 4),
                    "present": prob >= threshold,
                    "low_confidence": is_low_conf,
                    "spatial_cv_auc": round(cv_auc, 3) if cv_auc is not None and not np.isnan(cv_auc) else None,
                })

            # Sort by probability descending
            preds.sort(key=lambda x: -x["probability"])

            results.append({
                "lat": float(obs["lat"]),
                "lon": float(obs["lon"]),
                "day_of_year": int(obs["day_of_year"]),
                "temperature_c": float(obs["temperature_c"]),
                "effective_temperature_c": float(obs["temperature_c"]) + temperature_offset,
                "predictions": preds,
            })

        logger.debug("predict: %d observations → %d results", len(observations), len(results))
        return results

    # ------------------------------------------------------------------
    # POST /predict_grid
    # ------------------------------------------------------------------

    def predict_grid(
        self,
        lat_min: float,
        lat_max: float,
        lon_min: float,
        lon_max: float,
        day_of_year: int,
        temperature_offset: float = 0.0,
        n_lat: int = 20,
        n_lon: int = 20,
        species_filter: Optional[List[str]] = None,
        threshold: float = 0.5,
        include_low_confidence: bool = True,
    ) -> Dict[str, Any]:
        """
        Predict species probabilities over a regular spatial grid.

        Creates an n_lat × n_lon grid of (lat, lon) points, uses the sensor
        climatology mean temperature/humidity of the nearest training cell
        (approximated by the dataset mean ± spatial gradient is not modelled
        here — the scaler is used to produce a valid scaled input), and
        applies temperature_offset as a counterfactual shift.

        Note: grid points use the scaler's mean humidity (no spatial
        interpolation — humidity is not gridded in this pipeline; only
        temperature_offset is meaningful for counterfactual analysis).

        Parameters
        ──────────
        lat_min, lat_max, lon_min, lon_max : bounding box (decimal degrees)
        day_of_year          : int 1-365
        temperature_offset   : float °C added to baseline temperature
        n_lat, n_lon         : grid resolution
        species_filter       : if provided, only return these species
        threshold            : binary presence threshold

        Returns
        ───────
        {
            "grid_meta": {lat_min, lat_max, lon_min, lon_max, n_lat, n_lon,
                           day_of_year, temperature_offset},
            "lats": [float, ...],    # n_lat values
            "lons": [float, ...],    # n_lon values
            "species": [str, ...],   # S' species names in result
            "probabilities": [       # n_lat × n_lon × S' array
                [[float, ...], ...],
                ...
            ],
            "present": [             # same shape, boolean
                [[bool, ...], ...],
                ...
            ]
        }
        """
        lats = np.linspace(lat_min, lat_max, n_lat)
        lons = np.linspace(lon_min, lon_max, n_lon)

        # Use scaler mean for temperature and humidity as baseline
        # (temperature_offset then shifts temp; humidity stays at mean)
        baseline_temp = float(self._scaler.mean_[0])      # temperature_c mean
        baseline_hum  = float(self._scaler.mean_[1])      # humidity_pct mean

        # Build observations for every grid point
        grid_obs = []
        for lat in lats:
            for lon in lons:
                grid_obs.append({
                    "temperature_c": baseline_temp,
                    "humidity_pct": baseline_hum,
                    "lat": float(lat),
                    "lon": float(lon),
                    "day_of_year": int(day_of_year),
                })

        X_scaled = self._build_feature_batch(grid_obs, temperature_offset)
        probs_flat = self._model.predict_proba(X_scaled)  # (n_lat*n_lon, S)

        # Determine output species
        target_species = self._species_list
        if species_filter is not None:
            # Keep only requested species that exist in model
            valid_filter = [sp for sp in species_filter if sp in set(self._species_list)]
            if not valid_filter:
                raise ValueError(
                    f"None of the requested species are in the model: {species_filter}"
                )
            target_species = valid_filter

        if not include_low_confidence:
            target_species = [sp for sp in target_species if sp in self._filtered_set]

        # Column indices for target species
        sp_idx = {sp: j for j, sp in enumerate(self._species_list)}
        col_indices = [sp_idx[sp] for sp in target_species]

        # Reshape to (n_lat, n_lon, S')
        prob_grid = probs_flat[:, col_indices].reshape(n_lat, n_lon, len(target_species))

        # Build output
        probabilities = [
            [
                [round(float(prob_grid[i, k, s_i]), 4) for s_i in range(len(target_species))]
                for k in range(n_lon)
            ]
            for i in range(n_lat)
        ]
        present = [
            [
                [bool(prob_grid[i, k, s_i] >= threshold) for s_i in range(len(target_species))]
                for k in range(n_lon)
            ]
            for i in range(n_lat)
        ]

        logger.debug(
            "predict_grid: %d×%d grid, %d species, offset=%.1f°C",
            n_lat, n_lon, len(target_species), temperature_offset,
        )

        return {
            "grid_meta": {
                "lat_min": lat_min,
                "lat_max": lat_max,
                "lon_min": lon_min,
                "lon_max": lon_max,
                "n_lat": n_lat,
                "n_lon": n_lon,
                "day_of_year": day_of_year,
                "temperature_offset": temperature_offset,
            },
            "lats": lats.tolist(),
            "lons": lons.tolist(),
            "species": target_species,
            "probabilities": probabilities,
            "present": present,
        }

    # ------------------------------------------------------------------
    # GET /species
    # ------------------------------------------------------------------

    def list_species(
        self,
        only_high_confidence: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Return metadata for all (or only high-confidence) species.

        Response matches the API contract:
            [
                {
                    "taxon_name": str,
                    "common_name": str,
                    "iconic_taxon": str,
                    "model_index": int,       # column index in prediction output
                    "high_confidence": bool,  # spatial CV AUC ≥ MIN_SPECIES_AUC
                    "spatial_cv_auc": float | null,
                },
                ...
            ]
        Sorted alphabetically by taxon_name.
        """
        results = []
        for j, sp in enumerate(self._species_list):
            if only_high_confidence and sp not in self._filtered_set:
                continue
            cv_auc = self._cv_auc.get(sp)
            results.append({
                "taxon_name": sp,
                "common_name": self._metadata.get(sp, {}).get("common", sp),
                "iconic_taxon": self._metadata.get(sp, {}).get("iconic_taxon", "Unknown"),
                "model_index": j,
                "high_confidence": sp in self._filtered_set,
                "spatial_cv_auc": (
                    round(cv_auc, 3)
                    if cv_auc is not None and not np.isnan(cv_auc)
                    else None
                ),
            })
        results.sort(key=lambda x: x["taxon_name"])
        return results

    # ------------------------------------------------------------------
    # GET /health
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """
        Return health / metadata dict for the GET /health endpoint.

            {
                "status": "ok",
                "n_species": int,
                "n_high_confidence_species": int,
                "model_input_dim": int,
                "device": str,
                "loaded_at": str (ISO8601 UTC),
            }
        """
        return {
            "status": "ok",
            "n_species": len(self._species_list),
            "n_high_confidence_species": len(self._filtered_set),
            "model_input_dim": self._model.input_dim,
            "device": str(self._model.device),
            "loaded_at": self._loaded_at,
        }


# ---------------------------------------------------------------------------
# Convenience singleton loader
# ---------------------------------------------------------------------------


_predictor_instance: Optional[Predictor] = None


def get_predictor(reload: bool = False, **kwargs) -> Predictor:
    """
    Return a module-level Predictor singleton (load once, reuse everywhere).

    Call `get_predictor(reload=True)` to force a fresh load from disk.
    Pass keyword arguments to override default paths.
    """
    global _predictor_instance
    if _predictor_instance is None or reload:
        _predictor_instance = Predictor.load(**kwargs)
    return _predictor_instance


# ---------------------------------------------------------------------------
# Smoke test / CLI
# ---------------------------------------------------------------------------


def _smoke_test() -> None:
    """Quick round-trip test using saved model artefacts."""
    import pandas as pd
    from config import SAMPLED_PATH

    logger.info("Running inference.py smoke test…")

    predictor = Predictor.load()

    # Health check
    h = predictor.health()
    assert h["status"] == "ok"
    assert h["n_species"] > 0
    logger.info("Health: %s", h)

    # list_species
    sp_list = predictor.list_species()
    assert len(sp_list) == h["n_species"]
    logger.info("Species list: %d entries (first: %s)", len(sp_list), sp_list[0]["taxon_name"])

    # predict — use a few rows from the dataset
    df = pd.read_parquet(SAMPLED_PATH).head(5)
    obs = []
    for _, row in df.iterrows():
        obs.append({
            "temperature_c": float(row["temperature_c"]),
            "humidity_pct": float(row["humidity_pct"]),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "day_of_year": int(row["day_of_year"]),
        })

    results = predictor.predict(obs)
    assert len(results) == 5
    for r in results:
        assert "predictions" in r
        for p in r["predictions"]:
            assert 0.0 <= p["probability"] <= 1.0
    logger.info("predict: %d observations → %d results, %d predictions each",
                len(obs), len(results), len(results[0]["predictions"]))

    # predict with offset
    results_warm = predictor.predict(obs, temperature_offset=2.0)
    assert len(results_warm) == 5
    logger.info("predict (offset +2°C): effective_temp=%s",
                [r["effective_temperature_c"] for r in results_warm])

    # predict_grid
    grid = predictor.predict_grid(
        lat_min=32.85, lat_max=32.89,
        lon_min=-117.26, lon_max=-117.21,
        day_of_year=100,
        temperature_offset=0.0,
        n_lat=5, n_lon=5,
    )
    assert grid["grid_meta"]["n_lat"] == 5
    assert len(grid["lats"]) == 5
    assert len(grid["probabilities"]) == 5
    assert len(grid["probabilities"][0]) == 5
    logger.info("predict_grid: %d species per cell", len(grid["species"]))

    # predict_grid with species filter
    top_sp = [s["taxon_name"] for s in sp_list[:3]]
    grid_filtered = predictor.predict_grid(
        lat_min=32.85, lat_max=32.89,
        lon_min=-117.26, lon_max=-117.21,
        day_of_year=100,
        n_lat=3, n_lon=3,
        species_filter=top_sp,
    )
    assert grid_filtered["species"] == top_sp
    logger.info("predict_grid (filtered 3 species): OK")

    logger.info("Smoke test PASSED")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    _smoke_test()
