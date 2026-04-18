"""
config.py — Central configuration for the San Diego SDM pipeline.

All file paths are resolved relative to the project root (directory containing
this file). Import these constants rather than hardcoding paths in pipeline modules.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
PIPELINE_DIR = PROJECT_ROOT / "pipeline"

# ---------------------------------------------------------------------------
# Input data  (place raw CSV files here before running Stage 1)
# ---------------------------------------------------------------------------
HEAT_MAP_PATH = DATA_DIR / "heat_map.csv"
INAT_PATH = DATA_DIR / "inaturalist.csv"

# ---------------------------------------------------------------------------
# Stage outputs (written by pipeline, read by downstream stages)
# ---------------------------------------------------------------------------
ALIGNED_PATH = DATA_DIR / "aligned.parquet"    # Stage 1
SAMPLED_PATH = DATA_DIR / "sampled.parquet"    # Stage 2

# ---------------------------------------------------------------------------
# Model artefacts
# ---------------------------------------------------------------------------
SCALER_PATH = MODELS_DIR / "scaler.pkl"
MODEL_PATH = MODELS_DIR / "sdm_model.pt"
SPECIES_LABELS_PATH = MODELS_DIR / "species_labels.json"

# ---------------------------------------------------------------------------
# San Diego county bounding box
# ---------------------------------------------------------------------------
SD_LAT_MIN: float = 32.5
SD_LAT_MAX: float = 33.3
SD_LON_MIN: float = -117.6
SD_LON_MAX: float = -116.1
# Reference centre used for equirectangular projection (Stage 1 spatial matching)
SD_LAT_CENTER: float = (SD_LAT_MIN + SD_LAT_MAX) / 2   # 32.90 °N
SD_LON_CENTER: float = (SD_LON_MIN + SD_LON_MAX) / 2   # -116.85 °W

# ---------------------------------------------------------------------------
# Stage 1 — sensor matching thresholds
# ---------------------------------------------------------------------------
MAX_DIST_M: float = 500.0    # maximum spatial gap between observation and nearest sensor cell (metres)

# Sensor climatology grid resolution.  Each unique (lat, lon) pair is snapped to the
# nearest grid node; temperature and humidity are averaged across all sessions at that
# node.  0.0005° ≈ 55 m — fine enough to preserve the campus microclimate gradient.
SENSOR_GRID_DEG: float = 0.0005

# ---------------------------------------------------------------------------
# Stage 1 / 2 — species presence thresholds
# ---------------------------------------------------------------------------
MIN_PRESENCE_COUNT: int = 30   # discard species with fewer matched presence records
# Standard SDM practice uses 0.01° (~1.1 km), but our study area is only
# ~4.8 km × 4.8 km — at 0.01° the whole area has ~8 cells and thinning
# destroys almost all records.  0.0005° (~56 m) matches the sensor
# climatology grid: two observations in the same cell get identical features,
# so keeping one is both autocorrelation-correct and information-lossless.
GRID_CELL_DEG: float = 0.0005  # spatial thinning cell size (degrees ≈ 56 m)

# ---------------------------------------------------------------------------
# Stage 4 — MLP hyperparameters
# ---------------------------------------------------------------------------
INPUT_DIM: int = 6
HIDDEN_DIMS: list[int] = [128, 64, 32]
DROPOUT: float = 0.3
LEARNING_RATE: float = 1e-3
PATIENCE: int = 10
MIN_DELTA: float = 1e-4
BATCH_SIZE: int = 256
MAX_EPOCHS: int = 200

# ---------------------------------------------------------------------------
# Stage 5 — spatial cross-validation
# ---------------------------------------------------------------------------
CV_GRID_SIZE: int = 5         # n × n geographic block grid  → up to 25 folds
MIN_SPECIES_AUC: float = 0.65 # exclude species below this spatial CV AUC from final model
