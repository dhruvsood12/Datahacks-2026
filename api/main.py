"""
api/main.py — FastAPI backend for the Biosphere UCSD frontend.

Exposes four endpoints that exactly match the frontend contract in api.ts:

    GET  /health               → { ok: true }
    GET  /species              → Species[]
    GET  /heatmap_climatology  → ClimateCell[]
    POST /predict_grid         → SuitabilityCell[]

All longitude fields use "lng" (not "lon") to match the frontend types.

Start with:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Make sure project root is on sys.path when run directly
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import HEAT_MAP_PATH, MODELS_DIR
from pipeline.ingest import build_sensor_climatology, load_heat_map
from pipeline.inference import Predictor

logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — must match frontend ucsdData.ts UCSD_BOUNDS
# ---------------------------------------------------------------------------
BOUNDS = dict(west=-117.245, east=-117.220, south=32.873, north=32.892)
GRID_STEPS = 28          # 28×28 = 784 cells, matches frontend buildGrid(28)
BASELINE_DOY = 150       # day used to precompute per-species baseline / sensitivity

# Maps iNaturalist iconic_taxon_name → frontend category enum
ICONIC_TO_CATEGORY: dict[str, str] = {
    "Aves":           "bird",
    "Mammalia":       "mammal",
    "Reptilia":       "reptile",
    "Amphibia":       "reptile",
    "Plantae":        "plant",
    "Insecta":        "insect",
    "Arachnida":      "insect",
    "Fungi":          "plant",
    "Chromista":      "plant",
    "Animalia":       "mammal",
    "Protozoa":       "plant",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(taxon_name: str) -> str:
    """'Pinus torreyana' → 'pinus_torreyana'"""
    return taxon_name.lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Biosphere UCSD API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Module-level singletons — populated at startup
_predictor:      Optional[Predictor]  = None
_climate_cells:  Optional[List[dict]] = None
_species_cache:  Optional[List[dict]] = None
_slug_to_taxon:  Optional[dict]       = None


# ---------------------------------------------------------------------------
# Startup: load everything once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup() -> None:
    global _predictor, _climate_cells, _species_cache, _slug_to_taxon

    logger.info("Loading model artefacts…")
    _predictor = Predictor.load()

    logger.info("Building climatology cells…")
    _climate_cells = _build_climate_cells()

    logger.info("Building species metadata cache (runs 2 grid predictions)…")
    _species_cache, _slug_to_taxon = _build_species_cache()

    logger.info("Startup complete — %d species, %d climate cells",
                len(_species_cache), len(_climate_cells))


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _build_climate_cells() -> List[dict]:
    """
    Load sensor climatology, filter to UCSD bounds, normalise temperature
    to [0, 1] (cool=0, warm=1), return as ClimateCell list.
    Value 1.0 = warmest campus microclimate (exposed ridge / parking lot).
    Value 0.0 = coolest (canyon bottom / coast-facing slope).
    """
    sensor_df   = load_heat_map(HEAT_MAP_PATH)
    climatology = build_sensor_climatology(sensor_df)

    buf  = 0.008   # small buffer so edge cells are included
    mask = (
        (climatology["lat"] >= BOUNDS["south"] - buf) &
        (climatology["lat"] <= BOUNDS["north"] + buf) &
        (climatology["lon"] >= BOUNDS["west"]  - buf) &
        (climatology["lon"] <= BOUNDS["east"]  + buf)
    )
    clim = climatology[mask].copy()
    if clim.empty:
        logger.warning("No climatology cells within UCSD bounds — using full extent")
        clim = climatology.copy()

    t_min, t_max = clim["temperature_c"].min(), clim["temperature_c"].max()
    if t_max > t_min:
        clim = clim.assign(value=(clim["temperature_c"] - t_min) / (t_max - t_min))
    else:
        clim = clim.assign(value=0.5)

    return [
        {"lat": round(float(r["lat"]), 6),
         "lng": round(float(r["lon"]), 6),      # frontend uses "lng"
         "value": round(float(r["value"]), 4)}
        for _, r in clim.iterrows()
    ]


def _build_species_cache() -> tuple[List[dict], dict]:
    """
    Run two 28×28 predict_grid calls (0 °C and +1 °C) to compute per-species
    baseline probability and warming sensitivity, then join with metadata.
    """
    meta_path    = MODELS_DIR / "species_metadata.json"
    cv_auc_path  = MODELS_DIR / "species_auc_spatial_cv.json"
    filt_path    = MODELS_DIR / "species_labels_filtered.json"

    with open(meta_path) as f:
        meta = json.load(f)

    # Spatial CV AUC → per-species confidence score
    cv_auc_map: dict[str, float] = {}
    if cv_auc_path.exists():
        with open(cv_auc_path) as f:
            raw_cv = json.load(f)
        for sp, folds in raw_cv.items():
            valid = [v for v in folds if v is not None]
            cv_auc_map[sp] = float(np.mean(valid)) if valid else 0.70

    filtered_set: set[str] = set()
    if filt_path.exists():
        with open(filt_path) as f:
            filtered_set = set(json.load(f))

    # ── predict_grid at 0 °C and +1 °C ────────────────────────────────
    common_kwargs = dict(
        lat_min=BOUNDS["south"], lat_max=BOUNDS["north"],
        lon_min=BOUNDS["west"],  lon_max=BOUNDS["east"],
        day_of_year=BASELINE_DOY,
        n_lat=GRID_STEPS, n_lon=GRID_STEPS,
    )
    grid_base = _predictor.predict_grid(temperature_offset=0.0, **common_kwargs)
    grid_warm = _predictor.predict_grid(temperature_offset=1.0, **common_kwargs)

    species_names = grid_base["species"]                        # ordered list (S,)
    probs_base    = np.array(grid_base["probabilities"])        # (n_lat, n_lon, S)
    probs_warm    = np.array(grid_warm["probabilities"])

    mean_base = probs_base.mean(axis=(0, 1))                    # (S,)
    sensitivity  = probs_warm.mean(axis=(0, 1)) - mean_base     # (S,)

    cache: List[dict] = []
    slug_to_taxon: dict[str, str] = {}

    for idx, taxon in enumerate(species_names):
        m        = meta.get(taxon, {})
        common   = m.get("common", taxon)
        iconic   = m.get("iconic_taxon", "Unknown")
        category = ICONIC_TO_CATEGORY.get(iconic, "plant")

        # Confidence: spatial CV AUC clamped to [0.50, 0.99]
        conf = float(np.clip(cv_auc_map.get(taxon, 0.75), 0.50, 0.99))
        # If species didn't pass spatial CV threshold, cap at 0.70
        if taxon not in filtered_set and filtered_set:
            conf = min(conf, 0.70)

        slug = _slug(taxon)
        slug_to_taxon[slug] = taxon

        cache.append({
            "id":             slug,
            "commonName":     common,
            "scientificName": taxon,
            "category":       category,
            "baseline":       round(float(mean_base[idx]), 4),
            "sensitivity":    round(float(sensitivity[idx]), 4),
            "confidence":     round(conf, 4),
        })

    cache.sort(key=lambda x: x["commonName"])
    return cache, slug_to_taxon


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class PredictGridRequest(BaseModel):
    species_id:  str
    warming_c:   float
    day_of_year: int


class SuitabilityCell(BaseModel):
    lat:        float
    lng:        float
    suitability: float
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    """GET /health — { ok: true }"""
    return {"ok": True}


@app.get("/species")
def list_species() -> List[dict]:
    """GET /species — Species[] matching frontend interface."""
    return _species_cache


@app.get("/heatmap_climatology")
def heatmap_climatology() -> List[dict]:
    """GET /heatmap_climatology — ClimateCell[] for the real-climate heatmap layer."""
    return _climate_cells


@app.post("/predict_grid", response_model=List[SuitabilityCell])
def predict_grid(req: PredictGridRequest) -> List[SuitabilityCell]:
    """
    POST /predict_grid — run the SDM for one species at one warming scenario.

    Body: { species_id: str, warming_c: float, day_of_year: int }
    Returns a flat list of SuitabilityCell covering the 28×28 UCSD grid.

    confidence per cell:
        base_confidence (spatial CV AUC) reduced slightly with warming amount,
        clamped to [0.30, 0.99].  Mirrors the mock formula in ucsdData.ts.
    """
    taxon = _slug_to_taxon.get(req.species_id)
    if taxon is None:
        raise HTTPException(status_code=404,
                            detail=f"Species '{req.species_id}' not found. "
                                   f"Call GET /species for valid ids.")

    grid = _predictor.predict_grid(
        lat_min=BOUNDS["south"],
        lat_max=BOUNDS["north"],
        lon_min=BOUNDS["west"],
        lon_max=BOUNDS["east"],
        day_of_year=req.day_of_year,
        temperature_offset=req.warming_c,
        n_lat=GRID_STEPS,
        n_lon=GRID_STEPS,
        species_filter=[taxon],
        include_low_confidence=True,
    )

    lats  = grid["lats"]                            # (n_lat,)
    lons  = grid["lons"]                            # (n_lon,)
    probs = np.array(grid["probabilities"])          # (n_lat, n_lon, 1)

    # Per-species base confidence, reduced by warming (matches mock formula)
    sp_meta    = next((s for s in _species_cache if s["id"] == req.species_id), None)
    base_conf  = sp_meta["confidence"] if sp_meta else 0.75
    cell_conf  = float(np.clip(base_conf - abs(req.warming_c) * 0.04, 0.30, 0.99))

    cells: List[SuitabilityCell] = []
    for i, lat in enumerate(lats):
        for k, lon in enumerate(lons):
            cells.append(SuitabilityCell(
                lat=round(float(lat), 6),
                lng=round(float(lon), 6),           # "lng" not "lon"
                suitability=round(float(probs[i, k, 0]), 4),
                confidence=cell_conf,
            ))

    return cells
