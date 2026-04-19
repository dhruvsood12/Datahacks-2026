"""
api/main.py — FastAPI HTTP layer wrapping the trained SDM.

Run locally:
    uvicorn api.main:app --reload --port 8000

Endpoints
─────────
  GET  /health                — model status + counts
  GET  /species               — sorted species metadata + spatial CV AUC
  GET  /heatmap_climatology   — raw per-cell sensor temperature/humidity
  POST /predict_grid          — species probability grid w/ per-cell climate
                                + counterfactual warming offset
  GET  /bounds                — spatial extent of the climatology

The Predictor and climatology are loaded once at startup and reused on
every request.  The map bounding box is locked server-side to the actual
extent of the UCSD sensor data so callers can't ask for predictions in
regions the model never saw.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from api.climatology import climatology_bounds, load_climatology
from pipeline.inference import Predictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("api")

# ──────────────────────────────────────────────────────────────────────────
# App + CORS
# ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="UCSD Climate Shift API",
    description=(
        "Counterfactual species distribution model for the UCSD campus.  "
        "Trained on iNaturalist research-grade observations + a 13-session "
        "mobile temperature/humidity sensor walk."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    # Any localhost port — the demo runs on whatever port is free.
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────
# Singletons (loaded at startup)
# ──────────────────────────────────────────────────────────────────────────
_PREDICTOR: Optional[Predictor] = None


@app.on_event("startup")
def _warm_up() -> None:
    global _PREDICTOR
    logger.info("Warming up: loading Predictor + climatology …")
    _PREDICTOR = Predictor.load()
    load_climatology()
    logger.info("Ready.")


def _predictor() -> Predictor:
    if _PREDICTOR is None:
        raise HTTPException(503, "Predictor not yet loaded")
    return _PREDICTOR


# ──────────────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────────────
class PredictGridRequest(BaseModel):
    species: str = Field(..., description="Scientific (taxon) name to predict")
    day_of_year: int = Field(196, ge=1, le=366, description="Day of year, 1-366")
    temperature_offset: float = Field(
        0.0, ge=-5.0, le=10.0,
        description="Counterfactual °C to add to baseline temperature",
    )
    n_lat: int = Field(40, ge=5, le=80, description="Grid rows")
    n_lon: int = Field(40, ge=5, le=80, description="Grid cols")
    threshold: float = Field(0.5, ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return _predictor().health()


@app.get("/bounds")
def bounds() -> dict:
    return climatology_bounds()


@app.get("/species")
def species(only_high_confidence: bool = False) -> dict:
    """Return species sorted by spatial CV AUC desc (NaN AUCs at the bottom)."""
    raw = _predictor().list_species(only_high_confidence=only_high_confidence)

    def _key(s: dict) -> tuple:
        auc = s.get("spatial_cv_auc")
        # Sort: high AUC first, then None at end
        return (auc is None, -(auc or 0.0))

    raw.sort(key=_key)
    return {"species": raw, "count": len(raw)}


@app.get("/heatmap_climatology")
def heatmap_climatology() -> dict:
    """Return per-cell sensor climatology as a flat list."""
    df = load_climatology()
    return {
        "cells": [
            {
                "lat": float(r.lat),
                "lon": float(r.lon),
                "temperature_c": float(r.temperature_c),
                "humidity_pct": float(r.humidity_pct),
            }
            for r in df.itertuples(index=False)
        ],
        "count": int(len(df)),
        "temp_min": float(df["temperature_c"].min()),
        "temp_max": float(df["temperature_c"].max()),
    }


@app.post("/predict_grid")
def predict_grid(req: PredictGridRequest) -> dict:
    """
    Predict species probability over a spatial grid clamped to the climatology
    extent, with real per-cell temperature + humidity + counterfactual offset.
    """
    predictor = _predictor()

    # Validate species exists
    known = {s["taxon_name"] for s in predictor.list_species()}
    if req.species not in known:
        raise HTTPException(400, f"Unknown species: {req.species}")

    bounds_ = climatology_bounds()
    clim = load_climatology()

    grid = predictor.predict_grid(
        lat_min=bounds_["lat_min"],
        lat_max=bounds_["lat_max"],
        lon_min=bounds_["lon_min"],
        lon_max=bounds_["lon_max"],
        day_of_year=req.day_of_year,
        temperature_offset=req.temperature_offset,
        n_lat=req.n_lat,
        n_lon=req.n_lon,
        species_filter=[req.species],
        threshold=req.threshold,
        include_low_confidence=True,
        climatology=clim,
    )

    # Flatten (lat, lon, prob) so the frontend doesn't have to reshape
    cells = []
    lats = grid["lats"]
    lons = grid["lons"]
    probs = grid["probabilities"]  # n_lat × n_lon × 1
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            p = probs[i][j][0]
            cells.append({"lat": float(lat), "lon": float(lon), "prob": float(p)})

    # Pull species metadata + AUC for the badge
    sp_meta = next((s for s in predictor.list_species() if s["taxon_name"] == req.species), {})

    return {
        "species": req.species,
        "common_name": sp_meta.get("common_name", req.species),
        "iconic_taxon": sp_meta.get("iconic_taxon", "Unknown"),
        "spatial_cv_auc": sp_meta.get("spatial_cv_auc"),
        "high_confidence": sp_meta.get("high_confidence", False),
        "day_of_year": req.day_of_year,
        "temperature_offset": req.temperature_offset,
        "n_lat": req.n_lat,
        "n_lon": req.n_lon,
        "cells": cells,
        "n_above_threshold": sum(1 for c in cells if c["prob"] >= req.threshold),
        "n_total": len(cells),
        "mean_prob": round(sum(c["prob"] for c in cells) / len(cells), 4) if cells else 0.0,
    }
