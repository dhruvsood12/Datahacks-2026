"""
api/climatology.py — singleton sensor climatology loader for the API.

Loads heat_map.csv once and builds the per-location averaged climatology
that powers both /heatmap_climatology and the per-cell climate injection
into Predictor.predict_grid.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from pipeline.ingest import build_sensor_climatology, load_heat_map

logger = logging.getLogger(__name__)

_CLIMATOLOGY: Optional[pd.DataFrame] = None


def load_climatology() -> pd.DataFrame:
    """Return the cached sensor climatology DataFrame, building it on first call."""
    global _CLIMATOLOGY
    if _CLIMATOLOGY is None:
        logger.info("Building sensor climatology from heat_map.csv …")
        sensors = load_heat_map()
        _CLIMATOLOGY = build_sensor_climatology(sensors).reset_index(drop=True)
        logger.info("Climatology cached: %d cells", len(_CLIMATOLOGY))
    return _CLIMATOLOGY


def climatology_bounds() -> dict:
    """Return the spatial extent of the climatology — used for the demo bbox."""
    df = load_climatology()
    return {
        "lat_min": float(df["lat"].min()),
        "lat_max": float(df["lat"].max()),
        "lon_min": float(df["lon"].min()),
        "lon_max": float(df["lon"].max()),
        "n_cells": int(len(df)),
        "temp_min": float(df["temperature_c"].min()),
        "temp_max": float(df["temperature_c"].max()),
    }
