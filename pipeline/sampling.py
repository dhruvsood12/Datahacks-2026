"""
pipeline/sampling.py — Stage 2: Bias correction and negative sampling.

Two methodological choices here, both critical for model validity:

1.  Per-species spatial thinning
    Each species' presences are thinned to at most 1 record per GRID_CELL_DEG
    (0.01° ≈ 1.1 km) grid cell.  Thinning is done independently per species —
    if it were done globally, common species would consume all the quota and
    rare species would lose disproportionately more records.

2.  Target-group background sampling
    Background points are drawn from ALL iNaturalist observations in the study
    area (every quality grade, every species) that can be matched to the sensor
    climatology within MAX_DIST_M metres.  This is the standard MaxEnt
    target-group correction: because observers visit certain trails and spots far
    more than others, a uniform random background would teach the model to
    predict observer routes, not species habitat.  Using the iNat observation
    pool as the background makes the null distribution mirror actual sampling
    effort, so the model learns genuine habitat preference.

    Reference: Phillips et al. (2009) "Sample selection bias and
    presence-only species distribution models", Ecography 32:431-441.

Output schema for sampled.parquet:
    lat, lon, observed_on, taxon_name (NaN = background), common_name,
    iconic_taxon_name, temperature_c, humidity_pct, day_of_year, presence (0|1)

Correctness assumptions:
    - aligned.parquet from Stage 1 contains only research-grade presences
      matched to the sensor climatology.
    - Background pool is built from ALL iNat quality grades to maximise
      coverage of observer effort.
    - Spatial thinning uses a fixed random seed for reproducibility.
    - No background point is silently excluded; all drops are logged.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from config import (
    ALIGNED_PATH,
    HEAT_MAP_PATH,
    INAT_PATH,
    MAX_DIST_M,
    MIN_PRESENCE_COUNT,
    SAMPLED_PATH,
    SENSOR_GRID_DEG,
    GRID_CELL_DEG,
    SD_LAT_CENTER,
    SD_LON_CENTER,
)
from pipeline.ingest import (
    load_heat_map,
    build_sensor_climatology,
    _latlon_to_xy_metres,
)

logger = logging.getLogger(__name__)

_RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Per-species spatial thinning
# ---------------------------------------------------------------------------


def spatial_thin(
    df: pd.DataFrame,
    grid_deg: float = GRID_CELL_DEG,
    species_col: str = "taxon_name",
    lat_col: str = "lat",
    lon_col: str = "lon",
    seed: int = _RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Thin presence records to at most 1 per grid cell, independently per species.

    Each (lat, lon) is snapped to the nearest grid_deg node; if multiple records
    of the same species fall in the same cell, one is kept at random (seeded for
    reproducibility) and the rest are dropped.

    Correctness: thinning is per-species.  Global thinning would allow a common
    species to occupy every cell before a rare species gets to use any, which
    biases the thinned dataset toward rarity.

    Data split: aligned.parquet from Stage 1 — presence records only.
    Returns: (thinned_df, {species: n_dropped}) for all species with drops.
    """
    rng = np.random.default_rng(seed)
    thinned_frames = []
    drop_log: Dict[str, int] = {}

    for species, group in df.groupby(species_col):
        group = group.copy()
        group["_cell_lat"] = (group[lat_col] / grid_deg).round() * grid_deg
        group["_cell_lon"] = (group[lon_col] / grid_deg).round() * grid_deg

        # Shuffle so the kept record is random within each cell
        group = group.sample(frac=1, random_state=int(rng.integers(1 << 31)))

        thinned = group.drop_duplicates(subset=["_cell_lat", "_cell_lon"])
        n_dropped = len(group) - len(thinned)
        if n_dropped:
            drop_log[str(species)] = n_dropped

        thinned_frames.append(thinned.drop(columns=["_cell_lat", "_cell_lon"]))

    result = pd.concat(thinned_frames, ignore_index=True)
    total_dropped = sum(drop_log.values())
    logger.info(
        "Spatial thinning: %d → %d records  (%d dropped across %d species)",
        len(df), len(result), total_dropped, len(drop_log),
    )
    return result, drop_log


# ---------------------------------------------------------------------------
# Target-group background pool construction
# ---------------------------------------------------------------------------


def build_background_pool(
    inat_path: Path,
    climatology: pd.DataFrame,
    max_dist_m: float = MAX_DIST_M,
) -> pd.DataFrame:
    """
    Build the target-group background pool from ALL iNaturalist observations.

    Loads the raw iNat CSV including every quality grade (research, needs_id,
    casual), drops rows without coordinates, and spatially matches each
    observation to the nearest sensor climatology cell within max_dist_m metres.
    The result represents observer effort across the study area regardless of
    what was observed.

    Correctness: using ALL quality grades (not just research-grade) is
    intentional — even a casual sighting of a flower records that an observer
    was at that location, which is what we need to model sampling effort.

    Data split: raw iNat CSV (all quality grades) — no prior processing assumed.
    Returns: DataFrame with lat, lon, temperature_c, humidity_pct,
             day_of_year, observed_on, _matched_dist_m.
    """
    logger.info("Building target-group background pool from all iNat observations…")

    # Load all rows, all quality grades
    needed = ["latitude", "longitude", "observed_on", "time_observed_at"]
    df = pd.read_csv(inat_path, usecols=needed)

    n_raw = len(df)
    df = df.dropna(subset=["latitude", "longitude"]).copy()
    logger.info(
        "iNat pool: %d total rows, %d with coordinates",
        n_raw, len(df),
    )

    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df = df.dropna(subset=["observed_on"]).copy()
    df["day_of_year"] = df["observed_on"].dt.day_of_year

    # Spatial match to climatology
    clim_xy = _latlon_to_xy_metres(
        climatology["lat"].values, climatology["lon"].values
    )
    obs_xy = _latlon_to_xy_metres(df["latitude"].values, df["longitude"].values)

    tree = KDTree(clim_xy)
    dists, idx = tree.query(obs_xy, k=1, workers=-1)

    mask = dists <= max_dist_m
    n_no_match = int((~mask).sum())
    logger.info(
        "Background pool spatial match: %d/%d within %.0f m  (%d outside)",
        mask.sum(), len(df), max_dist_m, n_no_match,
    )

    pool = df[mask].copy().reset_index(drop=True)
    matched_clim = climatology.iloc[idx[mask]].reset_index(drop=True)

    pool["lat"]            = pool["latitude"]
    pool["lon"]            = pool["longitude"]
    pool["temperature_c"]  = matched_clim["temperature_c"].values
    pool["humidity_pct"]   = matched_clim["humidity_pct"].values
    pool["_matched_dist_m"] = dists[mask]

    out_cols = ["lat", "lon", "observed_on", "temperature_c",
                "humidity_pct", "day_of_year", "_matched_dist_m"]
    return pool[out_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Background sampling
# ---------------------------------------------------------------------------


def sample_backgrounds(
    pool: pd.DataFrame,
    n_samples: int,
    seed: int = _RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sample n_samples background points from the target-group pool.

    If n_samples >= len(pool), all pool rows are used (logged as a warning
    since the background distribution may under-represent sparse areas).
    Background rows receive presence=0 and NaN for all species-identity columns.

    Data split: output of build_background_pool.
    Returns: DataFrame with the sampled.parquet schema, presence=0.
    """
    if n_samples >= len(pool):
        logger.warning(
            "Requested %d background samples but pool only has %d rows — "
            "using entire pool.  Consider expanding MAX_DIST_M.",
            n_samples, len(pool),
        )
        sampled = pool.copy()
    else:
        sampled = pool.sample(n=n_samples, random_state=seed, replace=False).copy()

    sampled["taxon_name"]       = np.nan
    sampled["common_name"]      = np.nan
    sampled["iconic_taxon_name"] = np.nan
    sampled["presence"]         = 0

    logger.info("Sampled %d background points", len(sampled))
    return sampled


# ---------------------------------------------------------------------------
# Correctness check helpers
# ---------------------------------------------------------------------------


def check_background_coverage(
    presences: pd.DataFrame,
    backgrounds: pd.DataFrame,
    n_species: int = 3,
) -> None:
    """
    Log a warning if the background spatial distribution does not cover the
    presence distribution for representative species.

    For each of n_species representative species, computes the fraction of
    presence records whose nearest background point is within 1 km.  If this
    fraction falls below 80 % for any species, logs a warning to review the
    background pool before proceeding.

    Data split: thinned presences and sampled backgrounds from this stage.
    Correctness assumption: called after sampling — not valid on raw data.
    """
    if presences.empty or backgrounds.empty:
        logger.warning("Cannot run background coverage check — empty input.")
        return

    bg_xy = _latlon_to_xy_metres(
        backgrounds["lat"].values, backgrounds["lon"].values
    )
    bg_tree = KDTree(bg_xy)

    # Pick representative species: highest, median, and lowest record count
    counts = presences["taxon_name"].value_counts()
    indices = [0, len(counts) // 2, -1]
    species_sample = [counts.index[i] for i in indices][:n_species]

    all_ok = True
    for sp in species_sample:
        sp_df = presences[presences["taxon_name"] == sp]
        sp_xy = _latlon_to_xy_metres(sp_df["lat"].values, sp_df["lon"].values)
        dists, _ = bg_tree.query(sp_xy, k=1)
        frac_covered = (dists <= 1000).mean()
        status = "OK" if frac_covered >= 0.80 else "⚠ LOW"
        logger.info(
            "Background coverage for %-40s  %.0f %% within 1 km  [%s]",
            sp, frac_covered * 100, status,
        )
        if frac_covered < 0.80:
            all_ok = False

    if not all_ok:
        logger.warning(
            "Background coverage below 80 %% for at least one species.  "
            "The background pool may not adequately represent observer effort "
            "in areas where those species occur.  Review the spatial plot in "
            "02_sampling.ipynb before proceeding to Stage 3."
        )
    else:
        logger.info("Background coverage check passed for all sampled species.")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_sampling(
    aligned_path: Path = ALIGNED_PATH,
    inat_path: Path = INAT_PATH,
    heat_map_path: Path = HEAT_MAP_PATH,
    output_path: Path = SAMPLED_PATH,
    n_background_multiplier: float = 10.0,
    seed: int = _RANDOM_SEED,
) -> pd.DataFrame:
    """
    Full Stage 2 pipeline: thin presences → build background pool → sample →
    combine → validate → save.

    Steps:
      1. Load aligned.parquet (Stage 1 presences).
      2. Thin each species to 1 presence per GRID_CELL_DEG cell.
      3. Re-check that all species still meet MIN_PRESENCE_COUNT after thinning;
         drop any that fall below and log them.
      4. Rebuild sensor climatology, then build target-group background pool
         from ALL iNat observations matched to the climatology.
      5. Sample n_background_multiplier × total_thinned_presences backgrounds.
      6. Combine presences (presence=1) and backgrounds (presence=0).
      7. Run background coverage correctness check.
      8. Save to output_path as Parquet.

    Data split: aligned.parquet (presences) + raw iNat CSV (background pool).
    Returns: sampled DataFrame.
    """
    logger.info("=== Stage 2: Bias Correction & Negative Sampling ===")

    # 1 — Load presences
    presences = pd.read_parquet(aligned_path)
    logger.info(
        "Loaded %d presence records, %d species",
        len(presences), presences["taxon_name"].nunique(),
    )

    # 2 — Per-species spatial thinning
    thinned, thin_log = spatial_thin(presences, grid_deg=GRID_CELL_DEG, seed=seed)

    # 3 — Re-check MIN_PRESENCE_COUNT after thinning
    counts_after = thinned["taxon_name"].value_counts()
    below_threshold = counts_after[counts_after < MIN_PRESENCE_COUNT].index.tolist()
    if below_threshold:
        logger.warning(
            "%d species fell below %d records after thinning and will be excluded: %s",
            len(below_threshold), MIN_PRESENCE_COUNT, sorted(below_threshold),
        )
        thinned = thinned[~thinned["taxon_name"].isin(below_threshold)].reset_index(drop=True)

    n_presences = len(thinned)
    n_species_final = thinned["taxon_name"].nunique()
    logger.info(
        "After thinning: %d presence records, %d species",
        n_presences, n_species_final,
    )

    # 4 — Build background pool
    sensor_df   = load_heat_map(heat_map_path)
    climatology = build_sensor_climatology(sensor_df)
    bg_pool     = build_background_pool(inat_path, climatology)

    # 5 — Sample backgrounds
    n_bg = int(n_presences * n_background_multiplier)
    backgrounds = sample_backgrounds(bg_pool, n_samples=n_bg, seed=seed)

    # 6 — Combine
    thinned["presence"] = 1
    keep_cols = [
        "lat", "lon", "observed_on", "taxon_name", "common_name",
        "iconic_taxon_name", "temperature_c", "humidity_pct",
        "day_of_year", "presence",
    ]
    presences_out  = thinned[keep_cols]
    backgrounds_out = backgrounds[keep_cols]

    sampled = pd.concat([presences_out, backgrounds_out], ignore_index=True)
    sampled = sampled.sample(frac=1, random_state=seed).reset_index(drop=True)

    logger.info(
        "Combined dataset: %d rows  (%d presences, %d backgrounds)",
        len(sampled),
        (sampled["presence"] == 1).sum(),
        (sampled["presence"] == 0).sum(),
    )

    # 7 — Background coverage check
    check_background_coverage(thinned, backgrounds)

    # 8 — Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_parquet(output_path, index=False)
    logger.info("Saved sampled.parquet → %s", output_path)

    # Summary report
    _sampling_report(thinned, sampled, thin_log, below_threshold)

    return sampled


def _sampling_report(
    thinned: pd.DataFrame,
    sampled: pd.DataFrame,
    thin_log: Dict[str, int],
    dropped_post_thin: list,
) -> None:
    """Print a concise Stage 2 quality summary."""
    n_pres  = (sampled["presence"] == 1).sum()
    n_bg    = (sampled["presence"] == 0).sum()
    ratio   = n_bg / n_pres if n_pres else 0

    sep = "=" * 64
    lines = [
        "",
        sep,
        "  STAGE 2 — SAMPLING QUALITY REPORT",
        sep,
        f"  Species retained after thinning:     {thinned['taxon_name'].nunique():>6}",
        f"  Species dropped post-thinning:       {len(dropped_post_thin):>6}",
        f"  Total presence records (thinned):    {n_pres:>6,}",
        f"  Total background records:            {n_bg:>6,}",
        f"  Background : presence ratio:         {ratio:>6.1f}x",
        f"  Total rows in sampled.parquet:       {len(sampled):>6,}",
        "",
        f"  Species thinned most (top 5 drops):",
    ]
    for sp, n in sorted(thin_log.items(), key=lambda x: -x[1])[:5]:
        lines.append(f"    {n:4d} records removed  {sp}")
    lines.append(sep)
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    run_sampling()
