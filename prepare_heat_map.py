"""
prepare_heat_map.py — One-time preprocessing script for UCSD heat-map sensor data.

Reads all YYYYMMDD_UCSD_Campus_*.txt files from data/raw/, parses their
proprietary column format, and writes a single data/heat_map.csv that
pipeline/ingest.py expects.

Column format of each .txt file (no header, whitespace-separated):
  0  doy_decimal   decimal day-of-year (e.g. 214.817 = Aug 2 at 19:36 UTC)
  1  lat           decimal degrees (0.0 when GPS not locked)
  2  lon           decimal degrees (0.0 when GPS not locked)
  3  auxiliary     speed or heading — not used
  4  temperature_c degrees Celsius
  5  humidity_pct  percent relative humidity

Year is extracted from the first 4 characters of each filename (e.g. "2025").

Run once before running the Stage 1 notebook:
    python prepare_heat_map.py
"""

import logging
import re
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).parent / "data" / "heat_map.csv"

_COLS = ["doy_decimal", "lat", "lon", "auxiliary", "temperature_c", "humidity_pct"]

# Bounding box used as GPS sanity filter — rejects glitched nonzero coordinates.
# Slightly wider than the SD study area to keep all valid campus readings.
_LAT_MIN, _LAT_MAX = 32.5, 33.3
_LON_MIN, _LON_MAX = -117.6, -116.1


def _doy_to_timestamp(doy_decimal: "pd.Series", year: int) -> "pd.Series":
    """
    Convert a series of decimal day-of-year values to UTC-aware timestamps.

    DOY 1.0 = Jan 1 at midnight UTC.
    DOY 30.003 = Jan 30 at 0.003 * 24h = 4.3 min past midnight UTC.
    """
    year_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    return year_start + pd.to_timedelta(doy_decimal - 1.0, unit="D")


def load_single_file(path: Path) -> pd.DataFrame:
    """
    Load one .txt session file and return a cleaned DataFrame.

    Extracts year from filename prefix (first 4 chars).
    Drops rows where lat == 0 or lon == 0 (GPS not yet locked).
    """
    year_str = path.stem[:4]
    if not year_str.isdigit():
        raise ValueError(f"Cannot parse year from filename: {path.name}")
    year = int(year_str)

    df = pd.read_csv(path, sep=r"\s+", header=None, names=_COLS)

    n_before = len(df)
    # Two-stage GPS filter:
    #   1. Zero lat/lon = GPS not locked yet
    #   2. Outside SD bounding box = GPS glitch (nonzero but wrong coordinates)
    valid = (
        (df["lat"] != 0.0) & (df["lon"] != 0.0)
        & (df["lat"] >= _LAT_MIN) & (df["lat"] <= _LAT_MAX)
        & (df["lon"] >= _LON_MIN) & (df["lon"] <= _LON_MAX)
    )
    df = df[valid].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("  %s: dropped %d rows (no GPS fix or out-of-bounds coords)",
                    path.name, n_dropped)

    df["timestamp"] = _doy_to_timestamp(df["doy_decimal"], year)
    df["source_file"] = path.name

    return df[["timestamp", "lat", "lon", "temperature_c", "humidity_pct", "source_file"]]


def prepare(raw_dir: Path = RAW_DIR, output_path: Path = OUTPUT_PATH) -> pd.DataFrame:
    """
    Concatenate all session files into a single heat_map.csv.

    Sorts by timestamp, drops exact duplicates, and reports summary statistics.
    """
    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {raw_dir}")

    logger.info("Found %d session files in %s", len(txt_files), raw_dir)

    frames = []
    for f in txt_files:
        logger.info("Loading %s ...", f.name)
        try:
            frames.append(load_single_file(f))
        except Exception as e:
            logger.warning("Skipping %s — %s", f.name, e)

    if not frames:
        raise RuntimeError("All files failed to load.")

    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("timestamp", inplace=True)

    n_before = len(combined)
    combined.drop_duplicates(subset=["timestamp", "lat", "lon"], inplace=True)
    n_dupes = n_before - len(combined)
    if n_dupes:
        logger.info("Dropped %d duplicate rows", n_dupes)

    combined.reset_index(drop=True, inplace=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    logger.info("Saved %d rows → %s", len(combined), output_path)
    logger.info("Timestamp range: %s → %s",
                combined["timestamp"].min(), combined["timestamp"].max())
    logger.info("Lat range:       %.4f → %.4f",
                combined["lat"].min(), combined["lat"].max())
    logger.info("Lon range:       %.4f → %.4f",
                combined["lon"].min(), combined["lon"].max())
    logger.info("Temp range:      %.1f°C → %.1f°C",
                combined["temperature_c"].min(), combined["temperature_c"].max())
    logger.info("Humidity range:  %.1f%% → %.1f%%",
                combined["humidity_pct"].min(), combined["humidity_pct"].max())
    logger.info("Session dates:   %s",
                combined["source_file"].unique().tolist())

    return combined


if __name__ == "__main__":
    prepare()
