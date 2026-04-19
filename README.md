# UCSD Climate Shift

A counterfactual species distribution model (SDM) for the UCSD campus, built for **DataHacks 2026** (ML/AI track · Environment, Climate & Energy Sciences).

Drag a slider to warm UCSD by N°C and watch where each species' habitat moves on a real map of campus microclimate.

---

## What this is

We trained a multi-label MLP over **iNaturalist research-grade observations** within UCSD, using per-cell **temperature + humidity** from a 13-session mobile sensor walk as the climate features. The model learns where each species is happy in microclimate space, then we project that surface back onto the map under counterfactual warming.

- **Backend** — FastAPI wrapping a trained PyTorch SDM.
- **Frontend** — Next.js 16 + react-leaflet showing two layers: the sensor climatology and the predicted species probability surface, recomputed live as the warming offset changes.
- **Scope** — UCSD campus only (lat 32.853–32.894, lon -117.257 to -117.212). 1,641 climatology cells. 68 high-confidence species (spatial 5×5 block CV AUC ≥ 0.65).

---

## Run locally

Two terminals.

### 1. Backend (FastAPI on :8000)

```bash
cd repo
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r api/requirements.txt
uvicorn api.main:app --port 8000
```

The first request loads the Predictor + climatology — give it ~3 seconds.

### 2. Frontend (Next.js on :3100)

```bash
cd repo/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npx next dev --port 3100
```

Open <http://localhost:3100>.

---

## Endpoints

| Method | Path                    | Purpose |
|--------|-------------------------|---------|
| GET    | `/health`               | Predictor status + species counts |
| GET    | `/bounds`               | Spatial extent of the climatology |
| GET    | `/species`              | Species sorted by spatial CV AUC desc |
| GET    | `/heatmap_climatology`  | Per-cell sensor temperature + humidity |
| POST   | `/predict_grid`         | Probability grid for one species under a warming offset |

`POST /predict_grid` body:

```json
{
  "species": "Cotinis mutabilis",
  "day_of_year": 196,
  "temperature_offset": 2.0,
  "n_lat": 40,
  "n_lon": 40,
  "threshold": 0.5
}
```

Returns a flat list of `{lat, lon, prob}` cells plus species metadata and `n_above_threshold` / `n_total` counts.

---

## How the warming counterfactual works

`predict_grid` snaps each grid cell to its nearest real climatology cell (KDTree on equirectangular metres, max 500 m), then **adds** `temperature_offset` to the per-cell baseline temperature before scaling and inference. Humidity, lat, lon, and day-of-year are unchanged. So you're holding everything fixed except the warming, which is the whole point of a counterfactual.

If you run with `climatology=None`, the Predictor falls back to the scaler's mean temperature for every cell (for backwards compatibility with old notebooks).

---

## Stack

- **ML** — PyTorch MLP (6 features → 128 → 64 → 32 → N species), BCEWithLogitsLoss with per-species pos_weight, target-group background sampling, spatial 5×5 block cross-validation.
- **API** — FastAPI 0.115, uvicorn.
- **UI** — Next.js 16.2, React 19.2, TypeScript, Tailwind v4, Zustand, react-leaflet, CARTO dark basemap.

---

## Project layout

```
repo/
├── api/                   # FastAPI HTTP layer
│   ├── main.py
│   ├── climatology.py     # singleton sensor climatology loader
│   └── requirements.txt
├── pipeline/              # data ingest → training → inference
│   ├── ingest.py          # sensor matching, target-group BG sampling
│   ├── model.py           # SDM MLP definition
│   ├── train.py
│   └── inference.py       # Predictor.predict_grid (with KDTree climate injection)
├── frontend/              # Next.js v1 visualization
│   ├── app/
│   ├── components/Map.tsx, ControlPanel.tsx
│   └── lib/api.ts, store.ts
├── data/                  # raw + intermediate parquet
├── models/                # scaler.pkl, sdm_model.pt, species_labels.json
├── notebooks/
├── config.py
├── train.py
└── prepare_heat_map.py
```

---

## Authors

Dhruv Sood · Krishang — DataHacks 2026, UCSD.
