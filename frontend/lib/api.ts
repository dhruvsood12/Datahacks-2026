// frontend/lib/api.ts — typed client for the local FastAPI backend.

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// ─── Types mirroring the backend response shapes ──────────────────────────
export type Health = {
  status: string;
  n_species: number;
  n_high_confidence_species: number;
  model_input_dim: number;
  device: string;
  loaded_at: string;
};

export type Bounds = {
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  n_cells: number;
  temp_min: number;
  temp_max: number;
};

export type Species = {
  taxon_name: string;
  common_name: string;
  iconic_taxon: string;
  model_index: number;
  high_confidence: boolean;
  spatial_cv_auc: number | null;
};

export type ClimatologyCell = {
  lat: number;
  lon: number;
  temperature_c: number;
  humidity_pct: number;
};

export type ClimatologyResponse = {
  cells: ClimatologyCell[];
  count: number;
  temp_min: number;
  temp_max: number;
};

export type GridCell = {
  lat: number;
  lon: number;
  prob: number;
};

export type PredictGridResponse = {
  species: string;
  common_name: string;
  iconic_taxon: string;
  spatial_cv_auc: number | null;
  high_confidence: boolean;
  day_of_year: number;
  temperature_offset: number;
  n_lat: number;
  n_lon: number;
  cells: GridCell[];
  n_above_threshold: number;
  n_total: number;
};

// ─── Fetchers ─────────────────────────────────────────────────────────────
async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const getHealth = (): Promise<Health> =>
  fetch(`${API_URL}/health`).then((r) => jsonOrThrow<Health>(r));

export const getBounds = (): Promise<Bounds> =>
  fetch(`${API_URL}/bounds`).then((r) => jsonOrThrow<Bounds>(r));

export const getSpecies = (
  onlyHighConfidence = true,
): Promise<{ species: Species[]; count: number }> =>
  fetch(
    `${API_URL}/species?only_high_confidence=${onlyHighConfidence}`,
  ).then((r) => jsonOrThrow(r));

export const getClimatology = (): Promise<ClimatologyResponse> =>
  fetch(`${API_URL}/heatmap_climatology`).then((r) =>
    jsonOrThrow<ClimatologyResponse>(r),
  );

export const postPredictGrid = (req: {
  species: string;
  day_of_year: number;
  temperature_offset: number;
  n_lat?: number;
  n_lon?: number;
}): Promise<PredictGridResponse> =>
  fetch(`${API_URL}/predict_grid`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ n_lat: 40, n_lon: 40, ...req }),
  }).then((r) => jsonOrThrow<PredictGridResponse>(r));
