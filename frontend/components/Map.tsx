// frontend/components/Map.tsx — Leaflet map with climatology + species layers.

"use client";

import { useEffect } from "react";
import {
  CircleMarker,
  MapContainer,
  TileLayer,
  Tooltip,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";

import { useStore } from "@/lib/store";

// ─── Color helpers ────────────────────────────────────────────────────────
// Compact viridis-like ramp; takes t in [0,1] → "rgb(...)"
function viridis(t: number): string {
  const stops: [number, [number, number, number]][] = [
    [0.0, [68, 1, 84]],
    [0.25, [59, 82, 139]],
    [0.5, [33, 144, 141]],
    [0.75, [93, 201, 99]],
    [1.0, [253, 231, 37]],
  ];
  const x = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const [t0, c0] = stops[i];
    const [t1, c1] = stops[i + 1];
    if (x >= t0 && x <= t1) {
      const f = (x - t0) / (t1 - t0);
      const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
      const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
      const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
      return `rgb(${r},${g},${b})`;
    }
  }
  return "rgb(253,231,37)";
}

function FitBounds() {
  const map = useMap();
  const climatology = useStore((s) => s.climatology);
  useEffect(() => {
    if (!climatology || climatology.cells.length === 0) return;
    const lats = climatology.cells.map((c) => c.lat);
    const lons = climatology.cells.map((c) => c.lon);
    const sw: [number, number] = [Math.min(...lats), Math.min(...lons)];
    const ne: [number, number] = [Math.max(...lats), Math.max(...lons)];
    map.fitBounds([sw, ne], { padding: [20, 20] });
  }, [climatology, map]);
  return null;
}

export default function Map() {
  const climatology = useStore((s) => s.climatology);
  const currentGrid = useStore((s) => s.currentGrid);
  const showHeat = useStore.getState().climatology !== null;

  // UCSD center fallback
  const center: [number, number] = [32.8745, -117.235];

  const tempMin = climatology?.temp_min ?? 12;
  const tempMax = climatology?.temp_max ?? 23;

  return (
    <MapContainer
      center={center}
      zoom={15}
      scrollWheelZoom
      className="h-full w-full"
      style={{ background: "#0a0a0a" }}
    >
      <TileLayer
        attribution='&copy; <a href="https://carto.com/attributions">CARTO</a> &copy; OSM contributors'
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
      />

      <FitBounds />

      {/* Layer A — sensor climatology base layer (small, dim) */}
      {showHeat &&
        climatology!.cells.map((c, i) => (
          <CircleMarker
            key={`hm-${i}`}
            center={[c.lat, c.lon]}
            radius={3}
            pathOptions={{
              color: viridis((c.temperature_c - tempMin) / (tempMax - tempMin)),
              fillColor: viridis(
                (c.temperature_c - tempMin) / (tempMax - tempMin),
              ),
              fillOpacity: 0.35,
              weight: 0,
            }}
          />
        ))}

      {/* Layer B — species probability surface */}
      {currentGrid &&
        currentGrid.cells.map((c, i) => (
          <CircleMarker
            key={`gp-${i}`}
            center={[c.lat, c.lon]}
            radius={6}
            pathOptions={{
              color: viridis(c.prob),
              fillColor: viridis(c.prob),
              fillOpacity: Math.max(0.15, Math.min(0.85, c.prob)),
              weight: 0,
            }}
          >
            <Tooltip direction="top" offset={[0, -4]} opacity={0.9}>
              <div className="text-xs font-mono">
                p={c.prob.toFixed(3)} · {c.lat.toFixed(4)}, {c.lon.toFixed(4)}
              </div>
            </Tooltip>
          </CircleMarker>
        ))}
    </MapContainer>
  );
}
